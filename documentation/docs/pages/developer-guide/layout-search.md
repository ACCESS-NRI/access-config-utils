# How the layout search works

The search asks one question at every component of the tree: *given this component and this many
cores, what are all its valid layouts?* Cores go down the tree, and layouts come back up. This
page describes how that question is answered, the protocol the allocation modes implement, and how
to change the search safely. The module docstrings of
[`parallel_layouts`](../api/layout-search.md) and
[`parallel_allocation_strategies`](../api/allocation-strategies.md) give the reasoning in full.

## The two trees

A search is given two trees of the same shape: the `ParallelComponent` tree, which says what the
model is, and the `AllocationStrategy` tree, which says how a study wants its cores allocated.
They are kept apart so that one model serves many studies, and they are matched by name:
`resolve_root_strategy` checks, before the search starts, that the strategy tree names exactly
the sub-components of the component tree at every level, since there is no default mode. It also
turns every fraction of the total into a number of cores, once, so that the search itself only
ever sees counts.

## The recursion

One call of `iter_layouts` is one `_LayoutSearch`, which holds the two things fixed for the whole
search: the total number of cores, and a memo of the layouts of every sub-tree already worked out.

A **leaf** spends its cores itself. For each thread count in its range that divides its cores, it
runs `n_cores // threads_per_rank` ranks, and for each grid of those ranks its domain can be split
over, there is one candidate layout. A leaf with no domain has one candidate per thread count.
Each candidate is checked against the leaf's local constraints, its own and its strategy's.

A **parent** hands its cores down. Its strategies say every way of dividing them among its
sub-components, through `iter_core_splits`, or, for a parent whose sub-components share its cores,
every combination of counts that fits, through `iter_shared_core_shares`. For each division, the
question is put to every sub-component with its share, and every combination of one layout per
sub-component is a candidate layout of the parent, checked against the parent's group constraints
and then its local ones.

The numbers come from that combining step: two sub-components with 30 layouts each already give
900 combinations for one division. Two things keep it affordable:

- **Rejecting early.** A constraint is checked at its own component, so a layout it rejects is
  dropped before anything is built on it, and a division in which some sub-component has no layout
  is abandoned before its combinations are formed.
- **Asking once.** The layouts of a sub-tree depend only on the component, its cores, its thread
  range and its strategy, which is the memo's key. The atmosphere on 40 cores is asked about by
  every division giving it 40 cores, and worked out the first time only. The total number of cores
  is not in the key, since it is fixed for the life of the memo, which is why a search is not
  reused across totals and the memo is never kept at module level.

Only the layouts of the whole tree are yielded one at a time. Below the root, a sub-tree's layouts
are held as a tuple, since forming the combinations goes through each many times, so the memory a
search needs grows with the sub-tree answers rather than with the number of layouts.

```mermaid
sequenceDiagram
    participant iter_layouts
    participant Search as _LayoutSearch
    participant Strategies as parallel_allocation_strategies
    iter_layouts->>Strategies: resolve_root_strategy: match the names, resolve the fractions
    iter_layouts->>Search: run(root, all the cores)
    loop each component, from the root down
        alt a leaf
            Search->>Search: every usable thread count and process grid
        else a parent
            Search->>Strategies: iter_core_splits or iter_shared_core_shares
            Strategies-->>Search: the counts of each division
            Search->>Search: the layouts of each sub-component on its count, from the memo where it can
            Search->>Search: every combination, filtered by the constraints
        end
    end
    Search-->>iter_layouts: the layouts of the root, one at a time
```

## The allocation modes

The search never names an allocation mode. A mode is a subclass of `AllocationStrategy`, and what
it means is on its own class, behind a small protocol of `_`-prefixed members:

| Member | What it says |
|---|---|
| `_phase` | Where the mode is served among siblings that divide a parent's cores: fixed 0, weighted 1, free 2. It has no default, so a mode that forgets it fails on the first search, by name. |
| `_iter_core_shares(allocs, budget)` | Every way of sizing a group of siblings in this mode from the cores left to them. The cores reserved for the other modes are already taken out of the budget, so a mode never learns the others exist. |
| `_reserved_cores(allocs)` | How many cores the group needs at least, so that earlier phases leave them. 0 by default, which is always correct: it is a hint for pruning, not a rule. |
| `_describe_infeasible(allocs, names, budget)` | Why the group could not be served, for the log. `None` by default. |
| `_shared_core_range(parent_cores)` | The counts the mode offers under a shared parent, given the cores left of the range from the sub-component's offset. `None` by default, which means the mode cannot size such a sub-component. |

`iter_core_splits` is the one function that knows every mode at once: it groups the siblings by
mode, serves the groups in the order of their phases, and gives each the budget the earlier ones
left. It is a function of the sibling group rather than a method of one strategy because weighted
siblings share one multiplier, which no single strategy can choose. It sits beside the modes, in
`parallel_allocation_strategies`, so that `parallel_layouts` imports the modes' module and never
the reverse.

A shared parent does not use the phases at all: nothing is divided, so `iter_shared_core_shares`
asks each sub-component for its `_shared_core_range` and yields the combinations that cover the
range. `WeightedAllocation` and `WholeRangeAllocation` are mirror images: a weight is a share of a
divided budget and means nothing under a shared parent, and a whole range is all of a shared one
and means nothing under a divided parent. Each raises `ValueError` on the path it does not take.

### Adding a mode

A new mode is a new class in `parallel_allocation_strategies`, deriving from `AllocationStrategy`,
which:

1. declares its `_phase`;
2. implements `_iter_core_shares`;
3. overrides `_shared_core_range` if it can size a sub-component of a shared parent, and
   `_reserved_cores` and `_describe_infeasible` if it can help the pruning or the diagnostics;
4. validates its fields in `__post_init__`, as a frozen dataclass, raising `ValueError` naming the
   class and the field.

Nothing in `iter_core_splits` or `parallel_layouts` changes. If a change seems to need
`isinstance(strategy, SomeMode)`, or reading a mode's fields such as `min_cores` or `weight`, in
`parallel_layouts`, the logic belongs on the mode.

## Rules the search relies on

- Every public type is a frozen dataclass that validates its arguments in `__post_init__`, and is
  hashable, since components, strategies and constraints are part of the memo's key. A strategy's
  `subcomponents` mapping is left out of its hash and kept in its equality, which is what tells
  apart strategies that hash alike.
- Cores are what is divided, everywhere: a rank costs as many cores as it runs threads, so ranks
  are never divided.
- A constraint returns `False` for a layout it rejects, and raises `ValueError` for one it cannot
  evaluate, such as a domain rule on a component with no domain: answering `False` there would
  quietly empty the search.
- `iter_layouts` validates its arguments when it is called, and returns a lazy iterator: it is not
  a generator function, so that a mistake raises at the call rather than at the first `next()`.
- Each module logs under its own name, `access.config.parallel_layouts` and
  `access.config.parallel_allocation_strategies`, at `DEBUG` level, for every dead end.

## Testing a change to the search

The tests pin the order of the layouts, the text of the log messages and of the errors, but there
are no golden outputs of whole searches. To show that a change to the search keeps its behaviour,
compare whole enumerations: import the module as it is and a patched copy of it, run both over a
range of trees, from a single leaf to nested, threaded, constrained and shared ones, and over a
budget too small for them, and check that they yield the same layouts in the same order, log the
same messages and raise the same errors.
