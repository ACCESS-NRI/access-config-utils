# Allocating cores

Without further instructions, a search offers every sub-component every number of cores that fits.
This is far more than a study wants. An *allocation strategy* says how many cores each component may
receive. It is a tree of its own, with the same shape as the component tree: there is one strategy
for each component, keyed by the name of the component, under a `RootAllocation` for the root.

In the examples we use a model of three components with no domain, so that each way of dividing the
cores is one parallelisation layout. The `splits` function returns the core counts that each layout
gives the three components:

```python
from access.config.parallel_component import ParallelComponent
from access.config.parallel_layouts import iter_layouts

model = ParallelComponent(
    "model",
    subcomponents=(ParallelComponent("atm"), ParallelComponent("ocn"), ParallelComponent("ice")),
)


def splits(allocations, total_cores=24):
    layouts = iter_layouts(model, total_cores, allocations=allocations)
    return sorted({tuple(sub.n_cores for sub in layout.sub_layouts) for layout in layouts})
```

## The modes

The strategy of each component uses one of four modes. Each mode is a class of its own.

**`FixedAllocation`** gives a component an exact number of cores:

```python
from access.config.parallel_allocation_strategies import FixedAllocation, RootAllocation

allocations = RootAllocation(
    subcomponents={"atm": FixedAllocation(12), "ocn": FixedAllocation(8), "ice": FixedAllocation(4)}
)
print(splits(allocations))
```

This prints the only possible split:

```text
[(12, 8, 4)]
```

**`WeightedAllocation`** fixes the proportions between siblings, but not the number of cores.
Siblings with weights 3 and 2 receive `3 * k` and `2 * k` cores, for every multiplier `k` that fits:

```python
from access.config.parallel_allocation_strategies import WeightedAllocation

allocations = RootAllocation(
    subcomponents={"atm": WeightedAllocation(3), "ocn": WeightedAllocation(2), "ice": FixedAllocation(4)}
)
print(splits(allocations))
```

This prints:

```text
[(3, 2, 4), (6, 4, 4), (9, 6, 4), (12, 8, 4)]
```

**`FreeAllocation`** takes any number of cores between `min_cores`, which is 1 by default, and
`max_cores`, which is unbounded by default. With `core_step`, only the multiples of the step are
offered. This is how you give a component whole nodes, or a count that its decomposition needs. Here
we first use the bounds, and then a step of 4:

```python
from access.config.parallel_allocation_strategies import FreeAllocation

allocations = RootAllocation(
    subcomponents={
        "atm": FreeAllocation(min_cores=12),
        "ocn": FreeAllocation(min_cores=6, max_cores=8),
        "ice": FixedAllocation(4),
    }
)
print(splits(allocations))

allocations = RootAllocation(
    subcomponents={"atm": FreeAllocation(min_cores=8, core_step=4), "ocn": FixedAllocation(8), "ice": FixedAllocation(4)}
)
print(splits(allocations))
```

This prints one line for each of the two strategies:

```text
[(12, 6, 4), (12, 7, 4), (12, 8, 4), (13, 6, 4), (13, 7, 4), (14, 6, 4)]
[(8, 8, 4), (12, 8, 4)]
```

**`WholeRangeAllocation`** gives a sub-component of a [shared
parent](components.md#partitioned-and-shared-cores) the whole range of the parent, from its own
offset to the end. Use it to pin the sub-components of a shared parent to whatever range the parent
is given.

`FreeAllocation()`, with no bounds, leaves a component open. Every component of the tree needs a
strategy, because there is no default mode. At every level, the names in the strategy have to be
exactly those of the component tree. A name that is missing from the strategy, or that the component
tree does not have, raises `ValueError` when the search is started. For example, we leave out `ice`:

```python
try:
    iter_layouts(model, 24, allocations=RootAllocation(subcomponents={"atm": FreeAllocation(), "ocn": FreeAllocation()}))
except ValueError as error:
    print(error)
```

This prints:

```text
RootAllocation: no allocation given at root for {'ice'}. Every sub-component needs one - there is no default mode. Add a FreeAllocation() for the ones that should be left open.
```

If you give no allocation strategy at all, `iter_layouts` uses a `FreeAllocation()` for every
component, with one thread per rank.

## How the cores of a parent are divided

Siblings with different modes are served in a fixed order. First, the fixed counts are taken out of
the cores of the parent. Then the multiplier of the weighted siblings is chosen from what is left.
Finally, the free siblings divide the rest. The cores that the free siblings do not take are left
idle. A `MaxWastedCoreFractionConstraint` on the parent limits how many, as
[Constraints](constraints.md) shows. A parent is given its own cores in the same way by its parent,
and then divides them among its sub-components. The strategies of the sub-components of a parent
therefore go in the `subcomponents` of its strategy, as they do for the root.

Under a [shared parent](components.md#partitioned-and-shared-cores) nothing is divided. Each
sub-component is offered every count that fits its part of the range, starting from its offset. A
sub-component at an offset other than 0 is only offered the layouts that run one rank on each core.
You can use a fixed, a free or a whole-range strategy for such a sub-component. A weight cannot be
used, because it is a share of something that is divided, and it raises `ValueError` once the search
reaches it. A `WholeRangeAllocation` under a parent that divides its cores raises `ValueError` in
the same way.

## Fractions of the total

A fixed or free allocation can also be given as a fraction of the total number of cores of the
search, instead of a number of cores. One strategy tree can then be used for every size of a scaling
study. Here we use the same strategy on 24 and on 48 cores:

```python
allocations = RootAllocation(
    subcomponents={
        "atm": FixedAllocation(core_fraction=0.5),
        "ocn": FreeAllocation(min_core_fraction=0.25, max_core_fraction=0.3),
        "ice": FixedAllocation(2),
    }
)
print(splits(allocations, total_cores=24))
print(splits(allocations, total_cores=48))
```

This prints:

```text
[(12, 6, 2), (12, 7, 2), (12, 8, 2)]
[(24, 12, 2), (24, 13, 2), (24, 14, 2), (24, 15, 2)]
```

A fraction is always a fraction of the whole total, even for a component deep in the tree. It is
turned into a number of cores once, before the search starts. A fixed count is rounded down. A free
range is widened to whole cores: its lower bound is rounded down and its upper bound up. For
example, 0.3 of 24 cores is 7.2, so the upper bound is 8. A product that is within a hair of a whole
number is that number, so that 0.29 of 100 cores is 29 and not 28. If you give a free allocation
both a count and a fraction, the count limits the fraction: `min_cores` raises the lower bound, and
`max_cores` lowers the upper one. Bounds that leave no count at a given total raise `ValueError`
when the search is started.

## Threads

Each strategy can set a `thread_range`. This is the inclusive range of OpenMP threads per rank that
its component may run. A strategy without a range inherits the range of its parent, and the range of
the root is `(1, 1)` by default. A model therefore runs one thread per rank unless you say
otherwise. To thread one component only, set a range on its strategy. Here the atmosphere may run 1
to 4 threads per rank:

```python
allocations = RootAllocation(
    subcomponents={"atm": FixedAllocation(8, thread_range=(1, 4)), "ocn": FixedAllocation(8), "ice": FixedAllocation(4)}
)
for layout in iter_layouts(model, 24, allocations=allocations):
    atm = layout.sub_layouts[0]
    print(atm.n_ranks, "ranks of", atm.threads_per_rank, "threads")
```

This prints:

```text
8 ranks of 1 threads
4 ranks of 2 threads
2 ranks of 4 threads
```

A thread count is only used where it divides the cores of the component. For example, 3 threads do
not divide 8 cores, so there is no layout with 3 threads. The range on the strategy of a parent is
only the default that its sub-components inherit, because a parent runs no threads of its own.

## Constraints on a strategy

A strategy can also carry constraints, in its `local_constraints` and `group_constraints`. They are
checked together with those of its component. This is where a study adds its own requirements to
those of the component tree, as [Constraints](constraints.md) describes.
