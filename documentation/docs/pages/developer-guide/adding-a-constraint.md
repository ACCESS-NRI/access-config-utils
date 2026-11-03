# Adding a constraint

Constraints are the way to extend the search for parallelisation layouts. The search generates every
layout that a model can run, and the constraints reject those that a study does not want. In this
tutorial we go through what it takes to add a constraint to the package. You never need to change
`parallel_layouts` or `parallel_allocation_strategies` for this.
[Constraints](../user-guide/layouts/constraints.md) shows how constraints are used, and how to write
one outside the package.

## Should it be a constraint?

First check that your rule should be a constraint at all. A constraint filters: every candidate is
generated, and then judged. A rule about how many cores a component may have is better written as an
allocation: a `FixedAllocation`, the bounds or the `core_step` of a `FreeAllocation`, or a
`WholeRangeAllocation`. The counts that it rules out are then never generated. A rule that is a
share of the whole number of cores has to be an allocation, because a constraint never sees the
total. A constraint is for what an allocation cannot say: the shape of a grid, the size of a
subdomain, a relation between siblings, a thread count.

## Local or group

There are two base classes, both in `parallel_component`:

| Base class | `is_satisfied` takes | Attached to |
|---|---|---|
| `LocalConstraint` | `layout: ComponentLayout` | the component's `local_constraints` |
| `GroupConstraint` | `sub_layouts: tuple[ComponentLayout, ...]` | the **parent's** `group_constraints` |

A rule that can be decided from the layout of one component, including its `sub_layouts`, is local,
as `MaxWastedCoreFractionConstraint` is. A group constraint is for a rule that relates siblings to
each other. It names them: siblings are referred to by name, never by their position. You can also
attach either kind to an allocation strategy, of any mode, when the rule belongs to a study and not
to the model. There is nothing more to do for this, because the search checks both.

## Writing it

A constraint goes in `parallel_constraints`, in the section of its family: the grid of ranks, how
the cores are shared out, the block of the domain that each rank holds, or threads. It follows the
shape of the other constraints. As an example, we write one that sets a minimum number of ranks
along a dimension:

```py
@dataclass(frozen=True)
class MinRanksPerDimConstraint(LocalConstraint):
    """``grid[dim]`` must be at least ``min_ranks``.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        dim (int): Zero-based dimension index to check.
        min_ranks (int): Minimum ranks along that dimension. Must be >= 1.
    """

    dim: int
    min_ranks: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"MinRanksPerDimConstraint.dim must be >= 0, got {self.dim}.")
        if self.min_ranks < 1:
            raise ValueError(f"MinRanksPerDimConstraint.min_ranks must be >= 1, got {self.min_ranks}.")

    def is_satisfied(self, layout: ComponentLayout) -> bool:
        return _require_grid_dim(self, layout, self.dim) >= self.min_ranks
```

Every constraint does the following:

- It is a `@dataclass(frozen=True)`, which makes it hashable, because it is part of the memo key of
  the search. Its docstring documents every field and its valid range in an `Args:` section.
- `__post_init__` validates every field. It raises `ValueError` with a message that names the class
  and the field and gives the value. A constraint is public API, so this is done whether or not the
  search could produce an invalid value.
- `is_satisfied` **returns `False` only for a valid layout that it rejects, and raises `ValueError`
  for one that it cannot evaluate**. Examples of the second case are a domain rule on a component
  with no decomposition, a dimension out of range, and a sibling name that does not exist. Answering
  `False` there would empty the search and leave nothing to debug.
- The preconditions are checked with the `_require_*` helpers at the top of the module:
  `_require_decomposition`, `_require_grid_dim` and `_require_threads_per_rank`. In this way every
  constraint words them in the same way. A new precondition gets a helper of its own.
- `is_satisfied` changes nothing and stays cheap, because it is called once for every candidate
  layout.

## What a layout holds

The attributes of a layout are easy to confuse, and confusing them is the most common mistake in
this area:

- `layout.decomposition.grid` is an `MPICartesianGrid`, the number of **ranks** in each dimension;
- `layout.decomposition.domain.shape` is the number of **points** in each dimension;
- `layout.decomposition.mean_local_shape` is `shape[i] / grid[i]`, as floats. This is the *mean*
  size of the block of a rank, and not the size of the block of any rank where a dimension does not
  divide evenly. A rule that needs exact sizes uses `//`, as `MinSubdomainSizeConstraint` does, or
  also requires `DomainDivisibleByRanksConstraint`;
- `n_cores`, `used_cores`, `idle_cores` and `n_ranks` are four different counts;
- under a shared parent, `idle_cores` is always 0, because the sub-layouts cover the range. A rule
  about idle cores therefore always passes there;
- `threads_per_rank` is `None` on a parent.

## Tests

You test a constraint in `tests/test_parallel_constraints.py`, in a `class Test<Name>:` block in the
section of its family. The tests cover at least:

- each rejection in `__post_init__`, with `pytest.raises(ValueError, match="<field>")`;
- one layout that it accepts and one that it rejects, and the boundary value, because most of these
  rules are inequalities;
- every `ValueError` of `is_satisfied`, matched on a distinctive part of its message.

Build the layouts with the `leaf()`, `parent()`, `decomp()` and `grid()` helpers at the top of that
file. They give a constraint exactly the attributes that it reads, so the tests are about the rule
alone. The file must take `parallel_constraints` to 100% coverage on its own. A constraint that is
meant to prune a real search also gets a case in `tests/test_parallel_layouts.py`, which checks the
layouts that `iter_layouts` yields with it.

## Finishing

- Add what the constraint checks to the line of its family in the module docstring of
  `parallel_constraints`, and update the count of constraints in `AGENTS.md`.
- Add it to the table of [Constraints](../user-guide/layouts/constraints.md#the-ready-made-constraints).
- An `Examples:` block in its docstring runs as a doctest.
- Run the checks in [Contributing](contributing.md#before-committing).
