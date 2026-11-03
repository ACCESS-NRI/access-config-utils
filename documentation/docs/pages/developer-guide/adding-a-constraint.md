# Adding a constraint

Constraints are how the layout search is extended: the search generates every layout a model can
run, and the constraints reject those a study does not want. Adding one to the package never
needs a change to `parallel_layouts` or `parallel_allocation_strategies`.
[Constraints](../user-guide/layouts/constraints.md) shows how they are used, and how to write one
outside the package.

## Should it be a constraint?

A constraint filters: every candidate is generated, then judged. A rule saying how many cores a
component may have is better stated as an allocation, a `FixedAllocation`, the bounds or the
`core_step` of a `FreeAllocation`, or a `WholeRangeAllocation`, because the counts it rules out are
then never generated. A rule stated as a share of the whole number of cores has to be an
allocation, since a constraint never sees the total. A constraint is for what an allocation cannot
say: the shape of a grid, the size of a subdomain, a relation between siblings, a thread count.

## Local or group

Both base classes are in `parallel_component`:

| Base class | `is_satisfied` takes | Attached to |
|---|---|---|
| `LocalConstraint` | `layout: ComponentLayout` | the component's `local_constraints` |
| `GroupConstraint` | `sub_layouts: tuple[ComponentLayout, ...]` | the **parent's** `group_constraints` |

A rule that can be decided from one component's layout, its `sub_layouts` included, is local, as
`MaxWastedCoreFractionConstraint` is. A group constraint is for a rule that relates siblings to
each other, and it names them: siblings are referred to by name, never by their position. Either
kind can also be attached to an allocation strategy, of any mode, when it belongs to a study
rather than to the model, with nothing more to do: the search checks both.

## Writing it

A constraint goes in `parallel_constraints`, in the section of its family, the grid of ranks, how
the cores are shared out, the block of the domain each rank holds, or threads, and follows the
shape of the others:

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

What every constraint does:

- It is a `@dataclass(frozen=True)`, which makes it hashable: it is part of the search's memo key.
  Its docstring documents every field and its valid range in an `Args:` section.
- `__post_init__` validates every field, raising `ValueError` with a message that names the class
  and the field and gives the value: a constraint is public API, whether or not the search could
  produce an invalid value.
- `is_satisfied` **returns `False` only for a valid layout it rejects, and raises `ValueError` for
  one it cannot evaluate**: a domain rule on a component with no decomposition, a dimension out of
  range, a sibling name that does not exist. Answering `False` there would empty the search and
  leave nothing to debug.
- The preconditions are checked through the `_require_*` helpers at the top of the module,
  `_require_decomposition`, `_require_grid_dim` and `_require_threads_per_rank`, so that every
  constraint words them the same way. A new precondition gets a helper of its own.
- `is_satisfied` changes nothing and stays cheap: it is called once for every candidate layout.

## What a layout holds

These are easy to confuse, and confusing them is the most common mistake in this area:

- `layout.decomposition.grid` is an `MPICartesianGrid`, the number of **ranks** in each dimension;
- `layout.decomposition.domain.shape` is the number of **points** in each dimension;
- `layout.decomposition.mean_local_shape` is `shape[i] / grid[i]`, as floats: the *mean* size of a
  rank's block, not the size of any rank's block where a dimension does not divide evenly. A rule
  needing exact sizes uses `//`, as `MinSubdomainSizeConstraint` does, or requires
  `DomainDivisibleByRanksConstraint` too;
- `n_cores`, `used_cores`, `idle_cores` and `n_ranks` are four different counts;
- under a shared parent, `idle_cores` is always 0, since the sub-layouts cover the range, so a rule
  about idle cores always passes there;
- `threads_per_rank` is `None` on a parent.

## Tests

A constraint is tested in `tests/test_parallel_constraints.py`, in a `class Test<Name>:` block in
the section of its family, which covers at least:

- each rejection in `__post_init__`, with `pytest.raises(ValueError, match="<field>")`;
- one layout it accepts and one it rejects, and the boundary value, since most of these rules are
  inequalities;
- every `ValueError` of `is_satisfied`, matched on a distinctive part of its message.

The layouts are built with the `leaf()`, `parent()`, `decomp()` and `grid()` helpers at the top of
that file, which give a constraint exactly the attributes it reads, so the tests are about the
rule alone. The file must take `parallel_constraints` to 100% coverage on its own. A constraint
meant to prune a real search also gets a case in `tests/test_parallel_layouts.py`, checking the
layouts `iter_layouts` yields with it.

## Finishing

- Add what the constraint checks to its family's line in the module docstring of
  `parallel_constraints`, and update the count of constraints in `AGENTS.md`.
- Add it to the table of [Constraints](../user-guide/layouts/constraints.md#the-ready-made-constraints).
- An `Examples:` block in its docstring runs as a doctest.
- Run the checks in [Contributing](contributing.md#before-committing).
