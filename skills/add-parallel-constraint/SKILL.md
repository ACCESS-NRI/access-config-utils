---
name: add-parallel-constraint
description: Add a constraint to the parallel layout enumeration library in access-config-utils. Use when a new rule is needed to filter candidate MPI/OpenMP layouts — process grid shape, subdomain size or aspect ratio, rank distribution between components, thread counts — i.e. any new LocalConstraint or GroupConstraint in parallel_constraints.py.
---

# Adding a parallel layout constraint

Constraints are the extension point of the layout enumerator: `iter_layouts` generates every
structurally possible layout and constraints reject the ones that are not wanted. Adding a
constraint should never require touching `parallel_layouts.py` or
`parallel_allocation_strategies.py`.

## 1. Check it should be a constraint

A constraint filters: every candidate is generated and then judged. A rule that says how many
cores a component gets is better stated as an allocation (`FixedAllocation`, `FreeAllocation`
bounds or `core_step`, `WholeRangeAllocation`), because the counts it rules out are then never
generated. A rule stated as a share of the whole budget has to be an allocation, since a
constraint never sees the budget. Write a constraint for what an allocation cannot say: a grid
shape, a subdomain size, a relation between siblings, a thread count.

## 2. Choose the interface

Both live in `src/access/config/parallel_component.py`:

| | signature | attached to |
|---|---|---|
| `LocalConstraint` | `is_satisfied(layout: ComponentLayout) -> bool` | the component's own `local_constraints` |
| `GroupConstraint` | `is_satisfied(sub_layouts: tuple[ComponentLayout, ...]) -> bool` | the **parent's** `group_constraints` |

If the rule can be decided from one component's layout — including its `sub_layouts`, as
`MaxWastedCoreFractionConstraint` does — it is local. Only reach for `GroupConstraint` when the rule
genuinely relates two or more siblings to each other.

Both can also be attached to an allocation strategy — any mode, or the `RootAllocation` —
instead of a `ParallelComponent`, when the rule belongs to a particular allocation scenario rather
than to the model structure. Nothing extra is needed to support that: the enumerator concatenates
both sources.

## 3. Write it in `parallel_constraints.py`

Place it in the matching category section (Cartesian grid / Core distribution / Domain layout /
Thread), keeping the existing banner comments. Follow the established shape exactly:

```python
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

Non-negotiable points:

- `@dataclass(frozen=True)`, Google-style docstring with an `Args:` section giving each field's
  valid range (see the docstring conventions in `AGENTS.md`).
- Validate every field in `__post_init__`, raising `ValueError` whose message names the class and
  field and reports the offending value. Do this even when the enumerator would never produce an
  invalid value — the constraint is user-facing API.
- **Return `False` only for "valid layout, rejected". Raise `ValueError` for "cannot evaluate".**
  A domain constraint applied to a component with no `decomposition`, an out-of-range dimension
  index, or a group constraint naming a sibling that does not exist are all programming errors, and
  the existing constraints raise on all three. Returning `False` there would silently empty the
  result set and give the user nothing to debug.
- Raise those through the `_require_*` helpers at the top of the module (`_require_decomposition`,
  `_require_grid_dim`, `_require_threads_per_rank`), so every constraint words the error the same
  way. A new precondition gets a helper of its own beside them.
- Group constraints address siblings **by name**, never by index — see `RankRatioGroupConstraint`,
  which looks the names up in `sub_layouts` and raises if either is missing.
- Do not mutate anything. `is_satisfied` is called once per candidate layout in a hot loop; keep it
  cheap and side-effect free.

## 4. Know what you are constraining

Confusing these is the most common bug in this area:

- `layout.decomposition.grid` — `MPICartesianGrid`, **rank counts** per dimension.
- `layout.decomposition.domain.shape` — `Domain`, **grid points** per dimension.
- `layout.decomposition.mean_local_shape` — `shape[i] / grid[i]`, **floats**, the *mean* subdomain
  extent. Not any actual rank's extent: when a dimension does not divide evenly, real ranks differ.
  If your rule needs exact per-rank sizes, it must also require `DomainDivisibleByRanksConstraint`, or be
  expressed in terms of `//` like `MinSubdomainSizeConstraint` does.
- `layout.n_cores` (cores allocated) vs `layout.used_cores`/`layout.idle_cores` vs `layout.n_ranks`.
  Pick deliberately. A constraint never sees the whole search's budget, only the layout it judges: a
  rule stated as a share of `total_cores` is an allocation bound
  (`FixedAllocation(core_fraction=...)`, `FreeAllocation(max_core_fraction=...)`), resolved to a
  core count before the search starts.
- Under `CoreSharing.SHARED` a parent's `used_cores` is the part of its range at least one
  sub-layout sits on, overlaps counted once, and its `idle_cores` is always 0, because its
  sub-layouts must cover the range. A rule about idle cores is vacuous on such a parent
  (`MaxWastedCoreFractionConstraint` always passes there), and its siblings' counts are a product
  rather than a split of a budget — see `CoreSharing` in `parallel_component.py`.
- `layout.threads_per_rank` is `None` on a parent. If your rule needs it, go through
  `_require_threads_per_rank`.

## 5. Tests

Add a `class Test<ConstraintName>:` block to `tests/test_parallel_constraints.py`, in the same
category section. Cover, at minimum:

- each `__post_init__` rejection, via `pytest.raises(ValueError, match="<field>")`,
- one satisfying layout and one rejected layout,
- every `ValueError` path in `is_satisfied` (missing decomposition, out-of-range dim, unknown
  sibling name), matched on a distinctive fragment of the message,
- the boundary value, since these rules are almost all inequalities.

Build the layouts as the existing tests do, with the `leaf()`, `parent()`, `decomp()` and `grid()`
helpers at the top of the file. They return mocks carrying exactly the attributes a constraint
reads, which keeps these tests about the rule: how a real `ComponentLayout` derives `idle_cores`,
or a real decomposition its `mean_local_shape`, is asserted in those modules' own test files.

`tests/test_parallel_constraints.py` must take `parallel_constraints` to 100% on its own.

If the constraint is meant to prune a real enumeration, also add an integration case to
`tests/test_parallel_layouts.py` asserting `iter_layouts` yields the expected reduced set.

## 6. Finish

- Update the module docstring's list of families: add what the new rule checks to its family's
  line (as `EqualCoresGroupConstraint` did for core distribution), or add a family.
- Update the constraint count in the repository layout of `AGENTS.md`.
- Any `Examples:` block you add runs as a doctest (`tests/test_parallel_doctests.py`).
- Run the gate in [`skills/check/SKILL.md`](../check/SKILL.md).
