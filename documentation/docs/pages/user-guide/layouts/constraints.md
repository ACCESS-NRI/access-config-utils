# Constraints

Every parallelisation layout that a search finds is one that the model could run. This is still far
more than a study wants. Some layouts have process grids that are too elongated to exchange halos
efficiently, or subdomains that are too small to be worth a rank. Others give one component far more
cores than its cost deserves. A *constraint* rules such layouts out. It is a yes-or-no test: a
layout that fails it is dropped, and the search moves on to the next one.

In the examples we count the layouts of a model with two components on 24 cores. We start without
any constraint:

```python
from access.config.parallel_component import ParallelComponent
from access.config.parallel_domain import Domain
from access.config.parallel_layouts import iter_layouts


def count(component, allocations=None):
    return sum(1 for _ in iter_layouts(component, 24, allocations=allocations))


um7 = ParallelComponent("UM7", domain=Domain((192, 144)))
mom5 = ParallelComponent("MOM5", domain=Domain((360, 300)))
print(count(ParallelComponent("ESM", subcomponents=(um7, mom5))))
```

This prints:

```text
2377
```

## Local and group constraints

A *local* constraint judges the layout of one component on its own. It goes in the
`local_constraints` of that component. A *group* constraint judges the layouts of the sub-components
of a parent together, for example the ratio between two of them. It therefore goes in the
`group_constraints` of the parent, and not on either sibling. A group constraint names the siblings
that it compares.

Here we add three constraints. The model may leave no core idle, the grid of the atmosphere has to
be even in its first dimension, and the atmosphere has to run at least as many ranks as the ocean:

```python
from access.config.parallel_constraints import (
    MaxWastedCoreFractionConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
)

um7 = ParallelComponent("UM7", domain=Domain((192, 144)), local_constraints=(ProcessGridDimEvenConstraint(dim=0),))
esm = ParallelComponent(
    "ESM",
    subcomponents=(um7, mom5),
    local_constraints=(MaxWastedCoreFractionConstraint(max_fraction=0.0),),
    group_constraints=(RankRatioGroupConstraint(name_a="UM7", name_b="MOM5", min_ratio=1.0),),
)
print(count(esm))
```

This leaves far fewer layouts:

```text
76
```

## On the component tree, or on the strategy

You can attach constraints in two places. Those on the component tree are requirements: they are
true of every layout of the model, such as a grid that the model cannot run on. Those on an
[allocation strategy](allocations.md), in its `local_constraints` and `group_constraints`, are the
preferences of a study. They are checked together with those of the component. Here a study asks for
subdomains that are close to square:

```python
from access.config.parallel_allocation_strategies import FreeAllocation, RootAllocation
from access.config.parallel_constraints import SubdomainAspectRatioConstraint

near_square = (SubdomainAspectRatioConstraint(max_ratio=1.5),)
allocations = RootAllocation(
    subcomponents={
        "UM7": FreeAllocation(local_constraints=near_square),
        "MOM5": FreeAllocation(local_constraints=near_square),
    }
)
for layout in iter_layouts(esm, 24, allocations=allocations):
    print([(sub.name, sub.n_cores, sub.decomposition.grid.shape) for sub in layout.sub_layouts])
```

Only two layouts are left:

```text
[('UM7', 12, (4, 3)), ('MOM5', 12, (4, 3))]
[('UM7', 18, (6, 3)), ('MOM5', 6, (3, 2))]
```

What about a rule that is a share of the whole number of cores, such as "no more than half of them"?
This is not a constraint but an allocation: a [fraction](allocations.md#fractions-of-the-total) of
the total. An allocation never generates the layouts that a constraint would then have to rule out.

## The ready-made constraints

`access.config.parallel_constraints` holds the rules that are needed often enough to deserve a name.
They fall into four families.

| Constraint | Kind | Accepts a layout when |
|---|---|---|
| **The grid of ranks** | | |
| `ProcessGridDimEvenConstraint(dim)` | local | the grid's dimension `dim` is even |
| `ProcessGridDimDivisibleConstraint(dim, divisor)` | local | the grid's dimension `dim` is a multiple of `divisor` |
| **How the cores are shared out** | | |
| `MaxWastedCoreFractionConstraint(max_fraction)` | local, on a parent | at most `max_fraction` of the parent's cores are idle |
| `RankRatioGroupConstraint(name_a, name_b, min_ratio)` | group | sibling `name_a` runs at least `min_ratio` times the ranks of sibling `name_b` |
| `EqualCoresGroupConstraint(names=None)` | group | the named siblings, or all of them, have the same number of cores |
| **The block of the domain each rank holds** | | |
| `DomainDivisibleByRanksConstraint()` | local | every dimension of the domain divides evenly over the grid |
| `MinSubdomainSizeConstraint(min_size)` | local | each rank holds at least `min_size` points along every dimension |
| `SubdomainAspectRatioConstraint(max_ratio)` | local | the longest side of the mean block is at most `max_ratio` times the shortest |
| **Threads** | | |
| `MaxThreadsPerRankConstraint(max_threads)` | local, on a leaf | the leaf runs at most `max_threads` threads per rank |
| `FixedThreadsPerRankConstraint(n_threads)` | local, on a leaf | the leaf runs exactly `n_threads` threads per rank |

To bound a rank ratio on both sides, use two `RankRatioGroupConstraint`s, one in each direction. The
[API reference](../../api/constraints.md) describes each constraint in full.

A constraint that cannot judge the layout it is given raises `ValueError` instead of answering. This
happens with a grid or domain rule on a component with no domain, a thread rule on a parent, or an
idle-core rule on a leaf. Such a rule was attached in the wrong place. This is a mistake to fix, and
not a layout to reject.

## Writing a constraint

You can also write your own constraint. It is a class that derives from `LocalConstraint` or
`GroupConstraint`. Its `is_satisfied` method takes the layout of a component, or the tuple of the
sub-layouts of a parent, and returns whether to keep it. The class has to be hashable, because it is
part of what the search memoises, so make it a frozen dataclass. Here we write a constraint on the
number of ranks, and add it to the ocean of the previous example:

```python
from dataclasses import dataclass

from access.config.parallel_component import ComponentLayout, LocalConstraint


@dataclass(frozen=True)
class MaxRanksConstraint(LocalConstraint):
    """Keep the layouts running at most max_ranks MPI ranks."""

    max_ranks: int

    def is_satisfied(self, layout: ComponentLayout) -> bool:
        return layout.n_ranks <= self.max_ranks


allocations = RootAllocation(
    subcomponents={
        "UM7": FreeAllocation(local_constraints=near_square),
        "MOM5": FreeAllocation(local_constraints=(*near_square, MaxRanksConstraint(9))),
    }
)
for layout in iter_layouts(esm, 24, allocations=allocations):
    print([(sub.name, sub.n_cores, sub.decomposition.grid.shape) for sub in layout.sub_layouts])
```

Of the two layouts, only the one where the ocean runs at most 9 ranks is left:

```text
[('UM7', 18, (6, 3)), ('MOM5', 6, (3, 2))]
```

A constraint returns `False` for a layout that it rejects, and raises `ValueError` for one that it
cannot evaluate. [Adding a constraint](../../developer-guide/adding-a-constraint.md) describes how
to add one to the package.
