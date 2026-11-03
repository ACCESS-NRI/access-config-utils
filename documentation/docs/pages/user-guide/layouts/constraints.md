# Constraints

Every layout a search finds is one the model could run, which is still far more than a study
wants: process grids too elongated to exchange halos efficiently, subdomains too small to be worth
a rank, one component given far more cores than its cost deserves. A *constraint* rules such
layouts out. It is a yes-or-no test, and a layout it says no to is dropped; the search moves on
to the next one.

The examples on this page count the layouts of a two-component model on 24 cores:

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

```text
2377
```

## Local and group constraints

A *local* constraint judges one component's layout on its own, and is attached to the component's
`local_constraints`. A *group* constraint judges the layouts of a parent's sub-components
together, such as the ratio between two of them, and so is attached to the parent's
`group_constraints` rather than to either sibling. A group constraint names the siblings it
compares.

Here the model may leave no core idle, the atmosphere's grid has to be even in its first
dimension, and the atmosphere has to run at least as many ranks as the ocean:

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

```text
76
```

## On the component tree, or on the strategy

Constraints can be attached in two places. Those on the component tree are requirements: what is
true of every layout of the model, such as a grid the model cannot run on. Those on an
[allocation strategy](allocations.md), in its `local_constraints` and `group_constraints`, are
the preferences of a study, checked together with the component's own. Here a study asks for
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

```text
[('UM7', 12, (4, 3)), ('MOM5', 12, (4, 3))]
[('UM7', 18, (6, 3)), ('MOM5', 6, (3, 2))]
```

A rule stated as a share of the whole number of cores, such as "no more than half of them", is
not a constraint: it is an allocation, a [fraction](allocations.md#fractions-of-the-total) of the
total, which never generates the layouts it would rule out.

## The ready-made constraints

`access.config.parallel_constraints` holds the rules that recur often enough to be worth naming.
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

A rank ratio bounded on both sides takes two `RankRatioGroupConstraint`s, one each way. The
[API reference](../../api/constraints.md) describes each constraint in full.

A constraint that cannot judge the layout it is given raises `ValueError` rather than answering:
a grid or domain rule on a component with no domain, a thread rule on a parent, an idle-core rule
on a leaf. Such a rule was attached in the wrong place, which is a mistake to fix rather than a
layout to reject.

## Writing a constraint

A constraint of one's own is a class deriving from `LocalConstraint` or `GroupConstraint`, whose
`is_satisfied` takes a component's layout, or the tuple of a parent's sub-layouts, and returns
whether to keep it. It has to be hashable, since it is part of what the search memoises, so make
it a frozen dataclass:

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

```text
[('UM7', 18, (6, 3)), ('MOM5', 6, (3, 2))]
```

A constraint returns `False` for a layout it rejects, and raises `ValueError` for one it cannot
evaluate. [Adding a constraint](../../developer-guide/adding-a-constraint.md) describes how to add
one to the package.
