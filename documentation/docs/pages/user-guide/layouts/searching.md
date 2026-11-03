# Searching for layouts

`iter_layouts` runs a search:

```py
iter_layouts(component, total_cores, *, allocations=None)
```

It gives the root `component` all `total_cores`, divides them down the tree as `allocations`
says, and yields every layout that satisfies the constraints, one at a time. This page goes
through two searches, shaped like those of ACCESS-ESM1.6 and ACCESS-OM3, and then describes how to
use the results and how to keep a search small.

## A coupled model on four nodes

ACCESS-ESM1.6's pre-industrial control runs on four nodes of 104 cores: 208 for the UM, 196 for
MOM5 and 12 for CICE5. A study looking for the layouts near that one describes the model with the
constraints every layout of it has to meet:

```python
from access.config.parallel_component import ParallelComponent
from access.config.parallel_constraints import (
    MaxWastedCoreFractionConstraint,
    MinSubdomainSizeConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
    SubdomainAspectRatioConstraint,
)
from access.config.parallel_domain import Domain

um7 = ParallelComponent(
    "UM7",
    domain=Domain((192, 144)),
    local_constraints=(ProcessGridDimEvenConstraint(dim=0), SubdomainAspectRatioConstraint(max_ratio=2.0)),
)
mom5 = ParallelComponent(
    "MOM5", domain=Domain((360, 300)), local_constraints=(SubdomainAspectRatioConstraint(max_ratio=2.0),)
)
cice5 = ParallelComponent("CICE5", domain=Domain((360,)), local_constraints=(MinSubdomainSizeConstraint(min_size=1),))
esm16 = ParallelComponent(
    "ESM1.6",
    subcomponents=(um7, mom5, cice5),
    local_constraints=(MaxWastedCoreFractionConstraint(max_fraction=0.02),),
    group_constraints=(
        RankRatioGroupConstraint(name_a="MOM5", name_b="UM7", min_ratio=0.75),
        RankRatioGroupConstraint(name_a="UM7", name_b="MOM5", min_ratio=1.0),
    ),
)
```

and then states what it is looking for in its allocation strategy: a few cores either side of the
released counts, and subdomains closer to square than the model requires:

```python
from access.config.parallel_allocation_strategies import FreeAllocation, RootAllocation
from access.config.parallel_layouts import iter_layouts

near_square = (SubdomainAspectRatioConstraint(max_ratio=1.5),)
allocations = RootAllocation(
    subcomponents={
        "UM7": FreeAllocation(min_cores=205, max_cores=212, local_constraints=near_square),
        "MOM5": FreeAllocation(min_cores=192, max_cores=200, local_constraints=near_square),
        "CICE5": FreeAllocation(min_cores=11, max_cores=12),
    }
)
layouts = list(iter_layouts(esm16, 416, allocations=allocations))
print(len(layouts))
```

```text
12
```

Sorting them by the cores they leave idle puts the released layout first:

```python
def grid(layout):
    return "x".join(str(n) for n in layout.decomposition.grid.shape)


for layout in sorted(layouts, key=lambda layout: layout.idle_cores):
    um, ocean, ice = layout.sub_layouts
    print(f"UM7 {grid(um):>5}   MOM5 {grid(ocean):>5}   CICE5 {ice.n_ranks:2}   idle {layout.idle_cores}")
```

```text
UM7 16x13   MOM5 14x14   CICE5 12   idle 0
UM7 14x15   MOM5 13x15   CICE5 11   idle 0
UM7 14x15   MOM5 15x13   CICE5 11   idle 0
UM7 16x13   MOM5 13x15   CICE5 12   idle 1
UM7 16x13   MOM5 15x13   CICE5 12   idle 1
UM7 16x13   MOM5 14x14   CICE5 11   idle 1
UM7 16x13   MOM5 13x15   CICE5 11   idle 2
UM7 16x13   MOM5 15x13   CICE5 11   idle 2
UM7 14x15   MOM5 16x12   CICE5 12   idle 2
UM7 14x15   MOM5 16x12   CICE5 11   idle 3
UM7 16x13   MOM5 16x12   CICE5 12   idle 4
UM7 16x13   MOM5 16x12   CICE5 11   idle 5
```

`sorted` needs the whole list, which is fine for twelve layouts. The next section says why that is
not always so.

## A model whose components share their cores

In ACCESS-OM3, the coupler, the data atmosphere, the sea ice and the runoff run in turn on one
range of cores, beside the ocean on cores of its own. The four are the sub-components of a
[shared parent](components.md#partitioned-and-shared-cores), each given the whole of its range with
a `WholeRangeAllocation`, and the range and the ocean are given fractions of the total, so that
the same strategy serves any number of cores:

```python
from access.config.parallel_allocation_strategies import WholeRangeAllocation
from access.config.parallel_component import CoreSharing

names = ("cpl", "atm", "ice", "rof")
shared = ParallelComponent(
    "shared", core_sharing=CoreSharing.SHARED, subcomponents=tuple(ParallelComponent(name) for name in names)
)
ocn = ParallelComponent("ocn", domain=Domain((1440, 1152)))
om3 = ParallelComponent(
    "ACCESS-OM3",
    subcomponents=(shared, ocn),
    local_constraints=(MaxWastedCoreFractionConstraint(max_fraction=0.0),),
)
allocations = RootAllocation(
    subcomponents={
        "shared": FreeAllocation(
            min_core_fraction=0.05,
            max_core_fraction=0.1,
            subcomponents={name: WholeRangeAllocation() for name in names},
        ),
        "ocn": FreeAllocation(min_core_fraction=0.85, local_constraints=(SubdomainAspectRatioConstraint(max_ratio=1.5),)),
    }
)
for layout in iter_layouts(om3, 96, allocations=allocations):
    shared_layout, ocn_layout = layout.sub_layouts
    print(
        "shared", shared_layout.n_cores, [sub.n_cores for sub in shared_layout.sub_layouts],
        "ocn", ocn_layout.n_cores, ocn_layout.decomposition.grid.shape,
    )
```

```text
shared 5 [5, 5, 5, 5] ocn 91 (13, 7)
shared 6 [6, 6, 6, 6] ocn 90 (9, 10)
shared 6 [6, 6, 6, 6] ocn 90 (10, 9)
shared 8 [8, 8, 8, 8] ocn 88 (11, 8)
```

## Using the results

`iter_layouts` does not hold the layouts it finds: it yields them one at a time, and finds the
next only when it is asked for. The number of layouts grows steeply with the number of cores, so
this matters. A three-component model with no allocation strategy has 14 thousand layouts on 24
cores, 2.2 million on 96 and 9 million on 144, and holding all of them in a list is often what
costs the most. Ask only for what is needed:

- `next(iter_layouts(...))` gives the first layout, `itertools.islice` the first few;
- `sum(1 for _ in iter_layouts(...))` counts them without holding them;
- a loop can stop at the first layout it is looking for;
- `list(iter_layouts(...))`, and anything that needs every layout at once, such as `sorted`, is
  for a search known to be small.

The layouts come in the same order every time the same search is run. A study that wants them in
another order, by idle cores for instance, sorts them itself.

The search remembers the layouts of every sub-tree it has worked out, so that the same question
asked again, such as the atmosphere's layouts on 40 cores, is answered without being worked out
again. That memory lasts as long as the iterator: two calls of `iter_layouts` share nothing.

## Keeping a search small

What keeps a search affordable is the allocation strategy more than the constraints. A
constraint rejects a layout once it has been generated, while an allocation never generates the
layouts it rules out. So:

- pin the components that should not vary with `FixedAllocation`, and give the others narrow
  bounds or a `core_step`;
- give the sub-components of a shared parent a `WholeRangeAllocation` or a `FixedAllocation`:
  each of them is offered every count that fits independently of the others, so four free
  sub-components of a 275-core range make most of 275⁴ combinations;
- narrow the `thread_range` rather than filtering threads with a constraint, since every thread
  count in the range is tried;
- attach constraints as low in the tree as they apply: a constraint is checked at its own
  component, before the layouts above it are built.

`EqualCoresGroupConstraint` says that siblings have the same number of cores in one rule, but it
filters the combinations rather than generating fewer of them.

## When a search finds nothing

A search that finds nothing is not an error: the budget may not fit the fixed allocations, no
thread count in a component's range may divide its cores, or a constraint may rule everything
out. Each dead end is logged at `DEBUG` level, on the loggers of the two modules involved, which
say what was missing:

```python
import logging
import sys

from access.config.parallel_allocation_strategies import FixedAllocation

logging.basicConfig(stream=sys.stdout, format="%(name)s: %(message)s")
for name in ("access.config.parallel_layouts", "access.config.parallel_allocation_strategies"):
    logging.getLogger(name).setLevel(logging.DEBUG)

model = ParallelComponent("model", subcomponents=(ParallelComponent("atm"), ParallelComponent("ocn")))
allocations = RootAllocation(subcomponents={"atm": FixedAllocation(6), "ocn": FixedAllocation(4)})
print(list(iter_layouts(model, 8, allocations=allocations)))
```

```text
access.config.parallel_allocation_strategies: no core split: the fixed allocations {'atm': 6, 'ocn': 4} need 10 core(s), but only 8 are available
access.config.parallel_layouts: no valid layout for 'model' on 8 core(s)
[]
```

## Errors

`iter_layouts` checks its arguments when it is called, before any layout is asked for, and
raises:

- `TypeError` if `allocations` is not a `RootAllocation`;
- `ValueError` if `total_cores` is less than 1, if the strategy tree does not name exactly the
  components of the tree at every level, or if fractional bounds leave a component no count.

Two mistakes can only be found once the search reaches the component involved, and raise
`ValueError` on the first `next()` rather than on the call: a `WeightedAllocation` under a shared
parent, and a `WholeRangeAllocation` under a parent that divides its cores. So do a constraint
attached where it cannot be evaluated, and a sub-component with an offset whose thread range
allows more than one thread.

## Further examples

[access-profiling](https://github.com/ACCESS-NRI/access-profiling) runs its profiling and scaling
studies of the ACCESS models on these searches. Its models of ACCESS-ESM1.6 and ACCESS-OM3, and
its guides to finding the best layout of a budget and to scaling studies, show how a study
chooses its allocation strategies, and what they find.
