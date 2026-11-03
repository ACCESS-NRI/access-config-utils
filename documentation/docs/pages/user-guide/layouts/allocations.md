# Allocating cores

Left to itself, a search offers every sub-component every number of cores that fits, which is
far more than a study wants. An *allocation strategy* says how many cores each component may
receive. It is a tree of its own, mirroring the component tree: one strategy for each component,
keyed by the component's name, under a `RootAllocation` for the root.

The examples on this page use a model of three components with no domain, so that each way of
dividing the cores is one layout, and print the core counts each layout gives the components:

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

Each component's strategy uses one of four modes, each a class of its own.

**`FixedAllocation`** gives a component an exact number of cores:

```python
from access.config.parallel_allocation_strategies import FixedAllocation, RootAllocation

allocations = RootAllocation(
    subcomponents={"atm": FixedAllocation(12), "ocn": FixedAllocation(8), "ice": FixedAllocation(4)}
)
print(splits(allocations))
```

```text
[(12, 8, 4)]
```

**`WeightedAllocation`** fixes the proportions between siblings, and leaves open how many cores
that comes to: siblings of weights 3 and 2 receive `3 * k` and `2 * k` cores, for every multiplier
`k` that fits.

```python
from access.config.parallel_allocation_strategies import WeightedAllocation

allocations = RootAllocation(
    subcomponents={"atm": WeightedAllocation(3), "ocn": WeightedAllocation(2), "ice": FixedAllocation(4)}
)
print(splits(allocations))
```

```text
[(3, 2, 4), (6, 4, 4), (9, 6, 4), (12, 8, 4)]
```

**`FreeAllocation`** takes any number of cores between `min_cores`, 1 by default, and
`max_cores`, unbounded by default. `core_step` offers only its multiples, which is how a component
is given whole nodes, or a count its decomposition needs:

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

```text
[(12, 6, 4), (12, 7, 4), (12, 8, 4), (13, 6, 4), (13, 7, 4), (14, 6, 4)]
[(8, 8, 4), (12, 8, 4)]
```

**`WholeRangeAllocation`** gives a sub-component of a [shared parent](components.md#partitioned-and-shared-cores)
the whole of the parent's range, from its own offset to the end. It is how the sub-components of
a shared parent are pinned to whatever range the parent is given.

`FreeAllocation()`, with no bounds, leaves a component open. Every component of the tree needs a
strategy, with exactly the names of the component tree at every level, since there is no default
mode. A name missing from the strategy, or one the component tree does not have, raises
`ValueError` when the search is started:

```python
try:
    iter_layouts(model, 24, allocations=RootAllocation(subcomponents={"atm": FreeAllocation(), "ocn": FreeAllocation()}))
except ValueError as error:
    print(error)
```

```text
RootAllocation: no allocation given at root for {'ice'}. Every sub-component needs one - there is no default mode. Add a FreeAllocation() for the ones that should be left open.
```

Without any allocation strategy, `iter_layouts` uses a `FreeAllocation()` for every component,
with one thread per rank.

## How the cores of a parent are divided

Siblings in different modes are served in a fixed order: the fixed counts are taken out of the
parent's cores first, then the multiplier of the weighted siblings is chosen from what is left,
and then the free siblings divide the rest. Cores the free siblings do not take are left idle:
`MaxWastedCoreFractionConstraint` on the parent bounds how many, as
[Constraints](constraints.md) shows. A parent is sized the same way by its own parent, and then
divides the cores it receives among its sub-components, so the strategies of a parent's
sub-components go in its strategy's `subcomponents`, as the root's do.

Under a [shared parent](components.md#partitioned-and-shared-cores) nothing is divided: each
sub-component is offered every count that fits its part of the range, from its offset. Fixed,
free and whole-range strategies can size such a sub-component, but a weight, which is a share
of something divided, cannot, and raises `ValueError` once the search reaches it. A
`WholeRangeAllocation` under a parent that divides its cores raises `ValueError` the same way.

## Fractions of the total

A fixed or free allocation can be given as a fraction of the total number of cores of the search
rather than as a number of cores, which lets one strategy tree serve every size of a scaling
study:

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

```text
[(12, 6, 2), (12, 7, 2), (12, 8, 2)]
[(24, 12, 2), (24, 13, 2), (24, 14, 2), (24, 15, 2)]
```

A fraction is always one of the whole total, even for a component deep in the tree, and is turned
into a number of cores once, before the search starts. A fixed count is rounded down, and a free
range widened to whole cores, its lower bound rounded down and its upper bound up: 0.3 of 24 cores
is 7.2, so up to 8. A product within a hair of a whole number is that number, so that 0.29 of 100
cores is 29 rather than 28. Where a free allocation is given both ways, the count is a limit on the
fraction: `min_cores` raises the lower bound, and `max_cores` lowers the upper one. Bounds that
leave no count at a given total raise `ValueError` when the search is started.

## Threads

Each strategy may set a `thread_range`, the inclusive range of OpenMP threads per rank its
component may run. A strategy without one inherits its parent's, and the root's is `(1, 1)` by
default, so a model runs one thread per rank unless told otherwise. Setting a range on one
component is how it alone is threaded:

```python
allocations = RootAllocation(
    subcomponents={"atm": FixedAllocation(8, thread_range=(1, 4)), "ocn": FixedAllocation(8), "ice": FixedAllocation(4)}
)
for layout in iter_layouts(model, 24, allocations=allocations):
    atm = layout.sub_layouts[0]
    print(atm.n_ranks, "ranks of", atm.threads_per_rank, "threads")
```

```text
8 ranks of 1 threads
4 ranks of 2 threads
2 ranks of 4 threads
```

A thread count is used only where it divides the component's cores: 3 threads do not divide 8
cores, so there is no layout with them. The range on a parent's strategy is only the default its
sub-components inherit, since a parent runs no threads of its own.

## Constraints on a strategy

A strategy can also carry constraints, in its `local_constraints` and `group_constraints`, which
are checked together with those of its component. They are what a study adds to the requirements
the component tree states, as [Constraints](constraints.md) describes.
