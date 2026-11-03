# Components and layouts

A layout search needs two things: a description of the model, as a tree of components, and a
number of cores. It then finds every way of running the model on those cores. This page describes
the tree and the layouts found for it; [Allocating cores](allocations.md) and
[Constraints](constraints.md) describe how to narrow the search down, and
[Searching for layouts](searching.md) how to run it.

The classes of the layout search are imported from the modules that define them, not from
`access.config`.

## The component tree

A model is a tree of `ParallelComponent`s. A component is either a *leaf*, which runs work
itself, or a *parent*, which hands its cores to its sub-components, never both. A leaf may have a
`Domain`, the grid of points its work is decomposed over. Here is a model shaped like
ACCESS-ESM1.6, with an atmosphere, an ocean and a sea ice component:

```python
from access.config.parallel_component import ParallelComponent
from access.config.parallel_domain import Domain

um7 = ParallelComponent("UM7", domain=Domain((192, 144)))
mom5 = ParallelComponent("MOM5", domain=Domain((360, 300)))
cice5 = ParallelComponent("CICE5", domain=Domain((360,)))
esm = ParallelComponent("ESM1.6", subcomponents=(um7, mom5, cice5))
```

The names of a parent's sub-components have to be different, since they are how the allocation
strategies and the group constraints refer to them. The tree holds what is true of every way of
running the model: its structure, and, as [Constraints](constraints.md) shows, the requirements
every layout of it has to meet. What one study prefers belongs in its allocation strategy instead.

## Cores, ranks and threads

What is shared out is cores. A leaf runs the cores it is given as MPI ranks of one or more OpenMP
threads each, so it runs `n_cores // threads_per_rank` ranks, for every thread count in its range
that divides its cores exactly. A rank costs as many cores as it runs threads, which is why two
components on the same number of ranks can take very different shares of the cores. The range of
thread counts comes from the [allocation strategy](allocations.md#threads), and is a single
thread per rank by default.

The ranks of a leaf with a domain are arranged in a grid with as many dimensions as the domain,
an `MPICartesianGrid`, and there is a layout for every such grid:

```python
from access.config.parallel_domain import DomainDecompositionSpec
from access.config.parallel_mpi_grid import MPICartesianGrid

print([grid.shape for grid in MPICartesianGrid.iter_grids(2, 6)])
decomposition = DomainDecompositionSpec(um7.domain, MPICartesianGrid((2, 3)))
print(decomposition.n_ranks, decomposition.mean_local_shape)
```

```text
[(1, 6), (2, 3), (3, 2), (6, 1)]
6 (96.0, 48.0)
```

A `DomainDecompositionSpec` pairs a domain with a grid of ranks. It says how the domain is to be
split, and gives the mean size of the block each rank holds, but not how the points are actually
distributed among the ranks: that is up to the model. A leaf with no domain has one layout for
each usable thread count, with no grid.

## Layouts

A layout is a tree of `ComponentLayout`s of the same shape as the component tree, holding what
each component was given. Here is the first layout of the model on 8 cores:

```python
from access.config.parallel_layouts import iter_layouts


def show(layout, indent=""):
    grid = layout.decomposition.grid.shape if layout.decomposition is not None else None
    print(
        f"{indent}{layout.name}: {layout.n_cores} cores, {layout.n_ranks} ranks, "
        f"{layout.threads_per_rank} threads per rank, grid {grid}"
    )
    for sub in layout.sub_layouts:
        show(sub, indent + "    ")


layout = next(iter_layouts(esm, 8))
show(layout)
print(layout.used_cores, layout.idle_cores)
```

```text
ESM1.6: 8 cores, 3 ranks, None threads per rank, grid None
    UM7: 1 cores, 1 ranks, 1 threads per rank, grid (1, 1)
    MOM5: 1 cores, 1 ranks, 1 threads per rank, grid (1, 1)
    CICE5: 1 cores, 1 ranks, 1 threads per rank, grid (1,)
3 5
```

A layout holds:

- `name`, the component's;
- `n_cores`, the cores it was given;
- `n_ranks`, the ranks it runs, or, for a parent, the ranks its sub-components run;
- `threads_per_rank`, a leaf's thread count, which is `None` for a parent;
- `decomposition`, a leaf's `DomainDecompositionSpec`, or `None` when it has no domain;
- `sub_layouts`, the layouts of a parent's sub-components, in the order the component tree lists
  them;
- `used_cores` and `idle_cores`, how many of its cores are spent and how many are not.

A leaf spends its cores exactly. A parent may leave some of its cores to no sub-component at all,
as the root does above, with five of its eight cores idle. `idle_cores` counts those of one parent
only, not those its sub-components leave idle in turn. The layouts of the sub-components are found
by their position in `sub_layouts`, so pair them with the component tree's names, or read
`sub.name`, rather than counting on an order.

## Partitioned and shared cores

By default a parent *partitions* its cores: each sub-component receives cores of its own, and the
parent needs at least their total. A parent may instead have its sub-components *share* its
cores, all drawing on the same range and running on it in turn, as the components driven by NUOPC
in ACCESS-OM3 do. Where in the range each sub-component starts is its `core_offset`, the core
index ESMF calls `rootpe`:

```python
from access.config.parallel_component import CoreSharing

shared = ParallelComponent(
    "shared",
    core_sharing=CoreSharing.SHARED,
    subcomponents=(ParallelComponent("cpl"), ParallelComponent("ice", core_offset=2)),
)
for layout in iter_layouts(shared, 4):
    print([(sub.name, sub.core_offset, sub.n_cores) for sub in layout.sub_layouts])
```

```text
[('cpl', 0, 2), ('ice', 2, 2)]
[('cpl', 0, 3), ('ice', 2, 2)]
[('cpl', 0, 4), ('ice', 2, 1)]
[('cpl', 0, 4), ('ice', 2, 2)]
```

A shared parent's range has to be covered: every core of it is run on by at least one
sub-component, and none is left idle. That is why `cpl` never has a single core here, which would
leave the second core with nothing on it. A shared parent's `n_cores` is how far its furthest
sub-component reaches into the range, and its `n_ranks` the ranks they sit on between them.

Two rules come with offsets:

- Only the sub-components of a shared parent can have one. Under a partitioned parent, each
  sub-component starts where the last one ends, and an offset raises `ValueError`.
- A sub-component with an offset other than 0 runs one thread per rank, since its offset counts
  cores, and placement is assumed to be contiguous. Give such a sub-component a thread range of
  `(1, 1)`: a range allowing more makes the search raise `ValueError` when it reaches a thread
  count above one.

Each sub-component of a shared parent is offered every core count that fits, independently of the
others, so the number of combinations multiplies quickly. [Allocating cores](allocations.md)
describes how to pin them down.
