# Components and layouts

A parallelisation layout, or layout for short, says how the cores of a job are shared among the
components of a model. A layout search needs two things: a description of the model, as a tree of
components, and a number of cores. It then finds every way of running the model on those cores. Here
we look at the tree, and at the layouts that are found for it. [Allocating cores](allocations.md)
and [Constraints](constraints.md) describe how to narrow the search down, and [Searching for
layouts](searching.md) describes how to run it.

Note that the classes of the layout search are imported from the modules that define them, and not
from `access.config`.

## The component tree

A model is a tree of `ParallelComponent`s. A component is either a *leaf*, which does work itself,
or a *parent*, which hands its cores to its sub-components. It is never both. A leaf can have a
`Domain`, which is the grid of points that its work is decomposed over. As an example, we describe a
model with the shape of ACCESS-ESM1.6, with an atmosphere, an ocean and a sea ice component:

```python
from access.config.parallel_component import ParallelComponent
from access.config.parallel_domain import Domain

um7 = ParallelComponent("UM7", domain=Domain((192, 144)))
mom5 = ParallelComponent("MOM5", domain=Domain((360, 300)))
cice5 = ParallelComponent("CICE5", domain=Domain((360,)))
esm = ParallelComponent("ESM1.6", subcomponents=(um7, mom5, cice5))
```

The sub-components of a parent must have different names, because the allocation strategies and the
group constraints refer to them by name. The tree holds what is true of every way of running the
model. This is its structure and, as [Constraints](constraints.md) shows, the requirements that
every layout has to meet. The preferences of a particular study belong in its allocation strategy
instead.

## Cores, ranks and threads

The search shares out cores. A leaf runs the cores it is given as MPI ranks, each with one or more
OpenMP threads. It therefore runs `n_cores // threads_per_rank` ranks, for every thread count in its
range that divides its cores exactly. A rank costs as many cores as it runs threads. This is why two
components with the same number of ranks can take very different shares of the cores. The range of
thread counts comes from the [allocation strategy](allocations.md#threads), and is a single thread
per rank by default.

The ranks of a leaf with a domain are arranged in a grid with as many dimensions as the domain. This
grid is an `MPICartesianGrid`, and there is a layout for every such grid. For example, we list the
two-dimensional grids of 6 ranks, and decompose the domain of the atmosphere over one of them:

```python
from access.config.parallel_domain import DomainDecompositionSpec
from access.config.parallel_mpi_grid import MPICartesianGrid

print([grid.shape for grid in MPICartesianGrid.iter_grids(2, 6)])
decomposition = DomainDecompositionSpec(um7.domain, MPICartesianGrid((2, 3)))
print(decomposition.n_ranks, decomposition.mean_local_shape)
```

This prints the grids, and then the number of ranks and the mean size of a subdomain:

```text
[(1, 6), (2, 3), (3, 2), (6, 1)]
6 (96.0, 48.0)
```

A `DomainDecompositionSpec` pairs a domain with a grid of ranks. It says how the domain is to be
split, and gives the mean size of the block that each rank holds. It does not say how the points are
actually distributed among the ranks, which is up to the model. A leaf with no domain has one layout
for each thread count it can use, with no grid.

## Layouts

A layout is a tree of `ComponentLayout`s with the same shape as the component tree. It holds what
each component was given. Here we print the first layout of our model on 8 cores:

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

This prints the layout, and then the used and the idle cores of the root:

```text
ESM1.6: 8 cores, 3 ranks, None threads per rank, grid None
    UM7: 1 cores, 1 ranks, 1 threads per rank, grid (1, 1)
    MOM5: 1 cores, 1 ranks, 1 threads per rank, grid (1, 1)
    CICE5: 1 cores, 1 ranks, 1 threads per rank, grid (1,)
3 5
```

A layout holds:

- `name`, the name of the component;
- `n_cores`, the number of cores it was given;
- `n_ranks`, the number of ranks it runs or, for a parent, that its sub-components run;
- `threads_per_rank`, the thread count of a leaf, which is `None` for a parent;
- `decomposition`, the `DomainDecompositionSpec` of a leaf, or `None` when it has no domain;
- `sub_layouts`, the layouts of the sub-components of a parent, in the order in which the component
  tree lists them;
- `used_cores` and `idle_cores`, how many of its cores are used and how many are not.

A leaf uses all its cores. A parent can leave some of its cores to no sub-component at all, as the
root does above, where five of its eight cores are idle. `idle_cores` only counts the idle cores of
that parent, and not those that its sub-components leave idle in turn. The layouts of the
sub-components are identified only by their position in `sub_layouts`. To find the layout of a
particular component, pair `sub_layouts` with the names in the component tree, or read `sub.name`,
instead of assuming an order.

## Partitioned and shared cores

By default a parent *partitions* its cores: each sub-component receives cores of its own, and the
parent needs at least their total. A parent can instead make its sub-components *share* its cores.
They then all draw on the same range of cores and run on it in turn, as the components driven by
NUOPC in ACCESS-OM3 do. The `core_offset` of a sub-component says where in the range it starts. This
is the core index that ESMF calls `rootpe`. Here two components share 4 cores, and the second one
starts at the third core:

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

This prints the name, the offset and the cores of each sub-component, for every layout:

```text
[('cpl', 0, 2), ('ice', 2, 2)]
[('cpl', 0, 3), ('ice', 2, 2)]
[('cpl', 0, 4), ('ice', 2, 1)]
[('cpl', 0, 4), ('ice', 2, 2)]
```

The range of a shared parent has to be covered: at least one sub-component runs on every core of it,
and no core is left idle. This is why `cpl` never has a single core here, which would leave nothing
running on the second core. The `n_cores` of a shared parent is how far its furthest sub-component
reaches into the range. Its `n_ranks` is the number of ranks that its sub-components occupy between
them.

Offsets come with two rules:

- Only the sub-components of a shared parent can have an offset. Under a partitioned parent, each
  sub-component starts where the previous one ends, and an offset raises `ValueError`.
- A sub-component with an offset other than 0 runs one rank on each core. The reason is that the
  offset counts cores, and the placement is assumed to be contiguous. The search therefore skips its
  layouts with more than one thread per rank. If it is a parent, the search also skips its layouts
  that leave cores idle. A sub-component left with no layout is a dead end, which is reported at
  `DEBUG` level.

Each sub-component of a shared parent is offered every core count that fits, independently of the
others. The number of combinations therefore multiplies quickly. [Allocating cores](allocations.md)
describes how to pin them down.
