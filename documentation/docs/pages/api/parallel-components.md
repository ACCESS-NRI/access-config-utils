# Components and layouts

These modules hold the structure that a search for parallelisation layouts is run on, and what it
finds. The structure is a tree of
[`ParallelComponent`][access.config.parallel_component.ParallelComponent]s. Each layout is a tree of
[`ComponentLayout`][access.config.parallel_component.ComponentLayout]s of the same shape. The
domains and the process grids that the layout of a leaf is decomposed with are in `parallel_domain`
and `parallel_mpi_grid`. [Components and layouts](../user-guide/layouts/components.md) introduces
them.

::: access.config.parallel_component

::: access.config.parallel_domain

::: access.config.parallel_mpi_grid
