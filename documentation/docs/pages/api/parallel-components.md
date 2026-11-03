# Components and layouts

The structure a layout search is run on, and what it finds: a tree of
[`ParallelComponent`][access.config.parallel_component.ParallelComponent]s, and for each layout a
tree of [`ComponentLayout`][access.config.parallel_component.ComponentLayout]s of the same
shape. The domains and process grids a leaf's layout is decomposed with are in
`parallel_domain` and `parallel_mpi_grid`. [Components and
layouts](../user-guide/layouts/components.md) introduces them.

::: access.config.parallel_component

::: access.config.parallel_domain

::: access.config.parallel_mpi_grid
