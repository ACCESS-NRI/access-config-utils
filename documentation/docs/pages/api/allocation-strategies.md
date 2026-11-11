# Allocation strategies

An allocation strategy says how the cores of a parent are divided among its sub-components. It is a
tree of strategies with the same shape as the component tree, with one allocation mode at each node.
[Allocating cores](../user-guide/layouts/allocations.md) introduces the modes.

::: access.config.parallel_allocation_strategies

## Package-internal contract

These functions are only public because `parallel_layouts` calls them. They are the scheduling that
has to know every allocation mode at once, and they are not meant to be called from outside the
package. [How the layout search works](../developer-guide/layout-search.md) explains where they fit.

::: access.config.parallel_allocation_strategies.iter_core_splits
    options:
      heading_level: 3

::: access.config.parallel_allocation_strategies.iter_shared_core_shares
    options:
      heading_level: 3

::: access.config.parallel_allocation_strategies.resolve_root_strategy
    options:
      heading_level: 3
