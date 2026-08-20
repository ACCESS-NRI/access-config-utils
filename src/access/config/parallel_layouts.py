# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Enumerate the valid parallel layouts of a component tree under a core budget.

Given a total core budget, ``iter_layouts`` yields every valid layout for a component tree
one at a time. It is lazy by design, as layout counts can reach millions for realistic
numbers of core used in production.

How the cores are divided among a parent's sub-components is determined by the
``AllocationStrategy`` tree in ``access.config.parallel_allocation_strategies``, which the
caller passes as *allocations*. This module schedules the search — recursion, memoisation,
constraint filtering.

The search works its way down the component tree and back up again: cores are handed down,
and layouts are handed back up. The same question is asked at every component - *given this
component and this many cores, what are all its valid layouts?* - and how it is answered
depends only on whether the component has sub-components.

A component with no sub-components spends its cores itself. It must spend them exactly, so a
thread count is usable only when it divides the cores on offer, and the ranks that follow
are ``number of cores // threads_per_rank``. Each of those rank counts can be arranged into
a process grid over the component's domain in one or more ways, and every (thread count,
process grid) pair that comes out is one layout. A component with no domain still gives one
layout per usable thread count, with no grid attached.

A component with sub-components hands its cores down instead. It asks the allocation
strategies for every way of dividing its cores among the sub-components - this is where
fixed counts, ratios and free ranges do their work, and a division may leave cores idle -
and for each division it puts the same question to every sub-component, with the share that
division gave it. Each sub-component answers with all of its own layouts, and any choice of
one layout per sub-component is a layout of the parent, whose rank count is then the sum of
its sub-components' rank counts. So the parent's layouts for one division are every such
combination, and its layouts overall are those of all its divisions together.

That combining step is where the numbers come from: two sub-components with 30 layouts each
already give the parent 900 for a single division, before the next division is even tried.
It is why a three-component model yields 14k layouts on 24 cores and 9.0M on 144.

Two things keep this affordable. The first is that layouts are rejected as early as they can
be rather than at the end. A constraint attached to a component or to a strategy is applied
at that component, so an unusable sub-component layout is discarded before anything is built
on top of it, and a division in which some sub-component has no layout at all is dropped
before its combinations are formed. The second is that no question is answered twice. The
layouts of the atmosphere on 40 cores do not depend on what the ocean received, so the
answer is recorded the first time it is worked out and reused every later time the same
question comes up - and it comes up constantly, because every division that gives the
atmosphere 40 cores asks it again. For a three-component model on 48 cores this turns tens
of thousands of questions into a couple of hundred pieces of real work.

Thread ranges travel down with the cores. A strategy that sets a ``tpr_range`` replaces the
range it inherited, for its component and everything below it, while the rest inherit the
root's. This is why the same component on the same cores is not always the same question: a
different inherited range gives genuinely different layouts, so the range is recorded
alongside the component, its core count and its strategy.

Only the layouts of the whole tree are produced one at a time. Below the root, a
sub-component's layouts are held as a complete list, because forming the combinations means
going through each list many times over. Memory therefore grows with the recorded sub-tree
answers rather than with the number of complete layouts, which is what makes a search worth
millions of layouts possible at all.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator

from access.config.parallel_allocation_strategies import (
    AllocationStrategy,
    RootAllocation,
    iter_core_splits,
    resolve_root_strategy,
)
from access.config.parallel_component import ComponentLayout, ParallelComponent

__all__ = [
    "iter_layouts",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumeration internals
# ---------------------------------------------------------------------------


# Type of the per-call memo of already-enumerated subtrees. The key is everything the
# result depends on: (component, n_cores, tpr_range, strategy). ``total_cores`` is
# deliberately absent - it is fixed for the life of one search, which is exactly why the
# cache belongs to a ``_LayoutSearch`` instance rather than being passed around.
_SubtreeCache = dict[tuple[ParallelComponent, int, tuple[int, int], AllocationStrategy], tuple[ComponentLayout, ...]]


class _LayoutSearch:
    """One run of the enumerator, holding what stays fixed for the whole of it.

    Two things are invariant across a search — the core budget every constraint is judged
    against, and the subtree memo — while ``(component, n_cores, tpr_range, strategy)``
    varies with the recursion and is exactly the memo key. Making the invariant pair state
    leaves the recursive methods carrying only what actually changes.

    The cache is per-instance, and an instance is per ``iter_layouts`` call, which is what
    makes omitting ``total_cores`` from the key sound: a cache never outlives the budget it
    was filled under.
    """

    __slots__ = ("_cache", "_total_cores")

    def __init__(self, total_cores: int) -> None:
        self._total_cores = total_cores
        self._cache: _SubtreeCache = {}

    def run(self, component: ParallelComponent, strategy: RootAllocation) -> Iterator[ComponentLayout]:
        """Yield the layouts of *component* holding the whole budget, as the root does.

        The root's ``tpr_range`` seeds the inherited thread range, since there is nothing
        above it to inherit from.
        """
        return self._iter_component_layouts(component, self._total_cores, strategy.tpr_range, strategy)

    def _cached_component_layouts(
        self,
        component: ParallelComponent,
        n_cores: int,
        tpr_range: tuple[int, int],
        strategy: AllocationStrategy,
    ) -> tuple[ComponentLayout, ...]:
        """Return every valid layout for *component* at *n_cores*, from the memo if seen.

        Sibling core splits ask for the same subtree over and over — every split that
        gives the atmosphere 40 cores needs the same set of atmosphere layouts — and the
        answer depends only on ``(component, n_cores, tpr_range, strategy)``, not on what
        the siblings received. A three-component model on 48 cores makes tens of thousands
        of these requests for a couple of hundred distinct subtrees, so almost all of them
        are answered from here.

        ``ComponentLayout`` is immutable, so a cached tuple can be handed out again
        as-is rather than copied.
        """
        key = (component, n_cores, tpr_range, strategy)
        layouts = self._cache.get(key)
        if layouts is None:
            layouts = tuple(self._iter_component_layouts(component, n_cores, tpr_range, strategy))
            if not layouts:
                logger.debug(
                    "no valid layout for component %r on %d core(s) with %d-%d thread(s) per rank",
                    component.name,
                    n_cores,
                    *tpr_range,
                )
            self._cache[key] = layouts
        return layouts

    def _iter_component_layouts(
        self,
        component: ParallelComponent,
        n_cores: int,
        tpr_range: tuple[int, int],
        strategy: AllocationStrategy,
    ) -> Iterator[ComponentLayout]:
        """Yield all valid ``ComponentLayout`` objects for *component* given *n_cores*.

        The strategy's own ``tpr_range``, when it sets one, replaces the range inherited
        from the enclosing strategy, for this component and everything below it.
        """
        effective_tpr_range = strategy.resolve_tpr_range(tpr_range)
        all_constraints = component.local_constraints + strategy.local_constraints

        if component.subcomponents:
            candidates = self._iter_parent_candidates(component, n_cores, effective_tpr_range, strategy)
        else:
            candidates = _iter_leaf_candidates(component, n_cores, effective_tpr_range)

        # Hoisted: read once per component rather than once per candidate.
        total_cores = self._total_cores
        for candidate in candidates:
            # Unconstrained components are the common case, so skip building the
            # generator that all() would consume when there is nothing to check.
            if all_constraints and not all(c.is_satisfied(candidate, total_cores) for c in all_constraints):
                continue
            yield candidate

    def _iter_parent_candidates(
        self,
        component: ParallelComponent,
        n_cores: int,
        tpr_range: tuple[int, int],
        strategy: AllocationStrategy,
    ) -> Iterator[ComponentLayout]:
        """Yield the ways a parent can divide *n_cores* among its sub-components.

        A split survives only if every sub-component has at least one layout for its share
        and the combination satisfies the group constraints of both the component and its
        allocation strategy.
        """
        sub_names = tuple(sub.name for sub in component.subcomponents)
        sub_strategies = strategy.strategies_for(sub_names)
        all_group_constraints = component.group_constraints + strategy.group_constraints
        total_cores = self._total_cores

        for core_split in iter_core_splits(sub_strategies, n_cores, sub_names):
            sub_options = [
                self._cached_component_layouts(sub, c, tpr_range, sub_strategy)
                for sub, c, sub_strategy in zip(component.subcomponents, core_split, sub_strategies, strict=True)
            ]
            # A sub-component with no valid layout kills the whole split, so stop before
            # taking a product that is guaranteed to be empty.
            if not all(sub_options):
                continue
            for combo in itertools.product(*sub_options):
                if all_group_constraints and not all(
                    gc.is_satisfied(combo, total_cores) for gc in all_group_constraints
                ):
                    continue
                yield ComponentLayout(
                    name=component.name,
                    n_cores=n_cores,
                    n_ranks=sum(sub.n_ranks for sub in combo),
                    threads_per_rank=None,
                    decomposition=None,
                    sub_layouts=combo,
                )


def _iter_leaf_candidates(
    component: ParallelComponent,
    n_cores: int,
    tpr_range: tuple[int, int],
) -> Iterator[ComponentLayout]:
    """Yield the ways a leaf can spend *n_cores* as ranks and threads.

    A leaf spends its allocation exactly, so a thread count is usable only when it
    divides *n_cores*, giving ``n_cores // tpr`` ranks. Cores left idle are the
    business of the parent that chose not to hand them out.
    """
    min_tpr, max_tpr = tpr_range
    for tpr in range(min_tpr, max_tpr + 1):
        if n_cores % tpr != 0:
            logger.debug(
                "component %r cannot use %d thread(s) per rank: it does not divide its %d core(s)",
                component.name,
                tpr,
                n_cores,
            )
            continue
        n_ranks = n_cores // tpr
        for decomposition in component.iter_decompositions(n_ranks):
            yield ComponentLayout(
                name=component.name,
                n_cores=n_cores,
                n_ranks=n_ranks,
                threads_per_rank=tpr,
                decomposition=decomposition,
            )


def _log_if_empty(
    layouts: Iterator[ComponentLayout], component: ParallelComponent, total_cores: int
) -> Iterator[ComponentLayout]:
    """Pass *layouts* through, reporting the search as a dead end if it yields nothing."""
    found = False
    for layout in layouts:
        found = True
        yield layout
    if not found:
        logger.debug("no valid layout for %r on %d core(s)", component.name, total_cores)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def iter_layouts(
    component: ParallelComponent,
    total_cores: int,
    *,
    allocations: RootAllocation | None = None,
) -> Iterator[ComponentLayout]:
    """Yield the valid ``ComponentLayout`` trees for *component* one at a time.

    The root component receives all *total_cores*. Every parent divides the cores it holds
    among its sub-components according to *allocations*, and every leaf turns the cores it
    receives into ``n_cores // tpr`` MPI ranks of ``tpr`` OpenMP threads, for each thread
    count in its range that divides its allocation. Constraints filter the candidates at
    each level.

    Because cores rather than ranks are divided, components may use different thread counts:
    set ``AllocationStrategy.tpr_range`` on the sub-components that should be threaded, and
    the rest inherit the root's.

    This never holds the whole result set in memory, which is why it is the only entry
    point: the number of layouts grows steeply with the core budget (a three-component model
    yields 14k layouts on 24 cores, 2.2M on 96 and 9.0M on 144), so materialising them all
    to inspect the first few, or to filter them down, is often the dominant cost. Take
    ``list()`` of it only once the search is known to be small. Arguments are validated
    eagerly, before the first layout is requested.

    Note: an empty result has many possible causes, like fixed allocations that do not fit
    the budget, no thread count in a component's range that divides its core allocation, or
    a constraint that nothing satisfies. Each dead end is reported on the
    ``access.config.parallel_layouts`` logger at ``DEBUG`` level, naming the component and
    the counts involved, so raising the log level is the way to find out which. To see them,
    call ``logging.basicConfig()`` and then
    ``logging.getLogger("access.config.parallel_layouts").setLevel(logging.DEBUG)``.

    Args:
        component (ParallelComponent): Root of the component tree. It receives all of
            *total_cores*.
        total_cores (int): Total number of CPU cores available. Must be >= 1.
        allocations (RootAllocation | None): Allocation strategy for *component*, with one
            entry in ``subcomponents`` for every sub-component at every level - there is no
            default mode, so a component to be left open carries an explicit
            ``FreeAllocation()``. ``None`` (the default) builds an all-free tree at ``(1,
            1)`` thread. The root is a ``RootAllocation``, which states no core count of its
            own because *component* always receives the whole budget; its ``tpr_range`` is
            the thread range every component inherits unless it overrides it, and its
            ``subcomponents``, ``local_constraints`` and ``group_constraints`` are honoured
            as usual.

    Yields:
        ComponentLayout: Each valid layout tree, in enumeration order.

    Raises:
        TypeError: If *allocations* is not a ``RootAllocation``.
        ValueError: If *total_cores* < 1, or if the strategy tree does not name exactly
            *component*'s sub-components, at any depth.

    Examples:
        Take the first valid layout without enumerating the rest:

        >>> from access.config.parallel_domain import Domain
        >>> comp = ParallelComponent("model", domain=Domain((12, 8)))
        >>> next(iter_layouts(comp, total_cores=4)).decomposition.grid.shape
        (1, 4)

        Or collect a search small enough to fit in memory:

        >>> layouts = list(iter_layouts(comp, total_cores=4))
        >>> [layout.decomposition.grid.shape for layout in layouts]
        [(1, 4), (2, 2), (4, 1)]
    """
    if total_cores < 1:
        raise ValueError(f"total_cores must be >= 1, got {total_cores}.")
    root_strategy = resolve_root_strategy(component, allocations)
    # A search per call, so its memo lives and dies with the returned iterator and nothing
    # is retained between calls.
    return _log_if_empty(_LayoutSearch(total_cores).run(component, root_strategy), component, total_cores)
