# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Classes to specify how cores are divided among components.

As there are many different ways to allocate cores across components, one needs to be able
to specify how cores should be allocated to each component. This is done through the
``AllocationStrategy`` class, which describes how cores should be allocated to each
component. Using instances of ``AllocationStrategy``, one can then build a tree that mirrors
the component tree, and use it to enumerate all valid layouts for that component tree under
that allocation strategy. Because the allocation strategy tree is independent of the
component tree, one can use the same component tree with different allocation strategies to
explore different ways of distributing the cores.

Each node of the strategy tree uses exactly one allocation mode, and the mode is implemented
as one of the following subclasses of ``AllocationStrategy``:

* ``RootAllocation`` is always the root of the strategy tree.
* ``FixedAllocation`` names an exact core count.
* ``RatioAllocation`` pins the proportions between siblings, leaving their scale open.
* ``FreeAllocation`` takes any count within some bounds.

The ``RootAllocation`` states no core count of its own, because the root component always
receives the whole budget.

Just like components, allocation strategies may have constraints attached to them, which
filter out layouts that do not meet certain criteria. This enables to set more stringent
constraints than the ones defined in the component tree and further restrict the layouts
that are enumerated.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

from access.config.parallel_component import GroupConstraint, LocalConstraint, ParallelComponent

__all__ = [
    "AllocationStrategy",
    "RootAllocation",
    "FixedAllocation",
    "RatioAllocation",
    "FreeAllocation",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AllocationStrategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class AllocationStrategy(ABC):
    """Strategy to use when distributing cores within a given component.

    An allocation strategy instance determines how many cores a subcomponent receives from
    its parent. By creating a tree of allocation strategies that mirrors the component tree,
    one can describe how cores are distributed among sub-components, independently of the
    ``ParallelComponent`` structure (domains and constraints).

    Note that cores, not MPI ranks, are what gets divided. This means that a rank costs as
    many cores as it runs OpenMP threads, so two components on the same number of ranks can
    take very different shares of the total core count. A leaf component turns the cores it
    receives into ``n_cores // threads_per_rank`` MPI ranks, choosing its thread count from
    ``tpr_range``.

    Constrains can be attached to an allocation strategy. These will be added to the
    Component's own constraints and will be checked when enumerating layouts. This allows to
    set more stringent constraints than the ones defined in the component tree and further
    restrict the layouts that are enumerated.

    This class is abstract and holds data shared by every strategy. It is declared
    ``frozen`` (``frozen=True``) so that it is immutable and hashable, which is what lets a
    strategy form part of the layouts iterator (``iter_layouts``) subtree memoisation key.
    ``subcomponents`` is a mapping, so it is copied behind a read-only view on construction
    and left out of the generated hash; see the comment on the field itself.

    Every field here is keyword-only (``kw_only=True``), which keeps the positional slots
    free for the subclasses' own fields — hence ``FixedAllocation(12)`` rather than
    ``FixedAllocation(tpr_range=None, ..., n_cores=12)``.

    Args:
        tpr_range (tuple[int, int] | None): Inclusive ``(min, max)`` range of OpenMP
            threads per rank this component may use. ``None`` (the default) inherits the
            range from the enclosing strategy, and ultimately from the root's, which is
            never ``None``. Setting it here is how components get *different* thread counts:
            give the root ``(1, 1)`` and one leaf ``(1, 4)`` and only that leaf is threaded.
            A component only spends cores itself if it is a leaf, so on a parent this acts
            purely as the default inherited by its sub-components.
        subcomponents (Mapping[str, AllocationStrategy]): Allocation strategies for direct
            sub-components, keyed by component name. Omit (or pass ``{}``) for leaf
            components. Every sub-component of the matching component must appear, since
            there is no default mode; ``FreeAllocation()`` is the entry for one that should
            be left open. The mapping is copied on construction and exposed as a read-only
            view, so later changes to the dict passed in here do not affect the strategy.
        local_constraints (tuple[LocalConstraint, ...]): Constraints applied only when
            this allocation strategy is active, checked alongside the component's own
            ``ParallelComponent.local_constraints``. Use this for strategy-specific filters
            (e.g. aspect-ratio limits) that should not be hard-coded into the component
            definition.
        group_constraints (tuple[GroupConstraint, ...]): Group constraints applied only
            when this allocation strategy is active, checked alongside the
            ``ParallelComponent.group_constraints`` of the component this strategy is
            attached to. Like all group constraints they judge that component's
            sub-components together, so they belong on the parent's strategy rather than on
            a sibling's. Use this for strategy-specific cross-component constraints (e.g.
            rank-ratio limits) that should not be hard-coded into the component definition.

    Raises:
        ValueError: If ``tpr_range`` is not an inclusive ``(min, max)`` pair of thread
            counts >= 1.

    Examples:
        >>> strategy = RootAllocation(subcomponents={
        ...     "UM7":   FreeAllocation(tpr_range=(1, 4)),   # threaded
        ...     "MOM5":  FreeAllocation(),                   # inherits the thread range
        ...     "CICE5": FixedAllocation(12),
        ... })
        >>> strategy.subcomponents["CICE5"].n_cores
        12
    """

    tpr_range: tuple[int, int] | None = None
    # A read-only view over a dict, so unhashable as it stands: it is left out of the
    # generated __hash__ and kept in __eq__. Equal strategies therefore still hash equal,
    # which is all a dict key needs; two strategies differing only in their sub-tree
    # collide in the bucket and are separated by __eq__.
    subcomponents: Mapping[str, AllocationStrategy] = field(default_factory=dict, hash=False)
    local_constraints: tuple[LocalConstraint, ...] = ()
    group_constraints: tuple[GroupConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.tpr_range is not None:
            # Errors name the base class because that is where the field lives; every
            # strategy, root or mode, shares this one check.
            min_tpr, max_tpr = self.tpr_range
            if min_tpr < 1:
                raise ValueError(f"AllocationStrategy.tpr_range minimum must be >= 1, got {min_tpr}.")
            if max_tpr < min_tpr:
                raise ValueError(f"AllocationStrategy.tpr_range maximum ({max_tpr}) must be >= minimum ({min_tpr}).")
        # Copy before wrapping: MappingProxyType is a *view*, so proxying the caller's dict
        # directly would leave this frozen, hashable instance mutable through that dict.
        object.__setattr__(self, "subcomponents", MappingProxyType(dict(self.subcomponents)))

    # --- Per-mode contract -------------------------------------------------

    # Where this mode sits in the scheduling order. Groups are served in ascending phase
    # order, so a mode that must be carved out of the budget before its siblings can be
    # sized declares a lower number. There is deliberately no default: a mode that does
    # not declare one fails loudly on the first search rather than being served last by
    # accident. Ties break on class name, so the order is always deterministic.
    _phase: ClassVar[int]

    @classmethod
    @abstractmethod
    def _iter_core_shares(cls, allocs: Sequence[AllocationStrategy], budget: int) -> Iterator[tuple[int, ...]]:
        """Yield each way *allocs* — all of this mode — can share up to *budget* cores.

        Called once per sibling group of this mode, and only for modes that actually have
        siblings — the scheduler never builds an empty group, so *allocs* is never empty.

        Any cores held back for another mode's siblings are already subtracted from
        *budget*, so a mode never needs to know that the other modes exist.

        Args:
            allocs (Sequence[AllocationStrategy]): The siblings of this mode, in position
                order.
            budget (int): The most this group may take, in cores.

        Yields:
            tuple[int, ...]: One core count per allocation, in the same order.
        """

    @classmethod
    def _reserved_cores(cls, allocs: Sequence[AllocationStrategy]) -> int:
        """Return cores earlier phases should hold back for *allocs*.

        A **pruning hint only**: returning 0 is always correct and only costs the scheduler
        some wasted work, because a group that cannot be served simply yields nothing and
        the split is abandoned. Override it where the lower bound is cheap to compute and
        prunes usefully.

        Args:
            allocs (Sequence[AllocationStrategy]): The siblings of this mode.

        Returns:
            int: A lower bound on the cores this group needs. Defaults to 0.
        """
        return 0

    @classmethod
    def _describe_infeasible(
        cls, allocs: Sequence[AllocationStrategy], names: Sequence[str], budget: int
    ) -> str | None:
        """Return a DEBUG explanation of why *allocs* could take no share, or ``None``.

        Called only when this mode's group yielded nothing at all, so it is off every hot
        path. The default explains nothing, which is right for the modes whose emptiness is
        self-evident from the budget.

        Args:
            allocs (Sequence[AllocationStrategy]): The siblings of this mode.
            names (Sequence[str]): Their component names, in the same order.
            budget (int): The budget the group was offered.

        Returns:
            str | None: A message for the log, or ``None`` to stay quiet.
        """
        return None

    # --- Shared behaviour --------------------------------------------------

    def resolve_tpr_range(self, inherited: tuple[int, int]) -> tuple[int, int]:
        """Return this strategy's own thread range, or *inherited* when it sets none.

        Args:
            inherited (tuple[int, int]): The range in force at the enclosing strategy,
                ultimately the ``tpr_range`` argument of ``iter_layouts``.

        Returns:
            tuple[int, int]: The inclusive ``(min, max)`` range that applies to this
                component and, unless they override it, to everything below it.
        """
        return self.tpr_range if self.tpr_range is not None else inherited

    def strategies_for(self, names: Iterable[str]) -> tuple[AllocationStrategy, ...]:
        """Return the sub-strategies for *names*, in the order given.

        The mapping is keyed by component name while the enumerator works positionally,
        so this is where the two are lined up. Every name must have an entry: there is no
        default mode, and ``_validate_names`` has already established that the strategy
        tree names exactly the same components as the component tree.

        Args:
            names (Iterable[str]): Sub-component names, in the order the caller needs
                them - normally ``ParallelComponent.subcomponents`` declaration order.

        Returns:
            tuple[AllocationStrategy, ...]: One strategy per name, in the same order.

        Raises:
            KeyError: If any of *names* has no entry in ``subcomponents``.

        Examples:
            >>> s = FreeAllocation(
            ...     subcomponents={"atm": FreeAllocation(), "ocn": FixedAllocation(4)}
            ... )
            >>> [type(a).__name__ for a in s.strategies_for(("atm", "ocn"))]
            ['FreeAllocation', 'FixedAllocation']
        """
        return tuple(self.subcomponents[name] for name in names)

    def _validate_names(self, component: ParallelComponent, path: str = "root") -> None:
        """Check that this strategy tree names exactly *component*'s sub-components.

        The two trees must agree name for name, at every level. A name that matches
        nothing is almost always a typo, and silently ignoring it would hand back layouts
        that quietly disregard the caller's intent. A sub-component with no entry is
        equally an error: there is no default mode to fall back on, so the strategy tree
        simply does not say how that component is to be served.

        Args:
            component (ParallelComponent): The component this strategy is laid over.
            path (str): Dotted path to *component*, for the error message.

        Raises:
            ValueError: If, at any depth, the keys of ``subcomponents`` are not exactly
                the names of the matching component's sub-components.
        """
        sub_names = {sub.name for sub in component.subcomponents}
        given = set(self.subcomponents)
        if unknown := given - sub_names:
            raise ValueError(
                f"{type(self).__name__}: unknown component names in allocations at {path}: "
                f"{unknown}. Valid names: {sub_names}."
            )
        if missing := sub_names - given:
            raise ValueError(
                f"{type(self).__name__}: no allocation given at {path} for {missing}. "
                "Every sub-component needs one - there is no default mode. Add a "
                "FreeAllocation() for the ones that should be left open."
            )
        for sub in component.subcomponents:
            self.subcomponents[sub.name]._validate_names(sub, f"{path}.{sub.name}")


@dataclass(frozen=True, kw_only=True)
class RootAllocation(AllocationStrategy):
    """The allocation strategy of the root component, which holds the whole budget.

    This adds no fields to ``AllocationStrategy``, and that is the whole content of the
    class. Everything a root may legitimately state - a thread range, sub-strategies,
    constraints - already lives on the base, while every field that would size a component
    (``n_cores``, ``weight``, ``min_cores``, ``max_cores``) lives on one of the modes
    below. The root receives all of ``total_cores`` by definition, so a size given for it
    could only be ignored; making it a distinct type means it cannot be written down in
    the first place, rather than being written down and then rejected.

    A root is never one of a parent's sub-strategies, so it declares no ``_phase`` and
    nesting one raises ``AttributeError`` naming this class when the scheduler orders the
    group.

    Args:
        tpr_range (tuple[int, int]): Inclusive ``(min, max)`` range of OpenMP threads per
            rank, defaulting to ``(1, 1)`` — pure MPI, no threading. Unlike every other
            strategy this is never ``None``: it is the range in force for the whole
            application, inherited by every component that does not override it, so there
            is nothing above it to inherit from. See ``AllocationStrategy`` for the rest.

    Examples:
        >>> root = RootAllocation(
        ...     subcomponents={"atm": FreeAllocation(), "ocn": FixedAllocation(4)}
        ... )
        >>> (sorted(root.subcomponents), root.tpr_range)
        (['atm', 'ocn'], (1, 1))
    """

    tpr_range: tuple[int, int] = (1, 1)

    @classmethod
    def _iter_core_shares(cls, allocs: Sequence[AllocationStrategy], budget: int) -> Iterator[tuple[int, ...]]:
        """Refuse to take a share: a root is not one of a parent's sub-strategies.

        Deliberately not a generator, so the error surfaces on the call rather than on the
        first ``next()``. The scheduler never reaches this - it orders groups by ``_phase``
        first, which a root does not declare - so this is the answer to calling it directly.

        Raises:
            TypeError: Always.
        """
        raise TypeError(
            "RootAllocation describes the whole budget and takes no share of a parent's "
            "cores, so it cannot be nested under another strategy. Use FreeAllocation for "
            "a sub-component that should be left open."
        )


@dataclass(frozen=True)
class FixedAllocation(AllocationStrategy):
    """Fixed allocation: the component receives exactly ``n_cores`` cores.

    See ``AllocationStrategy`` for the fields shared by every mode.

    Args:
        n_cores (int): Cores this component must receive. Must be >= 1.

    Raises:
        ValueError: If ``n_cores`` < 1.

    Examples:
        >>> FixedAllocation(12).n_cores
        12
    """

    # Carved out of the budget before anything else: a fixed count is not negotiable, so
    # the siblings that can flex should flex around it.
    _phase: ClassVar[int] = 0

    n_cores: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_cores < 1:
            raise ValueError(f"FixedAllocation.n_cores must be >= 1, got {self.n_cores}.")

    @classmethod
    def _iter_core_shares(cls, allocs: Sequence[FixedAllocation], budget: int) -> Iterator[tuple[int, ...]]:
        """Yield the group's one share, if it fits *budget*.

        A fixed group has exactly one possible share, so this is arithmetic wearing an
        iterator's clothes — but wearing it is what lets the scheduler treat every mode
        alike, and yielding nothing is how "these do not fit" is expressed.

        Args:
            allocs (Sequence[FixedAllocation]): The fixed-allocated siblings.
            budget (int): The most this group may take, in cores.

        Yields:
            tuple[int, ...]: The one share, or nothing at all if it does not fit.

        Examples:
            >>> list(FixedAllocation._iter_core_shares([FixedAllocation(6)], 8))
            [(6,)]
            >>> list(FixedAllocation._iter_core_shares([FixedAllocation(6)], 4))
            []
        """
        if sum(alloc.n_cores for alloc in allocs) <= budget:
            yield tuple(alloc.n_cores for alloc in allocs)

    @classmethod
    def _describe_infeasible(cls, allocs: Sequence[FixedAllocation], names: Sequence[str], budget: int) -> str | None:
        """Name the fixed allocations and what they need, since the budget cannot say it."""
        return (
            f"the fixed allocations {dict(zip(names, (a.n_cores for a in allocs), strict=True))} "
            f"need {sum(a.n_cores for a in allocs)} core(s), but only {budget} are available"
        )


@dataclass(frozen=True)
class RatioAllocation(AllocationStrategy):
    """Ratio allocation: the component receives cores in proportion to sibling weights.

    There exists a shared integer multiplier ``k >= 1`` such that each ratio-allocated
    sibling receives ``k * weight`` cores. For example, weights (3, 2) can be allocated
    as (3, 2), (6, 4), (9, 6), and so on. The proportions are pinned; the scale is not.

    See ``AllocationStrategy`` for the fields shared by every mode.

    Args:
        weight (int): Relative weight among ratio-allocated siblings. Must be >= 1.

    Raises:
        ValueError: If ``weight`` < 1.

    Examples:
        >>> RatioAllocation(3).weight
        3
    """

    # After the fixed counts are carved out, but before the free siblings: the multiplier
    # is chosen from what fixed left behind, and the free siblings absorb the remainder.
    _phase: ClassVar[int] = 1

    weight: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weight < 1:
            raise ValueError(f"RatioAllocation.weight must be >= 1, got {self.weight}.")

    @classmethod
    def _iter_core_shares(cls, allocs: Sequence[RatioAllocation], budget: int) -> Iterator[tuple[int, ...]]:
        """Yield each way *allocs* can share up to *budget* cores.

        Ratio siblings share one integer multiplier ``k >= 1`` and receive ``k * weight``
        cores each, so the options are enumerated by increasing *k*, up to the largest one
        that fits *budget*.

        Any cores the caller must keep back for another mode's siblings are already
        subtracted from *budget*; this method does not know they exist.

        Args:
            allocs (Sequence[RatioAllocation]): The ratio-allocated siblings.
            budget (int): The most this group may take, in cores.

        Yields:
            tuple[int, ...]: One core count per allocation, in the same order.

        Examples:
            >>> allocs = [RatioAllocation(3), RatioAllocation(2)]
            >>> list(RatioAllocation._iter_core_shares(allocs, 12))
            [(3, 2), (6, 4)]
        """
        sum_weights = sum(alloc.weight for alloc in allocs)
        for k in range(1, budget // sum_weights + 1):
            yield tuple(k * alloc.weight for alloc in allocs)


@dataclass(frozen=True)
class FreeAllocation(AllocationStrategy):
    """Free allocation: the component takes any core count within its own bounds.

    This is the mode that constrains nothing, so a bare ``FreeAllocation()`` is what a
    component gets when the caller has no opinion about it. It still has to be written
    down: there is no default mode, and an omitted sub-component is an error rather than
    a free one.

    See ``AllocationStrategy`` for the fields shared by every mode.

    Args:
        min_cores (int): Lower bound on core count. Must be >= 1. Defaults to 1.
        max_cores (int | None): Upper bound on core count, or ``None`` for no upper
            bound. Must be >= ``min_cores`` when provided. Defaults to ``None``.

    Raises:
        ValueError: If ``min_cores`` < 1, or ``max_cores`` < ``min_cores``.

    Examples:
        >>> alloc = FreeAllocation(min_cores=4, max_cores=6)
        >>> (alloc.min_cores, alloc.max_cores)
        (4, 6)
    """

    # Served last: free siblings divide whatever the other modes left behind.
    _phase: ClassVar[int] = 2

    min_cores: int = 1
    max_cores: int | None = None

    @classmethod
    def _reserved_cores(cls, allocs: Sequence[FreeAllocation]) -> int:
        """Return the group's combined ``min_cores``, so earlier phases leave room."""
        return sum(alloc.min_cores for alloc in allocs)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_cores < 1:
            raise ValueError(f"FreeAllocation.min_cores must be >= 1, got {self.min_cores}.")
        if self.max_cores is not None and self.max_cores < self.min_cores:
            raise ValueError(f"FreeAllocation.max_cores ({self.max_cores}) must be >= min_cores ({self.min_cores}).")

    def _core_range(self, max_available: int) -> range:
        """Return the core counts this allocation may take from *max_available*.

        The range runs from ``min_cores`` up to the smaller of *max_available* and
        ``max_cores``, and is empty when not even ``min_cores`` fits.

        Args:
            max_available (int): The largest count the caller can spare, before this
                allocation's own ``max_cores`` is applied.

        Returns:
            range: The core counts to try, in increasing order.

        Examples:
            >>> FreeAllocation(min_cores=2, max_cores=4)._core_range(10)
            range(2, 5)
            >>> FreeAllocation(min_cores=2)._core_range(3)
            range(2, 4)
        """
        hi = max_available if self.max_cores is None else min(max_available, self.max_cores)
        return range(self.min_cores, hi + 1)

    @classmethod
    def _iter_core_shares(cls, allocs: Sequence[FreeAllocation], budget: int) -> Iterator[tuple[int, ...]]:
        """Yield each way *allocs* can share up to *budget* cores.

        Each allocation's ``min_cores``/``max_cores`` bounds are respected and the total
        assigned does not exceed *budget*. Any leftover budget is silently unused —
        callers may apply a ``MaxWastedCoreFractionConstraint`` on the parent component to
        restrict this slack.

        Any cores the caller must keep back for another mode's siblings are already
        subtracted from *budget*; this method does not know they exist.

        Args:
            allocs (Sequence[FreeAllocation]): The free-allocated siblings.
            budget (int): The most this group may take, in cores.

        Yields:
            tuple[int, ...]: One core count per allocation, in the same order.

        Examples:
            Bounds hold, and budget the group may not take is left idle:

            >>> allocs = [FreeAllocation(min_cores=4, max_cores=6)]
            >>> list(FreeAllocation._iter_core_shares(allocs, 8))
            [(4,), (5,), (6,)]
        """
        alloc = allocs[0]
        rest_allocs = allocs[1:]
        if not rest_allocs:
            # The last sibling may take the whole remaining budget. Recursing on the empty
            # tail would cost a generator per assignment just to produce one empty tuple.
            for c in alloc._core_range(budget):
                yield (c,)
            return

        rest_min = sum(a.min_cores for a in rest_allocs)

        for c in alloc._core_range(budget - rest_min):
            for rest in cls._iter_core_shares(rest_allocs, budget - c):
                yield (c, *rest)

    @classmethod
    def _tree_for(cls, component: ParallelComponent) -> FreeAllocation:
        """Return an all-free strategy tree mirroring *component*'s shape.

        ``resolve_root_strategy`` builds the default tree from this, one call per
        sub-component of the root: every component takes any core count, bounded only by
        what its parent holds. The root itself is a ``RootAllocation`` and so is not one
        of these.

        Args:
            component (ParallelComponent): Root of the component subtree to mirror.

        Returns:
            FreeAllocation: A strategy tree with one free node per component.
        """
        return cls(subcomponents={sub.name: cls._tree_for(sub) for sub in component.subcomponents})


# ---------------------------------------------------------------------------
# Sibling-group scheduling
# ---------------------------------------------------------------------------


def _group_by_mode(
    strategies: Sequence[AllocationStrategy],
) -> list[tuple[type[AllocationStrategy], list[tuple[int, AllocationStrategy]]]]:
    """Group *strategies* by mode and return the groups in scheduling order.

    Each entry keeps its position in *strategies*, because the caller has to scatter the
    shares back into a positional split. Groups come out ordered by ``_phase``, ties broken
    on class name so the order never depends on dict insertion.

    A mode that declares no ``_phase`` raises ``AttributeError`` naming its class — which
    is the intended failure for a new mode that forgot to say where it belongs.
    """
    groups: dict[type[AllocationStrategy], list[tuple[int, AllocationStrategy]]] = {}
    for i, strategy in enumerate(strategies):
        groups.setdefault(type(strategy), []).append((i, strategy))
    return sorted(groups.items(), key=lambda item: (item[0]._phase, item[0].__name__))


def _iter_group_shares(
    mode: type[AllocationStrategy],
    entries: Sequence[tuple[int, AllocationStrategy]],
    names: Sequence[str],
    budget: int,
) -> Iterator[tuple[int, ...]]:
    """Yield one mode-group's shares, logging the mode's explanation if there are none.

    The emptiness check is a flag rather than a materialised list, so the group stays lazy
    and a search that stops after the first layout does not enumerate the rest.
    """
    allocs = [strategy for _, strategy in entries]
    served = False
    for shares in mode._iter_core_shares(allocs, budget):
        served = True
        yield shares
    if not served and (why := mode._describe_infeasible(allocs, [names[i] for i, _ in entries], budget)):
        logger.debug("no core split: %s", why)


def _iter_phase_splits(
    groups: Sequence[tuple[type[AllocationStrategy], list[tuple[int, AllocationStrategy]]]],
    reserved_after: Sequence[int],
    names: Sequence[str],
    index: int,
    budget: int,
    split: list[int],
) -> Iterator[tuple[int, ...]]:
    """Serve *groups* from *index* onwards, scattering each group's share into *split*.

    Each group is offered what the earlier phases left, less what the later ones have
    reserved. *split* is copied before being written, so a partially filled split is never
    shared between branches of the recursion.
    """
    if index == len(groups):
        yield tuple(split)
        return

    mode, entries = groups[index]
    # Clamped at zero: once the later phases' reservations exceed the budget this group is
    # dead whatever the arithmetic says, and a negative "available" reads as nonsense in
    # the diagnostics.
    offered = max(budget - reserved_after[index], 0)
    for shares in _iter_group_shares(mode, entries, names, offered):
        filled = split.copy()
        for (position, _), cores in zip(entries, shares, strict=True):
            filled[position] = cores
        yield from _iter_phase_splits(groups, reserved_after, names, index + 1, budget - sum(shares), filled)


def iter_core_splits(
    strategies: Sequence[AllocationStrategy], parent_cores: int, names: Sequence[str]
) -> Iterator[tuple[int, ...]]:
    """Yield all valid core assignments to *strategies* summing to <= *parent_cores*.

    The siblings are grouped by mode and the groups served in ``_phase`` order — fixed
    counts carved out first, then the ratio multiplier, then the free remainder — with each
    group offered what the earlier phases left, less what the later ones have reserved.
    Nothing here knows which modes exist: a new mode joins the schedule by declaring a
    phase and implementing ``_iter_core_shares``.

    This is a sibling-group algorithm rather than per-node behaviour, which is why it sits
    at module level and not on ``AllocationStrategy``: the ratio siblings share one
    multiplier, so no single strategy can work out its own share without seeing the others.

    The total assigned may be less than *parent_cores*; the unused cores are left idle.
    Use ``MaxWastedCoreFractionConstraint`` on the parent component to restrict idle cores.

    Args:
        strategies (Sequence[AllocationStrategy]): One strategy per sub-component, in
            declaration order - normally straight from ``strategies_for``.
        parent_cores (int): Cores held by the parent, to be divided among *strategies*.
        names (Sequence[str]): The matching sub-component names, carried for diagnostics
            only.

    Yields:
        tuple[int, ...]: One core count per strategy, in the same order.
    """
    groups = _group_by_mode(strategies)

    if len(groups) == 1 and [i for i, _ in groups[0][1]] == list(range(len(strategies))):
        # One mode, holding every position in order: the share *is* the split, with nothing
        # to carve out first and nothing to rearrange. This is the default all-free
        # strategy tree, and by far the hottest shape.
        mode, entries = groups[0]
        yield from _iter_group_shares(mode, entries, names, parent_cores)
        return

    # What each phase must leave behind for the phases after it. A pruning hint only: a
    # group that gets squeezed out yields nothing anyway, this just finds out sooner.
    reserved_after = [0] * len(groups)
    for index in reversed(range(len(groups) - 1)):
        mode, entries = groups[index + 1]
        reserved_after[index] = reserved_after[index + 1] + mode._reserved_cores([s for _, s in entries])

    yield from _iter_phase_splits(groups, reserved_after, names, 0, parent_cores, [0] * len(strategies))


def resolve_root_strategy(component: ParallelComponent, allocations: RootAllocation | None) -> RootAllocation:
    """Validate *allocations* against the component tree, or build an all-free default.

    Being the root is a role a strategy is given rather than a property it has, so the
    ``RootAllocation`` type is checked here, at the boundary, and not at construction.
    The check is by ``isinstance`` rather than by annotation alone because this package
    ships no type information and callers are not required to run a type checker.

    Args:
        component (ParallelComponent): Root of the component tree to lay the strategy over.
        allocations (RootAllocation | None): The caller's strategy tree, or ``None`` to
            build an all-free one mirroring *component*.

    Returns:
        RootAllocation: The strategy tree to search under.

    Raises:
        TypeError: If *allocations* is not a ``RootAllocation``.
        ValueError: If the strategy tree does not name exactly *component*'s
            sub-components, at any depth.
    """
    if allocations is None:
        return RootAllocation(
            subcomponents={sub.name: FreeAllocation._tree_for(sub) for sub in component.subcomponents}
        )
    if not isinstance(allocations, RootAllocation):
        raise TypeError(
            f"allocations must be a RootAllocation, got {type(allocations).__name__}. The root "
            "component always receives the whole budget, so a strategy that states a core count "
            "or bound cannot describe it; set those on a sub-component, or change total_cores."
        )
    allocations._validate_names(component)
    return allocations
