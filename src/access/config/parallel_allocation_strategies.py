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
* ``WeightedAllocation`` pins the proportions between siblings, leaving their shared
  multiplier open.
* ``FreeAllocation`` takes any count within some bounds.

The ``RootAllocation`` states no core count of its own, because the root component always
receives the whole budget.

A few words for these pieces recur throughout the module:

* A **node** is one component's entry in the strategy tree.
* A node's **mode** is the ``AllocationStrategy`` subclass it uses, that is, *how* the node
  states its size. Siblings need not share a mode.
* A mode's **phase** is where it sits in the order sibling groups are served in: the fixed
  counts are carved out of the budget first, then the shared multiplier of the weighted
  siblings is chosen from what they left, then the free siblings absorb the remainder. See
  ``iter_core_splits``.
* The **weights** of weighted siblings are the integers they each carry, which together
  express a ratio: weights of 3 and 2 are the ratio 3:2. Their **shared multiplier** turns
  that ratio into core counts, one sibling receiving ``multiplier * weight`` cores. The
  weights pin the proportion; the multiplier is chosen afresh for each budget.

Note that "ratio" elsewhere in the package means something different - a bound on a
computed quotient, as in ``RankRatioGroupConstraint`` and
``SubdomainAspectRatioConstraint``. Only the weights of a sibling group express a ratio in
the sense above.

The phases order the siblings of a parent that *divides* its cores. A parent whose
sub-components share one range of them instead (``CoreSharing.SHARED``) divides nothing, so
its siblings are not scheduled against each other at all: each is offered every count that
fits, by ``iter_shared_core_shares`` rather than by ``iter_core_splits``. A mode reaches
that path through ``_shared_core_range`` rather than ``_iter_core_shares``, and
``WeightedAllocation``, which has no meaning where nothing is divided, declines it.

The two trees stand side by side, one strategy node per component, and it is the mode of
each node that differs::

    component tree          strategy tree
    --------------          -------------
    model                   RootAllocation(thread_range=(1, 4))
    |-- UM7                 |-- "UM7":   FreeAllocation(thread_range=(1, 4))
    |-- MOM5                |-- "MOM5":  WeightedAllocation(3)
    `-- CICE5               `-- "CICE5": FixedAllocation(12)

``FixedAllocation`` and ``FreeAllocation`` may state their core counts as a fraction of the
whole budget instead of as an absolute number, which is what makes one strategy tree usable
across a scaling study: ``FreeAllocation(min_core_fraction=0.45, max_core_fraction=0.55)``
asks for roughly half the machine, whatever the machine turns out to be. The fractions are
shares of ``iter_layouts``' ``total_cores``, and are turned into core counts once, before
the search starts.

Just like components, allocation strategies may have constraints attached to them, which
filter out layouts that do not meet certain criteria. This enables to set more stringent
constraints than the ones defined in the component tree and further restrict the layouts
that are enumerated.
"""

from __future__ import annotations

import itertools
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar, Self, cast

from access.config.parallel_component import GroupConstraint, LocalConstraint, ParallelComponent

__all__ = [
    "AllocationStrategy",
    "RootAllocation",
    "FixedAllocation",
    "WeightedAllocation",
    "FreeAllocation",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fractional bounds
# ---------------------------------------------------------------------------

# A share of the budget is snapped to the nearest whole number of cores before it is
# rounded, because the product is rarely exact: 0.29 * 100 is 28.999999999999996, and
# flooring that would hand back 28 cores for a bound that plainly says 29. A product this
# close to a whole number was meant to be that number.
_FRACTION_REL_TOL = 1e-9


def _validate_fraction(owner: str, name: str, value: float) -> None:
    """Check that *value* is a usable share of the budget.

    Args:
        owner (str): Class name, for the error message.
        name (str): Field name, for the error message.
        value (float): The fraction to check.

    Raises:
        ValueError: If *value* is not in ``(0.0, 1.0]``.
    """
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{owner}.{name} must be in (0.0, 1.0], got {value}.")


def _cores_from_fraction(fraction: float, total_cores: int, rounding: Callable[[float], int]) -> int:
    """Return the core count *fraction* of *total_cores* comes to, never less than 1.

    *rounding* decides which way a share falling between two core counts goes:
    ``math.floor`` for a lower bound and ``math.ceil`` for an upper one, so that a band
    written as a pair of fractions still contains the share it brackets, and ``round`` for
    a count meant to be a single value.

    Args:
        fraction (float): Share of the budget, in ``(0.0, 1.0]``.
        total_cores (int): The whole budget the share is taken from.
        rounding (Callable[[float], int]): How to turn a fractional share into a count.

    Returns:
        int: The core count, floored at 1 - however small the share, it still names a core.

    Examples:
        >>> import math
        >>> _cores_from_fraction(0.5, 415, math.floor)
        207
        >>> _cores_from_fraction(0.5, 415, math.ceil)
        208

        A share landing on a whole number stays there, even when the arithmetic does not:

        >>> 0.29 * 100
        28.999999999999996
        >>> _cores_from_fraction(0.29, 100, math.floor)
        29
    """
    exact = fraction * total_cores
    nearest = round(exact)
    if nearest >= 1 and math.isclose(exact, nearest, rel_tol=_FRACTION_REL_TOL):
        return nearest
    return max(1, rounding(exact))


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
    ``thread_range``.

    Constraints can be attached to an allocation strategy. These will be added to the
    component's own constraints and will be checked when enumerating layouts. This allows to
    set more stringent constraints than the ones defined in the component tree and further
    restrict the layouts that are enumerated.

    This class is abstract and holds data shared by every strategy. It is declared
    ``frozen`` (``frozen=True``) so that it is immutable and hashable, which is what lets a
    strategy form part of the layouts iterator (``iter_layouts``) subtree memoisation key.
    ``subcomponents`` is a mapping, so it is copied behind a read-only view on construction
    and left out of the generated hash; see the comment on the field itself.

    Every field here is keyword-only (``kw_only=True``), which keeps the positional slots
    free for the subclasses' own fields — hence ``FixedAllocation(12)`` rather than
    ``FixedAllocation(thread_range=None, ..., n_cores=12)``.

    Args:
        thread_range (tuple[int, int] | None): Inclusive ``(min, max)`` range of OpenMP
            threads per rank this component may use. ``None`` (the default) inherits the
            range from the enclosing strategy, and ultimately from the root's, which is
            never ``None``. Setting it here is how components get *different* thread counts:
            give the root ``(1, 1)`` and one leaf ``(1, 4)`` and only that leaf is threaded.
            Only a leaf component spends cores itself, so on a parent this acts purely
            as the default inherited by its sub-components.
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
        ValueError: If ``thread_range`` is not an inclusive ``(min, max)`` pair of thread
            counts >= 1.

    Examples:
        >>> strategy = RootAllocation(subcomponents={
        ...     "UM7":   FreeAllocation(thread_range=(1, 4)),   # threaded
        ...     "MOM5":  FreeAllocation(),                   # inherits the thread range
        ...     "CICE5": FixedAllocation(12),
        ... })
        >>> strategy.subcomponents["CICE5"].n_cores
        12
    """

    thread_range: tuple[int, int] | None = None
    # A read-only view over a dict, so unhashable as it stands: it is left out of the
    # generated __hash__ and kept in __eq__. Equal strategies therefore still hash equal,
    # which is all a dict key needs; two strategies differing only in their sub-tree
    # collide in the bucket and are separated by __eq__.
    subcomponents: Mapping[str, AllocationStrategy] = field(default_factory=dict, hash=False)
    local_constraints: tuple[LocalConstraint, ...] = ()
    group_constraints: tuple[GroupConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.thread_range is not None:
            # Errors name the base class because that is where the field lives; every
            # strategy, root or mode, shares this one check.
            min_threads, max_threads = self.thread_range
            if min_threads < 1:
                raise ValueError(f"AllocationStrategy.thread_range minimum must be >= 1, got {min_threads}.")
            if max_threads < min_threads:
                raise ValueError(
                    f"AllocationStrategy.thread_range maximum ({max_threads}) must be >= minimum ({min_threads})."
                )
        # Copy before wrapping: MappingProxyType is a *view*, so proxying the caller's dict
        # directly would leave this frozen, hashable instance mutable through that dict.
        object.__setattr__(self, "subcomponents", MappingProxyType(dict(self.subcomponents)))

    # --- Per-mode contract -------------------------------------------------

    # This mode's phase - see the module docstring. Sibling groups are served in
    # ascending phase order, so a mode that has to be carved out of the budget before its
    # siblings can be sized declares a lower number than they do. There is deliberately no
    # default: a mode that does not declare one fails loudly on the first search rather
    # than being served last by accident. Ties break on class name, so the order is always
    # deterministic.
    _phase: ClassVar[int]

    @classmethod
    @abstractmethod
    def _iter_core_shares(cls, allocs: Sequence[Any], budget: int) -> Iterator[tuple[int, ...]]:
        """Yield each way *allocs* — all of this mode — can share up to *budget* cores.

        Called once per sibling group of this mode, and only for modes that actually have
        siblings — the scheduler never builds an empty group, so *allocs* is never empty.

        Any cores held back for another mode's siblings are already subtracted from
        *budget*, so a mode never needs to know that the other modes exist.

        Args:
            allocs (Sequence[Any]): The siblings of this mode, in position order. Every
                override narrows this to its own class, which is what the ``Any`` is for -
                see the note on ``_group_by_mode`` for why the group is always homogeneous.
            budget (int): The most this group may take, in cores.

        Yields:
            tuple[int, ...]: One core count per allocation, in the same order.
        """

    @classmethod
    def _reserved_cores(cls, allocs: Sequence[Any]) -> int:
        """Return cores earlier phases should hold back for *allocs*.

        A **pruning hint only**: returning 0 is always correct and only costs the scheduler
        some wasted work, because a group that cannot be served simply yields nothing and
        the split is abandoned. Override it where the lower bound is cheap to compute and
        prunes usefully.

        Args:
            allocs (Sequence[Any]): The siblings of this mode, narrowed to its own class by
                the override.

        Returns:
            int: A lower bound on the cores this group needs. Defaults to 0.
        """
        return 0

    @classmethod
    def _describe_infeasible(cls, allocs: Sequence[Any], names: Sequence[str], budget: int) -> str | None:
        """Return a DEBUG explanation of why *allocs* could take no share, or ``None``.

        Called only when this mode's group yielded nothing at all, so it is off every hot
        path. The default explains nothing, which is right for the modes whose emptiness is
        self-evident from the budget.

        Args:
            allocs (Sequence[Any]): The siblings of this mode, narrowed to its own class by
                the override.
            names (Sequence[str]): Their component names, in the same order.
            budget (int): The budget the group was offered.

        Returns:
            str | None: A message for the log, or ``None`` to stay quiet.
        """
        return None

    def _resolve_own_fractions(self, total_cores: int, path: str) -> dict[str, object]:
        """Return the field changes turning this node's fractional bounds into core counts.

        Called once per node by ``_resolve_fractions``, before the search. The default
        states no change, which is the whole answer for a mode that cannot carry a
        fraction.

        Args:
            total_cores (int): The whole budget the fractions are shares of.
            path (str): Dotted path to this strategy, for the error message.

        Returns:
            dict[str, object]: Field names and their resolved values, empty when this
                strategy states no fractional bound.
        """
        return {}

    # --- Shared behaviour --------------------------------------------------

    def _shared_core_range(self, parent_cores: int) -> range | None:
        """Return the counts this sibling may take when it shares its parent's cores.

        Siblings of a ``CoreSharing.SHARED`` parent do not divide a budget between them:
        each takes what it needs from the same range, so each is offered every count that
        fits, independently of the others. That makes this a per-strategy question, unlike
        ``_iter_core_shares``, which serves a whole group.

        Args:
            parent_cores (int): Cores the parent holds, and so the most this sibling can
                take.

        Returns:
            range | None: The counts to try, or ``None`` if this mode has no meaning when
                cores are shared. The caller turns ``None`` into an error naming the
                sub-component, which it knows and this method does not.
        """
        return None

    def resolve_thread_range(self, inherited: tuple[int, int]) -> tuple[int, int]:
        """Return this strategy's own thread range, or *inherited* when it sets none.

        Args:
            inherited (tuple[int, int]): The range in force at the enclosing strategy,
                ultimately the ``thread_range`` argument of ``iter_layouts``.

        Returns:
            tuple[int, int]: The inclusive ``(min, max)`` range that applies to this
                component and, unless they override it, to everything below it.
        """
        return self.thread_range if self.thread_range is not None else inherited

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

    def _resolve_fractions(self, total_cores: int, path: str = "root") -> Self:
        """Return this strategy tree with every fractional bound turned into a core count.

        Resolving once, here, is what keeps the fractions out of the search: by the time
        the enumerator sees a strategy its bounds are plain core counts, so nothing
        downstream - the sibling scheduling, the per-mode share iterators, the subtree
        memo - needs to know that a fraction was ever written down.

        A tree stating no fractional bound anywhere is returned as it stands rather than
        rebuilt into an equal copy, so the common case costs nothing and a caller holding
        a reference to their strategy still holds the object the search runs on.

        Args:
            total_cores (int): The whole budget the fractions are shares of.
            path (str): Dotted path to this strategy, for the error messages.

        Returns:
            Self: The resolved strategy, or this one unchanged when it holds no fraction.

        Raises:
            ValueError: If a resolved lower bound comes out above its resolved upper bound.

        Examples:
            >>> band = FreeAllocation(min_core_fraction=0.25, max_core_fraction=0.5)
            >>> resolved = band._resolve_fractions(400)
            >>> (resolved.min_cores, resolved.max_cores)
            (100, 200)

            A tree with nothing to resolve is handed straight back:

            >>> absolute = FreeAllocation(min_cores=4)
            >>> absolute._resolve_fractions(400) is absolute
            True
        """
        changes = self._resolve_own_fractions(total_cores, path)
        subs = {name: sub._resolve_fractions(total_cores, f"{path}.{name}") for name, sub in self.subcomponents.items()}
        if any(sub is not self.subcomponents[name] for name, sub in subs.items()):
            changes["subcomponents"] = subs
        return replace(self, **changes) if changes else self


@dataclass(frozen=True, kw_only=True)
class RootAllocation(AllocationStrategy):
    """The allocation strategy of the root component, which holds the whole budget.

    This adds no fields to ``AllocationStrategy``, and that is the whole content of the
    class. Everything a root may legitimately state - a thread range, sub-strategies,
    constraints - already lives on the base, while every field that would size a component
    (``n_cores``, ``weight``, ``min_cores``, ``max_cores``, and the fractional bounds that
    stand in for them) lives on one of the modes below. The root receives all of
    ``total_cores`` by definition, so a size given for it could only be ignored; making it
    a distinct type means it cannot be written down in the first place, rather than being
    written down and then rejected.

    A root is never one of a parent's sub-strategies, so it declares no ``_phase``.
    Nesting one inside another strategy therefore raises ``AttributeError`` naming this
    class, as soon as the scheduler tries to order the sibling group it was put in.

    Args:
        thread_range (tuple[int, int]): Inclusive ``(min, max)`` range of OpenMP threads per
            rank, defaulting to ``(1, 1)`` — pure MPI, no threading. Unlike every other
            strategy this is never ``None``: it is the range in force for the whole
            application, inherited by every component that does not override it, so there
            is nothing above it to inherit from. See ``AllocationStrategy`` for the rest.

    Examples:
        >>> root = RootAllocation(
        ...     subcomponents={"atm": FreeAllocation(), "ocn": FixedAllocation(4)}
        ... )
        >>> (sorted(root.subcomponents), root.thread_range)
        (['atm', 'ocn'], (1, 1))
    """

    thread_range: tuple[int, int] = (1, 1)

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

    The count is given either directly or as ``core_fraction`` of the whole budget -
    exactly one of the two, since they say the same thing. A fraction is resolved before
    the search starts, so ``FixedAllocation(core_fraction=0.25)`` asks for 104 cores of 416
    and for 1300 of 5200. It lands on one count and stays there: where a component only
    admits certain core counts, a ``FreeAllocation`` over a band of fractions is the mode
    that can find one.

    A share that does not come to a whole number of cores is rounded **down**. That is what
    keeps a group of fixed siblings within the budget at every total: shares summing to no
    more than the whole budget resolve to counts summing to no more than the whole budget,
    for any total. Rounding to the nearest count instead would put two half-shares at 26
    cores each on a 51 core budget, and the search would find nothing on every odd total -
    which is exactly the sort of failure fractional bounds exist to avoid. The core the
    rounding gives up is not lost: it is left for whatever siblings can flex, or idle.

    Nothing here is particular to leaf components: a parent may be fixed too, and then
    divides exactly that count among its own sub-components. Several siblings of one parent
    may be fixed as well, in which case they are sized together as one group.

    See ``AllocationStrategy`` for the fields shared by every mode.

    Args:
        n_cores (int | None): Cores this component must receive. Must be >= 1. ``None`` (the
            default) takes the count from *core_fraction* instead.
        core_fraction (float | None): Cores this component must receive, as a share of the
            whole budget. Must be in ``(0.0, 1.0]``. ``None`` (the default) takes the count
            from *n_cores* instead.

    Raises:
        ValueError: If neither or both of ``n_cores`` and ``core_fraction`` are given, if
            ``n_cores`` < 1, or if ``core_fraction`` is outside ``(0.0, 1.0]``.

    Examples:
        >>> FixedAllocation(12).n_cores
        12

        A fraction names no count until the budget is known:

        >>> quarter = FixedAllocation(core_fraction=0.25)
        >>> (quarter.n_cores, quarter._resolve_fractions(416).n_cores)
        (None, 104)
    """

    # Carved out of the budget before anything else: a fixed count is not negotiable, so
    # the siblings that can flex should flex around it.
    _phase: ClassVar[int] = 0

    n_cores: int | None = None
    core_fraction: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.n_cores is None) == (self.core_fraction is None):
            raise ValueError(
                "FixedAllocation takes exactly one of n_cores and core_fraction, got "
                f"n_cores={self.n_cores} and core_fraction={self.core_fraction}."
            )
        if self.n_cores is not None and self.n_cores < 1:
            raise ValueError(
                f"FixedAllocation.n_cores must be >= 1, got {self.n_cores}. A share of the "
                "budget goes in core_fraction."
            )
        if self.core_fraction is not None:
            _validate_fraction("FixedAllocation", "core_fraction", self.core_fraction)

    def _resolve_own_fractions(self, total_cores: int, path: str) -> dict[str, object]:
        """Turn ``core_fraction`` into the one count it names on this budget."""
        if self.core_fraction is None:
            return {}
        # Round down rather than to nearest, so that fixed siblings whose shares fit the
        # budget resolve to counts that fit it too - see the note in the class docstring.
        return {"n_cores": _cores_from_fraction(self.core_fraction, total_cores, math.floor), "core_fraction": None}

    @staticmethod
    def _resolved_counts(allocs: Sequence[FixedAllocation]) -> tuple[int, ...]:
        """Return the core count each of *allocs* asks for, in the same order.

        ``n_cores`` is only optional until the search starts: ``__post_init__`` demands one
        of ``n_cores`` and ``core_fraction``, and ``_resolve_fractions`` has turned any
        fraction into a count before either caller here runs. The cast states that
        invariant, rather than testing for a ``None`` that cannot arrive.

        Args:
            allocs (Sequence[FixedAllocation]): The fixed-allocated siblings.

        Returns:
            tuple[int, ...]: Their resolved ``n_cores``, in position order.
        """
        return tuple(cast(int, alloc.n_cores) for alloc in allocs)

    def _shared_core_range(self, parent_cores: int) -> range | None:
        """Return this allocation's one count, if the parent is large enough to hold it.

        Args:
            parent_cores (int): Cores the parent holds.

        Returns:
            range | None: A range of exactly one count, empty if it does not fit.

        Examples:
            >>> FixedAllocation(6)._shared_core_range(8)
            range(6, 7)
            >>> FixedAllocation(6)._shared_core_range(4)
            range(6, 6)
        """
        (count,) = self._resolved_counts([self])
        return range(count, count + 1 if count <= parent_cores else count)

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
        counts = cls._resolved_counts(allocs)
        if sum(counts) <= budget:
            yield counts

    @classmethod
    def _describe_infeasible(cls, allocs: Sequence[FixedAllocation], names: Sequence[str], budget: int) -> str | None:
        """Name the fixed allocations and what they need, since the budget cannot say it."""
        counts = cls._resolved_counts(allocs)
        return (
            f"the fixed allocations {dict(zip(names, counts, strict=True))} "
            f"need {sum(counts)} core(s), but only {budget} are available"
        )


@dataclass(frozen=True)
class WeightedAllocation(AllocationStrategy):
    """Weighted allocation: the component receives cores in proportion to sibling weights.

    There exists a shared integer multiplier ``k >= 1`` such that each weighted sibling
    receives ``k * weight`` cores. For example, weights (3, 2) can be allocated as (3, 2),
    (6, 4), (9, 6), and so on: the weights pin the proportion, while the multiplier is
    chosen afresh for each budget.

    **The weights are read as a proportion, and reduced to lowest terms before the
    multiplier is chosen.** Weights of 6 and 4 therefore ask for exactly what weights of 3
    and 2 ask for, down to which budgets they fit: both take (3, 2) from six cores. Only
    the ratio between siblings is a request, never the size of the numbers used to write it
    down - so weights cannot be used to demand that a component be sized in blocks.

    A single weighted sibling is the exception, and keeps its weight as written: with no
    sibling to be in proportion to there is nothing to reduce against, and reducing it
    alone would leave the weight meaning nothing.

    A weight cannot size a sub-component of a parent that *shares* its cores
    (``CoreSharing.SHARED``): a weight states a share of a divided budget, and there
    nothing is divided. ``iter_shared_core_shares`` raises ``ValueError`` naming the
    sub-component rather than guessing at what the weight might have meant, so use
    ``FixedAllocation`` or ``FreeAllocation`` under such a parent.

    See ``AllocationStrategy`` for the fields shared by every mode.

    Args:
        weight (int): Relative weight among weighted siblings. Must be >= 1. The proportion
            is not written down in one place: each sibling carries its own weight, and what
            they express together is a ratio, so ``WeightedAllocation(3)`` beside
            ``WeightedAllocation(2)`` *is* the ratio 3:2. Weights only mean anything among
            direct siblings, so a proportion spanning three of them is written as one common
            set - A:B = 2:3 with B:C = 5:4 becomes A:B:C = 10:15:12. A weight fixes only a
            sibling's share relative to the others, never its absolute size; that is the
            shared multiplier's job. Weights are reduced to lowest terms among siblings
            before use, so 6 beside 4 is the same request as 3 beside 2. A weight standing
            alone is left as written.

    Raises:
        ValueError: If ``weight`` < 1.

    Examples:
        >>> WeightedAllocation(3).weight
        3
    """

    # After the fixed counts are carved out, but before the free siblings: the multiplier
    # is chosen from what fixed left behind, and the free siblings absorb the remainder.
    _phase: ClassVar[int] = 1

    weight: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weight < 1:
            raise ValueError(f"WeightedAllocation.weight must be >= 1, got {self.weight}.")

    @classmethod
    def _iter_core_shares(cls, allocs: Sequence[WeightedAllocation], budget: int) -> Iterator[tuple[int, ...]]:
        """Yield each way *allocs* can share up to *budget* cores.

        Weighted siblings share one integer multiplier ``k >= 1`` and receive
        ``k * weight`` cores each, so the options are enumerated by increasing *k*, up to
        the largest one that fits *budget*.

        Any cores the caller must keep back for another mode's siblings are already
        subtracted from *budget*; this method does not know they exist.

        Args:
            allocs (Sequence[WeightedAllocation]): The weighted siblings.
            budget (int): The most this group may take, in cores.

        Yields:
            tuple[int, ...]: One core count per allocation, in the same order.

        Examples:
            >>> allocs = [WeightedAllocation(3), WeightedAllocation(2)]
            >>> list(WeightedAllocation._iter_core_shares(allocs, 12))
            [(3, 2), (6, 4)]

            Weights are reduced first, so an unreduced proportion asks for the same thing:

            >>> allocs = [WeightedAllocation(6), WeightedAllocation(4)]
            >>> list(WeightedAllocation._iter_core_shares(allocs, 12))
            [(3, 2), (6, 4)]

            A lone weight is left as written, having nothing to be in proportion to:

            >>> list(WeightedAllocation._iter_core_shares([WeightedAllocation(2)], 5))
            [(2,), (4,)]
        """
        weights = tuple(alloc.weight for alloc in allocs)
        if len(weights) > 1:
            # Reduce to lowest terms, so equal proportions enumerate alike. Deliberately
            # skipped for a lone sibling: there is nothing for it to be in proportion to,
            # and reducing it would make its weight mean nothing at all.
            weights = tuple(weight // math.gcd(*weights) for weight in weights)
        sum_weights = sum(weights)
        for k in range(1, budget // sum_weights + 1):
            yield tuple(k * weight for weight in weights)


@dataclass(frozen=True)
class FreeAllocation(AllocationStrategy):
    """Free allocation: the component takes any core count within its own bounds.

    This is the mode that constrains nothing, so a bare ``FreeAllocation()`` is what a
    component gets when the caller has no opinion about it. It still has to be written
    down: there is no default mode, and an omitted sub-component is an error rather than
    a free one.

    A bound may be written as an absolute core count, as a fraction of the whole budget, or
    as both. The fractions are what make one strategy tree usable across a scaling study,
    since ``min_core_fraction=0.45`` means the same thing on 416 cores as it does on 5200.
    Where a bound is written both ways the absolute one is a guard rail the fraction cannot
    cross: ``min_cores`` is a floor under the resolved minimum, ``max_cores`` a cap over the
    resolved maximum.

    See ``AllocationStrategy`` for the fields shared by every mode.

    Args:
        min_cores (int): Lower bound on core count. Must be >= 1. Defaults to 1.
        max_cores (int | None): Upper bound on core count, or ``None`` for no upper
            bound. Must be >= ``min_cores`` when provided. Defaults to ``None``.
        min_core_fraction (float | None): Lower bound as a share of the whole budget, in
            ``(0.0, 1.0]``. Rounded down to a core count, then combined with *min_cores* by
            taking whichever is larger. ``None`` (the default) leaves *min_cores* to stand
            on its own.
        max_core_fraction (float | None): Upper bound as a share of the whole budget, in
            ``(0.0, 1.0]``. Rounded up to a core count, then combined with *max_cores* by
            taking whichever is smaller. ``None`` (the default) leaves *max_cores* to stand
            on its own.

    Raises:
        ValueError: If ``min_cores`` < 1, if ``max_cores`` < ``min_cores``, if either
            fraction falls outside ``(0.0, 1.0]``, or if ``max_core_fraction`` <
            ``min_core_fraction``.

    Examples:
        >>> alloc = FreeAllocation(min_cores=4, max_cores=6)
        >>> (alloc.min_cores, alloc.max_cores)
        (4, 6)

        A band of fractions resolves against the budget it is searched on:

        >>> band = FreeAllocation(min_core_fraction=0.45, max_core_fraction=0.55)
        >>> resolved = band._resolve_fractions(416)
        >>> (resolved.min_cores, resolved.max_cores)
        (187, 229)
    """

    # Served last: free siblings divide whatever the other modes left behind.
    _phase: ClassVar[int] = 2

    min_cores: int = 1
    max_cores: int | None = None
    min_core_fraction: float | None = None
    max_core_fraction: float | None = None

    @classmethod
    def _reserved_cores(cls, allocs: Sequence[FreeAllocation]) -> int:
        """Return the group's combined ``min_cores``, so earlier phases leave room."""
        return sum(alloc.min_cores for alloc in allocs)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_cores < 1:
            raise ValueError(
                f"FreeAllocation.min_cores must be >= 1, got {self.min_cores}. A share of the "
                "budget goes in min_core_fraction."
            )
        if self.max_cores is not None and self.max_cores < self.min_cores:
            raise ValueError(f"FreeAllocation.max_cores ({self.max_cores}) must be >= min_cores ({self.min_cores}).")
        self._validate_fractions()

    def _validate_fractions(self) -> None:
        """Check the fractional bounds on their own and against each other.

        Split out of ``__post_init__`` to keep each of the two halves - the absolute bounds
        and the fractional ones - readable on its own.

        Raises:
            ValueError: If either fraction falls outside ``(0.0, 1.0]``, or if
                ``max_core_fraction`` < ``min_core_fraction``.
        """
        if self.min_core_fraction is not None:
            _validate_fraction("FreeAllocation", "min_core_fraction", self.min_core_fraction)
        if self.max_core_fraction is not None:
            _validate_fraction("FreeAllocation", "max_core_fraction", self.max_core_fraction)
        if (
            self.min_core_fraction is not None
            and self.max_core_fraction is not None
            and self.max_core_fraction < self.min_core_fraction
        ):
            raise ValueError(
                f"FreeAllocation.max_core_fraction ({self.max_core_fraction}) must be >= "
                f"min_core_fraction ({self.min_core_fraction})."
            )

    def _resolve_own_fractions(self, total_cores: int, path: str) -> dict[str, object]:
        """Turn the fractional bounds into core counts, guarded by the absolute ones."""
        if self.min_core_fraction is None and self.max_core_fraction is None:
            return {}

        min_cores = self.min_cores
        if self.min_core_fraction is not None:
            min_cores = max(min_cores, _cores_from_fraction(self.min_core_fraction, total_cores, math.floor))

        max_cores = self.max_cores
        if self.max_core_fraction is not None:
            from_fraction = _cores_from_fraction(self.max_core_fraction, total_cores, math.ceil)
            max_cores = from_fraction if max_cores is None else min(max_cores, from_fraction)

        if max_cores is not None and max_cores < min_cores:
            # Rejected here rather than left to yield nothing, because an empty search
            # is the hardest failure to explain after the fact: the bounds contradict each
            # other only at this budget, and nothing downstream still knows the fractions
            # they came from.
            raise ValueError(
                f"FreeAllocation at {path}: on {total_cores} core(s) the bounds resolve to "
                f"min_cores={min_cores} and max_cores={max_cores}, which admits no core count. "
                "Widen the fractions, or drop the absolute bound guarding them."
            )

        return {
            "min_cores": min_cores,
            "max_cores": max_cores,
            "min_core_fraction": None,
            "max_core_fraction": None,
        }

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

    def _shared_core_range(self, parent_cores: int) -> range | None:
        """Return this allocation's own bounds, capped by what the parent holds.

        Args:
            parent_cores (int): Cores the parent holds.

        Returns:
            range | None: The counts to try, in increasing order.

        Examples:
            >>> FreeAllocation(min_cores=2, max_cores=4)._shared_core_range(10)
            range(2, 5)
        """
        return self._core_range(parent_cores)

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

    See the module docstring for what *mode* and *phase* mean.
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
    counts carved out first, then the shared multiplier, then the free remainder — with each
    group offered what the earlier phases left, less what the later ones have reserved.
    Nothing here knows which modes exist: a new mode joins the schedule by declaring a
    phase and implementing ``_iter_core_shares``. See the module docstring for what *mode*
    and *phase* mean, and the example below for the ordering in action.

    This is a sibling-group algorithm rather than per-node behaviour, which is why it sits
    at module level and not on ``AllocationStrategy``: the weighted siblings share one
    multiplier, so no single strategy can work out its own share without seeing the others.

    The total assigned may be less than *parent_cores*; the unused cores are left idle.
    Use ``MaxWastedCoreFractionConstraint`` on the parent component to restrict idle cores.

    Every bound here is an absolute core count: a strategy tree reaches the enumerator
    through ``resolve_root_strategy``, which has already turned any fractional bound into
    the count it names on this budget.

    Args:
        strategies (Sequence[AllocationStrategy]): One strategy per sub-component, in
            declaration order - normally straight from ``strategies_for``.
        parent_cores (int): Cores held by the parent, to be divided among *strategies*.
        names (Sequence[str]): The matching sub-component names, carried for diagnostics
            only.

    Yields:
        tuple[int, ...]: One core count per strategy, in the same order.

    Examples:
        Four siblings on 16 cores - ``A`` fixed at 4, ``B`` and ``C`` in a 3:2 ratio, ``D``
        free with at least one core - show the phases in order. ``A`` is carved out first,
        then each shared multiplier ``k``, then ``D`` takes what is left::

            (4, 3, 2, 1) ... (4, 3, 2, 7)   # k = 1
            (4, 6, 4, 1), (4, 6, 4, 2)      # k = 2

        There is no ``k = 3``: it would need 15 of the 12 cores ``A`` left behind. And the
        ``k = 2`` rows stop at 2 cores for ``D`` rather than 3 because ``D`` reserves its
        ``min_cores`` before the weighted group is served.
    """
    groups = _group_by_mode(strategies)

    if len(groups) == 1 and [i for i, _ in groups[0][1]] == list(range(len(strategies))):
        # One mode, holding every position in order: the share *is* the split, with nothing
        # to carve out first and nothing to rearrange. This is the default all-free
        # strategy tree, and the most common shape by far.
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


def iter_shared_core_shares(
    strategies: Sequence[AllocationStrategy],
    parent_cores: int,
    names: Sequence[str],
    offsets: Sequence[int],
) -> Iterator[tuple[int, ...]]:
    """Yield every core assignment to *strategies* when they share *parent_cores*.

    The counterpart of ``iter_core_splits`` for a ``CoreSharing.SHARED`` parent. There is
    no budget to divide: the siblings run on the same cores in turn, so each is offered
    every count that fits and the result is their product rather than a partition. The
    parent must reach as far as the furthest sibling gets, ``max(core_offset + count)``,
    which the layout the counts go into checks - and which is not always the largest count,
    since a smaller sibling may start further into the range.

    That product grows fast — four unconstrained siblings on 275 cores is 275**4 — and no
    constraint prunes it, since a shared parent's idle cores are measured against its
    furthest-reaching child alone. Pin the siblings, or bound them narrowly, unless the
    range is small.

    A sibling that starts partway into the range has that much less of it to spend, so each
    is offered what fits after its own offset.

    Args:
        strategies (Sequence[AllocationStrategy]): One strategy per sub-component, in
            declaration order.
        parent_cores (int): Cores the parent holds, available to each sibling from the core
            it starts at onwards.
        names (Sequence[str]): The matching sub-component names, for diagnostics.
        offsets (Sequence[int]): The core each sibling starts at, relative to the parent.

    Yields:
        tuple[int, ...]: One core count per strategy, in the same order.

    Raises:
        ValueError: If a strategy uses a mode that has no meaning when cores are shared.

    Examples:
        >>> allocs = [FixedAllocation(4), FreeAllocation(max_cores=2)]
        >>> list(iter_shared_core_shares(allocs, 4, ["a", "b"], [0, 0]))
        [(4, 1), (4, 2)]
        >>> list(iter_shared_core_shares(allocs, 4, ["a", "b"], [0, 3]))
        [(4, 1)]
    """
    ranges = []
    for strategy, name, offset in zip(strategies, names, offsets, strict=True):
        core_range = strategy._shared_core_range(parent_cores - offset)
        if core_range is None:
            raise ValueError(
                f"{type(strategy).__name__} cannot allocate {name!r}, whose parent shares its cores "
                "among its sub-components. A weight states a share of a divided budget, and nothing "
                "is divided here. Use FixedAllocation or FreeAllocation instead."
            )
        ranges.append(core_range)
    yield from itertools.product(*ranges)


def resolve_root_strategy(
    component: ParallelComponent, allocations: RootAllocation | None, total_cores: int
) -> RootAllocation:
    """Settle the strategy tree to search under: validate it, and resolve its fractions.

    Being the root is a role a strategy is given rather than a property it has, so the
    ``RootAllocation`` type is checked here, at the boundary, and not at construction.
    The check is by ``isinstance`` rather than by annotation alone because this package
    ships no type information and callers are not required to run a type checker.

    This is also where a bound written as a share of the budget becomes a core count, which
    is why the budget is an argument: the tree handed back always states absolute counts, so
    the enumerator never meets a fraction. A tree that holds none is returned unchanged.

    Args:
        component (ParallelComponent): Root of the component tree to lay the strategy over.
        allocations (RootAllocation | None): The caller's strategy tree, or ``None`` to
            build an all-free one mirroring *component*.
        total_cores (int): The whole budget, which the fractional bounds are shares of.

    Returns:
        RootAllocation: The strategy tree to search under, free of fractional bounds.

    Raises:
        TypeError: If *allocations* is not a ``RootAllocation``.
        ValueError: If the strategy tree does not name exactly *component*'s
            sub-components at any depth, or if a bound resolves to an empty range on
            *total_cores*.
    """
    if allocations is None:
        # An all-free tree states no bound at all, so there is nothing in it to resolve.
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
    return allocations._resolve_fractions(total_cores)
