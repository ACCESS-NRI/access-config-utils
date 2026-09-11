# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Classes and utilities for describing how a given application is parallelized.

In the model used here, a parallel application is represented as a tree of components. Each
component can be one of two types:
- parent components: these components distribute their cores to sub-components, and do not
  run any work themselves.
- leaves: these components run work themselves, and may have a domain of work that can be
  split across cores, eiher using MPI ranks and OpenMP threads, or some other
  parallelisation strategy.

A parent divides its cores between its sub-components by default, each of them taking a
disjoint subset. A parent may instead declare that its sub-components *share* its cores,
drawing on the same range and running on it in turn, as the NUOPC-driven components of
ACCESS-OM3 do. Which rule applies is the parent's ``CoreSharing``, and where in a shared
range a sub-component sits is its own ``core_offset``.

The component tree is implemented as a tree of ``ParallelComponent`` instances. This tree
describes the units of work and how they nest, but does not specify how many cores each
component actually receives, or how they are used. This is instead described by a tree of
``ComponentLayout`` instances, which have the same shape as the component tree, but record
the number of cores each component receives and how they are used. This allows to specify
a component tree once, and then explore different layouts for it, without having to modify
the components.

Components can also have constraints attached to them, which restrict the layouts that are
considered valid for that component. This allows one to enforce certain requirements on the
layouts, such as requiring that a component receives a certain number of cores, or that the
cores assigned to its sub-components follow some specified ratio. Constraints can be local
to a component or apply to groups of components. These are implemented as subclasses of
``LocalConstraint`` and ``GroupConstraint``, respectively.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto

from access.config import parallel_domain, parallel_mpi_grid


class CoreSharing(Enum):
    """How a parent component's cores are divided among its sub-components.

    ``PARTITIONED``, the default, gives each sub-component a disjoint subset of the parent's
    cores, so the parent holds at least their total. This describes most coupled models.
    ``SHARED`` has them all draw on the same range of cores and run on it in turn, so the
    parent holds at least as many cores as its furthest-reaching sub-component rather than
    their total, and as many ranks as they sit on between them.

    Where in the range each sub-component sits is its own ``core_offset``, so what a shared
    parent must be large enough for is the one that *reaches* furthest rather than the one
    that is biggest: ``max(core_offset + n_cores)``. Two sub-components may start at the
    same core and overlap, or divide the range between them; both are ordinary cases here.

    How large it must be and how much of it is spent are different questions once offsets
    are in play. A range whose sub-components all start partway into it is idle at the
    front, one with a gap between two of them is idle in the middle, and ``used_cores``
    counts neither - it counts the part at least one sub-component sits on.
    An offset means nothing under a partitioned parent, where each sub-component already
    starts where the last one ended, and is refused there.

    The offset is a core index, as ESMF means ``rootpe``. Its driver checks
    ``rootpe + npes <= ncpus``, where ``npes`` is the component's core count -
    ``nthreads * ntasks * pestride`` when threaded - so the index and the span are both
    counted in cores. Placement is assumed contiguous, ESMF's ``pestride = 1``, which the
    model has no way to express. An offset is therefore refused on a sub-component running
    more than one core per rank, where a core index and a PE index part company.

    Note that the interior of a ``SHARED`` parent is enumerated as a product rather than a
    partition: each child is offered every count that fits, independently of its siblings.
    Four unconstrained children on 275 cores is 275**4 combinations, and
    ``MaxWastedCoreFractionConstraint`` prunes few of them: a child sitting inside the
    range another already covers changes nothing about how much of it is spent. Give the
    children of a shared parent a ``FixedAllocation`` or a narrow ``FreeAllocation``
    unless the range is small. A
    ``WeightedAllocation`` cannot allocate one at all: a weight states a share of a divided
    budget, and nothing is divided here.

    Examples:
        An offset only means something under a parent whose cores are shared, so it is the
        ``SHARED`` rule that lets one be written down at all:

        >>> ice = ParallelComponent("ice", core_offset=4)
        >>> shared = ParallelComponent(
        ...     "ocean_side", subcomponents=(ice,), core_sharing=CoreSharing.SHARED
        ... )
        >>> [sub.core_offset for sub in shared.subcomponents]
        [4]
    """

    # Each sub-component takes a disjoint subset of the parent's cores.
    PARTITIONED = auto()
    # The sub-components draw on one range of cores and run on it in turn.
    SHARED = auto()


def _occupied(spans: Iterable[tuple[int, int]]) -> int:
    """Return how many positions at least one of *spans* covers.

    Each span is a ``(start, size)`` pair standing for ``[start, start + size)``. Spans
    that overlap are counted once, and positions no span covers are not counted at all -
    so this is what a component *spends*, as opposed to how far its sub-components reach.

    Args:
        spans (Iterable[tuple[int, int]]): The spans to measure, in any order.

    Returns:
        int: The number of positions covered, or 0 when there are no spans.

    Examples:
        >>> _occupied([(0, 4), (2, 4)])   # overlapping, counted once
        6
        >>> _occupied([(4, 4), (12, 4)])  # a gap between them, and nothing before the first
        8
    """
    covered = 0
    reach = 0
    for start, size in sorted(spans):
        end = start + size
        if end > reach:
            covered += end - max(start, reach)
            reach = end
    return covered


@dataclass(frozen=True)
class ComponentLayout:
    """Class describing how a certain component is parallelised.

    This class stores basic information about how a component is parallelised, including the
    number of cores, ranks, and threads used by the component, as well as any sub-layouts
    for its sub-components.

    How a component assigns its cores depends on whether it is a leaf or a parent:
    * A **leaf** component spends its cores itself, as ``n_ranks`` MPI ranks each running
      ``threads_per_rank`` OpenMP threads. It spends them exactly:
      ``n_ranks * threads_per_rank == n_cores``.
    * A **parent** component spends its cores on its sub-components, and has no thread
      count of its own - each sub-component may use a different one, so
      ``threads_per_rank`` is ``None``. Its ``core_sharing`` says how they receive them: a
      disjoint subset each, whose total cannot exceed ``n_cores``, or one range they share,
      which the furthest-reaching of them cannot reach past. Either way a parent may spend
      fewer cores than it holds, leaving the remainder idle (see ``idle_cores``).

    This class is frozen (``frozen=True``) so that a layout is an immutable, hashable value
    object, and assigning to a field raises ``dataclasses.FrozenInstanceError``. The layout
    enumerator relies on this property, as it memoises subtrees and hands the *same*
    instance to every result whose tree contains it, so a layout must never be modified in
    place.

    Args:
        name (str): Name of the ``ParallelComponent`` this layout resolves.
        n_cores (int): CPU cores allocated to this component. Must be >= 1.
        n_ranks (int): MPI ranks in this component's subtree: the ranks it runs itself
            for a leaf, the total over ``sub_layouts`` for a parent that partitions its
            cores, and the ranks they sit on between them for one whose sub-layouts share
            them. Must be >= 1.
        threads_per_rank (int | None): OpenMP threads per MPI rank, for a leaf. Must be
            >= 1 when set. ``None`` for a parent, whose sub-components may each use a
            different thread count.
        decomposition (DomainDecompositionSpec | None): The specification of how this
            component's domain is to be split across its ``MPICartesianGrid``, or
            ``None`` when the component has no domain. Only a leaf may have one.
        sub_layouts (tuple[ComponentLayout, ...]): Layouts for each direct sub-component,
            in the same order as ``ParallelComponent.subcomponents``. Names must be
            unique, and the cores they receive must fit within ``n_cores`` - under
            ``CoreSharing.PARTITIONED`` their total cannot exceed it, and under
            ``CoreSharing.SHARED`` none of them may reach past it - though either way
            they may come to less.
        core_sharing (CoreSharing): Whether ``sub_layouts`` divide this component's cores
            between them or share one range of them. Defaults to
            ``CoreSharing.PARTITIONED``, and says nothing about a leaf, which has no
            sub-layouts to combine.
        core_offset (int): The core this component starts at within its parent's range, as
            ESMF means ``rootpe``. Must be >= 0, and 0 unless the parent shares its cores
            among its sub-layouts. Defaults to 0.

    Raises:
        TypeError: If ``name`` is not a ``str``, or ``n_cores``/``n_ranks``/
            ``threads_per_rank`` are not ``int``.
        ValueError: If ``name`` is empty; ``n_cores``/``n_ranks``/``threads_per_rank``
            are < 1; ``core_offset`` is < 0; the ``decomposition`` grid does not use
            exactly ``n_ranks`` ranks; two sub-layouts share a name; or the leaf/parent
            rules above are broken - a leaf without a thread count, a parent with one, a
            leaf whose ranks and threads do not multiply to ``n_cores``, a parent whose
            sub-layouts overspend its cores or disagree with its rank total, a parent
            holding a decomposition, a sub-layout placed at an offset under a parent that
            partitions its cores, or one placed at an offset while running more than one
            core per rank.

    Examples:
        >>> from access.config.parallel_mpi_grid import MPICartesianGrid
        >>> atm_layout = ComponentLayout(
        ...     name="atmosphere",
        ...     n_cores=48,
        ...     n_ranks=12,
        ...     threads_per_rank=4,
        ...     decomposition=parallel_domain.DomainDecompositionSpec(
        ...         parallel_domain.Domain((192, 144)), MPICartesianGrid((4, 3))
        ...     ),
        ... )
    """

    name: str
    n_cores: int
    n_ranks: int
    threads_per_rank: int | None
    decomposition: parallel_domain.DomainDecompositionSpec | None
    sub_layouts: tuple[ComponentLayout, ...] = ()
    core_sharing: CoreSharing = CoreSharing.PARTITIONED
    core_offset: int = 0

    def __post_init__(self) -> None:
        self._validate_own_fields()
        if self.sub_layouts:
            self._validate_as_parent()
        else:
            self._validate_as_leaf()

    def _validate_own_fields(self) -> None:
        """Check this layout's own name and counts, independently of its kind."""
        if type(self.name) is not str:
            raise TypeError(f"ComponentLayout.name must be a str, got {type(self.name).__name__}.")
        if not self.name:
            raise ValueError("ComponentLayout.name must be non-empty.")
        self._validate_own_counts()
        if self.decomposition is not None and self.decomposition.n_ranks != self.n_ranks:
            raise ValueError(
                f"ComponentLayout {self.name!r}: decomposition grid "
                f"{self.decomposition.grid.shape} uses {self.decomposition.n_ranks} rank(s), "
                f"but n_ranks is {self.n_ranks}."
            )

    def _validate_own_counts(self) -> None:
        """Check the integer fields: the two counts, the threads and the core offset."""
        for field_name in ("n_cores", "n_ranks"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"ComponentLayout.{field_name} must be an int, got {type(value).__name__}.")
            if value < 1:
                raise ValueError(f"ComponentLayout.{field_name} must be >= 1, got {value}.")
        if self.core_offset < 0:
            raise ValueError(f"ComponentLayout.core_offset must be >= 0, got {self.core_offset}.")
        if self.threads_per_rank is not None:
            if type(self.threads_per_rank) is not int:
                raise TypeError(
                    f"ComponentLayout.threads_per_rank must be an int or None, "
                    f"got {type(self.threads_per_rank).__name__}."
                )
            if self.threads_per_rank < 1:
                raise ValueError(f"ComponentLayout.threads_per_rank must be >= 1, got {self.threads_per_rank}.")

    def _validate_as_leaf(self) -> None:
        """Check the rules for a component that spends its own cores."""
        if self.threads_per_rank is None:
            raise ValueError(
                f"ComponentLayout {self.name!r} has no sub-layouts, so it runs its own ranks and "
                "must state threads_per_rank."
            )
        if self.n_ranks * self.threads_per_rank != self.n_cores:
            raise ValueError(
                f"ComponentLayout {self.name!r}: {self.n_ranks} rank(s) x "
                f"{self.threads_per_rank} thread(s) is {self.n_ranks * self.threads_per_rank} core(s), "
                f"but n_cores is {self.n_cores}. A leaf must spend its cores exactly; idle cores "
                "belong to the parent that did not hand them out."
            )

    def _validate_sub_layout_offset(self, sub: ComponentLayout, shared: bool) -> None:
        """Check that *sub* may start where it says it does within this component.

        Args:
            sub (ComponentLayout): One of this component's sub-layouts.
            shared (bool): Whether the sub-layouts share this component's cores.

        Raises:
            ValueError: If *sub* states an offset under a component that partitions its
                cores, where each sub-layout already starts where the last one ended; or
                if it states one while running more than one core per rank, where a core
                index and a PE index are not the same thing.
        """
        if not sub.core_offset:
            return
        if not shared:
            raise ValueError(
                f"ComponentLayout {self.name!r}: sub-layout {sub.name!r} states a core offset of "
                f"{sub.core_offset}, but this component's sub-layouts divide its cores between "
                "them, so each one starts where the last ended and cannot be placed."
            )
        if sub.n_cores != sub.n_ranks:
            raise ValueError(
                f"ComponentLayout {self.name!r}: sub-layout {sub.name!r} starts at core "
                f"{sub.core_offset} but runs {sub.n_ranks} rank(s) over {sub.n_cores} core(s). "
                "An offset is a core index, as ESMF means rootpe, and only means the same thing "
                "as a PE index when a rank holds one core."
            )

    def _validate_as_parent(self) -> None:
        """Check the rules for a component that distributes its cores.

        Layout enumeration constructs millions of these, so this walks the sub-layouts
        once and allocates nothing on the success path.
        """
        if self.threads_per_rank is not None:
            raise ValueError(
                f"ComponentLayout {self.name!r} has sub-layouts, so it runs no ranks of its own and "
                f"cannot have threads_per_rank (got {self.threads_per_rank}); its sub-components may "
                "each use a different thread count."
            )
        if self.decomposition is not None:
            raise ValueError(
                f"ComponentLayout {self.name!r} has sub-layouts, so it cannot also hold a "
                "decomposition: its cores belong to its sub-components."
            )
        shared = self.core_sharing is CoreSharing.SHARED
        seen: set[str] = set()
        sub_cores = 0
        sub_ranks = 0
        for sub in self.sub_layouts:
            if sub.name in seen:
                dupes = [name for name, count in Counter(s.name for s in self.sub_layouts).items() if count > 1]
                raise ValueError(
                    f"ComponentLayout {self.name!r}: sub_layouts must have unique names; duplicates: {dupes}."
                )
            seen.add(sub.name)
            self._validate_sub_layout_offset(sub, shared)
            if shared:
                # Shared sub-components take turns on the same cores, so the parent has to
                # reach as far as the furthest of them gets, not hold their total. How
                # much of the range they sit on is a separate question, answered below.
                sub_cores = max(sub_cores, sub.core_offset + sub.n_cores)
            else:
                sub_cores += sub.n_cores
                sub_ranks += sub.n_ranks
        if shared:
            # The ranks a shared parent holds are the ones its sub-components sit on, so a
            # rank two of them share counts once and a stretch nobody is on not at all.
            sub_ranks = _occupied((sub.core_offset, sub.n_ranks) for sub in self.sub_layouts)
        if sub_cores > self.n_cores:
            raise ValueError(self._overspent_message(shared, sub_cores))
        if sub_ranks != self.n_ranks:
            raise ValueError(self._rank_total_message(shared, sub_ranks))

    def _overspent_message(self, shared: bool, sub_cores: int) -> str:
        """Return the complaint that this component's sub-layouts do not fit in its cores.

        Args:
            shared (bool): Whether the sub-layouts share this component's cores.
            sub_cores (int): What they came to, under whichever rule applies.

        Returns:
            str: The message, naming the rule that was applied.
        """
        detail = (
            f"its sub-layouts reach {sub_cores} core(s) into the range"
            if shared
            else f"sub-layouts use {sub_cores} core(s) in total"
        )
        return (
            f"ComponentLayout {self.name!r}: {detail}, which exceeds the {self.n_cores} core(s) "
            "assigned to this component."
        )

    def _rank_total_message(self, shared: bool, sub_ranks: int) -> str:
        """Return the complaint that ``n_ranks`` disagrees with the sub-layouts.

        Args:
            shared (bool): Whether the sub-layouts share this component's cores.
            sub_ranks (int): What they came to, under whichever rule applies.

        Returns:
            str: The message, naming the rule that was applied.
        """
        rule = (
            "A parent whose sub-components share its cores has as many ranks as they sit on between them."
            if shared
            else "A parent's n_ranks is the total over its subtree."
        )
        return (
            f"ComponentLayout {self.name!r}: n_ranks is {self.n_ranks}, but its sub-layouts hold "
            f"{sub_ranks} rank(s). {rule}"
        )

    def _combine(self, counts: Iterable[int]) -> int:
        """Total *counts* over the sub-layouts the way this component's sharing rule says.

        Partitioned sub-components hold disjoint things, so their counts add. Shared ones
        take turns on the same range, so what the parent spends is the part of that range
        at least one of them sits on: overlaps count once, and a stretch nobody is on does
        not count, wherever in the range it falls.

        Note that this is not the same as how far they *reach*, which is what the parent
        has to be large enough for. A range whose sub-components all start partway into it
        is idle at the front, and this counts that.

        Args:
            counts (Iterable[int]): One count per sub-layout, in order.

        Returns:
            int: The combined count, or 0 when there are no sub-layouts.
        """
        if self.core_sharing is CoreSharing.SHARED:
            return _occupied((sub.core_offset, count) for sub, count in zip(self.sub_layouts, counts, strict=True))
        return sum(counts)

    @property
    def is_leaf(self) -> bool:
        """Whether this component runs its own ranks rather than distributing cores."""
        return not self.sub_layouts

    @property
    def used_cores(self) -> int:
        """Cores this component actually spends, as opposed to the ``n_cores`` it holds.

        A leaf spends ``n_ranks × threads_per_rank``. A parent spends what its sub-layouts
        occupy between them: their total when it partitions its cores, and the part of the
        range at least one of them sits on when they share it. Either may be less than
        ``n_cores``; the difference is ``idle_cores``.
        """
        if self.threads_per_rank is not None:
            return self.n_ranks * self.threads_per_rank
        return self._combine(sub.n_cores for sub in self.sub_layouts)

    @property
    def idle_cores(self) -> int:
        """Cores allocated to this component but not spent by any sub-component.

        Under ``CoreSharing.SHARED`` these are the cores no sub-layout sits on, wherever
        in the range they fall - before the first, between two, or after the last. They
        are not the cores a sub-layout's siblings left unused: the siblings are meant to
        be on the same cores as it, and a core they share is spent once.
        """
        return self.n_cores - self.used_cores


class LocalConstraint(ABC):
    """Abstract base for constraints on a single component's layout.

    Instances of subclasses should be placed in ``ParallelComponent.local_constraints``.
    """

    @abstractmethod
    def is_satisfied(self, layout: ComponentLayout) -> bool:
        """Return ``True`` if the constraint is satisfied.

        Args:
            layout (ComponentLayout): The candidate layout for the component this
                constraint is attached to.
        """


class GroupConstraint(ABC):
    """Abstract base for constraints on a set of sibling component layouts.

    Because these constraints act on the layouts of several sibling sub-components
    simultaneously, they need to be added to the *parent* component, not to the individual
    sub-components. That means instances of subclasses must be placed in
    ``ParallelComponent.group_constraints`` of the *parent* component.
    """

    @abstractmethod
    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...]) -> bool:
        """Return ``True`` if the constraint is satisfied.

        Args:
            sub_layouts (tuple[ComponentLayout, ...]): Candidate layouts for every direct
                sub-component of the parent this constraint is attached to, in
                declaration order.
        """


@dataclass(frozen=True)
class ParallelComponent:
    """A parallelisable unit of work that is part of a larger parallel application.

    A component either does work itself or distributes its cores, never both:

    * A **leaf** may have a ``Domain``, whose work is decomposed across the component's
      MPI ranks using a cartesian process grid. It turns the cores it is given into
      ranks and threads.
    * A **parent** has sub-components that receive its cores - a disjoint subset each, or
      one range they share and run on in turn, according to its ``core_sharing`` - and no
      domain of its own.

    Either kind may carry constraints. Constrains are expressed as subclasses of
    ``LocalConstraint`` and ``GroupConstraint``, and are used to filter candidate layouts.
    There are two kinds:

    * ``LocalConstraint`` instances that filter candidate layouts for *this* component.
    * ``GroupConstraint`` instances that filter the *combined* layouts of all
      sub-components.

    This class is frozen (``frozen=True``) so that a component tree is immutable and
    hashable. This allows it to be validated once, reused across searches, and used as part
    of the enumerator's subtree memoisation key. Assigning to a field of this class raises
    ``dataclasses.FrozenInstanceError``.

    Args:
        name (str): Human-readable identifier. Must be non-empty.
        domain (Domain | None): Grid to be decomposed across this component's ranks, or
            ``None``. Only a component without sub-components may have a domain: a
            component that hands its cores out to children does not run a decomposition
            of its own.
        subcomponents (tuple[ParallelComponent, ...]): Direct child components. Each
            receives a disjoint subset of this component's cores, or a share of one range
            of them when ``core_sharing`` says so. All names must be unique; a
            ``ValueError`` is raised on construction if any two sub-components
            share a name. Cannot be combined with ``domain``.
        local_constraints (tuple[LocalConstraint, ...]): Constraints to be checked against
            this component's own layout.
        group_constraints (tuple[GroupConstraint, ...]): Constraints to be checked against
            the *joint* layouts of all sub-components. Must be placed on the parent, not the
            individual sub-components.
        core_sharing (CoreSharing): Whether ``subcomponents`` divide this component's cores
            between them or all draw on one range of them, running on it in turn. Defaults
            to ``CoreSharing.PARTITIONED``. See ``CoreSharing`` for what a shared parent
            costs to enumerate, and which allocation modes can size one's children.
        core_offset (int): The core this component starts at within its parent's range, as
            ESMF means ``rootpe``. Must be >= 0, and only a parent that shares its cores
            may have sub-components stating one. Defaults to 0.

    Raises:
        ValueError: If ``name`` is empty; ``core_offset`` is < 0; two sub-components share
            a name; a component has both a ``domain`` and sub-components; or a component
            that partitions its cores has sub-components stating a ``core_offset``.

    Examples:
        >>> atm = ParallelComponent("atmosphere", domain=parallel_domain.Domain((192, 144)))
        >>> ocn = ParallelComponent("ocean",      domain=parallel_domain.Domain((360, 300)))
        >>> ice = ParallelComponent("ice")
        >>> coupled = ParallelComponent("coupled_model", subcomponents=(atm, ocn, ice))
    """

    name: str
    domain: parallel_domain.Domain | None = None
    subcomponents: tuple[ParallelComponent, ...] = ()
    local_constraints: tuple[LocalConstraint, ...] = ()
    group_constraints: tuple[GroupConstraint, ...] = ()
    core_sharing: CoreSharing = CoreSharing.PARTITIONED
    core_offset: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParallelComponent.name must be non-empty.")
        if self.core_offset < 0:
            raise ValueError(f"ParallelComponent {self.name!r}: core_offset must be >= 0, got {self.core_offset}.")
        if self.core_sharing is CoreSharing.PARTITIONED:
            placed = [sub.name for sub in self.subcomponents if sub.core_offset]
            if placed:
                raise ValueError(
                    f"ParallelComponent {self.name!r} divides its cores between its sub-components, so "
                    f"each starts where the last ended, but {placed} state a core offset of their own. "
                    "Offsets only mean something under a component whose cores are shared."
                )
        names = [sub.name for sub in self.subcomponents]
        if len(names) != len(set(names)):
            dupes = [n for n, c in Counter(names).items() if c > 1]
            raise ValueError(f"ParallelComponent subcomponents must have unique names; duplicates: {dupes}.")
        if self.domain is not None and self.subcomponents:
            # Sub-components receive *disjoint* subsets of this component's cores, so a
            # domain here would be decomposed over cores the children are also using.
            raise ValueError(
                f"ParallelComponent {self.name!r} has both a domain and sub-components. "
                "Sub-components each take a disjoint subset of this component's cores, so there "
                "are none left for the component's own ranks. Put the domain on a leaf component "
                "instead."
            )

    def iter_decompositions(self, n_ranks: int) -> Iterator[parallel_domain.DomainDecompositionSpec | None]:
        """Yield every way this component's domain can be split over *n_ranks* ranks.

        A component without a domain still yields exactly once, with ``None``, so a caller
        enumerating layouts can treat the decomposition as one more dimension of the search
        rather than special-casing the components that have nothing to decompose.

        The grids come from ``MPICartesianGrid.iter_grids``, which validates its arguments,
        so a bad *n_ranks* is reported when this generator is first advanced rather than
        when it is created.

        Args:
            n_ranks (int): MPI ranks to split the domain over. Must be >= 1.

        Yields:
            DomainDecompositionSpec | None: One specification per process grid using
                exactly *n_ranks* ranks, in ``MPICartesianGrid.iter_grids`` order, or a
                single ``None`` when this component has no domain.

        Raises:
            TypeError: If *n_ranks* is not an ``int``, once iteration starts.
            ValueError: If *n_ranks* is < 1, once iteration starts.

        Examples:
            >>> ocn = ParallelComponent("ocean", domain=parallel_domain.Domain((360, 300)))
            >>> [spec.grid.shape for spec in ocn.iter_decompositions(4)]
            [(1, 4), (2, 2), (4, 1)]

            A component with no domain still yields once, so the caller's loop is uniform:

            >>> list(ParallelComponent("coupler").iter_decompositions(4))
            [None]
        """
        if self.domain is None:
            yield None
            return
        for grid in parallel_mpi_grid.MPICartesianGrid.iter_grids(self.domain.ndim, n_ranks):
            yield parallel_domain.DomainDecompositionSpec(self.domain, grid)
