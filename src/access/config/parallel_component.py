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
from collections.abc import Iterator
from dataclasses import dataclass

from access.config import parallel_domain, parallel_mpi_grid


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
    * A **parent** component spends its cores by assigning disjoint subsets of them to its
      sub-components, and has no thread count of its own - each sub-component may use a
      different one, so ``threads_per_rank`` is ``None``. A parent may hand out fewer
      cores than it holds, leaving the remainder idle (see ``idle_cores``).

    This class is frozen (``frozen=True``) so that a layout is an immutable, hashable value
    object, and assigning to a field raises ``dataclasses.FrozenInstanceError``. The layout
    enumerator relies on this property, as it memoises subtrees and hands the *same*
    instance to every result whose tree contains it, so a layout must never be modified in
    place.

    Args:
        name (str): Name of the ``ParallelComponent`` this layout resolves.
        n_cores (int): CPU cores allocated to this component. Must be >= 1.
        n_ranks (int): MPI ranks in this component's subtree: the ranks it runs itself
            for a leaf, or the total over ``sub_layouts`` for a parent. Must be >= 1.
        threads_per_rank (int | None): OpenMP threads per MPI rank, for a leaf. Must be
            >= 1 when set. ``None`` for a parent, whose sub-components may each use a
            different thread count.
        decomposition (DomainDecompositionSpec | None): The specification of how this
            component's domain is to be split across its ``MPICartesianGrid``, or
            ``None`` when the component has no domain. Only a leaf may have one.
        sub_layouts (tuple[ComponentLayout, ...]): Layouts for each direct sub-component,
            in the same order as ``ParallelComponent.subcomponents``. Names must be
            unique, and the cores they receive must fit within ``n_cores`` -
            sub-components receive disjoint subsets of this component's cores, so their
            total cannot exceed it, though it may be less.

    Raises:
        TypeError: If ``name`` is not a ``str``, or ``n_cores``/``n_ranks``/
            ``threads_per_rank`` are not ``int``.
        ValueError: If ``name`` is empty; ``n_cores``/``n_ranks``/``threads_per_rank``
            are < 1; the ``decomposition`` grid does not use exactly ``n_ranks`` ranks;
            two sub-layouts share a name; or the leaf/parent rules above are broken - a
            leaf without a thread count, a parent with one, a leaf whose ranks and
            threads do not multiply to ``n_cores``, a parent whose sub-layouts overspend
            its cores or disagree with its rank total, or a parent holding a
            decomposition.

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
        for field_name in ("n_cores", "n_ranks"):
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"ComponentLayout.{field_name} must be an int, got {type(value).__name__}.")
            if value < 1:
                raise ValueError(f"ComponentLayout.{field_name} must be >= 1, got {value}.")
        if self.threads_per_rank is not None:
            if type(self.threads_per_rank) is not int:
                raise TypeError(
                    f"ComponentLayout.threads_per_rank must be an int or None, "
                    f"got {type(self.threads_per_rank).__name__}."
                )
            if self.threads_per_rank < 1:
                raise ValueError(f"ComponentLayout.threads_per_rank must be >= 1, got {self.threads_per_rank}.")
        if self.decomposition is not None and self.decomposition.n_ranks != self.n_ranks:
            raise ValueError(
                f"ComponentLayout {self.name!r}: decomposition grid "
                f"{self.decomposition.grid.shape} uses {self.decomposition.n_ranks} rank(s), "
                f"but n_ranks is {self.n_ranks}."
            )

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
            sub_cores += sub.n_cores
            sub_ranks += sub.n_ranks
        if sub_cores > self.n_cores:
            raise ValueError(
                f"ComponentLayout {self.name!r}: sub-layouts use {sub_cores} core(s) in total, "
                f"which exceeds the {self.n_cores} core(s) assigned to this component."
            )
        if sub_ranks != self.n_ranks:
            raise ValueError(
                f"ComponentLayout {self.name!r}: n_ranks is {self.n_ranks}, but its sub-layouts hold "
                f"{sub_ranks} rank(s). A parent's n_ranks is the total over its subtree."
            )

    @property
    def is_leaf(self) -> bool:
        """Whether this component runs its own ranks rather than distributing cores."""
        return not self.sub_layouts

    @property
    def used_cores(self) -> int:
        """Cores this component actually spends, as opposed to the ``n_cores`` it holds.

        A leaf spends ``n_ranks × threads_per_rank``; a parent spends the total handed to
        its sub-layouts, which may be less than ``n_cores``. The difference is
        ``idle_cores``.
        """
        if self.threads_per_rank is not None:
            return self.n_ranks * self.threads_per_rank
        return sum(sub.n_cores for sub in self.sub_layouts)

    @property
    def idle_cores(self) -> int:
        """Cores allocated to this component but not handed to any sub-component."""
        return self.n_cores - self.used_cores


class LocalConstraint(ABC):
    """Abstract base for constraints on a single component's layout.

    Instances of subclasses should be placed in ``ParallelComponent.local_constraints``.
    """

    @abstractmethod
    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        """Return ``True`` if the constraint is satisfied.

        Args:
            layout (ComponentLayout): The candidate layout for the component this
                constraint is attached to.
            total_cores (int): The whole search's core budget, for constraints expressed
                as a share of it.
        """


class GroupConstraint(ABC):
    """Abstract base for constraints on a set of sibling component layouts.

    Because these constraints act on the layouts of several sibling sub-components
    simultaneously, they need to be added to the *parent* component, not to the individual
    sub-components. That means instances of subclasses must be placed in
    ``ParallelComponent.group_constraints`` of the *parent* component.
    """

    @abstractmethod
    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_cores: int) -> bool:
        """Return ``True`` if the constraint is satisfied.

        Args:
            sub_layouts (tuple[ComponentLayout, ...]): Candidate layouts for every direct
                sub-component of the parent this constraint is attached to, in
                declaration order.
            total_cores (int): The whole search's core budget, for constraints expressed
                as a share of it.
        """


@dataclass(frozen=True)
class ParallelComponent:
    """A parallelisable unit of work that is part of a larger parallel application.

    A component either does work itself or distributes its cores, never both:

    * A **leaf** may have a ``Domain``, whose work is decomposed across the component's
      MPI ranks using a cartesian process grid. It turns the cores it is given into
      ranks and threads.
    * A **parent** has sub-components that each receive a disjoint subset of the
      component's cores, and no domain of its own.

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
            receives a disjoint subset of this component's cores. All names must be
            unique; a ``ValueError`` is raised on construction if any two sub-components
            share a name. Cannot be combined with ``domain``.
        local_constraints (tuple[LocalConstraint, ...]): Constraints to be checked against
            this component's own layout.
        group_constraints (tuple[GroupConstraint, ...]): Constraints to be checked against
            the *joint* layouts of all sub-components. Must be placed on the parent, not the
            individual sub-components.

    Raises:
        ValueError: If ``name`` is empty; two sub-components share a name; or a component
        has both a ``domain`` and sub-components.

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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ParallelComponent.name must be non-empty.")
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
