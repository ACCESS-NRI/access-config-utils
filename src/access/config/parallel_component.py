"""Component-tree model for climate model parallelisation.

This module owns the structural model used by
:func:`access.config.layouts.enumerate_layouts`:

- :class:`ComponentLayout` describes the resolved layout for one component.
- :class:`LocalConstraint` and :class:`GroupConstraint` define constraint hooks.
- :class:`ParallelComponent` describes the parallelisable tree structure.

See :mod:`access.config.parallelisation` for rank-allocation strategies and
:mod:`access.config.domain_parallelisation` for domain/decomposition models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

from access.config import domain_parallelisation


@dataclass(frozen=True)
class ComponentLayout:
    """The resolved parallelisation assignment for one component.

    Parameters
    ----------
    name : str
        ParallelComponent name.
    n_ranks : int
        MPI ranks assigned to this component.  Must be >= 1.
    threads_per_rank : int
        OpenMP threads per MPI rank.  Must be >= 1.
    decomposition : CartesianDecomposition | None
        How this component's domain is tiled across the MPI cartesian grid, or
        ``None`` when the component has no domain.
    sub_layouts : tuple[ComponentLayout, ...]
        Layouts for each direct sub-component, in the same order as
        :attr:`~access.config.parallel_component.ParallelComponent.subcomponents`.
    """

    name: str
    n_ranks: int
    threads_per_rank: int
    decomposition: domain_parallelisation.DomainCartesianDecomposition | None
    sub_layouts: tuple[ComponentLayout, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ComponentLayout.name must be non-empty.")
        if self.n_ranks < 1:
            raise ValueError(f"ComponentLayout.n_ranks must be >= 1, got {self.n_ranks}.")
        if self.threads_per_rank < 1:
            raise ValueError(f"ComponentLayout.threads_per_rank must be >= 1, got {self.threads_per_rank}.")

    @property
    def total_cores(self) -> int:
        """CPU cores consumed by this component (``n_ranks × threads_per_rank``)."""
        return self.n_ranks * self.threads_per_rank


class LocalConstraint(ABC):
    """Abstract base for constraints on a single component's layout.

    Place instances of subclasses in :attr:`ParallelComponent.local_constraints`.
    Concrete implementations live in :mod:`access.config.constraints`.
    """

    @abstractmethod
    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        """Return ``True`` if the constraint is satisfied."""


class GroupConstraint(ABC):
    """Abstract base for constraints on a set of sibling component layouts.

    Place instances of subclasses in :attr:`ParallelComponent.group_constraints` of the
    *parent* component, because these constraints need access to all siblings'
    layouts simultaneously.

    Concrete implementations live in :mod:`access.config.constraints`.
    """

    @abstractmethod
    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_ranks: int) -> bool:
        """Return ``True`` if the constraint is satisfied."""


@dataclass(frozen=True)
class ParallelComponent:
    """A parallelisable unit in a climate model.

    A component may have:

    * An optional :class:`~access.config.domain_parallelisation.Domain` whose work
      is decomposed across the component's MPI ranks using a cartesian process grid.
    * Sub-components that each receive a disjoint subset of the component's ranks.
    * :class:`LocalConstraint` instances that filter candidate layouts for *this*
      component.
    * :class:`GroupConstraint` instances that filter the *combined* layouts of all
      sub-components.

    Parameters
    ----------
    name : str
        Human-readable identifier.  Must be non-empty.
    domain : access.config.domain_parallelisation.Domain | None
        Grid decomposed across this component's ranks, or ``None``.
    subcomponents : tuple[ParallelComponent, ...]
        Direct child components.  Each receives a disjoint rank subset.
        All names must be unique; a ``ValueError`` is raised on construction
        if any two sub-components share a name.
    local_constraints : tuple[LocalConstraint, ...]
        Constraints checked against this component's own layout.
    group_constraints : tuple[GroupConstraint, ...]
        Constraints checked against the *joint* layouts of all sub-components.
        Must be placed on the parent, not the individual sub-components.

    Examples
    --------
    >>> atm = ParallelComponent("atmosphere", domain=domain_parallelisation.Domain((192, 144)))
    >>> ocn = ParallelComponent("ocean",      domain=domain_parallelisation.Domain((360, 300)))
    >>> ice = ParallelComponent("ice")
    >>> coupled = ParallelComponent("coupled_model", subcomponents=(atm, ocn, ice))
    """

    name: str
    domain: domain_parallelisation.Domain | None = None
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
