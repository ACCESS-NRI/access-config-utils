# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Parallel component and domain definitions for climate model simulations.

This module defines the structural building blocks used to describe how a coupled
climate model can be parallelised:

- Rank-allocation types (:class:`FixedAllocation`, :class:`RatioAllocation`, :class:`FreeAllocation`)
  that specify how MPI ranks are distributed to a component.
- Constraint abstract base classes (:class:`LocalConstraint`, :class:`GroupConstraint`)
  that filter which layouts are considered valid.
- :class:`ParallelComponent`: a parallelisable unit that may carry a domain, sub-components,
  and constraints.

See :mod:`access.config.domain_parallelisation` for domain/decomposition models,
and :mod:`access.config.layouts` for layout result types, concrete constraint
implementations, and the :func:`~access.config.layouts.enumerate_layouts` entry
point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from access.config import domain_parallelisation

if TYPE_CHECKING:
    from access.config.layouts import ComponentLayout

# ---------------------------------------------------------------------------
# Rank allocation types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedAllocation:
    """The component must receive exactly ``n_ranks`` MPI ranks.

    Parameters
    ----------
    n_ranks : int
        Number of ranks.  Must be >= 1.
    """

    n_ranks: int

    def __post_init__(self) -> None:
        if self.n_ranks < 1:
            raise ValueError(f"FixedAllocation.n_ranks must be >= 1, got {self.n_ranks}.")


@dataclass(frozen=True)
class RatioAllocation:
    """The component receives ranks in exact proportion to sibling weights.

    When a parent's available ranks are distributed among siblings that use
    :class:`RatioAllocation`, their rank counts are taken to be exact integer
    multiples of the declared weights: there exists a shared integer
    multiplier ``k`` such that each sibling receives ``k * weight`` ranks.

    For example, weights (3, 2) can be allocated as (3, 2), (6, 4), (9, 6),
    and so on. Ranks are not assigned by rounding ``weight / total_weight *
    available``, and any remainder that cannot be expressed via a common
    integer multiplier is not distributed among ratio-allocated siblings by a
    largest-remainder rule.

    Parameters
    ----------
    weight : int
        Relative weight.  Must be >= 1.
    """

    weight: int

    def __post_init__(self) -> None:
        if self.weight < 1:
            raise ValueError(f"RatioAllocation.weight must be >= 1, got {self.weight}.")


@dataclass(frozen=True)
class FreeAllocation:
    """The enumeration engine may assign any rank count in ``[min_ranks, max_ranks]``.

    Parameters
    ----------
    min_ranks : int
        Minimum number of ranks.  Must be >= 1.  Defaults to 1.
    max_ranks : int | None
        Maximum number of ranks, or ``None`` for no upper bound.
            Must be >= ``min_ranks`` when provided.
    """

    min_ranks: int = 1
    max_ranks: int | None = None

    def __post_init__(self) -> None:
        if self.min_ranks < 1:
            raise ValueError(f"FreeAllocation.min_ranks must be >= 1, got {self.min_ranks}.")
        if self.max_ranks is not None and self.max_ranks < self.min_ranks:
            raise ValueError(f"FreeAllocation.max_ranks ({self.max_ranks}) must be >= min_ranks ({self.min_ranks}).")


#: Type alias covering all rank-allocation specifications.
RankAllocation = FixedAllocation | RatioAllocation | FreeAllocation


# ---------------------------------------------------------------------------
# AllocationStrategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationStrategy:
    """Rank-allocation assignment for one node in a component tree.

    An :class:`AllocationStrategy` tree is passed to
    :func:`~access.config.layouts.enumerate_layouts` to describe how ranks are
    distributed among sub-components, independently of the
    :class:`ParallelComponent` structure (domains and constraints).

    Parameters
    ----------
    allocation : RankAllocation
        How this component receives ranks from its parent.
    subcomponents : dict[str, AllocationStrategy]
        Allocation strategies for direct sub-components, keyed by component name.
        Omit (or pass ``{}``) for leaf components.
    local_constraints : tuple[LocalConstraint, ...]
        Constraints applied only when this allocation strategy is active, checked
        alongside the component's own :attr:`~ParallelComponent.local_constraints`.
        Use this for strategy-specific filters (e.g. aspect-ratio limits) that
        should not be hard-coded into the component definition.
    group_constraints : tuple[GroupConstraint, ...]
        Group constraints applied only when this allocation strategy is active,
        checked alongside the parent component's own
        :attr:`~ParallelComponent.group_constraints`.
        Use this for strategy-specific cross-component constraints (e.g.
        rank-ratio limits) that should not be hard-coded into the component
        definition.

    Examples
    --------
    >>> AllocationStrategy(FreeAllocation(), subcomponents={
    ...     "UM7":   AllocationStrategy(FreeAllocation()),
    ...     "MOM5":  AllocationStrategy(FreeAllocation()),
    ...     "CICE5": AllocationStrategy(FixedAllocation(12)),
    ... })
    """

    allocation: RankAllocation
    subcomponents: dict[str, AllocationStrategy] = field(default_factory=dict)
    local_constraints: tuple[LocalConstraint, ...] = ()
    group_constraints: tuple[GroupConstraint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subcomponents", MappingProxyType(self.subcomponents))

    def __hash__(self) -> int:
        return hash(
            (
                self.allocation,
                tuple(sorted(self.subcomponents.items())),
                self.local_constraints,
                self.group_constraints,
            )
        )


# ---------------------------------------------------------------------------
# Constraint abstract base classes
# ---------------------------------------------------------------------------


class LocalConstraint(ABC):
    """Abstract base for constraints on a single component's layout.

    Place instances of subclasses in :attr:`ParallelComponent.local_constraints`.
    Concrete implementations live in :mod:`access.config.layouts`.
    """

    @abstractmethod
    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        """Return ``True`` if the constraint is satisfied.

        Parameters
        ----------
        layout : ComponentLayout
            Candidate layout for the component being validated.
        total_ranks : int
            System-wide total MPI ranks (useful for fractional-rank constraints).
        """


class GroupConstraint(ABC):
    """Abstract base for constraints on a set of sibling component layouts.

    Place instances of subclasses in :attr:`ParallelComponent.group_constraints` of the
    *parent* component, because these constraints need access to all siblings'
    layouts simultaneously.

    Concrete implementations live in :mod:`access.config.layouts`.
    """

    @abstractmethod
    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_ranks: int) -> bool:
        """Return ``True`` if the constraint is satisfied.

        Parameters
        ----------
        sub_layouts : tuple[ComponentLayout, ...]
            Candidate layouts for all direct sub-components of the parent, in the
            same order as :attr:`ParallelComponent.subcomponents`.
        total_ranks : int
            System-wide total MPI ranks.
        """


# ---------------------------------------------------------------------------
# ParallelComponent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelComponent:
    """A parallelisable unit in a climate model.

    A component may have:

        * An optional :class:`~access.config.domain_parallelisation.Domain` whose work
            is decomposed across the component's
      MPI ranks using a cartesian process grid.
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
