# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Parallel component and domain definitions for climate model simulations.

This module defines the structural building blocks used to describe how a coupled
climate model can be parallelised:

- :class:`Domain`: an N-dimensional rectangular grid.
- :class:`CartesianDecomposition`: how a domain is tiled across an MPI cartesian process grid.
- Rank-allocation types (:class:`FixedRanks`, :class:`RatioAllocation`, :class:`FreeAllocation`)
  that specify how MPI ranks are distributed to a component.
- Constraint abstract base classes (:class:`LocalConstraint`, :class:`GroupConstraint`)
  that filter which layouts are considered valid.
- :class:`ParallelComponent`: a parallelisable unit that may carry a domain, sub-components,
  and constraints.

See :mod:`access.config.layouts` for layout result types, concrete constraint
implementations, and the :func:`~access.config.layouts.enumerate_layouts` entry point.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from access.config.layouts import ComponentLayout

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    """An N-dimensional rectangular grid.

    Parameters
    ----------
    shape : tuple[int, ...]
        Size of the grid along each spatial dimension.  Must be non-empty and
        every entry must be >= 1.

    Examples
    --------
    >>> Domain(shape=(360, 300))          # 2-D grid
    >>> Domain(shape=(192, 144, 85))      # 3-D grid
    """

    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("Domain shape must have at least one dimension.")
        if any(d < 1 for d in self.shape):
            raise ValueError(f"All dimension sizes must be >= 1, got shape={self.shape}.")

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of grid points (product of all dimension sizes)."""
        return math.prod(self.shape)


# ---------------------------------------------------------------------------
# CartesianDecomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CartesianDecomposition:
    """Decomposition of a domain across an MPI cartesian process grid.

    Parameters
    ----------
    domain : Domain
        The domain being decomposed.
    grid : tuple[int, ...]
        Number of MPI sub-domains along each dimension.  Must have the same
        length as ``domain.shape`` and all entries must be >= 1.

    Examples
    --------
    >>> CartesianDecomposition(Domain((360, 300)), grid=(6, 5))
    """

    domain: Domain
    grid: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.grid) != self.domain.ndim:
            raise ValueError(f"grid has {len(self.grid)} dimension(s), but domain has {self.domain.ndim}.")
        if any(g < 1 for g in self.grid):
            raise ValueError(f"All grid entries must be >= 1, got grid={self.grid}.")

    @property
    def n_ranks(self) -> int:
        """Total MPI ranks used by this decomposition (product of grid entries)."""
        return math.prod(self.grid)

    @property
    def local_shape(self) -> tuple[float, ...]:
        """Average local sub-domain size along each dimension (``domain.shape[i] / grid[i]``).

        Values are floating-point; use :class:`~access.config.layouts.UniformSubdomainConstraint`
        to enforce exact integer divisibility.
        """
        return tuple(d / g for d, g in zip(self.domain.shape, self.grid, strict=True))


# ---------------------------------------------------------------------------
# Rank allocation types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedRanks:
    """The component must receive exactly ``n_ranks`` MPI ranks.

    Parameters
    ----------
    n_ranks : int
        Number of ranks.  Must be >= 1.
    """

    n_ranks: int

    def __post_init__(self) -> None:
        if self.n_ranks < 1:
            raise ValueError(f"FixedRanks.n_ranks must be >= 1, got {self.n_ranks}.")


@dataclass(frozen=True)
class RatioAllocation:
    """The component receives a proportional share of the available ranks.

    When a parent's available ranks are distributed among siblings that all use
    :class:`RatioAllocation`, each sibling receives
    ``round(weight / total_weight * available)`` ranks, with ties broken via the
    largest-remainder method.

    For example, weights (3, 2) with 10 available ranks gives 6 and 4 ranks.

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
RankAllocation = FixedRanks | RatioAllocation | FreeAllocation


# ---------------------------------------------------------------------------
# AllocationSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationSpec:
    """Rank-allocation assignment for one node in a component tree.

    An :class:`AllocationSpec` tree is passed to
    :func:`~access.config.layouts.enumerate_layouts` to describe how ranks are
    distributed among sub-components, independently of the
    :class:`ParallelComponent` structure (domains and constraints).

    Parameters
    ----------
    allocation : RankAllocation
        How this component receives ranks from its parent.
    subcomponents : dict[str, AllocationSpec]
        Allocation specs for direct sub-components, keyed by component name.
        Omit (or pass ``{}``) for leaf components.
    local_constraints : tuple[LocalConstraint, ...]
        Constraints applied only when this allocation spec is active, checked
        alongside the component's own :attr:`~ParallelComponent.local_constraints`.
        Use this for strategy-specific filters (e.g. aspect-ratio limits) that
        should not be hard-coded into the component definition.
    group_constraints : tuple[GroupConstraint, ...]
        Group constraints applied only when this allocation spec is active,
        checked alongside the parent component's own
        :attr:`~ParallelComponent.group_constraints`.
        Use this for strategy-specific cross-component constraints (e.g.
        rank-ratio limits) that should not be hard-coded into the component
        definition.

    Examples
    --------
    >>> AllocationSpec(FreeAllocation(), subcomponents={
    ...     "UM7":   AllocationSpec(FreeAllocation()),
    ...     "MOM5":  AllocationSpec(FreeAllocation()),
    ...     "CICE5": AllocationSpec(FixedRanks(12)),
    ... })
    """

    allocation: RankAllocation
    subcomponents: dict[str, AllocationSpec] = field(default_factory=dict)
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

    * An optional :class:`Domain` whose work is decomposed across the component's
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
    domain : Domain | None
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

    Notes
    -----
    Rank-allocation strategies (how ranks are distributed among sub-components)
    are specified separately via :class:`AllocationSpec` and passed to
    :func:`~access.config.layouts.enumerate_layouts`.  This decoupling allows
    the same :class:`ParallelComponent` tree to be enumerated under different strategies.

    Examples
    --------
    >>> atm = ParallelComponent("atmosphere", domain=Domain((192, 144)))
    >>> ocn = ParallelComponent("ocean",      domain=Domain((360, 300)))
    >>> ice = ParallelComponent("ice")
    >>> coupled = ParallelComponent("coupled_model", subcomponents=(atm, ocn, ice))
    """

    name: str
    domain: Domain | None = None
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
