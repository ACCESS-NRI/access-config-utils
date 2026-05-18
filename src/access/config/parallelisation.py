# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Rank-allocation types for climate model component trees.

This module owns the allocation strategy model used by
:func:`access.config.layouts.enumerate_layouts`:

- :class:`FixedAllocation`, :class:`RatioAllocation`, and :class:`FreeAllocation`
    describe how MPI ranks are distributed to child components.
- :class:`AllocationStrategy` combines a rank allocation with optional child
    strategies and constraint filters.

See :mod:`access.config.parallel_component` for the component-tree model and
:mod:`access.config.domain_parallelisation` for domain/decomposition models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from access.config.parallel_component import (
    GroupConstraint,
    LocalConstraint,
)

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


# The component-tree model lives in access.config.parallel_component.
