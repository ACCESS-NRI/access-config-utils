# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Rank-allocation strategy for climate model component trees.

This module owns the allocation strategy model used by
:func:`access.config.layouts.enumerate_layouts`:

- :class:`AllocationStrategy` encodes how MPI ranks are distributed to one node in
    a component tree, along with optional child strategies and constraint filters.

The allocation mode is determined by the fields set on :class:`AllocationStrategy`:

- **fixed** – set ``n_ranks`` to assign exactly that many ranks.
- **ratio** – set ``weight`` to distribute ranks proportionally among siblings.
- **free** (default) – set ``min_ranks`` / ``max_ranks`` to let the enumeration
    engine assign any rank count in the given range.

See :mod:`access.config.parallel_component` for the component-tree model and
:mod:`access.config.domain_parallelisation` for domain/decomposition models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from access.config.parallel_component import (
    GroupConstraint,
    LocalConstraint,
)

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

    Exactly one allocation mode is active, determined by the fields that are set:

    * **Fixed** (``n_ranks`` is not ``None``) – the component receives exactly
      ``n_ranks`` MPI ranks.
    * **Ratio** (``weight`` is not ``None``) – the component receives ranks in
      exact proportion to sibling weights.  There exists a shared integer
      multiplier ``k >= 1`` such that each ratio-allocated sibling receives
      ``k * weight`` ranks.  For example, weights (3, 2) can be allocated as
      (3, 2), (6, 4), (9, 6), and so on.
    * **Free** (default, both ``n_ranks`` and ``weight`` are ``None``) – the
      enumeration engine may assign any rank count in ``[min_ranks, max_ranks]``
      from the budget remaining after fixed and ratio allocations.

    Parameters
    ----------
    n_ranks : int | None
        Fixed allocation: the component must receive exactly this many MPI ranks.
        Must be >= 1 when set.  Cannot be combined with ``weight``,
        ``min_ranks`` != 1, or ``max_ranks`` != ``None``.
    weight : int | None
        Ratio allocation: relative weight among ratio-allocated siblings.
        Must be >= 1 when set.  Cannot be combined with ``n_ranks``,
        ``min_ranks`` != 1, or ``max_ranks`` != ``None``.
    min_ranks : int
        Free-mode lower bound on rank count.  Must be >= 1.  Defaults to 1.
        Ignored when ``n_ranks`` or ``weight`` is set.
    max_ranks : int | None
        Free-mode upper bound on rank count, or ``None`` for no upper bound.
        Must be >= ``min_ranks`` when provided.  Defaults to ``None``.
        Ignored when ``n_ranks`` or ``weight`` is set.
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
    >>> AllocationStrategy(subcomponents={
    ...     "UM7":   AllocationStrategy(),
    ...     "MOM5":  AllocationStrategy(),
    ...     "CICE5": AllocationStrategy(n_ranks=12),
    ... })
    """

    n_ranks: int | None = None
    weight: int | None = None
    min_ranks: int = 1
    max_ranks: int | None = None
    subcomponents: dict[str, AllocationStrategy] = field(default_factory=dict)
    local_constraints: tuple[LocalConstraint, ...] = ()
    group_constraints: tuple[GroupConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.n_ranks is not None and self.weight is not None:
            raise ValueError("AllocationStrategy: n_ranks and weight cannot both be set.")
        if self.n_ranks is not None:
            if self.n_ranks < 1:
                raise ValueError(f"AllocationStrategy.n_ranks must be >= 1, got {self.n_ranks}.")
            if self.min_ranks != 1 or self.max_ranks is not None:
                raise ValueError("AllocationStrategy: min_ranks/max_ranks cannot be set alongside n_ranks.")
        elif self.weight is not None:
            if self.weight < 1:
                raise ValueError(f"AllocationStrategy.weight must be >= 1, got {self.weight}.")
            if self.min_ranks != 1 or self.max_ranks is not None:
                raise ValueError("AllocationStrategy: min_ranks/max_ranks cannot be set alongside weight.")
        else:
            if self.min_ranks < 1:
                raise ValueError(f"AllocationStrategy.min_ranks must be >= 1, got {self.min_ranks}.")
            if self.max_ranks is not None and self.max_ranks < self.min_ranks:
                raise ValueError(
                    f"AllocationStrategy.max_ranks ({self.max_ranks}) must be >= min_ranks ({self.min_ranks})."
                )
        object.__setattr__(self, "subcomponents", MappingProxyType(self.subcomponents))

    @property
    def allocation_mode(self) -> Literal["fixed", "ratio", "free"]:
        """The active allocation mode: ``"fixed"``, ``"ratio"``, or ``"free"``."""
        if self.n_ranks is not None:
            return "fixed"
        if self.weight is not None:
            return "ratio"
        return "free"

    def __hash__(self) -> int:
        return hash(
            (
                self.n_ranks,
                self.weight,
                self.min_ranks,
                self.max_ranks,
                tuple(sorted(self.subcomponents.items())),
                self.local_constraints,
                self.group_constraints,
            )
        )


# The component-tree model lives in access.config.parallel_component.
