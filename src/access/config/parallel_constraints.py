# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Concrete local and group constraints for parallel component layouts.

This module defines reusable constraint implementations grouped into four
categories:

1. Cartesian grid constraints.
2. Rank distribution constraints.
3. Domain-layout constraints.
4. Threading constraints.

Constraint interfaces are defined in :mod:`access.config.parallel_component`.
"""

from __future__ import annotations

from dataclasses import dataclass

from access.config.parallel_component import (
    ComponentLayout,
    GroupConstraint,
    LocalConstraint,
)

# ---------------------------------------------------------------------------
# Category 1 — Cartesian grid constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessGridDimEvenConstraint(LocalConstraint):
    """``grid[dim]`` must be even.

    Parameters
    ----------
    dim : int
        Zero-based dimension index to check.
    """

    dim: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"ProcessGridDimEvenConstraint.dim must be >= 0, got {self.dim}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        return layout.decomposition.grid[self.dim] % 2 == 0


@dataclass(frozen=True)
class ProcessGridDimDivisibleConstraint(LocalConstraint):
    """``grid[dim]`` must be divisible by ``divisor``.

    Parameters
    ----------
    dim : int
        Zero-based dimension index to check.
    divisor : int
        Required divisor.  Must be >= 1.
    """

    dim: int
    divisor: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"ProcessGridDimDivisibleConstraint.dim must be >= 0, got {self.dim}.")
        if self.divisor < 1:
            raise ValueError(f"ProcessGridDimDivisibleConstraint.divisor must be >= 1, got {self.divisor}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        return layout.decomposition.grid[self.dim] % self.divisor == 0


@dataclass(frozen=True)
class ProcessGridAspectRatioConstraint(LocalConstraint):
    """``max(grid) / min(grid)`` must not exceed ``max_ratio``.

    This enforces near-square decompositions and helps balance communication costs.

    Parameters
    ----------
    max_ratio : float
        Maximum allowed ratio between the largest and smallest grid dimension.
        Must be >= 1.0.
    """

    max_ratio: float

    def __post_init__(self) -> None:
        if self.max_ratio < 1.0:
            raise ValueError(f"ProcessGridAspectRatioConstraint.max_ratio must be >= 1.0, got {self.max_ratio}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        g = layout.decomposition.grid
        return max(g) / min(g) <= self.max_ratio


# ---------------------------------------------------------------------------
# Category 2 — Distribution constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxRankFractionConstraint(LocalConstraint):
    """This component must not use more than ``max_fraction`` of the total MPI ranks.

    Parameters
    ----------
    max_fraction : float
        Maximum fraction of ``total_ranks`` allowed.  Must be in ``(0.0, 1.0]``.
    """

    max_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 < self.max_fraction <= 1.0):
            raise ValueError(f"MaxRankFractionConstraint.max_fraction must be in (0, 1], got {self.max_fraction}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return layout.n_ranks <= self.max_fraction * total_ranks


@dataclass(frozen=True)
class MaxWastedRankFractionConstraint(LocalConstraint):
    """The fraction of idle (unused) ranks must not exceed ``max_fraction``.

    "Wasted" ranks are those assigned to this component but not distributed to
    any sub-component: ``layout.n_ranks - sum(sub.n_ranks for sub in
    layout.sub_layouts)``.  For leaf components (no sub-components), this
    constraint is always satisfied.

    This constraint is typically placed on a *parent* component to limit how
    many of the available cores are left idle.  It mirrors the
    ``max_wasted_ncores_frac`` post-filter used in the legacy
    ``esm1p6_layout_input.py`` search utilities.

    Parameters
    ----------
    max_fraction : float
        Maximum allowed fraction of idle ranks.  Must be in ``[0.0, 1.0]``.
        Use ``0.0`` to require all ranks to be consumed exactly.
    """

    max_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_fraction <= 1.0):
            raise ValueError(
                f"MaxWastedRankFractionConstraint.max_fraction must be in [0.0, 1.0], got {self.max_fraction}."
            )

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if not layout.sub_layouts:
            return True  # leaf: no sub-components to waste anything
        used = sum(sub.n_ranks for sub in layout.sub_layouts)
        wasted = layout.n_ranks - used
        return wasted <= self.max_fraction * layout.n_ranks


@dataclass(frozen=True)
class RankRatioGroupConstraint(GroupConstraint):
    """``n_ranks(name_a) / n_ranks(name_b)`` must be >= ``min_ratio``.

    Place this on the *parent* component's :attr:`~ParallelComponent.group_constraints`.
    Uses sub-component names rather than positional indices, so the constraint
    remains correct regardless of subcomponent ordering.

    Parameters
    ----------
    name_a : str
        Name of the numerator sub-component.
    name_b : str
        Name of the denominator sub-component.
    min_ratio : float
        Minimum required ratio.  Must be > 0.
    """

    name_a: str
    name_b: str
    min_ratio: float

    def __post_init__(self) -> None:
        if self.min_ratio <= 0.0:
            raise ValueError(f"RankRatioGroupConstraint.min_ratio must be > 0, got {self.min_ratio}.")

    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_ranks: int) -> bool:
        a = next((lay for lay in sub_layouts if lay.name == self.name_a), None)
        b = next((lay for lay in sub_layouts if lay.name == self.name_b), None)
        if a is None or b is None:
            raise ValueError(
                f"RankRatioGroupConstraint: sub-component names {self.name_a!r} and/or "
                f"{self.name_b!r} not found in sub_layouts "
                f"(available: {[lay.name for lay in sub_layouts]})."
            )
        return a.n_ranks >= self.min_ratio * b.n_ranks


# ---------------------------------------------------------------------------
# Category 3 — Domain layout constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformSubdomainConstraint(LocalConstraint):
    """Every local sub-domain must have exactly the same size (exact integer division).

    Equivalent to requiring ``domain.shape[i] % grid[i] == 0`` for every dimension.
    """

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        d = layout.decomposition
        return all(dim % g == 0 for dim, g in zip(d.domain.shape, d.grid, strict=True))


@dataclass(frozen=True)
class SubdomainSizeToleranceConstraint(LocalConstraint):
    """The ratio of the largest to smallest local sub-domain must not exceed ``tolerance``.

    For each dimension ``i``, ``ceil(shape[i]/grid[i]) / floor(shape[i]/grid[i])``
    must be <= ``tolerance``.  This models load-imbalance tolerance.

    Parameters
    ----------
    tolerance : float
        Maximum allowed size ratio.  Must be >= 1.0.
    """

    tolerance: float

    def __post_init__(self) -> None:
        if self.tolerance < 1.0:
            raise ValueError(f"SubdomainSizeToleranceConstraint.tolerance must be >= 1.0, got {self.tolerance}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        d = layout.decomposition
        for dim_size, g in zip(d.domain.shape, d.grid, strict=True):
            lo = dim_size // g
            if lo == 0:
                return False
            hi = (dim_size + g - 1) // g
            if hi / lo > self.tolerance:
                return False
        return True


@dataclass(frozen=True)
class MinSubdomainSizeConstraint(LocalConstraint):
    """Each local sub-domain must have at least ``min_size`` grid points per dimension.

    Checks ``floor(shape[i] / grid[i]) >= min_size`` for every dimension.

    Parameters
    ----------
    min_size : int
        Minimum number of grid points per sub-domain dimension.  Must be >= 1.
    """

    min_size: int

    def __post_init__(self) -> None:
        if self.min_size < 1:
            raise ValueError(f"MinSubdomainSizeConstraint.min_size must be >= 1, got {self.min_size}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        d = layout.decomposition
        return all(dim // g >= self.min_size for dim, g in zip(d.domain.shape, d.grid, strict=True))


@dataclass(frozen=True)
class SubdomainAspectRatioConstraint(LocalConstraint):
    """``max(local_shape) / min(local_shape)`` must not exceed ``max_ratio``.

    Constrains the aspect ratio of the local subdomain in *physical* space
    (``domain.shape[i] / grid[i]`` per dimension), not the MPI process grid.
    Use this to ensure that the grid-point block held by each rank is near-square,
    which typically improves halo-exchange balance.

    For leaf components with no domain this constraint is always satisfied.

    Parameters
    ----------
    max_ratio : float
        Maximum allowed ratio between the longest and shortest local subdomain
        dimension.  Must be >= 1.0.
    """

    max_ratio: float

    def __post_init__(self) -> None:
        if self.max_ratio < 1.0:
            raise ValueError(f"SubdomainAspectRatioConstraint.max_ratio must be >= 1.0, got {self.max_ratio}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        ls = layout.decomposition.local_shape
        return max(ls) / min(ls) <= self.max_ratio


# ---------------------------------------------------------------------------
# Category 4 — Thread constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxThreadsPerRankConstraint(LocalConstraint):
    """``threads_per_rank`` must not exceed ``max_threads``.

    Parameters
    ----------
    max_threads : int
        Maximum allowed threads per rank.  Must be >= 1.
    """

    max_threads: int

    def __post_init__(self) -> None:
        if self.max_threads < 1:
            raise ValueError(f"MaxThreadsPerRankConstraint.max_threads must be >= 1, got {self.max_threads}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return layout.threads_per_rank <= self.max_threads


@dataclass(frozen=True)
class FixedThreadsPerRankConstraint(LocalConstraint):
    """``threads_per_rank`` must equal exactly ``n_threads``.

    Parameters
    ----------
    n_threads : int
        Required thread count per rank.  Must be >= 1.
    """

    n_threads: int

    def __post_init__(self) -> None:
        if self.n_threads < 1:
            raise ValueError(f"FixedThreadsPerRankConstraint.n_threads must be >= 1, got {self.n_threads}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return layout.threads_per_rank == self.n_threads


@dataclass(frozen=True)
class ThreadsDivisorConstraint(LocalConstraint):
    """``divisor`` must be divisible by ``threads_per_rank``.

    Useful when a component's work tiles evenly only for specific thread counts
    (e.g., ``threads_per_rank`` must divide 8).

    Parameters
    ----------
    divisor : int
        Value that must be divisible by ``threads_per_rank``.  Must be >= 1.
    """

    divisor: int

    def __post_init__(self) -> None:
        if self.divisor < 1:
            raise ValueError(f"ThreadsDivisorConstraint.divisor must be >= 1, got {self.divisor}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return self.divisor % layout.threads_per_rank == 0
