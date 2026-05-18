# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Domain models and cartesian decomposition helpers for parallel layouts.

This module owns:

- :class:`Domain`: an N-dimensional rectangular grid.
- :class:`DomainCartesianDecomposition`: tiling of a domain across an MPI cartesian process grid.
- :func:`iter_cartesian_decompositions`: enumeration of valid process grids for a domain and rank count.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    """An N-dimensional rectangular grid.

    Parameters
    ----------
    shape : tuple[int, ...]
        Size of the grid along each spatial dimension. Must be non-empty and
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


@dataclass(frozen=True)
class DomainCartesianDecomposition:
    """Decomposition of a domain across an MPI cartesian process grid.

    Parameters
    ----------
    domain : Domain
        The domain being decomposed.
    grid : tuple[int, ...]
        Number of MPI sub-domains along each dimension. Must have the same
        length as ``domain.shape`` and all entries must be >= 1.

    Examples
    --------
    >>> DomainCartesianDecomposition(Domain((360, 300)), grid=(6, 5))
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


def _iter_divisors(n: int) -> Iterator[int]:
    """Yield the positive divisors of ``n`` in ascending order."""
    small_divisors: list[int] = []
    large_divisors: list[int] = []
    limit = math.isqrt(n)
    for divisor in range(1, limit + 1):
        if n % divisor != 0:
            continue
        small_divisors.append(divisor)
        complement = n // divisor
        if complement != divisor:
            large_divisors.append(complement)
    yield from small_divisors
    yield from reversed(large_divisors)


def _iter_grids(ndim: int, n_ranks: int) -> Iterator[tuple[int, ...]]:
    """Yield all ndim-tuples whose product equals n_ranks."""
    if ndim == 1:
        yield (n_ranks,)
        return
    for d0 in _iter_divisors(n_ranks):
        for rest in _iter_grids(ndim - 1, n_ranks // d0):
            yield (d0, *rest)


def iter_cartesian_decompositions(domain: Domain, n_ranks: int) -> Iterator[DomainCartesianDecomposition]:
    """Yield all :class:`CartesianDecomposition` objects for *domain* using *n_ranks* ranks.

    Enumerates every N-tuple ``(g0, g1, ..., g_{N-1})`` such that
    ``g0 * g1 * ... * g_{N-1} == n_ranks`` and ``gi >= 1`` for all ``i``.

    Parameters
    ----------
    domain : Domain
        The domain to be decomposed.
    n_ranks : int
        Total number of MPI ranks. Must be >= 1.

    Yields
    ------
    CartesianDecomposition
        Each valid decomposition in lexicographic grid order.

    Examples
    --------
    >>> list(iter_cartesian_decompositions(Domain((10, 10)), 4))
    [CartesianDecomposition(domain=..., grid=(1, 4)),
     CartesianDecomposition(domain=..., grid=(2, 2)),
     CartesianDecomposition(domain=..., grid=(4, 1))]
    """
    if n_ranks < 1:
        raise ValueError(f"iter_cartesian_decompositions: n_ranks must be >= 1, got {n_ranks}.")

    for grid in _iter_grids(domain.ndim, n_ranks):
        yield DomainCartesianDecomposition(domain, grid)
