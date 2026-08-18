# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Classes and utilities for working with MPI cartesian process grids.

The Cartesian grid describes how MPI ranks are arranged in a multi-dimensional grid and it
is often used for domain decomposition in parallel applications. This module provides the
``MPICartesianGrid`` class to represent such grids and includes methods for enumerating all
possible grid shapes given a number of ranks and dimensions.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class MPICartesianGrid:
    """An N-dimensional MPI cartesian process grid.

    Instances represent a grid shape as a value object: the class is frozen
    (``frozen=True``), so two grids compare equal when their shapes match, and
    they can be used in hash-based collections such as sets and dict keys. The
    shape must be non-empty and every entry must be >= 1.

    Note: the grid is a pure MPI concept and knows nothing about the domain it may
    be used to decompose.

    Args:
        shape (tuple[int, ...]): Number of MPI ranks along each dimension.

    Examples:
        >>> MPICartesianGrid(shape=(6, 5)).n_ranks    # 2-D grid of 30 ranks
        30
        >>> MPICartesianGrid(shape=(2, 2, 2)).n_ranks # 3-D grid of 8 ranks
        8
    """

    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple:
            raise TypeError(f"MPICartesianGrid shape must be a tuple of ints, got {type(self.shape).__name__}.")
        if not self.shape:
            raise ValueError("MPICartesianGrid shape must have at least one dimension.")
        if any(type(g) is not int for g in self.shape):
            raise TypeError(f"All grid entries must be ints, got shape={self.shape}.")
        if any(g < 1 for g in self.shape):
            raise ValueError(f"All grid entries must be >= 1, got shape={self.shape}.")

    @property
    def ndim(self) -> int:
        """Number of grid dimensions."""
        return len(self.shape)

    @property
    def n_ranks(self) -> int:
        """Total number of MPI ranks (product of all grid dimensions)."""
        return math.prod(self.shape)

    def __len__(self) -> int:
        return len(self.shape)

    def __getitem__(self, index: int) -> int:
        return self.shape[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self.shape)

    @classmethod
    def iter_grids(cls, ndim: int, n_ranks: int) -> Iterator[MPICartesianGrid]:
        """Yield every ``ndim``-dimensional grid that uses exactly *n_ranks* ranks.

        In a more mathematical way, it enumerates every N-tuple ``(g0, g1, ..., g_{N-1})``
        such that ``g0 * g1 * ... * g_{N-1} == n_ranks`` and ``gi >= 1`` for all ``i``.

        Args:
            ndim (int): Number of grid dimensions. Must be >= 1.
            n_ranks (int): Total number of MPI ranks. Must be >= 1.

        Yields:
            MPICartesianGrid: Each valid grid, in lexicographic order of its shape.

        Examples:
            >>> [g.shape for g in MPICartesianGrid.iter_grids(2, 4)]
            [(1, 4), (2, 2), (4, 1)]
        """
        if type(ndim) is not int:
            raise TypeError(f"MPICartesianGrid.iter_grids: ndim must be an int, got {type(ndim).__name__}.")
        if type(n_ranks) is not int:
            raise TypeError(f"MPICartesianGrid.iter_grids: n_ranks must be an int, got {type(n_ranks).__name__}.")
        if ndim < 1:
            raise ValueError(f"MPICartesianGrid.iter_grids: ndim must be >= 1, got {ndim}.")
        if n_ranks < 1:
            raise ValueError(f"MPICartesianGrid.iter_grids: n_ranks must be >= 1, got {n_ranks}.")

        for shape in _iter_grid_shapes(ndim, n_ranks):
            yield cls(shape)


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


def _iter_grid_shapes(ndim: int, n_ranks: int) -> Iterator[tuple[int, ...]]:
    """Yield all ndim-tuples whose product equals n_ranks, in lexicographic order."""
    if ndim == 1:
        yield (n_ranks,)
        return
    for d0 in _iter_divisors(n_ranks):
        for rest in _iter_grid_shapes(ndim - 1, n_ranks // d0):
            yield (d0, *rest)
