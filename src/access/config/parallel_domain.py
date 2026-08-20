# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Parallel domain models and domain-decomposition specifications.

This module defines two frozen value objects:
- ``Domain``: an N-dimensional rectangular grid.
- ``DomainDecompositionSpec``: *how* a domain is to be split across an MPI cartesian
  process grid.

Note that these are types only: nothing here performs a decomposition or represents an
actual partition. A specification records the domain and the process grid it is to be split
over and nothing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from access.config import parallel_mpi_grid


@dataclass(frozen=True)
class Domain:
    """An N-dimensional rectangular grid, representing some work to be parallelised.

    Usually this is a physical domain, such as a 2-D or 3-D spatial grid, but it can be any
    N-dimensional quantity representing some work to be parallelised.

    The domain is a frozen value object, so it is immutable and hashable. This means two
    domains with the same shape compare equal and can be used in sets and as dict keys.

    Args:
        shape (tuple[int, ...]): Size of the domain along each dimension. Must be a
            non-empty tuple and every entry must be an ``int`` >= 1.

    Examples:
        >>> Domain(shape=(360, 300)).ndim         # 2-D domain
        2
        >>> Domain(shape=(192, 144, 85)).size     # 3-D domain
        2350080
    """

    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple:
            raise TypeError(f"Domain shape must be a tuple of ints, got {type(self.shape).__name__}.")
        if not self.shape:
            raise ValueError("Domain shape must have at least one dimension.")
        if any(type(d) is not int for d in self.shape):
            raise TypeError(f"All dimension sizes must be ints, got shape={self.shape}.")
        if any(d < 1 for d in self.shape):
            raise ValueError(f"All dimension sizes must be >= 1, got shape={self.shape}.")

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements in the domain (product of all dimension sizes)."""
        return math.prod(self.shape)


@dataclass(frozen=True)
class DomainDecompositionSpec:
    """Specification of how a domain is to be split across an MPI cartesian process grid.

    This class associates a domain with a process grid, describing how the domain is to be
    split over the ranks of the grid. It does not represent an actual partition: it does not
    record the extents or offsets of any particular rank, nor does it record which ranks
    absorb the remainder when a dimension does not divide evenly.

    The class is frozen (``frozen=True``), so it is immutable and hashable, as are the
    ``Domain`` and ``MPICartesianGrid`` it holds. This makes sharing specifications between
    layouts safe.

    Args:
        domain (Domain): The domain to be split.
        grid (MPICartesianGrid): The MPI cartesian process grid to split the domain over,
            with one sub-domain per rank. Must have the same number of dimensions as
            ``domain``.

    Examples:
        >>> from access.config.parallel_mpi_grid import MPICartesianGrid
        >>> spec = DomainDecompositionSpec(Domain((360, 300)), MPICartesianGrid((6, 5)))
        >>> spec.n_ranks, spec.mean_local_shape
        (30, (60.0, 60.0))
    """

    domain: Domain
    grid: parallel_mpi_grid.MPICartesianGrid

    def __post_init__(self) -> None:
        if self.grid.ndim != self.domain.ndim:
            raise ValueError(f"grid has {self.grid.ndim} dimension(s), but domain has {self.domain.ndim}.")

    @property
    def n_ranks(self) -> int:
        """Total MPI ranks the domain is split over (product of the grid entries)."""
        return self.grid.n_ranks

    @property
    def mean_local_shape(self) -> tuple[float, ...]:
        """Mean sub-domain extent along each dimension (``domain.shape[i] / grid[i]``).

        These are averages, not the extents of any particular rank: the values are
        floating-point, and when a dimension does not divide evenly individual ranks hold
        more or fewer grid points than stated here.
        """
        return tuple(d / g for d, g in zip(self.domain.shape, self.grid, strict=True))
