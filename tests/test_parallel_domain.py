# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_domain."""

import dataclasses

import pytest

from access.config.parallel_domain import (
    Domain,
    DomainCartesianDecomposition,
    _iter_divisors,
    _iter_grids,
    iter_cartesian_decompositions,
)


@pytest.fixture(scope="module")
def domain_1d() -> Domain:
    return Domain(shape=(100,))


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(360, 300))


@pytest.fixture(scope="module")
def domain_3d() -> Domain:
    return Domain(shape=(192, 144, 85))


class TestDomain:
    def test_basic_1d(self, domain_1d: Domain) -> None:
        assert domain_1d.ndim == 1
        assert domain_1d.size == 100

    def test_basic_2d(self, domain_2d: Domain) -> None:
        assert domain_2d.ndim == 2
        assert domain_2d.size == 360 * 300

    def test_basic_3d(self, domain_3d: Domain) -> None:
        assert domain_3d.ndim == 3
        assert domain_3d.size == 192 * 144 * 85

    def test_empty_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            Domain(shape=())

    def test_zero_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            Domain(shape=(0, 10))

    def test_negative_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            Domain(shape=(10, -5))

    def test_frozen(self, domain_2d: Domain) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            domain_2d.shape = (1, 2)  # type: ignore[misc]


class TestDomainCartesianDecomposition:
    def test_basic(self, domain_2d: Domain) -> None:
        decomp = DomainCartesianDecomposition(domain_2d, grid=(6, 5))
        assert decomp.n_ranks == 30
        assert decomp.local_shape == (360 / 6, 300 / 5)

    def test_n_ranks_1d(self, domain_1d: Domain) -> None:
        decomp = DomainCartesianDecomposition(domain_1d, grid=(7,))
        assert decomp.n_ranks == 7

    def test_local_shape_non_integer(self, domain_2d: Domain) -> None:
        decomp = DomainCartesianDecomposition(domain_2d, grid=(7, 1))
        assert abs(decomp.local_shape[0] - 360 / 7) < 1e-12

    def test_wrong_ndim_raises(self, domain_2d: Domain) -> None:
        with pytest.raises(ValueError, match="dimension"):
            DomainCartesianDecomposition(domain_2d, grid=(6,))

    def test_zero_grid_entry_raises(self, domain_2d: Domain) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            DomainCartesianDecomposition(domain_2d, grid=(0, 5))

    def test_frozen(self, domain_2d: Domain) -> None:
        decomp = DomainCartesianDecomposition(domain_2d, grid=(2, 2))
        with pytest.raises(dataclasses.FrozenInstanceError):
            decomp.grid = (1, 4)  # type: ignore[misc]


class TestIterCartesianDecompositions:
    def test_invalid_n_ranks_raises(self, domain_1d: Domain) -> None:
        with pytest.raises(ValueError, match="n_ranks"):
            list(iter_cartesian_decompositions(domain_1d, n_ranks=0))

    def test_1d_single(self, domain_1d: Domain) -> None:
        decomps = list(iter_cartesian_decompositions(domain_1d, n_ranks=1))
        assert len(decomps) == 1
        assert decomps[0].grid == (1,)

    def test_1d_multiple(self, domain_1d: Domain) -> None:
        decomps = list(iter_cartesian_decompositions(domain_1d, n_ranks=4))
        grids = [d.grid for d in decomps]
        assert (4,) in grids
        assert all(d.n_ranks == 4 for d in decomps)

    def test_2d_rank_1(self, domain_2d: Domain) -> None:
        decomps = list(iter_cartesian_decompositions(domain_2d, n_ranks=1))
        assert len(decomps) == 1
        assert decomps[0].grid == (1, 1)

    def test_2d_rank_4(self, domain_2d: Domain) -> None:
        decomps = list(iter_cartesian_decompositions(domain_2d, n_ranks=4))
        grids = [d.grid for d in decomps]
        assert (1, 4) in grids
        assert (2, 2) in grids
        assert (4, 1) in grids
        assert all(d.n_ranks == 4 for d in decomps)

    def test_2d_prime_rank(self, domain_2d: Domain) -> None:
        decomps = list(iter_cartesian_decompositions(domain_2d, n_ranks=7))
        grids = [d.grid for d in decomps]
        assert (1, 7) in grids
        assert (7, 1) in grids
        assert len(grids) == 2

    def test_3d(self) -> None:
        domain = Domain(shape=(4, 4, 4))
        decomps = list(iter_cartesian_decompositions(domain, n_ranks=8))
        grids = [d.grid for d in decomps]
        assert all(d.n_ranks == 8 for d in decomps)
        assert (2, 2, 2) in grids

    def test_all_products_correct(self, domain_2d: Domain) -> None:
        for n in [1, 2, 3, 6, 12]:
            for d in iter_cartesian_decompositions(domain_2d, n_ranks=n):
                assert d.n_ranks == n

    def test_domain_attached(self, domain_2d: Domain) -> None:
        decomps = list(iter_cartesian_decompositions(domain_2d, n_ranks=4))
        assert all(d.domain is domain_2d for d in decomps)


class TestIterDivisors:
    def test_prime(self) -> None:
        assert list(_iter_divisors(7)) == [1, 7]

    def test_composite_sorted(self) -> None:
        assert list(_iter_divisors(12)) == [1, 2, 3, 4, 6, 12]

    def test_perfect_square_no_duplicate_root(self) -> None:
        assert list(_iter_divisors(16)) == [1, 2, 4, 8, 16]


class TestIterGrids:
    def test_ndim_1(self) -> None:
        assert list(_iter_grids(1, 4)) == [(4,)]

    def test_ndim_2(self) -> None:
        assert list(_iter_grids(2, 4)) == [(1, 4), (2, 2), (4, 1)]

    def test_ndim_3_products(self) -> None:
        for grid in _iter_grids(3, 8):
            assert len(grid) == 3
            assert grid[0] * grid[1] * grid[2] == 8
