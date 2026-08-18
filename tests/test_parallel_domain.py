# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_domain."""

import dataclasses

import pytest

from access.config.parallel_domain import (
    Domain,
    DomainDecompositionSpec,
)
from access.config.parallel_mpi_grid import MPICartesianGrid


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

    def test_non_tuple_shape_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a tuple of ints"):
            Domain(shape=[10, 10])  # type: ignore[arg-type]

    def test_non_int_dimension_raises(self) -> None:
        with pytest.raises(TypeError, match="must be ints"):
            Domain(shape=(10.5, 4))  # type: ignore[arg-type]

    def test_bool_dimension_raises(self) -> None:
        with pytest.raises(TypeError, match="must be ints"):
            Domain(shape=(True, 4))  # type: ignore[arg-type]

    def test_frozen(self, domain_2d: Domain) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            domain_2d.shape = (1, 2)  # type: ignore[misc]


class TestDomainDecompositionSpec:
    def test_basic(self, domain_2d: Domain) -> None:
        decomp = DomainDecompositionSpec(domain_2d, grid=MPICartesianGrid((6, 5)))
        assert decomp.n_ranks == 30
        assert decomp.mean_local_shape == (360 / 6, 300 / 5)

    def test_n_ranks_1d(self, domain_1d: Domain) -> None:
        decomp = DomainDecompositionSpec(domain_1d, grid=MPICartesianGrid((7,)))
        assert decomp.n_ranks == 7

    def test_mean_local_shape_non_integer(self, domain_2d: Domain) -> None:
        decomp = DomainDecompositionSpec(domain_2d, grid=MPICartesianGrid((7, 1)))
        assert abs(decomp.mean_local_shape[0] - 360 / 7) < 1e-12

    def test_wrong_ndim_raises(self, domain_2d: Domain) -> None:
        with pytest.raises(ValueError, match="dimension"):
            DomainDecompositionSpec(domain_2d, grid=MPICartesianGrid((6,)))

    def test_frozen(self, domain_2d: Domain) -> None:
        decomp = DomainDecompositionSpec(domain_2d, grid=MPICartesianGrid((2, 2)))
        with pytest.raises(dataclasses.FrozenInstanceError):
            decomp.grid = MPICartesianGrid((1, 4))  # type: ignore[misc]
