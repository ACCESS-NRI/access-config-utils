# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_mpi_grid."""

import dataclasses

import pytest

from access.config.parallel_mpi_grid import (
    MPICartesianGrid,
    _iter_divisors,
    _iter_grid_shapes,
)


@pytest.fixture(scope="module")
def grid_1d() -> MPICartesianGrid:
    return MPICartesianGrid(shape=(7,))


@pytest.fixture(scope="module")
def grid_2d() -> MPICartesianGrid:
    return MPICartesianGrid(shape=(6, 5))


@pytest.fixture(scope="module")
def grid_3d() -> MPICartesianGrid:
    return MPICartesianGrid(shape=(2, 3, 4))


class TestMPICartesianGrid:
    def test_basic_1d(self, grid_1d: MPICartesianGrid) -> None:
        assert grid_1d.ndim == 1
        assert grid_1d.n_ranks == 7

    def test_basic_2d(self, grid_2d: MPICartesianGrid) -> None:
        assert grid_2d.ndim == 2
        assert grid_2d.n_ranks == 6 * 5

    def test_basic_3d(self, grid_3d: MPICartesianGrid) -> None:
        assert grid_3d.ndim == 3
        assert grid_3d.n_ranks == 2 * 3 * 4

    def test_empty_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            MPICartesianGrid(shape=())

    @pytest.mark.parametrize("shape", [[6, 5], 30, "65"])
    def test_non_tuple_shape_raises(self, shape: object) -> None:
        # Sequences that are not tuples are rejected, so that the shape stays hashable
        # and immutable.
        with pytest.raises(TypeError, match="must be a tuple of ints"):
            MPICartesianGrid(shape=shape)  # type: ignore[arg-type]

    @pytest.mark.parametrize("shape", [(6.0, 5), (6, "5"), (True, 5)])
    def test_non_int_entry_raises(self, shape: tuple[object, ...]) -> None:
        # bool is a subclass of int, but is still not a valid number of ranks.
        with pytest.raises(TypeError, match="All grid entries must be ints"):
            MPICartesianGrid(shape=shape)  # type: ignore[arg-type]

    def test_zero_entry_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            MPICartesianGrid(shape=(0, 5))

    def test_negative_entry_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            MPICartesianGrid(shape=(4, -2))

    def test_frozen(self, grid_2d: MPICartesianGrid) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            grid_2d.shape = (1, 2)  # type: ignore[misc]

    def test_len(self, grid_3d: MPICartesianGrid) -> None:
        assert len(grid_3d) == 3

    def test_getitem(self, grid_2d: MPICartesianGrid) -> None:
        assert grid_2d[0] == 6
        assert grid_2d[1] == 5
        assert grid_2d[-1] == 5

    def test_iter(self, grid_3d: MPICartesianGrid) -> None:
        assert list(grid_3d) == [2, 3, 4]

    def test_max_min(self, grid_2d: MPICartesianGrid) -> None:
        assert max(grid_2d) == 6
        assert min(grid_2d) == 5

    def test_zip_strict(self, grid_2d: MPICartesianGrid) -> None:
        # strict=True ensures this pairing is length-matched: a 2D grid must be zipped with
        # exactly two values, instead of silently dropping an extra dimension or value.
        assert list(zip((360, 300), grid_2d, strict=True)) == [(360, 6), (300, 5)]

    def test_equality(self) -> None:
        assert MPICartesianGrid((2, 2)) == MPICartesianGrid((2, 2))
        assert MPICartesianGrid((2, 2)) != MPICartesianGrid((1, 4))

    def test_hashable(self) -> None:
        # As a frozen value object, equal grids must hash the same way and deduplicate
        # in sets.
        assert len({MPICartesianGrid((2, 2)), MPICartesianGrid((2, 2)), MPICartesianGrid((4, 1))}) == 2


class TestIterGrids:
    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            list(MPICartesianGrid.iter_grids(0, 4))

    def test_invalid_n_ranks_raises(self) -> None:
        with pytest.raises(ValueError, match="n_ranks"):
            list(MPICartesianGrid.iter_grids(2, 0))

    @pytest.mark.parametrize(
        ("ndim", "n_ranks"),
        [
            (2.0, 4),
            (2, 4.0),
            ("2", 4),
            (2, "4"),
        ],
    )
    def test_invalid_types_raises(self, ndim: object, n_ranks: object) -> None:
        with pytest.raises(TypeError):
            list(MPICartesianGrid.iter_grids(ndim, n_ranks))

    def test_yields_grid_instances(self) -> None:
        assert all(isinstance(g, MPICartesianGrid) for g in MPICartesianGrid.iter_grids(2, 12))

    def test_ndim_1(self) -> None:
        assert [g.shape for g in MPICartesianGrid.iter_grids(1, 4)] == [(4,)]

    def test_ndim_2(self) -> None:
        assert [g.shape for g in MPICartesianGrid.iter_grids(2, 4)] == [(1, 4), (2, 2), (4, 1)]

    def test_ndim_2_prime(self) -> None:
        assert [g.shape for g in MPICartesianGrid.iter_grids(2, 7)] == [(1, 7), (7, 1)]

    def test_single_rank(self) -> None:
        assert [g.shape for g in MPICartesianGrid.iter_grids(3, 1)] == [(1, 1, 1)]

    def test_ndim_3_products(self) -> None:
        grids = list(MPICartesianGrid.iter_grids(3, 8))
        assert all(g.ndim == 3 for g in grids)
        assert all(g.n_ranks == 8 for g in grids)
        assert MPICartesianGrid((2, 2, 2)) in grids


class TestIterDivisors:
    def test_prime(self) -> None:
        assert list(_iter_divisors(7)) == [1, 7]

    def test_composite_sorted(self) -> None:
        assert list(_iter_divisors(12)) == [1, 2, 3, 4, 6, 12]

    def test_perfect_square_no_duplicate_root(self) -> None:
        assert list(_iter_divisors(16)) == [1, 2, 4, 8, 16]


class TestIterGridShapes:
    def test_ndim_1(self) -> None:
        assert list(_iter_grid_shapes(1, 4)) == [(4,)]

    def test_ndim_2(self) -> None:
        assert list(_iter_grid_shapes(2, 4)) == [(1, 4), (2, 2), (4, 1)]

    def test_ndim_3_products(self) -> None:
        for shape in _iter_grid_shapes(3, 8):
            assert len(shape) == 3
            assert shape[0] * shape[1] * shape[2] == 8
