# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for concrete constraints in access.config.parallel_constraints."""

from unittest.mock import MagicMock

import pytest

from access.config.parallel_constraints import (
    DomainDivisibleByRanksConstraint,
    FixedThreadsPerRankConstraint,
    MaxThreadsPerRankConstraint,
    MaxWastedCoreFractionConstraint,
    MinSubdomainSizeConstraint,
    ProcessGridDimDivisibleConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
    SubdomainAspectRatioConstraint,
)

# ---------------------------------------------------------------------------
# Mock layouts
# ---------------------------------------------------------------------------
#
# A constraint reads a handful of attributes off the layout it judges and computes nothing
# else, so the layouts here are mocks carrying exactly those attributes. Stating them
# directly keeps these tests about the rules: how a real ComponentLayout derives
# ``idle_cores``, or a real decomposition its ``mean_local_shape``, belongs to the modules
# that own them, and is asserted in their own test files.


def grid(*shape: int) -> MagicMock:
    """A mock process grid: constraints read its ndim, index it and iterate it."""
    g = MagicMock()
    g.ndim = len(shape)
    g.__getitem__.side_effect = shape.__getitem__
    g.__iter__.side_effect = lambda: iter(shape)
    return g


def decomp(
    *,
    grid_shape: tuple[int, ...],
    domain_shape: tuple[int, ...] = (12, 8),
    mean_local_shape: tuple[int, ...] | None = None,
) -> MagicMock:
    """A mock decomposition, carrying only what the domain-layout constraints read."""
    d = MagicMock()
    d.grid = grid(*grid_shape)
    d.domain.shape = domain_shape
    d.mean_local_shape = mean_local_shape
    return d


def leaf(
    name: str = "x", n_ranks: int = 1, threads_per_rank: int = 1, decomposition: MagicMock | None = None
) -> MagicMock:
    """A mock leaf layout, spending exactly ``n_ranks * threads_per_rank`` cores."""
    layout = MagicMock()
    layout.name = name  # assigned, never passed: ``name=`` is reserved by Mock itself
    layout.n_ranks = n_ranks
    layout.threads_per_rank = threads_per_rank
    layout.n_cores = n_ranks * threads_per_rank
    layout.decomposition = decomposition
    layout.is_leaf = True
    return layout


def parent(name: str, n_cores: int, *, idle_cores: int = 0) -> MagicMock:
    """A mock parent layout: it has sub-layouts, so no thread count of its own."""
    layout = MagicMock()
    layout.name = name
    layout.n_cores = n_cores
    layout.threads_per_rank = None
    layout.decomposition = None
    layout.is_leaf = False
    layout.idle_cores = idle_cores
    return layout


# ---------------------------------------------------------------------------
# Category 1 — Cartesian grid constraints
# ---------------------------------------------------------------------------


class TestProcessGridDimEvenConstraint:
    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            ProcessGridDimEvenConstraint(dim=-1)

    def test_even_grid(self) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = leaf("x", 4, 1, decomp(grid_shape=(2, 2)))
        assert c.is_satisfied(layout)

    def test_odd_grid_fails(self) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = leaf("x", 3, 1, decomp(grid_shape=(3, 1)))
        assert not c.is_satisfied(layout)

    def test_no_decomposition_raises(self) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout)

    def test_dim_out_of_range_raises(self) -> None:
        c = ProcessGridDimEvenConstraint(dim=2)
        layout = leaf("x", 4, 1, decomp(grid_shape=(2, 2)))
        with pytest.raises(ValueError, match="out of range"):
            c.is_satisfied(layout)


class TestProcessGridDimDivisibleConstraint:
    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            ProcessGridDimDivisibleConstraint(dim=-1, divisor=2)

    def test_divisible(self) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = leaf("x", 4, 1, decomp(grid_shape=(1, 4)))
        assert c.is_satisfied(layout)

    def test_not_divisible(self) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = leaf("x", 6, 1, decomp(grid_shape=(2, 3)))
        assert not c.is_satisfied(layout)

    def test_invalid_divisor_raises(self) -> None:
        with pytest.raises(ValueError, match="divisor"):
            ProcessGridDimDivisibleConstraint(dim=0, divisor=0)

    def test_no_decomposition_raises(self) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=0, divisor=4)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout)

    def test_dim_out_of_range_raises(self) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=2, divisor=2)
        layout = leaf("x", 4, 1, decomp(grid_shape=(2, 2)))
        with pytest.raises(ValueError, match="out of range"):
            c.is_satisfied(layout)


# ---------------------------------------------------------------------------
# Category 2 — Core distribution constraints
# ---------------------------------------------------------------------------


class TestMaxWastedCoreFractionConstraint:
    def test_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="max_fraction"):
            MaxWastedCoreFractionConstraint(max_fraction=-0.1)

    def test_no_sublayouts_raises(self) -> None:
        c = MaxWastedCoreFractionConstraint(max_fraction=0.0)
        layout = leaf("leaf", 8)
        with pytest.raises(ValueError, match="requires layout to have sub-layouts"):
            c.is_satisfied(layout)

    def test_no_waste_satisfies_zero_fraction(self) -> None:
        c = MaxWastedCoreFractionConstraint(max_fraction=0.0)
        assert c.is_satisfied(parent("p", 8, idle_cores=0))

    def test_waste_exceeding_fraction_fails(self) -> None:
        c = MaxWastedCoreFractionConstraint(max_fraction=0.1)
        # 2 of 8 cores idle = 0.25 > 0.1
        assert not c.is_satisfied(parent("p", 8, idle_cores=2))

    def test_exact_boundary_is_satisfied(self) -> None:
        # 7 of 70 cores idle is exactly 0.1; comparing against 0.1 * 70 rejects it.
        c = MaxWastedCoreFractionConstraint(max_fraction=0.1)
        assert c.is_satisfied(parent("p", 70, idle_cores=7))


class TestRankRatioGroupConstraint:
    def test_satisfied(self) -> None:
        c = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=1.5)
        a = leaf("a", 6, 1)
        b = leaf("b", 4, 1)
        assert c.is_satisfied((a, b))

    def test_not_satisfied(self) -> None:
        c = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0)
        a = leaf("a", 6, 1)
        b = leaf("b", 4, 1)
        # 6 / 4 = 1.5 < 2.0
        assert not c.is_satisfied((a, b))

    def test_unknown_name_raises(self) -> None:
        c = RankRatioGroupConstraint(name_a="TYPO", name_b="b", min_ratio=1.0)
        a = leaf("a", 6, 1)
        b = leaf("b", 4, 1)
        with pytest.raises(ValueError, match="not found"):
            c.is_satisfied((a, b))

    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="min_ratio"):
            RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=0.0)


# ---------------------------------------------------------------------------
# Category 3 — Domain layout constraints
# ---------------------------------------------------------------------------


class TestDomainDivisibleByRanksConstraint:
    def test_uniform(self) -> None:
        # 12 / 3 == 4, 8 / 2 == 4 — both exact
        c = DomainDivisibleByRanksConstraint()
        layout = leaf("x", 6, 1, decomp(grid_shape=(3, 2)))
        assert c.is_satisfied(layout)

    def test_non_uniform(self) -> None:
        # 12 / 5 is not integer
        c = DomainDivisibleByRanksConstraint()
        layout = leaf("x", 5, 1, decomp(grid_shape=(5, 1)))
        assert not c.is_satisfied(layout)

    @pytest.mark.parametrize(("n_ranks", "satisfied"), [(12, True), (7, False)])
    def test_extent_is_the_dividend(self, n_ranks: int, satisfied: bool) -> None:
        # The rank count must divide the extent, not the other way round: on a 360-point
        # dimension 12 ranks tile it exactly and 7 do not. ProcessGridDimDivisibleConstraint
        # states the reverse relation and cannot express this for any divisor it is given.
        c = DomainDivisibleByRanksConstraint()
        layout = leaf("x", n_ranks, 1, decomp(grid_shape=(n_ranks,), domain_shape=(360,)))
        assert c.is_satisfied(layout) is satisfied

    def test_no_decomposition_raises(self) -> None:
        c = DomainDivisibleByRanksConstraint()
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout)


class TestSubdomainAspectRatioConstraint:
    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ratio"):
            SubdomainAspectRatioConstraint(max_ratio=0.9)

    def test_no_decomposition_raises(self) -> None:
        c = SubdomainAspectRatioConstraint(max_ratio=2.0)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout)

    def test_aspect_ratio_check(self) -> None:
        c = SubdomainAspectRatioConstraint(max_ratio=1.2)
        # 6 / 4 is a ratio of 1.5, which exceeds 1.2
        layout = leaf("x", 4, 1, decomp(grid_shape=(2, 2), mean_local_shape=(6, 4)))
        assert not c.is_satisfied(layout)


class TestMinSubdomainSizeConstraint:
    def test_large_enough(self) -> None:
        # 12/3=4, 8/2=4 — both >= 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = leaf("x", 6, 1, decomp(grid_shape=(3, 2)))
        assert c.is_satisfied(layout)

    def test_too_small(self) -> None:
        # 8/8=1 < 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = leaf("x", 8, 1, decomp(grid_shape=(1, 8)))
        assert not c.is_satisfied(layout)

    def test_invalid_min_size_raises(self) -> None:
        with pytest.raises(ValueError, match="min_size"):
            MinSubdomainSizeConstraint(min_size=0)

    def test_no_decomposition_raises(self) -> None:
        c = MinSubdomainSizeConstraint(min_size=4)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout)


# ---------------------------------------------------------------------------
# Category 4 — Thread constraints
# ---------------------------------------------------------------------------


class TestMaxThreadsPerRankConstraint:
    def test_within_limit(self) -> None:
        c = MaxThreadsPerRankConstraint(max_threads=4)
        layout = leaf("x", 8, 4)
        assert c.is_satisfied(layout)

    def test_exceeds_limit(self) -> None:
        c = MaxThreadsPerRankConstraint(max_threads=4)
        layout = leaf("x", 4, 8)
        assert not c.is_satisfied(layout)

    def test_invalid_max_raises(self) -> None:
        with pytest.raises(ValueError, match="max_threads"):
            MaxThreadsPerRankConstraint(max_threads=0)

    def test_parent_raises(self) -> None:
        # A parent's sub-components may each use a different thread count, so it has none.
        c = MaxThreadsPerRankConstraint(max_threads=4)
        with pytest.raises(ValueError, match="requires layout.threads_per_rank"):
            c.is_satisfied(parent("p", 8))


class TestFixedThreadsPerRankConstraint:
    def test_matches(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        layout = leaf("x", 8, 4)
        assert c.is_satisfied(layout)

    def test_does_not_match(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        layout = leaf("x", 8, 2)
        assert not c.is_satisfied(layout)

    def test_invalid_n_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="n_threads"):
            FixedThreadsPerRankConstraint(n_threads=0)

    def test_parent_raises(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        with pytest.raises(ValueError, match="requires layout.threads_per_rank"):
            c.is_satisfied(parent("p", 8))
