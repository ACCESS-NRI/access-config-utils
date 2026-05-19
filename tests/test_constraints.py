# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for concrete constraints in access.config.constraints."""

import pytest

from access.config.constraints import (
    FixedThreadsPerRankConstraint,
    MaxRankFractionConstraint,
    MaxThreadsPerRankConstraint,
    MaxWastedRankFractionConstraint,
    MinSubdomainSizeConstraint,
    ProcessGridAspectRatioConstraint,
    ProcessGridDimDivisibleConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
    SubdomainAspectRatioConstraint,
    SubdomainSizeToleranceConstraint,
    ThreadsDivisorConstraint,
    UniformSubdomainConstraint,
)
from access.config.domain_parallelisation import Domain, DomainCartesianDecomposition
from access.config.parallel_component import ComponentLayout


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(12, 8))


# ---------------------------------------------------------------------------
# Category 1 — Cartesian grid constraints
# ---------------------------------------------------------------------------


class TestProcessGridDimEvenConstraint:
    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            ProcessGridDimEvenConstraint(dim=-1)

    def test_even_grid(self, domain_2d: Domain) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = ComponentLayout("x", 4, 1, DomainCartesianDecomposition(domain_2d, (2, 2)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_odd_grid_fails(self, domain_2d: Domain) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = ComponentLayout("x", 3, 1, DomainCartesianDecomposition(domain_2d, (3, 1)))
        assert not c.is_satisfied(layout, total_ranks=3)

    def test_no_decomposition_passes(self) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)


class TestProcessGridDimDivisibleConstraint:
    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            ProcessGridDimDivisibleConstraint(dim=-1, divisor=2)

    def test_divisible(self, domain_2d: Domain) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = ComponentLayout("x", 4, 1, DomainCartesianDecomposition(domain_2d, (1, 4)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_not_divisible(self, domain_2d: Domain) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = ComponentLayout("x", 6, 1, DomainCartesianDecomposition(domain_2d, (2, 3)))
        assert not c.is_satisfied(layout, total_ranks=6)

    def test_invalid_divisor_raises(self) -> None:
        with pytest.raises(ValueError, match="divisor"):
            ProcessGridDimDivisibleConstraint(dim=0, divisor=0)

    def test_no_decomposition_passes(self) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=0, divisor=4)
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)


class TestProcessGridAspectRatioConstraint:
    def test_square_passes(self, domain_2d: Domain) -> None:
        c = ProcessGridAspectRatioConstraint(max_ratio=2.0)
        layout = ComponentLayout("x", 4, 1, DomainCartesianDecomposition(domain_2d, (2, 2)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_elongated_fails(self, domain_2d: Domain) -> None:
        c = ProcessGridAspectRatioConstraint(max_ratio=2.0)
        # (1, 12): ratio = 12 > 2
        layout = ComponentLayout("x", 12, 1, DomainCartesianDecomposition(domain_2d, (1, 12)))
        assert not c.is_satisfied(layout, total_ranks=12)

    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ratio"):
            ProcessGridAspectRatioConstraint(max_ratio=0.5)

    def test_no_decomposition_passes(self) -> None:
        c = ProcessGridAspectRatioConstraint(max_ratio=1.5)
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)


# ---------------------------------------------------------------------------
# Category 2 — Distribution constraints
# ---------------------------------------------------------------------------


class TestMaxRankFractionConstraint:
    def test_within_limit(self) -> None:
        c = MaxRankFractionConstraint(max_fraction=0.5)
        layout = ComponentLayout("x", n_ranks=50, threads_per_rank=1, decomposition=None)
        assert c.is_satisfied(layout, total_ranks=100)

    def test_at_limit(self) -> None:
        c = MaxRankFractionConstraint(max_fraction=0.5)
        layout = ComponentLayout("x", n_ranks=50, threads_per_rank=1, decomposition=None)
        assert c.is_satisfied(layout, total_ranks=100)

    def test_exceeds_limit(self) -> None:
        c = MaxRankFractionConstraint(max_fraction=0.5)
        layout = ComponentLayout("x", n_ranks=51, threads_per_rank=1, decomposition=None)
        assert not c.is_satisfied(layout, total_ranks=100)

    def test_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="max_fraction"):
            MaxRankFractionConstraint(max_fraction=0.0)

        with pytest.raises(ValueError, match="max_fraction"):
            MaxRankFractionConstraint(max_fraction=1.1)


class TestMaxWastedRankFractionConstraint:
    def test_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="max_fraction"):
            MaxWastedRankFractionConstraint(max_fraction=-0.1)

    def test_leaf_layout_always_satisfied(self) -> None:
        c = MaxWastedRankFractionConstraint(max_fraction=0.0)
        layout = ComponentLayout("leaf", n_ranks=8, threads_per_rank=1, decomposition=None, sub_layouts=())
        assert c.is_satisfied(layout, total_ranks=8)


class TestRankRatioGroupConstraint:
    def test_satisfied(self) -> None:
        c = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=1.5)
        a = ComponentLayout("a", n_ranks=6, threads_per_rank=1, decomposition=None)
        b = ComponentLayout("b", n_ranks=4, threads_per_rank=1, decomposition=None)
        assert c.is_satisfied((a, b), total_ranks=10)

    def test_not_satisfied(self) -> None:
        c = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0)
        a = ComponentLayout("a", n_ranks=6, threads_per_rank=1, decomposition=None)
        b = ComponentLayout("b", n_ranks=4, threads_per_rank=1, decomposition=None)
        # 6 / 4 = 1.5 < 2.0
        assert not c.is_satisfied((a, b), total_ranks=10)

    def test_unknown_name_raises(self) -> None:
        c = RankRatioGroupConstraint(name_a="TYPO", name_b="b", min_ratio=1.0)
        a = ComponentLayout("a", n_ranks=6, threads_per_rank=1, decomposition=None)
        b = ComponentLayout("b", n_ranks=4, threads_per_rank=1, decomposition=None)
        with pytest.raises(ValueError, match="not found"):
            c.is_satisfied((a, b), total_ranks=10)

    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="min_ratio"):
            RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=0.0)


# ---------------------------------------------------------------------------
# Category 3 — Domain layout constraints
# ---------------------------------------------------------------------------


class TestUniformSubdomainConstraint:
    def test_uniform(self, domain_2d: Domain) -> None:
        # 12 / 3 == 4, 8 / 2 == 4 — both exact
        c = UniformSubdomainConstraint()
        layout = ComponentLayout("x", 6, 1, DomainCartesianDecomposition(domain_2d, (3, 2)))
        assert c.is_satisfied(layout, total_ranks=6)

    def test_non_uniform(self, domain_2d: Domain) -> None:
        # 12 / 5 is not integer
        c = UniformSubdomainConstraint()
        layout = ComponentLayout("x", 5, 1, DomainCartesianDecomposition(domain_2d, (5, 1)))
        assert not c.is_satisfied(layout, total_ranks=5)

    def test_no_decomposition_passes(self) -> None:
        c = UniformSubdomainConstraint()
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)


class TestSubdomainSizeToleranceConstraint:
    def test_very_large_dimension_uses_integer_ceil(self) -> None:
        domain = Domain(shape=(10**400,))
        c = SubdomainSizeToleranceConstraint(tolerance=1.0)
        layout = ComponentLayout("x", 2, 1, DomainCartesianDecomposition(domain, (2,)))
        assert c.is_satisfied(layout, total_ranks=2)

    def test_exact_passes(self, domain_2d: Domain) -> None:
        c = SubdomainSizeToleranceConstraint(tolerance=1.0)
        layout = ComponentLayout("x", 6, 1, DomainCartesianDecomposition(domain_2d, (3, 2)))
        assert c.is_satisfied(layout, total_ranks=6)

    def test_within_tolerance(self) -> None:
        # Domain (13,), grid (4,): floor=3, ceil=4, ratio=4/3≈1.33 <= 1.5
        domain = Domain(shape=(13,))
        c = SubdomainSizeToleranceConstraint(tolerance=1.5)
        layout = ComponentLayout("x", 4, 1, DomainCartesianDecomposition(domain, (4,)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_exceeds_tolerance(self) -> None:
        # Domain (13,), grid (4,): ratio=4/3≈1.33 — fails at tolerance=1.2
        domain = Domain(shape=(13,))
        c = SubdomainSizeToleranceConstraint(tolerance=1.2)
        layout = ComponentLayout("x", 4, 1, DomainCartesianDecomposition(domain, (4,)))
        assert not c.is_satisfied(layout, total_ranks=4)

    def test_invalid_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            SubdomainSizeToleranceConstraint(tolerance=0.9)

    def test_no_decomposition_passes(self) -> None:
        c = SubdomainSizeToleranceConstraint(tolerance=1.5)
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)

    def test_zero_sized_local_chunk_fails(self) -> None:
        # Domain (2,), grid (4,) -> floor(2/4) = 0, which must fail.
        domain = Domain(shape=(2,))
        c = SubdomainSizeToleranceConstraint(tolerance=10.0)
        layout = ComponentLayout("x", 4, 1, DomainCartesianDecomposition(domain, (4,)))
        assert not c.is_satisfied(layout, total_ranks=4)


class TestSubdomainAspectRatioConstraint:
    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ratio"):
            SubdomainAspectRatioConstraint(max_ratio=0.9)

    def test_no_decomposition_passes(self) -> None:
        c = SubdomainAspectRatioConstraint(max_ratio=2.0)
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)

    def test_aspect_ratio_check(self) -> None:
        domain = Domain(shape=(12, 8))
        c = SubdomainAspectRatioConstraint(max_ratio=1.2)
        layout = ComponentLayout("x", 4, 1, DomainCartesianDecomposition(domain, (2, 2)))
        # local_shape = (6, 4) -> ratio 1.5, which exceeds 1.2
        assert not c.is_satisfied(layout, total_ranks=4)


class TestMinSubdomainSizeConstraint:
    def test_large_enough(self, domain_2d: Domain) -> None:
        # 12/3=4, 8/2=4 — both >= 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = ComponentLayout("x", 6, 1, DomainCartesianDecomposition(domain_2d, (3, 2)))
        assert c.is_satisfied(layout, total_ranks=6)

    def test_too_small(self, domain_2d: Domain) -> None:
        # 8/8=1 < 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = ComponentLayout("x", 8, 1, DomainCartesianDecomposition(domain_2d, (1, 8)))
        assert not c.is_satisfied(layout, total_ranks=8)

    def test_invalid_min_size_raises(self) -> None:
        with pytest.raises(ValueError, match="min_size"):
            MinSubdomainSizeConstraint(min_size=0)

    def test_no_decomposition_passes(self) -> None:
        c = MinSubdomainSizeConstraint(min_size=4)
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)


# ---------------------------------------------------------------------------
# Category 4 — Thread constraints
# ---------------------------------------------------------------------------


class TestMaxThreadsPerRankConstraint:
    def test_within_limit(self) -> None:
        c = MaxThreadsPerRankConstraint(max_threads=4)
        layout = ComponentLayout("x", 8, 4, None)
        assert c.is_satisfied(layout, total_ranks=8)

    def test_exceeds_limit(self) -> None:
        c = MaxThreadsPerRankConstraint(max_threads=4)
        layout = ComponentLayout("x", 4, 8, None)
        assert not c.is_satisfied(layout, total_ranks=4)

    def test_invalid_max_raises(self) -> None:
        with pytest.raises(ValueError, match="max_threads"):
            MaxThreadsPerRankConstraint(max_threads=0)


class TestFixedThreadsPerRankConstraint:
    def test_matches(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        layout = ComponentLayout("x", 8, 4, None)
        assert c.is_satisfied(layout, total_ranks=8)

    def test_does_not_match(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        layout = ComponentLayout("x", 8, 2, None)
        assert not c.is_satisfied(layout, total_ranks=8)

    def test_invalid_n_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="n_threads"):
            FixedThreadsPerRankConstraint(n_threads=0)


class TestThreadsDivisorConstraint:
    def test_divides(self) -> None:
        c = ThreadsDivisorConstraint(divisor=8)
        layout = ComponentLayout("x", 4, 4, None)
        assert c.is_satisfied(layout, total_ranks=4)

    def test_does_not_divide(self) -> None:
        c = ThreadsDivisorConstraint(divisor=8)
        layout = ComponentLayout("x", 4, 3, None)
        assert not c.is_satisfied(layout, total_ranks=4)

    def test_invalid_divisor_raises(self) -> None:
        with pytest.raises(ValueError, match="divisor"):
            ThreadsDivisorConstraint(divisor=0)
