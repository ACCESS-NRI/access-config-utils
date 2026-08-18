# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for concrete constraints in access.config.parallel_constraints."""

import pytest

from access.config.parallel_component import ComponentLayout
from access.config.parallel_constraints import (
    FixedThreadsPerRankConstraint,
    MaxCoreFractionConstraint,
    MaxThreadsPerRankConstraint,
    MaxWastedCoreFractionConstraint,
    MinSubdomainSizeConstraint,
    ProcessGridAspectRatioConstraint,
    ProcessGridDimDivisibleConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
    SubdomainAspectRatioConstraint,
    UniformSubdomainConstraint,
)
from access.config.parallel_domain import Domain, DomainDecompositionSpec
from access.config.parallel_mpi_grid import MPICartesianGrid


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(12, 8))


def leaf(
    name: str = "x",
    n_ranks: int = 1,
    tpr: int = 1,
    decomposition: DomainDecompositionSpec | None = None,
) -> ComponentLayout:
    """A leaf layout: it spends exactly ``n_ranks * tpr`` cores."""
    return ComponentLayout(
        name,
        n_cores=n_ranks * tpr,
        n_ranks=n_ranks,
        threads_per_rank=tpr,
        decomposition=decomposition,
    )


def parent(name: str, n_cores: int, subs: tuple[ComponentLayout, ...]) -> ComponentLayout:
    """A parent layout holding *n_cores*, of which the sub-layouts may use only some."""
    return ComponentLayout(
        name,
        n_cores=n_cores,
        n_ranks=sum(sub.n_ranks for sub in subs),
        threads_per_rank=None,
        decomposition=None,
        sub_layouts=subs,
    )


# ---------------------------------------------------------------------------
# Category 1 — Cartesian grid constraints
# ---------------------------------------------------------------------------


class TestProcessGridDimEvenConstraint:
    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            ProcessGridDimEvenConstraint(dim=-1)

    def test_even_grid(self, domain_2d: Domain) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = leaf("x", 4, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 2))))
        assert c.is_satisfied(layout, total_cores=4)

    def test_odd_grid_fails(self, domain_2d: Domain) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = leaf("x", 3, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((3, 1))))
        assert not c.is_satisfied(layout, total_cores=3)

    def test_no_decomposition_raises(self) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout, total_cores=4)

    def test_dim_out_of_range_raises(self, domain_2d: Domain) -> None:
        c = ProcessGridDimEvenConstraint(dim=5)
        layout = leaf("x", 4, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 2))))
        with pytest.raises(ValueError, match="out of range"):
            c.is_satisfied(layout, total_cores=4)


class TestProcessGridDimDivisibleConstraint:
    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            ProcessGridDimDivisibleConstraint(dim=-1, divisor=2)

    def test_divisible(self, domain_2d: Domain) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = leaf("x", 4, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((1, 4))))
        assert c.is_satisfied(layout, total_cores=4)

    def test_not_divisible(self, domain_2d: Domain) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = leaf("x", 6, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 3))))
        assert not c.is_satisfied(layout, total_cores=6)

    def test_invalid_divisor_raises(self) -> None:
        with pytest.raises(ValueError, match="divisor"):
            ProcessGridDimDivisibleConstraint(dim=0, divisor=0)

    def test_no_decomposition_raises(self) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=0, divisor=4)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout, total_cores=4)

    def test_dim_out_of_range_raises(self, domain_2d: Domain) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=5, divisor=2)
        layout = leaf("x", 4, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 2))))
        with pytest.raises(ValueError, match="out of range"):
            c.is_satisfied(layout, total_cores=4)


class TestProcessGridAspectRatioConstraint:
    def test_square_passes(self, domain_2d: Domain) -> None:
        c = ProcessGridAspectRatioConstraint(max_ratio=2.0)
        layout = leaf("x", 4, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 2))))
        assert c.is_satisfied(layout, total_cores=4)

    def test_elongated_fails(self, domain_2d: Domain) -> None:
        c = ProcessGridAspectRatioConstraint(max_ratio=2.0)
        # (1, 12): ratio = 12 > 2
        layout = leaf("x", 12, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((1, 12))))
        assert not c.is_satisfied(layout, total_cores=12)

    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ratio"):
            ProcessGridAspectRatioConstraint(max_ratio=0.5)

    def test_no_decomposition_raises(self) -> None:
        c = ProcessGridAspectRatioConstraint(max_ratio=1.5)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout, total_cores=4)


# ---------------------------------------------------------------------------
# Category 2 — Core distribution constraints
# ---------------------------------------------------------------------------


class TestMaxCoreFractionConstraint:
    def test_within_limit(self) -> None:
        c = MaxCoreFractionConstraint(max_fraction=0.5)
        assert c.is_satisfied(leaf("x", 40), total_cores=100)

    def test_at_limit(self) -> None:
        c = MaxCoreFractionConstraint(max_fraction=0.5)
        assert c.is_satisfied(leaf("x", 50), total_cores=100)

    def test_exceeds_limit(self) -> None:
        c = MaxCoreFractionConstraint(max_fraction=0.5)
        assert not c.is_satisfied(leaf("x", 51), total_cores=100)

    def test_counts_cores_not_ranks(self) -> None:
        # 30 ranks x 2 threads is 60 of 100 cores, over the limit, even though 30 ranks
        # alone would be under it. This is the distinction the constraint exists to make.
        c = MaxCoreFractionConstraint(max_fraction=0.5)
        assert not c.is_satisfied(leaf("x", 30, 2), total_cores=100)
        assert c.is_satisfied(leaf("x", 30, 1), total_cores=100)

    def test_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="max_fraction"):
            MaxCoreFractionConstraint(max_fraction=0.0)

        with pytest.raises(ValueError, match="max_fraction"):
            MaxCoreFractionConstraint(max_fraction=1.1)

    @pytest.mark.parametrize(("n_cores", "total_cores", "max_fraction"), [(29, 100, 0.29), (7, 70, 0.1), (3, 30, 0.1)])
    def test_exact_boundary_is_satisfied(self, n_cores: int, total_cores: int, max_fraction: float) -> None:
        # A component using exactly max_fraction of the cores must pass.  Comparing
        # against max_fraction * total_cores fails here (0.29 * 100 < 29).
        c = MaxCoreFractionConstraint(max_fraction=max_fraction)
        assert c.is_satisfied(leaf("x", n_cores), total_cores=total_cores)

    def test_zero_total_cores_raises(self) -> None:
        c = MaxCoreFractionConstraint(max_fraction=0.5)
        with pytest.raises(ValueError, match="requires total_cores >= 1"):
            c.is_satisfied(leaf("x", 1), total_cores=0)


class TestMaxWastedCoreFractionConstraint:
    def test_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="max_fraction"):
            MaxWastedCoreFractionConstraint(max_fraction=-0.1)

    def test_no_sublayouts_raises(self) -> None:
        c = MaxWastedCoreFractionConstraint(max_fraction=0.0)
        layout = leaf("leaf", 8)
        with pytest.raises(ValueError, match="requires layout to have sub-layouts"):
            c.is_satisfied(layout, total_cores=8)

    def test_no_waste_satisfies_zero_fraction(self) -> None:
        c = MaxWastedCoreFractionConstraint(max_fraction=0.0)
        assert c.is_satisfied(parent("p", 8, (leaf("a", 8),)), total_cores=8)

    def test_waste_exceeding_fraction_fails(self) -> None:
        c = MaxWastedCoreFractionConstraint(max_fraction=0.1)
        # 2 of 8 cores idle = 0.25 > 0.1
        assert not c.is_satisfied(parent("p", 8, (leaf("a", 6),)), total_cores=8)

    def test_counts_cores_not_ranks(self) -> None:
        # A threaded child spends 8 cores on 4 ranks, so nothing is idle even though the
        # rank counts differ from the core counts.
        c = MaxWastedCoreFractionConstraint(max_fraction=0.0)
        assert c.is_satisfied(parent("p", 8, (leaf("a", 4, 2),)), total_cores=8)

    def test_exact_boundary_is_satisfied(self) -> None:
        # 7 of 70 cores idle is exactly 0.1; comparing against 0.1 * 70 rejects it.
        c = MaxWastedCoreFractionConstraint(max_fraction=0.1)
        assert c.is_satisfied(parent("p", 70, (leaf("a", 63),)), total_cores=70)


class TestRankRatioGroupConstraint:
    def test_satisfied(self) -> None:
        c = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=1.5)
        a = leaf("a", 6, 1)
        b = leaf("b", 4, 1)
        assert c.is_satisfied((a, b), total_cores=10)

    def test_not_satisfied(self) -> None:
        c = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0)
        a = leaf("a", 6, 1)
        b = leaf("b", 4, 1)
        # 6 / 4 = 1.5 < 2.0
        assert not c.is_satisfied((a, b), total_cores=10)

    def test_unknown_name_raises(self) -> None:
        c = RankRatioGroupConstraint(name_a="TYPO", name_b="b", min_ratio=1.0)
        a = leaf("a", 6, 1)
        b = leaf("b", 4, 1)
        with pytest.raises(ValueError, match="not found"):
            c.is_satisfied((a, b), total_cores=10)

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
        layout = leaf("x", 6, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((3, 2))))
        assert c.is_satisfied(layout, total_cores=6)

    def test_non_uniform(self, domain_2d: Domain) -> None:
        # 12 / 5 is not integer
        c = UniformSubdomainConstraint()
        layout = leaf("x", 5, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((5, 1))))
        assert not c.is_satisfied(layout, total_cores=5)

    def test_no_decomposition_raises(self) -> None:
        c = UniformSubdomainConstraint()
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout, total_cores=4)


class TestSubdomainAspectRatioConstraint:
    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ratio"):
            SubdomainAspectRatioConstraint(max_ratio=0.9)

    def test_no_decomposition_raises(self) -> None:
        c = SubdomainAspectRatioConstraint(max_ratio=2.0)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout, total_cores=4)

    def test_aspect_ratio_check(self) -> None:
        domain = Domain(shape=(12, 8))
        c = SubdomainAspectRatioConstraint(max_ratio=1.2)
        layout = leaf("x", 4, 1, DomainDecompositionSpec(domain, MPICartesianGrid((2, 2))))
        # mean_local_shape = (6, 4) -> ratio 1.5, which exceeds 1.2
        assert not c.is_satisfied(layout, total_cores=4)


class TestMinSubdomainSizeConstraint:
    def test_large_enough(self, domain_2d: Domain) -> None:
        # 12/3=4, 8/2=4 — both >= 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = leaf("x", 6, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((3, 2))))
        assert c.is_satisfied(layout, total_cores=6)

    def test_too_small(self, domain_2d: Domain) -> None:
        # 8/8=1 < 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = leaf("x", 8, 1, DomainDecompositionSpec(domain_2d, MPICartesianGrid((1, 8))))
        assert not c.is_satisfied(layout, total_cores=8)

    def test_invalid_min_size_raises(self) -> None:
        with pytest.raises(ValueError, match="min_size"):
            MinSubdomainSizeConstraint(min_size=0)

    def test_no_decomposition_raises(self) -> None:
        c = MinSubdomainSizeConstraint(min_size=4)
        layout = leaf("x", 4, 1)
        with pytest.raises(ValueError, match="requires layout.decomposition"):
            c.is_satisfied(layout, total_cores=4)


# ---------------------------------------------------------------------------
# Category 4 — Thread constraints
# ---------------------------------------------------------------------------


class TestMaxThreadsPerRankConstraint:
    def test_within_limit(self) -> None:
        c = MaxThreadsPerRankConstraint(max_threads=4)
        layout = leaf("x", 8, 4)
        assert c.is_satisfied(layout, total_cores=8)

    def test_exceeds_limit(self) -> None:
        c = MaxThreadsPerRankConstraint(max_threads=4)
        layout = leaf("x", 4, 8)
        assert not c.is_satisfied(layout, total_cores=4)

    def test_invalid_max_raises(self) -> None:
        with pytest.raises(ValueError, match="max_threads"):
            MaxThreadsPerRankConstraint(max_threads=0)

    def test_parent_raises(self) -> None:
        # A parent's sub-components may each use a different thread count, so it has none.
        c = MaxThreadsPerRankConstraint(max_threads=4)
        with pytest.raises(ValueError, match="requires layout.threads_per_rank"):
            c.is_satisfied(parent("p", 8, (leaf("a", 8),)), total_cores=8)


class TestFixedThreadsPerRankConstraint:
    def test_matches(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        layout = leaf("x", 8, 4)
        assert c.is_satisfied(layout, total_cores=8)

    def test_does_not_match(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        layout = leaf("x", 8, 2)
        assert not c.is_satisfied(layout, total_cores=8)

    def test_invalid_n_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="n_threads"):
            FixedThreadsPerRankConstraint(n_threads=0)

    def test_parent_raises(self) -> None:
        c = FixedThreadsPerRankConstraint(n_threads=4)
        with pytest.raises(ValueError, match="requires layout.threads_per_rank"):
            c.is_satisfied(parent("p", 8, (leaf("a", 8),)), total_cores=8)
