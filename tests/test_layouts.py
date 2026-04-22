# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.layouts (ComponentLayout, constraints, enumeration)."""

import dataclasses

import pytest

from access.config.layouts import (
    ComponentLayout,
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
    enumerate_layouts,
    iter_cartesian_decompositions,
)
from access.config.parallelisation import (
    AllocationSpec,
    CartesianDecomposition,
    Domain,
    FixedRanks,
    FreeAllocation,
    ParallelComponent,
    RatioAllocation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def domain_1d() -> Domain:
    return Domain(shape=(100,))


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(12, 8))


@pytest.fixture(scope="module")
def domain_prime() -> Domain:
    """Domain whose total size is a prime — forces 1×n or n×1 decompositions."""
    return Domain(shape=(7, 11))


@pytest.fixture(scope="module")
def leaf_no_domain() -> ParallelComponent:
    return ParallelComponent("leaf")


@pytest.fixture(scope="module")
def leaf_with_domain(domain_2d: Domain) -> ParallelComponent:
    return ParallelComponent("leaf", domain=domain_2d)


@pytest.fixture(scope="module")
def two_child_tree(domain_2d: Domain) -> ParallelComponent:
    """Parent with two ratio-allocated children (weights 3 and 2)."""
    atm = ParallelComponent("atm", domain=domain_2d)
    ocn = ParallelComponent("ocn", domain=domain_2d)
    return ParallelComponent("coupled", subcomponents=(atm, ocn))


# ---------------------------------------------------------------------------
# ComponentLayout validation
# ---------------------------------------------------------------------------


class TestComponentLayout:
    def test_basic(self, domain_2d: Domain) -> None:
        decomp = CartesianDecomposition(domain_2d, grid=(2, 2))
        layout = ComponentLayout("comp", n_ranks=4, threads_per_rank=1, decomposition=decomp)
        assert layout.total_cores == 4

    def test_total_cores(self) -> None:
        layout = ComponentLayout("comp", n_ranks=8, threads_per_rank=4, decomposition=None)
        assert layout.total_cores == 32

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ComponentLayout("", n_ranks=4, threads_per_rank=1, decomposition=None)

    def test_zero_ranks_raises(self) -> None:
        with pytest.raises(ValueError, match="n_ranks"):
            ComponentLayout("c", n_ranks=0, threads_per_rank=1, decomposition=None)

    def test_zero_tpr_raises(self) -> None:
        with pytest.raises(ValueError, match="threads_per_rank"):
            ComponentLayout("c", n_ranks=4, threads_per_rank=0, decomposition=None)

    def test_frozen(self, domain_2d: Domain) -> None:
        layout = ComponentLayout("c", n_ranks=4, threads_per_rank=1, decomposition=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            layout.n_ranks = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# iter_cartesian_decompositions
# ---------------------------------------------------------------------------


class TestIterCartesianDecompositions:
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
        # 7 is prime: only (1,7) and (7,1)
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
        # (2,2,2) must be among them
        assert (2, 2, 2) in grids

    def test_all_products_correct(self, domain_2d: Domain) -> None:
        for n in [1, 2, 3, 6, 12]:
            for d in iter_cartesian_decompositions(domain_2d, n_ranks=n):
                assert d.n_ranks == n

    def test_domain_attached(self, domain_2d: Domain) -> None:
        decomps = list(iter_cartesian_decompositions(domain_2d, n_ranks=4))
        assert all(d.domain is domain_2d for d in decomps)


# ---------------------------------------------------------------------------
# Category 1 — Cartesian grid constraints
# ---------------------------------------------------------------------------


class TestProcessGridDimEvenConstraint:
    def test_even_grid(self, domain_2d: Domain) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = ComponentLayout("x", 4, 1, CartesianDecomposition(domain_2d, (2, 2)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_odd_grid_fails(self, domain_2d: Domain) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = ComponentLayout("x", 3, 1, CartesianDecomposition(domain_2d, (3, 1)))
        assert not c.is_satisfied(layout, total_ranks=3)

    def test_no_decomposition_passes(self) -> None:
        c = ProcessGridDimEvenConstraint(dim=0)
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)


class TestProcessGridDimDivisibleConstraint:
    def test_divisible(self, domain_2d: Domain) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = ComponentLayout("x", 4, 1, CartesianDecomposition(domain_2d, (1, 4)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_not_divisible(self, domain_2d: Domain) -> None:
        c = ProcessGridDimDivisibleConstraint(dim=1, divisor=4)
        layout = ComponentLayout("x", 6, 1, CartesianDecomposition(domain_2d, (2, 3)))
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
        layout = ComponentLayout("x", 4, 1, CartesianDecomposition(domain_2d, (2, 2)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_elongated_fails(self, domain_2d: Domain) -> None:
        c = ProcessGridAspectRatioConstraint(max_ratio=2.0)
        # (1, 12): ratio = 12 > 2
        layout = ComponentLayout("x", 12, 1, CartesianDecomposition(domain_2d, (1, 12)))
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
        layout = ComponentLayout("x", 6, 1, CartesianDecomposition(domain_2d, (3, 2)))
        assert c.is_satisfied(layout, total_ranks=6)

    def test_non_uniform(self, domain_2d: Domain) -> None:
        # 12 / 5 is not integer
        c = UniformSubdomainConstraint()
        layout = ComponentLayout("x", 5, 1, CartesianDecomposition(domain_2d, (5, 1)))
        assert not c.is_satisfied(layout, total_ranks=5)

    def test_no_decomposition_passes(self) -> None:
        c = UniformSubdomainConstraint()
        layout = ComponentLayout("x", 4, 1, None)
        assert c.is_satisfied(layout, total_ranks=4)


class TestSubdomainSizeToleranceConstraint:
    def test_exact_passes(self, domain_2d: Domain) -> None:
        c = SubdomainSizeToleranceConstraint(tolerance=1.0)
        layout = ComponentLayout("x", 6, 1, CartesianDecomposition(domain_2d, (3, 2)))
        assert c.is_satisfied(layout, total_ranks=6)

    def test_within_tolerance(self) -> None:
        # Domain (13,), grid (4,): floor=3, ceil=4, ratio=4/3≈1.33 <= 1.5
        domain = Domain(shape=(13,))
        c = SubdomainSizeToleranceConstraint(tolerance=1.5)
        layout = ComponentLayout("x", 4, 1, CartesianDecomposition(domain, (4,)))
        assert c.is_satisfied(layout, total_ranks=4)

    def test_exceeds_tolerance(self) -> None:
        # Domain (13,), grid (4,): ratio=4/3≈1.33 — fails at tolerance=1.2
        domain = Domain(shape=(13,))
        c = SubdomainSizeToleranceConstraint(tolerance=1.2)
        layout = ComponentLayout("x", 4, 1, CartesianDecomposition(domain, (4,)))
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
        layout = ComponentLayout("x", 4, 1, CartesianDecomposition(domain, (4,)))
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
        layout = ComponentLayout("x", 4, 1, CartesianDecomposition(domain, (2, 2)))
        # local_shape = (6, 4) -> ratio 1.5, which exceeds 1.2
        assert not c.is_satisfied(layout, total_ranks=4)


class TestMinSubdomainSizeConstraint:
    def test_large_enough(self, domain_2d: Domain) -> None:
        # 12/3=4, 8/2=4 — both >= 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = ComponentLayout("x", 6, 1, CartesianDecomposition(domain_2d, (3, 2)))
        assert c.is_satisfied(layout, total_ranks=6)

    def test_too_small(self, domain_2d: Domain) -> None:
        # 8/8=1 < 2
        c = MinSubdomainSizeConstraint(min_size=2)
        layout = ComponentLayout("x", 8, 1, CartesianDecomposition(domain_2d, (1, 8)))
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


# ---------------------------------------------------------------------------
# enumerate_layouts — leaf components
# ---------------------------------------------------------------------------


class TestEnumerateLayoutsLeaf:
    def test_leaf_no_domain(self, leaf_no_domain: ParallelComponent) -> None:
        # FreeAllocation with default min_ranks=1; total_cores=4, tpr=1 → n_ranks=4, no decomp
        layouts = enumerate_layouts(leaf_no_domain, total_cores=4)
        assert len(layouts) == 1
        layout = layouts[0]
        assert layout.name == "leaf"
        assert layout.n_ranks == 4
        assert layout.threads_per_rank == 1
        assert layout.decomposition is None

    def test_leaf_with_domain(self, leaf_with_domain: ParallelComponent) -> None:
        # domain=(12,8), 4 ranks → decompositions: (1,4),(2,2),(4,1)
        layouts = enumerate_layouts(leaf_with_domain, total_cores=4)
        grids = [layout.decomposition.grid for layout in layouts]
        assert (1, 4) in grids
        assert (2, 2) in grids
        assert (4, 1) in grids
        assert len(layouts) == 3

    def test_tpr_range(self) -> None:
        # total_cores=8, tpr_range=(1,2): tpr=1→8 ranks, tpr=2→4 ranks
        comp = ParallelComponent("c")
        layouts = enumerate_layouts(comp, total_cores=8, tpr_range=(1, 2))
        tprs = {layout.threads_per_rank for layout in layouts}
        assert 1 in tprs
        assert 2 in tprs

    def test_tpr_not_divisor_skipped(self) -> None:
        # total_cores=9, tpr_range=(1,4): only tpr=1,3,9 divide 9
        comp = ParallelComponent("c")
        layouts = enumerate_layouts(comp, total_cores=9, tpr_range=(1, 4))
        tprs = {layout.threads_per_rank for layout in layouts}
        assert 2 not in tprs
        assert 4 not in tprs
        assert 1 in tprs
        assert 3 in tprs

    def test_fixed_ranks_single_layout(self) -> None:
        # Root gets all total_ranks regardless of allocation spec.
        comp = ParallelComponent("c")
        layouts = enumerate_layouts(comp, total_cores=4)
        assert len(layouts) == 1
        assert layouts[0].n_ranks == 4

    def test_unknown_allocation_name_raises(self) -> None:
        a = ParallelComponent("a")
        parent = ParallelComponent("p", subcomponents=(a,))
        with pytest.raises(ValueError, match="unknown component names"):
            enumerate_layouts(
                parent,
                total_cores=4,
                allocations=AllocationSpec(FreeAllocation(), subcomponents={"TYPO": AllocationSpec(FixedRanks(4))}),
            )

    def test_invalid_total_cores_raises(self, leaf_no_domain: ParallelComponent) -> None:
        with pytest.raises(ValueError, match="total_cores"):
            enumerate_layouts(leaf_no_domain, total_cores=0)

    def test_invalid_tpr_range_raises(self, leaf_no_domain: ParallelComponent) -> None:
        with pytest.raises(ValueError, match="tpr_range"):
            enumerate_layouts(leaf_no_domain, total_cores=4, tpr_range=(0, 1))

        with pytest.raises(ValueError, match="tpr_range"):
            enumerate_layouts(leaf_no_domain, total_cores=4, tpr_range=(3, 1))


# ---------------------------------------------------------------------------
# enumerate_layouts — component trees
# ---------------------------------------------------------------------------


class TestEnumerateLayoutsTree:
    def test_two_fixed_children(self, domain_2d: Domain) -> None:
        atm = ParallelComponent("atm", domain=domain_2d)
        ocn = ParallelComponent("ocn", domain=domain_2d)
        coupled = ParallelComponent("coupled", subcomponents=(atm, ocn))
        layouts = enumerate_layouts(
            coupled,
            total_cores=10,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={"atm": AllocationSpec(FixedRanks(6)), "ocn": AllocationSpec(FixedRanks(4))},
            ),
        )
        assert len(layouts) > 0
        for layout in layouts:
            atm_sub = next(sl for sl in layout.sub_layouts if sl.name == "atm")
            ocn_sub = next(sl for sl in layout.sub_layouts if sl.name == "ocn")
            assert atm_sub.n_ranks == 6
            assert ocn_sub.n_ranks == 4

    def test_ratio_children_exact_split(self, domain_2d: Domain) -> None:
        # weights 3:2 with 10 ranks and no waste → k=2: 2*3=6, 2*2=4.
        # MaxWastedRankFractionConstraint(0.0) enforces exact consumption.
        atm = ParallelComponent("atm", domain=domain_2d)
        ocn = ParallelComponent("ocn", domain=domain_2d)
        coupled = ParallelComponent(
            "coupled",
            subcomponents=(atm, ocn),
            local_constraints=(MaxWastedRankFractionConstraint(max_fraction=0.0),),
        )
        layouts = enumerate_layouts(
            coupled,
            total_cores=10,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "atm": AllocationSpec(RatioAllocation(weight=3)),
                    "ocn": AllocationSpec(RatioAllocation(weight=2)),
                },
            ),
        )
        assert len(layouts) > 0
        for layout in layouts:
            atm_sub = next(sl for sl in layout.sub_layouts if sl.name == "atm")
            ocn_sub = next(sl for sl in layout.sub_layouts if sl.name == "ocn")
            assert atm_sub.n_ranks == 6
            assert ocn_sub.n_ranks == 4

    def test_ratio_only_unique_k(self) -> None:
        # weights 1:1, total_cores=6, no waste allowed → only k=3 (3+3=6).
        # Without the waste constraint, k=1 (waste=4) and k=2 (waste=2) are also valid.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent(
            "p",
            subcomponents=(a, b),
            local_constraints=(MaxWastedRankFractionConstraint(max_fraction=0.0),),
        )
        layouts = enumerate_layouts(
            parent,
            total_cores=6,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(RatioAllocation(weight=1)),
                    "b": AllocationSpec(RatioAllocation(weight=1)),
                },
            ),
        )
        assert len(layouts) == 1
        assert layouts[0].sub_layouts[0].n_ranks == 3
        assert layouts[0].sub_layouts[1].n_ranks == 3

    def test_ratio_with_free_sibling_multiple_k(self) -> None:
        # With a free sibling absorbing the remainder, multiple k values are valid.
        # weights 1:1, total_cores=9:
        # k=1: ratio=2, free=7; k=2: ratio=4, free=5; k=3: ratio=6, free=3; k=4: ratio=8, free=1.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        spare = ParallelComponent("spare")
        parent = ParallelComponent("p", subcomponents=(a, b, spare))
        layouts = enumerate_layouts(
            parent,
            total_cores=9,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(RatioAllocation(weight=1)),
                    "b": AllocationSpec(RatioAllocation(weight=1)),
                    "spare": AllocationSpec(FreeAllocation(min_ranks=1)),
                },
            ),
        )
        ratio_totals = {layout.sub_layouts[0].n_ranks + layout.sub_layouts[1].n_ranks for layout in layouts}
        assert 2 in ratio_totals  # k=1
        assert 4 in ratio_totals  # k=2
        assert 6 in ratio_totals  # k=3
        assert 8 in ratio_totals  # k=4

    def test_free_allocation_children(self) -> None:
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=4,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(FreeAllocation(min_ranks=1, max_ranks=4)),
                    "b": AllocationSpec(FreeAllocation(min_ranks=1, max_ranks=4)),
                },
            ),
        )
        # All pairs (r_a, r_b) with 1<=r_a,r_b<=4 and r_a+r_b<=4
        rank_pairs = {(sl.n_ranks for sl in layout.sub_layouts) for layout in layouts}
        assert len(rank_pairs) == 6  # (1,1),(1,2),(1,3),(2,1),(2,2),(3,1)

    def test_infeasible_fixed_children_empty(self) -> None:
        # Fixed ranks (8 + 8 = 16) > total_cores=10
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=10,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={"a": AllocationSpec(FixedRanks(8)), "b": AllocationSpec(FixedRanks(8))},
            ),
        )
        assert layouts == []

    def test_sub_layout_names_match(self) -> None:
        a = ParallelComponent("alpha")
        b = ParallelComponent("beta")
        parent = ParallelComponent("root", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=5,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={"alpha": AllocationSpec(FixedRanks(2)), "beta": AllocationSpec(FixedRanks(3))},
            ),
        )
        assert len(layouts) == 1
        sub_names = [sl.name for sl in layouts[0].sub_layouts]
        assert sub_names == ["alpha", "beta"]

    def test_nested_subcomponents(self) -> None:
        leaf1 = ParallelComponent("leaf1")
        leaf2 = ParallelComponent("leaf2")
        mid = ParallelComponent("mid", subcomponents=(leaf1, leaf2))
        root = ParallelComponent("root", subcomponents=(mid,))
        layouts = enumerate_layouts(
            root,
            total_cores=4,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "mid": AllocationSpec(
                        FixedRanks(4),
                        subcomponents={
                            "leaf1": AllocationSpec(FixedRanks(2)),
                            "leaf2": AllocationSpec(FixedRanks(2)),
                        },
                    ),
                },
            ),
        )
        assert len(layouts) == 1
        mid_layout = layouts[0].sub_layouts[0]
        assert mid_layout.name == "mid"
        assert len(mid_layout.sub_layouts) == 2


# ---------------------------------------------------------------------------
# enumerate_layouts — constraints filter layouts
# ---------------------------------------------------------------------------


class TestEnumerateLayoutsConstraints:
    def test_uniform_subdomain_filters(self, domain_2d: Domain) -> None:
        # domain=(12,8), 6 ranks: decompositions include (6,1)[12/6=2,8/1=8],
        # (3,2)[12/3=4,8/2=4], (2,3)[12/2=6,8/3=2.67→non-uniform], (1,6)[12/1=12,8/6→non-uniform]
        comp = ParallelComponent(
            "c",
            domain=domain_2d,
            local_constraints=(UniformSubdomainConstraint(),),
        )
        layouts = enumerate_layouts(comp, total_cores=6)
        for layout in layouts:
            d = layout.decomposition
            assert all(dim % g == 0 for dim, g in zip(d.domain.shape, d.grid, strict=True))

    def test_min_subdomain_filters(self) -> None:
        domain = Domain(shape=(8, 8))
        # With min_size=3: floor(8/g)>=3 → g<=2; grid entries must be <=2
        comp = ParallelComponent(
            "c",
            domain=domain,
            local_constraints=(MinSubdomainSizeConstraint(min_size=3),),
        )
        layouts = enumerate_layouts(comp, total_cores=4)
        for layout in layouts:
            d = layout.decomposition
            assert all(dim // g >= 3 for dim, g in zip(d.domain.shape, d.grid, strict=True))

    def test_max_rank_fraction_filters(self) -> None:
        a = ParallelComponent("a", local_constraints=(MaxRankFractionConstraint(0.5),))
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=10,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(FreeAllocation(min_ranks=1)),
                    "b": AllocationSpec(FreeAllocation(min_ranks=1)),
                },
            ),
        )
        # a must have <= 5 ranks out of 10 total
        for layout in layouts:
            a_layout = next(sl for sl in layout.sub_layouts if sl.name == "a")
            assert a_layout.n_ranks <= 5

    def test_group_constraint_filters(self) -> None:
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        # a must have at least 2× ranks of b
        parent = ParallelComponent(
            "p",
            subcomponents=(a, b),
            group_constraints=(RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0),),
        )
        layouts = enumerate_layouts(
            parent,
            total_cores=9,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(FreeAllocation(min_ranks=1)),
                    "b": AllocationSpec(FreeAllocation(min_ranks=1)),
                },
            ),
        )
        for layout in layouts:
            a_layout = layout.sub_layouts[0]
            b_layout = layout.sub_layouts[1]
            assert a_layout.n_ranks >= 2.0 * b_layout.n_ranks

    def test_fixed_tpr_constraint_filters(self) -> None:
        comp = ParallelComponent(
            "c",
            local_constraints=(FixedThreadsPerRankConstraint(n_threads=4),),
        )
        # total_cores=8, tpr_range=(1,8): only tpr=4 satisfies the constraint (and 8/4=2 ranks)
        layouts = enumerate_layouts(comp, total_cores=8, tpr_range=(1, 8))
        assert all(layout.threads_per_rank == 4 for layout in layouts)
        assert len(layouts) == 1

    def test_alloc_spec_constraint_filters(self, domain_2d: Domain) -> None:
        # domain_2d=(12,8); FixedRanks(4) → decomps (1,4),(2,2),(4,1).
        # ProcessGridDimEvenConstraint on the AllocationSpec filters out (1,4) → 2 pass.
        a = ParallelComponent("a", domain=domain_2d)
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=5,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(FixedRanks(4), local_constraints=(ProcessGridDimEvenConstraint(dim=0),)),
                    "b": AllocationSpec(FixedRanks(1)),
                },
            ),
        )
        assert len(layouts) == 2  # (2,2) and (4,1); (1,4) filtered
        for layout in layouts:
            assert layout.sub_layouts[0].decomposition.grid[0] % 2 == 0

    def test_alloc_spec_group_constraint_filters(self) -> None:
        # group_constraints on AllocationSpec filter sibling combos;
        # the component itself has no group constraints here.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=9,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(FreeAllocation(min_ranks=1)),
                    "b": AllocationSpec(FreeAllocation(min_ranks=1)),
                },
                group_constraints=(RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0),),
            ),
        )
        for layout in layouts:
            a_layout = layout.sub_layouts[0]
            b_layout = layout.sub_layouts[1]
            assert a_layout.n_ranks >= 2.0 * b_layout.n_ranks

    def test_alloc_spec_and_component_group_constraints_both_applied(self) -> None:
        # Both component and alloc-spec group constraints must be satisfied simultaneously.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent(
            "p",
            subcomponents=(a, b),
            group_constraints=(RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0),),
        )
        layouts = enumerate_layouts(
            parent,
            total_cores=9,
            allocations=AllocationSpec(
                FreeAllocation(),
                subcomponents={
                    "a": AllocationSpec(FreeAllocation(min_ranks=1)),
                    "b": AllocationSpec(FreeAllocation(min_ranks=1)),
                },
                group_constraints=(RankRatioGroupConstraint(name_a="b", name_b="a", min_ratio=0.25),),
            ),
        )
        for layout in layouts:
            a_layout = layout.sub_layouts[0]
            b_layout = layout.sub_layouts[1]
            assert a_layout.n_ranks >= 2.0 * b_layout.n_ranks
            assert b_layout.n_ranks >= 0.25 * a_layout.n_ranks

    def test_grid_even_constraint_filters(self) -> None:
        domain = Domain(shape=(12, 8))
        comp = ParallelComponent(
            "c",
            domain=domain,
            local_constraints=(ProcessGridDimEvenConstraint(dim=0),),
        )
        layouts = enumerate_layouts(comp, total_cores=6)
        for layout in layouts:
            assert layout.decomposition.grid[0] % 2 == 0
