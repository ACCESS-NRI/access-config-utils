# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_layouts (ComponentLayout and enumeration)."""

import dataclasses

import pytest

from access.config.parallel_component import ComponentLayout, ParallelComponent
from access.config.parallel_constraints import (
    FixedThreadsPerRankConstraint,
    MaxRankFractionConstraint,
    MaxWastedRankFractionConstraint,
    MinSubdomainSizeConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
    UniformSubdomainConstraint,
)
from access.config.parallel_domain import Domain, DomainCartesianDecomposition
from access.config.parallel_layouts import (
    AllocationStrategy,
    enumerate_layouts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(12, 8))


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
        decomp = DomainCartesianDecomposition(domain_2d, grid=(2, 2))
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
        grids = []
        for layout in layouts:
            assert layout.decomposition is not None
            grids.append(layout.decomposition.grid)
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
                allocations=AllocationStrategy(subcomponents={"TYPO": AllocationStrategy(n_ranks=4)}),
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
            allocations=AllocationStrategy(
                subcomponents={
                    "atm": AllocationStrategy(n_ranks=6),
                    "ocn": AllocationStrategy(n_ranks=4),
                },
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
            allocations=AllocationStrategy(
                subcomponents={
                    "atm": AllocationStrategy(weight=3),
                    "ocn": AllocationStrategy(weight=2),
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
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(weight=1),
                    "b": AllocationStrategy(weight=1),
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
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(weight=1),
                    "b": AllocationStrategy(weight=1),
                    "spare": AllocationStrategy(min_ranks=1),
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
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(min_ranks=1, max_ranks=4),
                    "b": AllocationStrategy(min_ranks=1, max_ranks=4),
                },
            ),
        )
        # All pairs (r_a, r_b) with 1<=r_a,r_b<=4 and r_a+r_b<=4
        rank_pairs = {tuple(sl.n_ranks for sl in layout.sub_layouts) for layout in layouts}
        assert len(rank_pairs) == 6  # (1,1),(1,2),(1,3),(2,1),(2,2),(3,1)

    def test_infeasible_fixed_children_empty(self) -> None:
        # Fixed ranks (8 + 8 = 16) > total_cores=10
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=10,
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(n_ranks=8),
                    "b": AllocationStrategy(n_ranks=8),
                },
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
            allocations=AllocationStrategy(
                subcomponents={
                    "alpha": AllocationStrategy(n_ranks=2),
                    "beta": AllocationStrategy(n_ranks=3),
                },
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
            allocations=AllocationStrategy(
                subcomponents={
                    "mid": AllocationStrategy(
                        n_ranks=4,
                        subcomponents={
                            "leaf1": AllocationStrategy(n_ranks=2),
                            "leaf2": AllocationStrategy(n_ranks=2),
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
            assert d is not None
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
            assert d is not None
            assert all(dim // g >= 3 for dim, g in zip(d.domain.shape, d.grid, strict=True))

    def test_max_rank_fraction_filters(self) -> None:
        a = ParallelComponent("a", local_constraints=(MaxRankFractionConstraint(0.5),))
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=10,
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(min_ranks=1),
                    "b": AllocationStrategy(min_ranks=1),
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
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(min_ranks=1),
                    "b": AllocationStrategy(min_ranks=1),
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
        # domain_2d=(12,8); FixedAllocation(4) → decomps (1,4),(2,2),(4,1).
        # ProcessGridDimEvenConstraint on the AllocationStrategy filters out (1,4) → 2 pass.
        a = ParallelComponent("a", domain=domain_2d)
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=5,
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(n_ranks=4, local_constraints=(ProcessGridDimEvenConstraint(dim=0),)),
                    "b": AllocationStrategy(n_ranks=1),
                },
            ),
        )
        assert len(layouts) == 2  # (2,2) and (4,1); (1,4) filtered
        for layout in layouts:
            sub_layout = layout.sub_layouts[0]
            assert sub_layout.decomposition is not None
            assert sub_layout.decomposition.grid[0] % 2 == 0

    def test_alloc_spec_group_constraint_filters(self) -> None:
        # group_constraints on AllocationStrategy filter sibling combos;
        # the component itself has no group constraints here.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = enumerate_layouts(
            parent,
            total_cores=9,
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(min_ranks=1),
                    "b": AllocationStrategy(min_ranks=1),
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
            allocations=AllocationStrategy(
                subcomponents={
                    "a": AllocationStrategy(min_ranks=1),
                    "b": AllocationStrategy(min_ranks=1),
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
            assert layout.decomposition is not None
            assert layout.decomposition.grid[0] % 2 == 0


# ---------------------------------------------------------------------------
# AllocationStrategy
# ---------------------------------------------------------------------------


class TestAllocationStrategy:
    # --- Fixed mode ---

    def test_fixed_valid(self) -> None:
        spec = AllocationStrategy(n_ranks=12)
        assert spec.n_ranks == 12
        assert spec.allocation_mode == "fixed"
        assert spec.subcomponents == {}

    def test_fixed_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            AllocationStrategy(n_ranks=0)

    def test_fixed_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            AllocationStrategy(n_ranks=-4)

    def test_fixed_with_min_ranks_raises(self) -> None:
        with pytest.raises(ValueError, match="min_ranks/max_ranks"):
            AllocationStrategy(n_ranks=4, min_ranks=2)

    def test_fixed_with_max_ranks_raises(self) -> None:
        with pytest.raises(ValueError, match="min_ranks/max_ranks"):
            AllocationStrategy(n_ranks=4, max_ranks=8)

    # --- Ratio mode ---

    def test_ratio_valid(self) -> None:
        spec = AllocationStrategy(weight=3)
        assert spec.weight == 3
        assert spec.allocation_mode == "ratio"

    def test_ratio_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            AllocationStrategy(weight=0)

    def test_ratio_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            AllocationStrategy(weight=-1)

    def test_ratio_with_min_ranks_raises(self) -> None:
        with pytest.raises(ValueError, match="min_ranks/max_ranks"):
            AllocationStrategy(weight=2, min_ranks=2)

    def test_ratio_with_max_ranks_raises(self) -> None:
        with pytest.raises(ValueError, match="min_ranks/max_ranks"):
            AllocationStrategy(weight=2, max_ranks=8)

    # --- Both n_ranks and weight raises ---

    def test_fixed_and_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot both be set"):
            AllocationStrategy(n_ranks=4, weight=2)

    # --- Free mode ---

    def test_free_defaults(self) -> None:
        spec = AllocationStrategy()
        assert spec.min_ranks == 1
        assert spec.max_ranks is None
        assert spec.allocation_mode == "free"

    def test_free_explicit_bounds(self) -> None:
        spec = AllocationStrategy(min_ranks=4, max_ranks=16)
        assert spec.min_ranks == 4
        assert spec.max_ranks == 16

    def test_free_equal_bounds(self) -> None:
        spec = AllocationStrategy(min_ranks=8, max_ranks=8)
        assert spec.min_ranks == spec.max_ranks == 8

    def test_free_zero_min_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            AllocationStrategy(min_ranks=0)

    def test_free_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ranks"):
            AllocationStrategy(min_ranks=4, max_ranks=2)

    # --- Subcomponents and constraints ---

    def test_with_subcomponents(self) -> None:
        child_a = AllocationStrategy()
        child_b = AllocationStrategy(n_ranks=12)
        parent = AllocationStrategy(subcomponents={"a": child_a, "b": child_b})
        assert len(parent.subcomponents) == 2
        assert parent.subcomponents["b"].n_ranks == 12

    def test_with_local_constraints(self) -> None:
        from access.config.parallel_constraints import FixedThreadsPerRankConstraint

        c = FixedThreadsPerRankConstraint(n_threads=1)
        spec = AllocationStrategy(local_constraints=(c,))
        assert spec.local_constraints == (c,)

    def test_group_constraints_default_empty(self) -> None:
        spec = AllocationStrategy()
        assert spec.group_constraints == ()

    def test_with_group_constraints(self) -> None:
        from access.config.parallel_constraints import RankRatioGroupConstraint

        gc = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=1.0)
        spec = AllocationStrategy(group_constraints=(gc,))
        assert spec.group_constraints == (gc,)

    def test_frozen(self) -> None:
        import dataclasses

        spec = AllocationStrategy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.n_ranks = 4  # type: ignore[misc]

    def test_hash_includes_structure(self) -> None:
        child = AllocationStrategy(n_ranks=2)
        spec_a = AllocationStrategy(subcomponents={"child": child})
        spec_b = AllocationStrategy(subcomponents={"child": child})
        assert hash(spec_a) == hash(spec_b)
