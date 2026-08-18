# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_layouts (allocation strategies and enumeration)."""

import logging

import pytest

from access.config.parallel_allocation_strategies import (
    AllocationStrategy,
    FixedAllocation,
    FreeAllocation,
    RatioAllocation,
    RootAllocation,
)
from access.config.parallel_component import ComponentLayout, ParallelComponent
from access.config.parallel_constraints import (
    FixedThreadsPerRankConstraint,
    MaxCoreFractionConstraint,
    MaxWastedCoreFractionConstraint,
    MinSubdomainSizeConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
    UniformSubdomainConstraint,
)
from access.config.parallel_domain import Domain
from access.config.parallel_layouts import iter_layouts
from access.config.parallel_mpi_grid import MPICartesianGrid

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


# ---------------------------------------------------------------------------
# iter_layouts — leaf components
# ---------------------------------------------------------------------------


class TestIterLayoutsLeaf:
    def test_leaf_no_domain(self, leaf_no_domain: ParallelComponent) -> None:
        # Free allocation with default min_cores=1; total_cores=4, tpr=1 → n_ranks=4,
        # no decomp
        layouts = list(iter_layouts(leaf_no_domain, total_cores=4))
        assert len(layouts) == 1
        layout = layouts[0]
        assert layout.name == "leaf"
        assert layout.n_ranks == 4
        assert layout.threads_per_rank == 1
        assert layout.decomposition is None

    def test_leaf_with_domain(self, leaf_with_domain: ParallelComponent, domain_2d: Domain) -> None:
        # domain=(12,8), 4 ranks → decompositions: (1,4),(2,2),(4,1)
        layouts = list(iter_layouts(leaf_with_domain, total_cores=4))
        shapes = []
        for layout in layouts:
            assert layout.decomposition is not None
            # Each spec pairs the component's own domain with a 4-rank cartesian grid.
            assert layout.decomposition.domain is domain_2d
            assert isinstance(layout.decomposition.grid, MPICartesianGrid)
            assert layout.decomposition.n_ranks == 4
            shapes.append(layout.decomposition.grid.shape)
        assert (1, 4) in shapes
        assert (2, 2) in shapes
        assert (4, 1) in shapes
        assert len(layouts) == 3

    def test_leaf_with_domain_prime_ranks(self, leaf_with_domain: ParallelComponent) -> None:
        # 7 ranks is prime → only the two degenerate grids (1,7) and (7,1)
        layouts = list(iter_layouts(leaf_with_domain, total_cores=7))
        shapes = []
        for layout in layouts:
            assert layout.decomposition is not None
            shapes.append(layout.decomposition.grid.shape)
        assert sorted(shapes) == [(1, 7), (7, 1)]

    def test_tpr_range(self) -> None:
        # total_cores=8, tpr_range=(1,2): tpr=1→8 ranks, tpr=2→4 ranks
        comp = ParallelComponent("c")
        layouts = list(iter_layouts(comp, total_cores=8, allocations=RootAllocation(tpr_range=(1, 2))))
        tprs = {layout.threads_per_rank for layout in layouts}
        assert 1 in tprs
        assert 2 in tprs

    def test_tpr_not_divisor_skipped(self) -> None:
        # total_cores=9, tpr_range=(1,4): only tpr=1,3,9 divide 9
        comp = ParallelComponent("c")
        layouts = list(iter_layouts(comp, total_cores=9, allocations=RootAllocation(tpr_range=(1, 4))))
        tprs = {layout.threads_per_rank for layout in layouts}
        assert 2 not in tprs
        assert 4 not in tprs
        assert 1 in tprs
        assert 3 in tprs

    def test_fixed_ranks_single_layout(self) -> None:
        # The root receives the whole budget.
        comp = ParallelComponent("c")
        layouts = list(iter_layouts(comp, total_cores=4))
        assert len(layouts) == 1
        assert layouts[0].n_ranks == 4

    def test_leaf_spends_its_cores_exactly(self) -> None:
        comp = ParallelComponent("c")
        for layout in iter_layouts(comp, total_cores=12, allocations=RootAllocation(tpr_range=(1, 4))):
            assert layout.n_ranks * layout.threads_per_rank == 12
            assert layout.idle_cores == 0

    def test_missing_allocation_raises(self) -> None:
        """No default mode, so a sub-component left out is an error, not a free one."""
        parent = ParallelComponent("p", subcomponents=(ParallelComponent("a"), ParallelComponent("b")))
        with pytest.raises(ValueError, match=r"no allocation given at root for \{'b'\}"):
            iter_layouts(
                parent,
                total_cores=4,
                allocations=RootAllocation(subcomponents={"a": FreeAllocation()}),
            )

    def test_missing_allocation_is_caught_at_depth(self) -> None:
        leaf = ParallelComponent("leaf")
        mid = ParallelComponent("mid", subcomponents=(leaf,))
        parent = ParallelComponent("p", subcomponents=(mid,))
        with pytest.raises(ValueError, match=r"no allocation given at root.mid for \{'leaf'\}"):
            iter_layouts(
                parent,
                total_cores=4,
                allocations=RootAllocation(subcomponents={"mid": FreeAllocation()}),
            )

    @pytest.mark.parametrize(
        "allocations",
        [
            FixedAllocation(n_cores=4),
            RatioAllocation(weight=2),
            FreeAllocation(min_cores=2),
            FreeAllocation(max_cores=8),
            FreeAllocation(min_cores=2, max_cores=8),
        ],
    )
    def test_a_sizing_strategy_cannot_be_the_root(
        self, leaf_no_domain: ParallelComponent, allocations: AllocationStrategy
    ) -> None:
        # The root always gets the whole budget, so a size given for it could only be
        # ignored — say so instead of returning layouts that contradict what was asked
        # for. RootAllocation has no field to write these in, so what reaches the
        # enumerator is the wrong type rather than a bad value.
        with pytest.raises(TypeError, match="allocations must be a RootAllocation"):
            iter_layouts(leaf_no_domain, total_cores=4, allocations=allocations)

    def test_root_non_rank_fields_are_honoured(self, domain_2d: Domain) -> None:
        # subcomponents, local_constraints and group_constraints still apply at the root.
        comp = ParallelComponent("c", domain=domain_2d)
        layouts = list(
            iter_layouts(
                comp,
                total_cores=6,
                allocations=RootAllocation(local_constraints=(ProcessGridDimEvenConstraint(dim=0),)),
            )
        )
        assert layouts
        for layout in layouts:
            assert layout.decomposition is not None
            assert layout.decomposition.grid[0] % 2 == 0

    def test_an_invalid_tpr_range_is_rejected_before_the_search(self) -> None:
        # The thread range belongs to the root strategy now, so a bad one never reaches
        # the enumerator; RootAllocation itself refuses to be built. See
        # test_parallel_allocation_strategies for the range rules themselves.
        with pytest.raises(ValueError, match="AllocationStrategy.tpr_range"):
            RootAllocation(tpr_range=(3, 1))


# ---------------------------------------------------------------------------
# Per-component thread counts
# ---------------------------------------------------------------------------


class TestHeterogeneousThreading:
    """Cores, not ranks, are divided, so components can use different thread counts."""

    def test_components_can_use_different_thread_counts(self) -> None:
        um = ParallelComponent("UM7", domain=Domain((192, 144)))
        mom = ParallelComponent("MOM5", domain=Domain((360, 300)))
        root = ParallelComponent("ESM", subcomponents=(um, mom))

        layouts = list(
            iter_layouts(
                root,
                total_cores=32,
                allocations=RootAllocation(
                    subcomponents={
                        "UM7": FixedAllocation(n_cores=16, tpr_range=(2, 2)),
                        "MOM5": FixedAllocation(n_cores=16, tpr_range=(1, 1)),
                    }
                ),
            )
        )

        assert layouts
        for layout in layouts:
            um_layout, mom_layout = layout.sub_layouts
            # Equal core shares, different thread counts, so different rank counts.
            assert (um_layout.n_cores, um_layout.threads_per_rank, um_layout.n_ranks) == (16, 2, 8)
            assert (mom_layout.n_cores, mom_layout.threads_per_rank, mom_layout.n_ranks) == (16, 1, 16)
            assert layout.n_ranks == 24
            assert layout.idle_cores == 0

    def test_tpr_range_is_inherited_and_overridden(self) -> None:
        threaded = ParallelComponent("threaded")
        plain = ParallelComponent("plain")
        root = ParallelComponent("root", subcomponents=(threaded, plain))

        layouts = list(
            iter_layouts(
                root,
                total_cores=8,
                allocations=RootAllocation(
                    tpr_range=(1, 1),  # the inherited default
                    subcomponents={
                        "threaded": FixedAllocation(n_cores=4, tpr_range=(2, 2)),  # overrides it
                        "plain": FixedAllocation(n_cores=4),  # inherits it
                    },
                ),
            )
        )

        assert len(layouts) == 1
        threaded_layout, plain_layout = layouts[0].sub_layouts
        assert threaded_layout.threads_per_rank == 2
        assert plain_layout.threads_per_rank == 1

    def test_a_parent_tpr_range_reaches_its_descendants(self) -> None:
        leaf_component = ParallelComponent("leaf")
        mid = ParallelComponent("mid", subcomponents=(leaf_component,))
        root = ParallelComponent("root", subcomponents=(mid,))

        layouts = list(
            iter_layouts(
                root,
                total_cores=8,
                allocations=RootAllocation(
                    subcomponents={
                        "mid": FixedAllocation(
                            n_cores=8,
                            tpr_range=(4, 4),
                            subcomponents={"leaf": FreeAllocation()},
                        )
                    },
                ),
            )
        )

        # The range set on "mid" reaches the leaf below it, so only core counts divisible
        # by 4 are spendable: the free leaf may take 4 of mid's 8 cores, or all 8.
        assert layouts
        for layout in layouts:
            leaf_layout = layout.sub_layouts[0].sub_layouts[0]
            assert leaf_layout.threads_per_rank == 4
        assert {layout.sub_layouts[0].sub_layouts[0].n_cores for layout in layouts} == {4, 8}

    def test_a_parent_has_no_thread_count(self) -> None:
        root = ParallelComponent("root", subcomponents=(ParallelComponent("a"),))
        allocations = RootAllocation(tpr_range=(1, 2), subcomponents={"a": FreeAllocation()})
        for layout in iter_layouts(root, total_cores=4, allocations=allocations):
            assert layout.threads_per_rank is None
            assert layout.sub_layouts[0].threads_per_rank is not None

    def test_thread_count_must_divide_the_allocation(self) -> None:
        # 9 cores cannot be spent at 2 threads per rank, so only 1 and 3 survive.
        comp = ParallelComponent("c")
        root = ParallelComponent("root", subcomponents=(comp,))
        layouts = list(
            iter_layouts(
                root,
                total_cores=9,
                allocations=RootAllocation(tpr_range=(1, 4), subcomponents={"c": FixedAllocation(n_cores=9)}),
            )
        )
        assert {layout.sub_layouts[0].threads_per_rank for layout in layouts} == {1, 3}


# ---------------------------------------------------------------------------
# iter_layouts — laziness and eager validation
# ---------------------------------------------------------------------------


class TestIterLayoutsLaziness:
    def test_is_lazy(self, leaf_with_domain: ParallelComponent) -> None:
        # Pulling one layout must not enumerate the rest: 4 ranks over a 2-D domain
        # has three decompositions, and only the first is produced here.
        layouts = iter_layouts(leaf_with_domain, total_cores=4)
        first = next(layouts)
        assert first.decomposition is not None
        assert first.decomposition.grid.shape == (1, 4)
        assert len(list(layouts)) == 2

    def test_validates_eagerly(self, leaf_no_domain: ParallelComponent) -> None:
        # A plain generator function would defer these until the first next() call.
        with pytest.raises(ValueError, match="total_cores"):
            iter_layouts(leaf_no_domain, total_cores=0)

        with pytest.raises(ValueError, match="unknown component names"):
            iter_layouts(
                ParallelComponent("p", subcomponents=(ParallelComponent("a"),)),
                total_cores=4,
                allocations=RootAllocation(subcomponents={"TYPO": FixedAllocation(n_cores=4)}),
            )

    def test_empty_search_yields_nothing(self) -> None:
        comp = ParallelComponent(
            "c", local_constraints=(FixedThreadsPerRankConstraint(n_threads=8),), domain=Domain((4, 4))
        )
        assert list(iter_layouts(comp, total_cores=4)) == []


# ---------------------------------------------------------------------------
# Diagnostics for an empty result
# ---------------------------------------------------------------------------


class TestEmptyResultDiagnostics:
    def test_unsatisfiable_constraint_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        comp = ParallelComponent("c", local_constraints=(FixedThreadsPerRankConstraint(n_threads=8),))
        with caplog.at_level(logging.DEBUG, logger="access.config.parallel_layouts"):
            assert list(iter_layouts(comp, total_cores=4)) == []
        assert "no valid layout for 'c' on 4 core(s)" in caplog.text

    def test_oversubscribed_fixed_allocation_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        parent = ParallelComponent("p", subcomponents=(ParallelComponent("a"), ParallelComponent("b")))
        allocations = RootAllocation(subcomponents={"a": FixedAllocation(n_cores=8), "b": FixedAllocation(n_cores=8)})
        # The core-split diagnostics live with the scheduler, in the allocation module.
        with caplog.at_level(logging.DEBUG, logger="access.config.parallel_allocation_strategies"):
            assert list(iter_layouts(parent, total_cores=4, allocations=allocations)) == []
        assert "the fixed allocations {'a': 8, 'b': 8} need 16 core(s), but only 4 are available" in caplog.text

    def test_skipped_thread_count_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        comp = ParallelComponent("c")
        with caplog.at_level(logging.DEBUG, logger="access.config.parallel_layouts"):
            list(iter_layouts(comp, total_cores=9, allocations=RootAllocation(tpr_range=(1, 2))))
        assert "component 'c' cannot use 2 thread(s) per rank: it does not divide its 9 core(s)" in caplog.text

    def test_unsatisfiable_subcomponent_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        # A leaf that cannot be laid out names itself, so the failing node is identifiable
        # even in a tree where the parent has options.
        child = ParallelComponent(
            "child", domain=Domain((4, 4)), local_constraints=(MinSubdomainSizeConstraint(min_size=99),)
        )
        parent = ParallelComponent("p", subcomponents=(child,))
        with caplog.at_level(logging.DEBUG, logger="access.config.parallel_layouts"):
            assert list(iter_layouts(parent, total_cores=2)) == []
        assert "no valid layout for component 'child' on 1 core(s) with 1-1 thread(s) per rank" in caplog.text

    def test_quiet_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        comp = ParallelComponent("c", local_constraints=(FixedThreadsPerRankConstraint(n_threads=8),))
        with caplog.at_level(logging.WARNING, logger="access.config.parallel_layouts"):
            assert list(iter_layouts(comp, total_cores=4)) == []
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# iter_layouts — component trees
# ---------------------------------------------------------------------------


class TestIterLayoutsTree:
    def test_two_fixed_children(self, domain_2d: Domain) -> None:
        atm = ParallelComponent("atm", domain=domain_2d)
        ocn = ParallelComponent("ocn", domain=domain_2d)
        coupled = ParallelComponent("coupled", subcomponents=(atm, ocn))
        layouts = list(
            iter_layouts(
                coupled,
                total_cores=10,
                allocations=RootAllocation(
                    subcomponents={
                        "atm": FixedAllocation(n_cores=6),
                        "ocn": FixedAllocation(n_cores=4),
                    },
                ),
            )
        )
        assert len(layouts) > 0
        for layout in layouts:
            atm_sub = next(sl for sl in layout.sub_layouts if sl.name == "atm")
            ocn_sub = next(sl for sl in layout.sub_layouts if sl.name == "ocn")
            assert atm_sub.n_ranks == 6
            assert ocn_sub.n_ranks == 4

    def test_ratio_children_exact_split(self, domain_2d: Domain) -> None:
        # weights 3:2 with 10 ranks and no waste → k=2: 2*3=6, 2*2=4.
        # MaxWastedCoreFractionConstraint(0.0) enforces exact consumption.
        atm = ParallelComponent("atm", domain=domain_2d)
        ocn = ParallelComponent("ocn", domain=domain_2d)
        coupled = ParallelComponent(
            "coupled",
            subcomponents=(atm, ocn),
            local_constraints=(MaxWastedCoreFractionConstraint(max_fraction=0.0),),
        )
        layouts = list(
            iter_layouts(
                coupled,
                total_cores=10,
                allocations=RootAllocation(
                    subcomponents={
                        "atm": RatioAllocation(weight=3),
                        "ocn": RatioAllocation(weight=2),
                    },
                ),
            )
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
            local_constraints=(MaxWastedCoreFractionConstraint(max_fraction=0.0),),
        )
        layouts = list(
            iter_layouts(
                parent,
                total_cores=6,
                allocations=RootAllocation(
                    subcomponents={
                        "a": RatioAllocation(weight=1),
                        "b": RatioAllocation(weight=1),
                    },
                ),
            )
        )
        assert len(layouts) == 1
        assert layouts[0].sub_layouts[0].n_ranks == 3
        assert layouts[0].sub_layouts[1].n_ranks == 3

    def test_ratio_with_free_sibling_multiple_k(self) -> None:
        # With a free sibling absorbing the remainder, multiple k values are valid.
        # weights 1:1, total_cores=9:
        # k=1: ratio=2, free=7; k=2: ratio=4, free=5; k=3: ratio=6, free=3;
        # k=4: ratio=8, free=1.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        spare = ParallelComponent("spare")
        parent = ParallelComponent("p", subcomponents=(a, b, spare))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=9,
                allocations=RootAllocation(
                    subcomponents={
                        "a": RatioAllocation(weight=1),
                        "b": RatioAllocation(weight=1),
                        "spare": FreeAllocation(min_cores=1),
                    },
                ),
            )
        )
        ratio_totals = {layout.sub_layouts[0].n_ranks + layout.sub_layouts[1].n_ranks for layout in layouts}
        assert 2 in ratio_totals  # k=1
        assert 4 in ratio_totals  # k=2
        assert 6 in ratio_totals  # k=3
        assert 8 in ratio_totals  # k=4

    def test_free_bounds_are_honoured_and_spare_cores_left_idle(self) -> None:
        # Replaces the FreeAllocation class doctest, which could not survive the module
        # split: the allocation module must not reach back into the enumerator.
        comp = ParallelComponent("m", subcomponents=(ParallelComponent("a"),))
        allocations = RootAllocation(subcomponents={"a": FreeAllocation(min_cores=4, max_cores=6)})
        layouts = list(iter_layouts(comp, 8, allocations=allocations))
        assert sorted({layout.sub_layouts[0].n_cores for layout in layouts}) == [4, 5, 6]

    def test_free_allocation_children(self) -> None:
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=4,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FreeAllocation(min_cores=1, max_cores=4),
                        "b": FreeAllocation(min_cores=1, max_cores=4),
                    },
                ),
            )
        )
        # All pairs (r_a, r_b) with 1<=r_a,r_b<=4 and r_a+r_b<=4
        rank_pairs = {tuple(sl.n_ranks for sl in layout.sub_layouts) for layout in layouts}
        assert len(rank_pairs) == 6  # (1,1),(1,2),(1,3),(2,1),(2,2),(3,1)

    def test_infeasible_fixed_children_empty(self) -> None:
        # Fixed ranks (8 + 8 = 16) > total_cores=10
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=10,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FixedAllocation(n_cores=8),
                        "b": FixedAllocation(n_cores=8),
                    },
                ),
            )
        )
        assert layouts == []

    def test_sub_layout_names_match(self) -> None:
        a = ParallelComponent("alpha")
        b = ParallelComponent("beta")
        parent = ParallelComponent("root", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=5,
                allocations=RootAllocation(
                    subcomponents={
                        "alpha": FixedAllocation(n_cores=2),
                        "beta": FixedAllocation(n_cores=3),
                    },
                ),
            )
        )
        assert len(layouts) == 1
        sub_names = [sl.name for sl in layouts[0].sub_layouts]
        assert sub_names == ["alpha", "beta"]

    def test_nested_subcomponents(self) -> None:
        leaf1 = ParallelComponent("leaf1")
        leaf2 = ParallelComponent("leaf2")
        mid = ParallelComponent("mid", subcomponents=(leaf1, leaf2))
        root = ParallelComponent("root", subcomponents=(mid,))
        layouts = list(
            iter_layouts(
                root,
                total_cores=4,
                allocations=RootAllocation(
                    subcomponents={
                        "mid": FixedAllocation(
                            n_cores=4,
                            subcomponents={
                                "leaf1": FixedAllocation(n_cores=2),
                                "leaf2": FixedAllocation(n_cores=2),
                            },
                        ),
                    },
                ),
            )
        )
        assert len(layouts) == 1
        mid_layout = layouts[0].sub_layouts[0]
        assert mid_layout.name == "mid"
        assert len(mid_layout.sub_layouts) == 2

    def test_repeated_subtrees_are_enumerated_once(self, domain_2d: Domain) -> None:
        # Many rank splits give a sub-component the same number of ranks, and the
        # resulting sub-layouts do not depend on what the siblings got.  The
        # enumerator memoises them, so every occurrence of a given sub-layout is
        # literally the same object rather than an equal copy.
        a = ParallelComponent("a", domain=domain_2d)
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))

        layouts = list(iter_layouts(parent, total_cores=6))

        occurrences: dict[tuple[int, tuple[int, ...]], list[ComponentLayout]] = {}
        for layout in layouts:
            a_layout = layout.sub_layouts[0]
            assert a_layout.decomposition is not None
            key = (a_layout.n_ranks, a_layout.decomposition.grid.shape)
            occurrences.setdefault(key, []).append(a_layout)

        assert any(len(group) > 1 for group in occurrences.values()), "expected some sub-layout to recur"
        for group in occurrences.values():
            assert all(item is group[0] for item in group)


# ---------------------------------------------------------------------------
# iter_layouts — constraints filter layouts
# ---------------------------------------------------------------------------


class TestIterLayoutsConstraints:
    def test_uniform_subdomain_filters(self, domain_2d: Domain) -> None:
        # domain=(12,8), 6 ranks: decompositions include (6,1)[12/6=2,8/1=8],
        # (3,2)[12/3=4,8/2=4], (2,3)[12/2=6,8/3=2.67→non-uniform],
        # (1,6)[12/1=12,8/6→non-uniform]
        comp = ParallelComponent(
            "c",
            domain=domain_2d,
            local_constraints=(UniformSubdomainConstraint(),),
        )
        layouts = list(iter_layouts(comp, total_cores=6))
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
        layouts = list(iter_layouts(comp, total_cores=4))
        for layout in layouts:
            d = layout.decomposition
            assert d is not None
            assert all(dim // g >= 3 for dim, g in zip(d.domain.shape, d.grid, strict=True))

    def test_max_rank_fraction_filters(self) -> None:
        a = ParallelComponent("a", local_constraints=(MaxCoreFractionConstraint(0.5),))
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=10,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FreeAllocation(min_cores=1),
                        "b": FreeAllocation(min_cores=1),
                    },
                ),
            )
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
        layouts = list(
            iter_layouts(
                parent,
                total_cores=9,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FreeAllocation(min_cores=1),
                        "b": FreeAllocation(min_cores=1),
                    },
                ),
            )
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
        # total_cores=8, tpr_range=(1,8): only tpr=4 satisfies the constraint
        # (and 8/4=2 ranks)
        layouts = list(iter_layouts(comp, total_cores=8, allocations=RootAllocation(tpr_range=(1, 8))))
        assert all(layout.threads_per_rank == 4 for layout in layouts)
        assert len(layouts) == 1

    def test_strategy_constraint_filters(self, domain_2d: Domain) -> None:
        # domain_2d=(12,8); FixedAllocation(4) → decomps (1,4),(2,2),(4,1).
        # ProcessGridDimEvenConstraint on the AllocationStrategy filters out (1,4) → 2 pass.
        a = ParallelComponent("a", domain=domain_2d)
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=5,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FixedAllocation(n_cores=4, local_constraints=(ProcessGridDimEvenConstraint(dim=0),)),
                        "b": FixedAllocation(n_cores=1),
                    },
                ),
            )
        )
        assert len(layouts) == 2  # (2,2) and (4,1); (1,4) filtered
        for layout in layouts:
            sub_layout = layout.sub_layouts[0]
            assert sub_layout.decomposition is not None
            assert sub_layout.decomposition.grid[0] % 2 == 0

    def test_strategy_group_constraint_filters(self) -> None:
        # group_constraints on AllocationStrategy filter sibling combos;
        # the component itself has no group constraints here.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=9,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FreeAllocation(min_cores=1),
                        "b": FreeAllocation(min_cores=1),
                    },
                    group_constraints=(RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0),),
                ),
            )
        )
        for layout in layouts:
            a_layout = layout.sub_layouts[0]
            b_layout = layout.sub_layouts[1]
            assert a_layout.n_ranks >= 2.0 * b_layout.n_ranks

    def test_strategy_and_component_group_constraints_both_applied(self) -> None:
        # Both component and alloc-spec group constraints must be satisfied simultaneously.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent(
            "p",
            subcomponents=(a, b),
            group_constraints=(RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=2.0),),
        )
        layouts = list(
            iter_layouts(
                parent,
                total_cores=9,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FreeAllocation(min_cores=1),
                        "b": FreeAllocation(min_cores=1),
                    },
                    group_constraints=(RankRatioGroupConstraint(name_a="b", name_b="a", min_ratio=0.25),),
                ),
            )
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
        layouts = list(iter_layouts(comp, total_cores=6))
        for layout in layouts:
            assert layout.decomposition is not None
            assert layout.decomposition.grid[0] % 2 == 0
