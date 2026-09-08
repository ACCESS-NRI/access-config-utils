# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_layouts (allocation strategies and enumeration).

This file is deliberately **not** isolated from the rest of the parallel half, and is the
integration suite for it, as ``test_parser.py`` is for the parser half. The other
``parallel_*`` test files mock their collaborators, so something has to drive the real stack
end to end: real component trees, real allocation strategies, real domains and grids. If the
layouts enumerated here are still right, the isolation elsewhere has not hidden a break.

The constraint doubles below are the one exception, and are about subject rather than
isolation: these tests are for the enumerator, so borrowing a rule from
``parallel_constraints`` would test two things at once.
"""

import logging
from dataclasses import dataclass

import pytest

from access.config.parallel_allocation_strategies import (
    AllocationStrategy,
    FixedAllocation,
    FreeAllocation,
    RootAllocation,
    WeightedAllocation,
)
from access.config.parallel_component import (
    ComponentLayout,
    GroupConstraint,
    LocalConstraint,
    ParallelComponent,
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
# Constraint doubles
# ---------------------------------------------------------------------------
#
# These tests are about the enumerator, not about the constraint library, so they use
# doubles implementing the ABCs rather than anything from parallel_constraints - whose own
# behaviour is test_parallel_constraints.py's business. Each double exists to show that the
# enumerator hands the right information to the right kind of constraint.
#
# Those carrying fields are frozen dataclasses because a constraint ends up inside
# ParallelComponent.local_constraints, which is frozen, hashable, and part of the search's
# memo key: an unhashable double would break the search rather than the assertion. They are
# deliberately dumber than the real constraints, with no precondition checks.


class _RejectAll(LocalConstraint):
    """Satisfied by nothing, so the search must come back empty."""

    def is_satisfied(self, layout: ComponentLayout) -> bool:
        return False


@dataclass(frozen=True)
class _GridDimEven(LocalConstraint):
    """Needs the decomposition the enumerator attached to the candidate."""

    dim: int

    def is_satisfied(self, layout: ComponentLayout) -> bool:
        assert layout.decomposition is not None
        return layout.decomposition.grid[self.dim] % 2 == 0


@dataclass(frozen=True)
class _ThreadsEqual(LocalConstraint):
    """Needs the thread count only a leaf has."""

    n_threads: int

    def is_satisfied(self, layout: ComponentLayout) -> bool:
        return layout.threads_per_rank == self.n_threads


class _NoIdleCores(LocalConstraint):
    """Needs a parent's idle-core accounting: every core must be handed out."""

    def is_satisfied(self, layout: ComponentLayout) -> bool:
        return layout.idle_cores == 0


@dataclass(frozen=True)
class _MinRankRatio(GroupConstraint):
    """Needs every sibling's layout, found by name rather than by position."""

    name_a: str
    name_b: str
    factor: float

    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...]) -> bool:
        a = next(lay for lay in sub_layouts if lay.name == self.name_a)
        b = next(lay for lay in sub_layouts if lay.name == self.name_b)
        return a.n_ranks >= self.factor * b.n_ranks


# ---------------------------------------------------------------------------
# iter_layouts — leaf components
# ---------------------------------------------------------------------------


class TestIterLayoutsLeaf:
    def test_leaf_no_domain(self, leaf_no_domain: ParallelComponent) -> None:
        # Free allocation with default min_cores=1; total_cores=4, threads_per_rank=1
        # → n_ranks=4, no decomp
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

    def test_thread_range(self) -> None:
        # total_cores=8, thread_range=(1,2): threads_per_rank=1→8 ranks,
        # threads_per_rank=2→4 ranks
        comp = ParallelComponent("c")
        layouts = list(iter_layouts(comp, total_cores=8, allocations=RootAllocation(thread_range=(1, 2))))
        thread_counts = {layout.threads_per_rank for layout in layouts}
        assert 1 in thread_counts
        assert 2 in thread_counts

    def test_threads_per_rank_not_divisor_skipped(self) -> None:
        # total_cores=9, thread_range=(1,4): only threads_per_rank=1,3,9 divide 9
        comp = ParallelComponent("c")
        layouts = list(iter_layouts(comp, total_cores=9, allocations=RootAllocation(thread_range=(1, 4))))
        thread_counts = {layout.threads_per_rank for layout in layouts}
        assert 2 not in thread_counts
        assert 4 not in thread_counts
        assert 1 in thread_counts
        assert 3 in thread_counts

    def test_fixed_ranks_single_layout(self) -> None:
        # The root receives the whole budget.
        comp = ParallelComponent("c")
        layouts = list(iter_layouts(comp, total_cores=4))
        assert len(layouts) == 1
        assert layouts[0].n_ranks == 4

    def test_leaf_spends_its_cores_exactly(self) -> None:
        comp = ParallelComponent("c")
        for layout in iter_layouts(comp, total_cores=12, allocations=RootAllocation(thread_range=(1, 4))):
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
            WeightedAllocation(weight=2),
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
                allocations=RootAllocation(local_constraints=(_GridDimEven(dim=0),)),
            )
        )
        assert layouts
        for layout in layouts:
            assert layout.decomposition is not None
            assert layout.decomposition.grid[0] % 2 == 0

    def test_an_invalid_thread_range_is_rejected_before_the_search(self) -> None:
        # The thread range belongs to the root strategy now, so a bad one never reaches
        # the enumerator; RootAllocation itself refuses to be built. See
        # test_parallel_allocation_strategies for the range rules themselves.
        with pytest.raises(ValueError, match="AllocationStrategy.thread_range"):
            RootAllocation(thread_range=(3, 1))


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
                        "UM7": FixedAllocation(n_cores=16, thread_range=(2, 2)),
                        "MOM5": FixedAllocation(n_cores=16, thread_range=(1, 1)),
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

    def test_thread_range_is_inherited_and_overridden(self) -> None:
        threaded = ParallelComponent("threaded")
        plain = ParallelComponent("plain")
        root = ParallelComponent("root", subcomponents=(threaded, plain))

        layouts = list(
            iter_layouts(
                root,
                total_cores=8,
                allocations=RootAllocation(
                    thread_range=(1, 1),  # the inherited default
                    subcomponents={
                        "threaded": FixedAllocation(n_cores=4, thread_range=(2, 2)),  # overrides it
                        "plain": FixedAllocation(n_cores=4),  # inherits it
                    },
                ),
            )
        )

        assert len(layouts) == 1
        threaded_layout, plain_layout = layouts[0].sub_layouts
        assert threaded_layout.threads_per_rank == 2
        assert plain_layout.threads_per_rank == 1

    def test_a_parent_thread_range_reaches_its_descendants(self) -> None:
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
                            thread_range=(4, 4),
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
        allocations = RootAllocation(thread_range=(1, 2), subcomponents={"a": FreeAllocation()})
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
                allocations=RootAllocation(thread_range=(1, 4), subcomponents={"c": FixedAllocation(n_cores=9)}),
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
        comp = ParallelComponent("c", local_constraints=(_RejectAll(),), domain=Domain((4, 4)))
        assert list(iter_layouts(comp, total_cores=4)) == []


# ---------------------------------------------------------------------------
# Diagnostics for an empty result
# ---------------------------------------------------------------------------


class TestEmptyResultDiagnostics:
    def test_unsatisfiable_constraint_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        comp = ParallelComponent("c", local_constraints=(_RejectAll(),))
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
            list(iter_layouts(comp, total_cores=9, allocations=RootAllocation(thread_range=(1, 2))))
        assert "component 'c' cannot use 2 thread(s) per rank: it does not divide its 9 core(s)" in caplog.text

    def test_unsatisfiable_subcomponent_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        # A leaf that cannot be laid out names itself, so the failing node is identifiable
        # even in a tree where the parent has options.
        child = ParallelComponent("child", domain=Domain((4, 4)), local_constraints=(_RejectAll(),))
        parent = ParallelComponent("p", subcomponents=(child,))
        with caplog.at_level(logging.DEBUG, logger="access.config.parallel_layouts"):
            assert list(iter_layouts(parent, total_cores=2)) == []
        assert "no valid layout for component 'child' on 1 core(s) with 1-1 thread(s) per rank" in caplog.text

    def test_quiet_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        comp = ParallelComponent("c", local_constraints=(_RejectAll(),))
        with caplog.at_level(logging.WARNING, logger="access.config.parallel_layouts"):
            assert list(iter_layouts(comp, total_cores=4)) == []
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# iter_layouts — fractional bounds
# ---------------------------------------------------------------------------


class TestIterLayoutsFractionalBounds:
    """One strategy tree, several budgets - what a scaling study needs."""

    @pytest.mark.parametrize(("total_cores", "expected_atm"), [(16, 8), (64, 32), (160, 80)])
    def test_one_strategy_serves_every_budget(self, domain_2d: Domain, total_cores: int, expected_atm: int) -> None:
        # The same object at every size: written in absolute cores this would need three
        # strategies, or a function rebuilding one per budget.
        atm = ParallelComponent("atm", domain=domain_2d)
        ocn = ParallelComponent("ocn", domain=domain_2d)
        coupled = ParallelComponent("coupled", subcomponents=(atm, ocn))
        allocations = RootAllocation(
            subcomponents={
                "atm": FixedAllocation(core_fraction=0.5),
                "ocn": FreeAllocation(min_core_fraction=0.25, max_core_fraction=0.5),
            }
        )

        layouts = list(iter_layouts(coupled, total_cores=total_cores, allocations=allocations))

        assert layouts
        for layout in layouts:
            atm_sub = next(sl for sl in layout.sub_layouts if sl.name == "atm")
            ocn_sub = next(sl for sl in layout.sub_layouts if sl.name == "ocn")
            assert atm_sub.n_cores == expected_atm
            assert total_cores // 4 <= ocn_sub.n_cores <= total_cores // 2

    def test_the_caller_keeps_their_strategy_object(self, domain_2d: Domain) -> None:
        # Resolution rebuilds the tree it is handed; it must not touch the caller's.
        allocations = RootAllocation(subcomponents={"atm": FixedAllocation(core_fraction=0.5)})
        comp = ParallelComponent("coupled", subcomponents=(ParallelComponent("atm", domain=domain_2d),))

        assert list(iter_layouts(comp, total_cores=16, allocations=allocations))
        assert allocations.subcomponents["atm"] == FixedAllocation(core_fraction=0.5)

    def test_bounds_that_cross_on_this_budget_are_reported_eagerly(self, domain_2d: Domain) -> None:
        # Like every other argument error, this surfaces on the call rather than on the
        # first layout requested.
        comp = ParallelComponent("coupled", subcomponents=(ParallelComponent("atm", domain=domain_2d),))
        allocations = RootAllocation(subcomponents={"atm": FreeAllocation(min_cores=64, max_core_fraction=0.1)})

        with pytest.raises(ValueError, match="admits no core count"):
            iter_layouts(comp, total_cores=16, allocations=allocations)


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
        assert len(layouts) == 12
        for layout in layouts:
            atm_sub = next(sl for sl in layout.sub_layouts if sl.name == "atm")
            ocn_sub = next(sl for sl in layout.sub_layouts if sl.name == "ocn")
            assert atm_sub.n_ranks == 6
            assert ocn_sub.n_ranks == 4

    def test_weighted_children_exact_split(self, domain_2d: Domain) -> None:
        # weights 3:2 with 10 ranks and no waste → k=2: 2*3=6, 2*2=4.
        # Requiring no idle cores is what forces exact consumption.
        atm = ParallelComponent("atm", domain=domain_2d)
        ocn = ParallelComponent("ocn", domain=domain_2d)
        coupled = ParallelComponent(
            "coupled",
            subcomponents=(atm, ocn),
            local_constraints=(_NoIdleCores(),),
        )
        layouts = list(
            iter_layouts(
                coupled,
                total_cores=10,
                allocations=RootAllocation(
                    subcomponents={
                        "atm": WeightedAllocation(weight=3),
                        "ocn": WeightedAllocation(weight=2),
                    },
                ),
            )
        )
        assert len(layouts) == 12
        for layout in layouts:
            atm_sub = next(sl for sl in layout.sub_layouts if sl.name == "atm")
            ocn_sub = next(sl for sl in layout.sub_layouts if sl.name == "ocn")
            assert atm_sub.n_ranks == 6
            assert ocn_sub.n_ranks == 4

    def test_weighted_only_unique_k(self) -> None:
        # weights 1:1, total_cores=6, no waste allowed → only k=3 (3+3=6). The smaller
        # multipliers this rules out are the subject of the next test.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent(
            "p",
            subcomponents=(a, b),
            local_constraints=(_NoIdleCores(),),
        )
        layouts = list(
            iter_layouts(
                parent,
                total_cores=6,
                allocations=RootAllocation(
                    subcomponents={
                        "a": WeightedAllocation(weight=1),
                        "b": WeightedAllocation(weight=1),
                    },
                ),
            )
        )
        assert len(layouts) == 1
        assert layouts[0].sub_layouts[0].n_ranks == 3
        assert layouts[0].sub_layouts[1].n_ranks == 3

    def test_weighted_leaves_cores_idle_when_nothing_forbids_it(self) -> None:
        # The same weights and budget as above, but with no constraint on idle cores, so
        # every multiplier that fits survives and the smaller ones simply waste cores.
        # Weighted siblings pin their proportion to each other, not to the budget.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=6,
                allocations=RootAllocation(
                    subcomponents={
                        "a": WeightedAllocation(weight=1),
                        "b": WeightedAllocation(weight=1),
                    },
                ),
            )
        )
        # k = 1, 2, 3 → (1, 1), (2, 2), (3, 3), leaving 4, 2 and 0 cores idle.
        by_share = {layout.sub_layouts[0].n_cores: layout for layout in layouts}
        assert sorted(by_share) == [1, 2, 3]
        for share, layout in by_share.items():
            assert layout.sub_layouts[1].n_cores == share  # the 1:1 proportion holds
            assert layout.idle_cores == 6 - 2 * share

    def test_weighted_with_free_sibling_multiple_k(self) -> None:
        # With a free sibling absorbing the remainder, multiple k values are valid.
        # weights 1:1, total_cores=9:
        # k=1: weighted=2, free=7; k=2: weighted=4, free=5; k=3: weighted=6, free=3;
        # k=4: weighted=8, free=1.
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
                        "a": WeightedAllocation(weight=1),
                        "b": WeightedAllocation(weight=1),
                        "spare": FreeAllocation(min_cores=1),
                    },
                ),
            )
        )
        weighted_components_totals = {
            layout.sub_layouts[0].n_ranks + layout.sub_layouts[1].n_ranks for layout in layouts
        }
        assert 2 in weighted_components_totals  # k=1
        assert 4 in weighted_components_totals  # k=2
        assert 6 in weighted_components_totals  # k=3
        assert 8 in weighted_components_totals  # k=4

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
        assert mid_layout.sub_layouts[0].name == "leaf1"
        assert mid_layout.sub_layouts[1].name == "leaf2"
        assert mid_layout.sub_layouts[0].n_ranks == 2
        assert mid_layout.sub_layouts[1].n_ranks == 2

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
    def test_max_core_fraction_caps_a_component(self) -> None:
        # A cap stated as a share of the whole budget is an allocation bound, resolved to a
        # core count before the search starts, rather than a constraint on finished layouts.
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=10,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FreeAllocation(min_cores=1, max_core_fraction=0.5),
                        "b": FreeAllocation(min_cores=1),
                    },
                ),
            )
        )
        assert layouts
        # a may take at most 5 of the 10 cores, and at 1 thread per rank that caps its ranks
        for layout in layouts:
            a_layout = next(sl for sl in layout.sub_layouts if sl.name == "a")
            assert a_layout.n_cores <= 5
            assert a_layout.n_ranks <= 5
        # the cap must actually bind, or the test proves nothing
        assert max(next(sl for sl in lay.sub_layouts if sl.name == "a").n_cores for lay in layouts) == 5

    def test_group_constraint_filters(self) -> None:
        a = ParallelComponent("a")
        b = ParallelComponent("b")
        # a must have at least 2× ranks of b
        parent = ParallelComponent(
            "p",
            subcomponents=(a, b),
            group_constraints=(_MinRankRatio(name_a="a", name_b="b", factor=2.0),),
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

    def test_a_local_constraint_can_filter_on_thread_count(self) -> None:
        comp = ParallelComponent(
            "c",
            local_constraints=(_ThreadsEqual(n_threads=4),),
        )
        # total_cores=8, thread_range=(1,8): only threads_per_rank=4 satisfies the
        # constraint (and 8/4=2 ranks)
        layouts = list(iter_layouts(comp, total_cores=8, allocations=RootAllocation(thread_range=(1, 8))))
        assert all(layout.threads_per_rank == 4 for layout in layouts)
        assert len(layouts) == 1

    def test_strategy_constraint_filters(self, domain_2d: Domain) -> None:
        # domain_2d=(12,8); FixedAllocation(4) → decomps (1,4),(2,2),(4,1).
        # A grid constraint on the AllocationStrategy filters out (1,4) → 2 pass.
        a = ParallelComponent("a", domain=domain_2d)
        b = ParallelComponent("b")
        parent = ParallelComponent("p", subcomponents=(a, b))
        layouts = list(
            iter_layouts(
                parent,
                total_cores=5,
                allocations=RootAllocation(
                    subcomponents={
                        "a": FixedAllocation(n_cores=4, local_constraints=(_GridDimEven(dim=0),)),
                        "b": FixedAllocation(n_cores=1),
                    },
                ),
            )
        )
        assert len(layouts) == 2  # (2,2) and (4,1); (1,4) filtered
        for layout in layouts:
            sub_layout = next(sl for sl in layout.sub_layouts if sl.name == "a")
            assert sub_layout.decomposition is not None
            assert sub_layout.decomposition.grid[0] % 2 == 0
            # The constraint filters a's grid without disturbing its sibling.
            b_layout = next(sl for sl in layout.sub_layouts if sl.name == "b")
            assert (b_layout.n_cores, b_layout.n_ranks) == (1, 1)

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
                    group_constraints=(_MinRankRatio(name_a="a", name_b="b", factor=2.0),),
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
            group_constraints=(_MinRankRatio(name_a="a", name_b="b", factor=2.0),),
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
                    group_constraints=(_MinRankRatio(name_a="b", name_b="a", factor=0.25),),
                ),
            )
        )
        for layout in layouts:
            a_layout = layout.sub_layouts[0]
            b_layout = layout.sub_layouts[1]
            assert a_layout.n_ranks >= 2.0 * b_layout.n_ranks
            assert b_layout.n_ranks >= 0.25 * a_layout.n_ranks

    def test_a_local_constraint_can_filter_on_the_grid(self) -> None:
        domain = Domain(shape=(12, 8))
        comp = ParallelComponent(
            "c",
            domain=domain,
            local_constraints=(_GridDimEven(dim=0),),
        )
        layouts = list(iter_layouts(comp, total_cores=6))
        for layout in layouts:
            assert layout.decomposition is not None
            assert layout.decomposition.grid[0] % 2 == 0
