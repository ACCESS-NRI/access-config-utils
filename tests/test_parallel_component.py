# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_component (layouts, ABCs, tree model)."""

import dataclasses
from unittest.mock import patch

import pytest

from access.config import parallel_component
from access.config.parallel_component import (
    ComponentLayout,
    CoreSharing,
    GroupConstraint,
    LocalConstraint,
    ParallelComponent,
    _occupied,
)
from access.config.parallel_domain import Domain, DomainDecompositionSpec
from access.config.parallel_mpi_grid import MPICartesianGrid


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(360, 300))


class _AlwaysTrueLocalConstraint(LocalConstraint):
    def is_satisfied(self, layout: ComponentLayout) -> bool:
        return True


class _AlwaysTrueGroupConstraint(GroupConstraint):
    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...]) -> bool:
        return True


def leaf(name: str = "x", n_ranks: int = 1, threads_per_rank: int = 1) -> ComponentLayout:
    """A leaf layout spending exactly ``n_ranks * threads_per_rank`` cores."""
    return ComponentLayout(
        name, n_cores=n_ranks * threads_per_rank, n_ranks=n_ranks, threads_per_rank=threads_per_rank, decomposition=None
    )


class TestComponentLayout:
    def test_leaf_with_decomposition(self, domain_2d: Domain) -> None:
        decomp = DomainDecompositionSpec(domain_2d, grid=MPICartesianGrid((2, 2)))
        layout = ComponentLayout("comp", n_cores=4, n_ranks=4, threads_per_rank=1, decomposition=decomp)
        assert layout.is_leaf
        assert layout.used_cores == 4
        assert layout.idle_cores == 0

    def test_threaded_leaf(self) -> None:
        layout = ComponentLayout("comp", n_cores=32, n_ranks=8, threads_per_rank=4, decomposition=None)
        assert layout.used_cores == 32
        assert layout.idle_cores == 0

    def test_parent_derives_its_totals(self) -> None:
        # A parent spends its cores through its children and has no thread count itself.
        subs = (leaf("a", 4, 2), leaf("b", 6))
        layout = ComponentLayout(
            "p", n_cores=16, n_ranks=10, threads_per_rank=None, decomposition=None, sub_layouts=subs
        )
        assert not layout.is_leaf
        assert layout.used_cores == 14
        assert layout.idle_cores == 2

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ComponentLayout("", n_cores=4, n_ranks=4, threads_per_rank=1, decomposition=None)

    def test_zero_ranks_raises(self) -> None:
        with pytest.raises(ValueError, match="n_ranks"):
            ComponentLayout("c", n_cores=4, n_ranks=0, threads_per_rank=1, decomposition=None)

    def test_zero_cores_raises(self) -> None:
        with pytest.raises(ValueError, match="n_cores"):
            ComponentLayout("c", n_cores=0, n_ranks=1, threads_per_rank=1, decomposition=None)

    def test_zero_threads_per_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="threads_per_rank"):
            ComponentLayout("c", n_cores=4, n_ranks=4, threads_per_rank=0, decomposition=None)

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            leaf("c", 4).n_ranks = 8  # type: ignore[misc]

    def test_non_str_name_raises(self) -> None:
        with pytest.raises(TypeError, match="name must be a str"):
            ComponentLayout(4, n_cores=4, n_ranks=4, threads_per_rank=1, decomposition=None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field_name", ["n_cores", "n_ranks"])
    def test_non_int_count_raises(self, field_name: str) -> None:
        kwargs: dict[str, object] = {"n_cores": 4, "n_ranks": 4, "threads_per_rank": 1, "decomposition": None}
        kwargs[field_name] = 2.5
        with pytest.raises(TypeError, match=f"{field_name} must be an int"):
            ComponentLayout("c", **kwargs)  # type: ignore[arg-type]

    def test_non_int_threads_per_rank_raises(self) -> None:
        with pytest.raises(TypeError, match="threads_per_rank must be an int or None"):
            ComponentLayout("c", n_cores=4, n_ranks=4, threads_per_rank=2.5, decomposition=None)  # type: ignore[arg-type]

    def test_decomposition_rank_mismatch_raises(self, domain_2d: Domain) -> None:
        # A (2, 2) grid is 4 ranks, which contradicts n_ranks=3.
        decomp = DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 2)))
        with pytest.raises(ValueError, match="uses 4 rank"):
            ComponentLayout("c", n_cores=3, n_ranks=3, threads_per_rank=1, decomposition=decomp)

    def test_leaf_must_spend_its_cores_exactly(self) -> None:
        # 4 ranks x 1 thread is 4 cores, not 8: the missing cores would be unaccounted for.
        with pytest.raises(ValueError, match="but n_cores is 8"):
            ComponentLayout("c", n_cores=8, n_ranks=4, threads_per_rank=1, decomposition=None)

    def test_leaf_without_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="must state threads_per_rank"):
            ComponentLayout("c", n_cores=4, n_ranks=4, threads_per_rank=None, decomposition=None)

    def test_parent_with_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot have threads_per_rank"):
            ComponentLayout(
                "p", n_cores=8, n_ranks=4, threads_per_rank=1, decomposition=None, sub_layouts=(leaf("a", 4),)
            )

    def test_parent_with_decomposition_raises(self, domain_2d: Domain) -> None:
        decomp = DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 2)))
        with pytest.raises(ValueError, match="cannot also hold a decomposition"):
            ComponentLayout(
                "p", n_cores=8, n_ranks=4, threads_per_rank=None, decomposition=decomp, sub_layouts=(leaf("a", 4),)
            )

    def test_duplicate_sub_layout_names_raises(self) -> None:
        subs = (leaf("a"), leaf("a"))
        with pytest.raises(ValueError, match="unique names; duplicates: \\['a'\\]"):
            ComponentLayout("p", n_cores=4, n_ranks=2, threads_per_rank=None, decomposition=None, sub_layouts=subs)

    def test_sub_layouts_exceeding_cores_raises(self) -> None:
        subs = (leaf("a", 3), leaf("b", 3))
        with pytest.raises(ValueError, match="use 6 core\\(s\\) in total"):
            ComponentLayout("p", n_cores=4, n_ranks=6, threads_per_rank=None, decomposition=None, sub_layouts=subs)

    def test_parent_rank_total_must_match(self) -> None:
        subs = (leaf("a", 3),)
        with pytest.raises(ValueError, match="sub-layouts hold 3 rank"):
            ComponentLayout("p", n_cores=8, n_ranks=4, threads_per_rank=None, decomposition=None, sub_layouts=subs)

    def test_parent_may_leave_cores_idle(self) -> None:
        # Under-using the parent's cores is allowed by design;
        # MaxWastedCoreFractionConstraint
        # is what bounds the slack.
        layout = ComponentLayout(
            "p", n_cores=4, n_ranks=1, threads_per_rank=None, decomposition=None, sub_layouts=(leaf("a", 1),)
        )
        assert layout.idle_cores == 3


class TestConstraintAbcs:
    def test_local_constraint_subclass_runs(self, domain_2d: Domain) -> None:
        layout = ComponentLayout(
            "c",
            n_cores=4,
            n_ranks=4,
            threads_per_rank=1,
            decomposition=DomainDecompositionSpec(domain_2d, MPICartesianGrid((2, 2))),
        )
        constraint = _AlwaysTrueLocalConstraint()
        assert constraint.is_satisfied(layout)

    def test_group_constraint_subclass_runs(self) -> None:
        layouts = (leaf("a", 2), leaf("b", 2))
        constraint = _AlwaysTrueGroupConstraint()
        assert constraint.is_satisfied(layouts)


class TestParallelComponent:
    def test_minimal(self) -> None:
        component = ParallelComponent(name="atm")
        assert component.name == "atm"
        assert component.domain is None
        assert component.subcomponents == ()
        assert component.local_constraints == ()
        assert component.group_constraints == ()

    def test_with_domain(self, domain_2d: Domain) -> None:
        component = ParallelComponent(name="ocean", domain=domain_2d)
        assert component.domain is domain_2d

    def test_with_subcomponents(self) -> None:
        atm = ParallelComponent("atm")
        ocn = ParallelComponent("ocn")
        coupled = ParallelComponent("coupled", subcomponents=(atm, ocn))
        assert len(coupled.subcomponents) == 2
        assert coupled.subcomponents[0].name == "atm"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ParallelComponent(name="")

    def test_duplicate_subcomponent_names_raises(self) -> None:
        a1 = ParallelComponent("a")
        a2 = ParallelComponent("a")
        with pytest.raises(ValueError, match="unique names"):
            ParallelComponent("root", subcomponents=(a1, a2))

    def test_domain_with_subcomponents_raises(self, domain_2d: Domain) -> None:
        # Sub-components take disjoint subsets of the parent's cores, so a parent domain
        # would be decomposed over cores its children are also using.
        child = ParallelComponent("child")
        with pytest.raises(ValueError, match="both a domain and sub-components"):
            ParallelComponent("parent", domain=domain_2d, subcomponents=(child,))

    def test_with_constraints(self) -> None:
        local = _AlwaysTrueLocalConstraint()
        group = _AlwaysTrueGroupConstraint()
        component = ParallelComponent("root", local_constraints=(local,), group_constraints=(group,))
        assert component.local_constraints == (local,)
        assert component.group_constraints == (group,)

    def test_frozen(self) -> None:
        component = ParallelComponent("ice")
        with pytest.raises(dataclasses.FrozenInstanceError):
            component.name = "sea_ice"  # type: ignore[misc]


class TestIterDecompositions:
    def test_one_spec_per_grid_in_iter_grids_order(self, domain_2d: Domain) -> None:
        # Which grids exist for a rank count is parallel_mpi_grid's arithmetic, asserted in
        # its own test file. Supply them here, so this checks only what iter_decompositions
        # itself contributes: one spec per grid, in order, each carrying the component's
        # own domain.
        component = ParallelComponent("ocean", domain=domain_2d)
        supplied = [MPICartesianGrid((3, 2)), MPICartesianGrid((6, 1))]

        with patch.object(parallel_component, "parallel_mpi_grid") as grid_module:
            grid_module.MPICartesianGrid.iter_grids.return_value = iter(supplied)
            specs = list(component.iter_decompositions(6))
            # It must ask for grids of the domain's dimensionality and the rank count given.
            grid_module.MPICartesianGrid.iter_grids.assert_called_once_with(domain_2d.ndim, 6)

        assert specs == [DomainDecompositionSpec(domain_2d, grid) for grid in supplied]

    def test_every_spec_carries_the_components_own_domain(self, domain_2d: Domain) -> None:
        component = ParallelComponent("ocean", domain=domain_2d)
        assert all(spec.domain is domain_2d for spec in component.iter_decompositions(6))

    def test_a_component_without_a_domain_yields_none_once(self) -> None:
        # The single None is what lets the enumerator treat the decomposition as one more
        # dimension of the search rather than special-casing domainless components.
        assert list(ParallelComponent("coupler").iter_decompositions(4)) == [None]

    def test_a_bad_rank_count_is_reported_on_first_advance(self, domain_2d: Domain) -> None:
        # A plain generator, like MPICartesianGrid.iter_grids: creating it checks nothing.
        specs = ParallelComponent("ocean", domain=domain_2d).iter_decompositions(0)
        with pytest.raises(ValueError, match="n_ranks must be >= 1"):
            next(specs)


def _leaf(name: str, n_cores: int) -> ComponentLayout:
    """A single-threaded leaf holding *n_cores* cores, for building parents by hand."""
    return ComponentLayout(name=name, n_cores=n_cores, n_ranks=n_cores, threads_per_rank=1, decomposition=None)


class TestSharedCores:
    """A parent whose sub-components take turns on its cores instead of dividing them."""

    def _pool(self, n_cores: int, children: tuple[ComponentLayout, ...]) -> ComponentLayout:
        return ComponentLayout(
            name="pool",
            n_cores=n_cores,
            n_ranks=_occupied((child.core_offset, child.n_ranks) for child in children),
            threads_per_rank=None,
            decomposition=None,
            sub_layouts=children,
            core_sharing=CoreSharing.SHARED,
        )

    def test_holds_its_largest_child_not_their_total(self) -> None:
        pool = self._pool(24, (_leaf("ice", 24), _leaf("atm", 12), _leaf("rof", 12)))
        assert pool.used_cores == 24
        assert pool.idle_cores == 0
        assert pool.n_ranks == 24

    def test_counts_cores_left_over_as_idle(self) -> None:
        pool = self._pool(32, (_leaf("ice", 24), _leaf("atm", 12)))
        assert pool.used_cores == 24
        assert pool.idle_cores == 8

    def test_rejects_a_child_larger_than_the_parent(self) -> None:
        with pytest.raises(ValueError, match="reach 40 core"):
            self._pool(24, (_leaf("ice", 40),))

    def test_rejects_a_rank_total_that_is_not_the_largest_child(self) -> None:
        with pytest.raises(ValueError, match="share its cores"):
            ComponentLayout(
                name="pool",
                n_cores=24,
                n_ranks=36,  # the sum of the children, which is the partitioned rule
                threads_per_rank=None,
                decomposition=None,
                sub_layouts=(_leaf("ice", 24), _leaf("atm", 12)),
                core_sharing=CoreSharing.SHARED,
            )

    def test_partitioned_is_the_default_and_still_sums(self) -> None:
        children = (_leaf("atm", 24), _leaf("ocn", 12))
        parent = ComponentLayout(
            name="model", n_cores=64, n_ranks=36, threads_per_rank=None, decomposition=None, sub_layouts=children
        )
        assert parent.core_sharing is CoreSharing.PARTITIONED
        assert parent.used_cores == 36
        assert parent.idle_cores == 28

    def test_reaches_as_far_as_its_furthest_child_not_its_biggest(self) -> None:
        """A child starting partway in can need more of the range than the largest one."""

        pool = self._pool(28, (_leaf("ice", 24), dataclasses.replace(_leaf("rof", 12), core_offset=16)))
        assert pool.used_cores == 28, "rof starts at 16 and runs 12, so the range has to reach 28"
        assert pool.idle_cores == 0
        assert pool.n_ranks == 28

    def test_counts_only_the_cores_its_children_sit_on(self) -> None:
        """Cores before the first child are idle, not spent.

        Every child starting partway into the range leaves a gap at the front that belongs
        to nobody. Measuring how far the children reach would count it as spent.
        """

        pool = self._pool(
            24,
            (
                dataclasses.replace(_leaf("a", 12), core_offset=4),
                dataclasses.replace(_leaf("b", 10), core_offset=8),
            ),
        )
        assert pool.used_cores == 14, "the children cover cores 4 to 17"
        assert pool.idle_cores == 10, "cores 0 to 3 and 18 to 23 belong to nobody"
        assert pool.n_ranks == 14

    def test_counts_a_gap_between_two_children_as_idle(self) -> None:
        pool = self._pool(12, (_leaf("a", 4), dataclasses.replace(_leaf("b", 4), core_offset=8)))
        assert pool.used_cores == 8, "cores 4 to 7 lie between them and are spent by neither"
        assert pool.idle_cores == 4

    def test_counts_a_core_two_children_share_once(self) -> None:
        pool = self._pool(24, (_leaf("a", 24), dataclasses.replace(_leaf("b", 12), core_offset=8)))
        assert pool.used_cores == 24, "b sits inside a's range, so it adds nothing"
        assert pool.idle_cores == 0

    def test_rejects_a_child_reaching_past_the_parent(self) -> None:
        with pytest.raises(ValueError, match="reach 28 core"):
            self._pool(24, (_leaf("ice", 24), dataclasses.replace(_leaf("rof", 12), core_offset=16)))

    def test_rejects_an_offset_on_a_multi_threaded_child(self) -> None:
        """An offset is a core index, so it only means a PE index at one core per rank."""

        threaded = ComponentLayout(
            name="ice", n_cores=24, n_ranks=12, threads_per_rank=2, decomposition=None, core_offset=4
        )
        with pytest.raises(ValueError, match="only means the same thing as a PE index"):
            self._pool(28, (threaded,))

    def test_partitioned_parents_place_their_children_themselves(self) -> None:
        with pytest.raises(ValueError, match="cannot be placed"):
            ComponentLayout(
                name="model",
                n_cores=64,
                n_ranks=36,
                threads_per_rank=None,
                decomposition=None,
                sub_layouts=(_leaf("atm", 24), dataclasses.replace(_leaf("ocn", 12), core_offset=24)),
            )

    def test_component_rejects_offsets_under_a_partitioned_parent(self) -> None:
        with pytest.raises(ValueError, match="Offsets only mean something"):
            ParallelComponent("model", subcomponents=(ParallelComponent("atm", core_offset=4),))

    def test_component_rejects_a_negative_offset(self) -> None:
        with pytest.raises(ValueError, match="core_offset must be >= 0"):
            ParallelComponent("atm", core_offset=-1)

    def test_layout_rejects_a_negative_offset(self) -> None:
        # The layout validates the bound itself rather than trusting the component it
        # resolves: it is public, and a caller may build one by hand.
        with pytest.raises(ValueError, match="core_offset must be >= 0"):
            ComponentLayout(name="ice", n_cores=4, n_ranks=4, threads_per_rank=1, decomposition=None, core_offset=-1)

    def test_component_declares_the_rule(self) -> None:
        assert ParallelComponent("model").core_sharing is CoreSharing.PARTITIONED
        pool = ParallelComponent("pool", subcomponents=(ParallelComponent("ice"),), core_sharing=CoreSharing.SHARED)
        assert pool.core_sharing is CoreSharing.SHARED
