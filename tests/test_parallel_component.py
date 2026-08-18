# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_component (layouts, ABCs, tree model)."""

import dataclasses

import pytest

from access.config.parallel_component import ComponentLayout, GroupConstraint, LocalConstraint, ParallelComponent
from access.config.parallel_domain import Domain, DomainDecompositionSpec
from access.config.parallel_mpi_grid import MPICartesianGrid


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(360, 300))


class _AlwaysTrueLocalConstraint(LocalConstraint):
    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        return True


class _AlwaysTrueGroupConstraint(GroupConstraint):
    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_cores: int) -> bool:
        return True


def leaf(name: str = "x", n_ranks: int = 1, tpr: int = 1) -> ComponentLayout:
    """A leaf layout spending exactly ``n_ranks * tpr`` cores."""
    return ComponentLayout(name, n_cores=n_ranks * tpr, n_ranks=n_ranks, threads_per_rank=tpr, decomposition=None)


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

    def test_zero_tpr_raises(self) -> None:
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
        assert constraint.is_satisfied(layout, total_cores=4)

    def test_group_constraint_subclass_runs(self) -> None:
        layouts = (leaf("a", 2), leaf("b", 2))
        constraint = _AlwaysTrueGroupConstraint()
        assert constraint.is_satisfied(layouts, total_cores=4)


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
        component = ParallelComponent("ocean", domain=domain_2d)
        specs = list(component.iter_decompositions(4))
        assert specs == [
            DomainDecompositionSpec(domain_2d, MPICartesianGrid(shape)) for shape in ((1, 4), (2, 2), (4, 1))
        ]
        assert [spec.grid.shape for spec in specs] == [g.shape for g in MPICartesianGrid.iter_grids(2, 4)]

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
