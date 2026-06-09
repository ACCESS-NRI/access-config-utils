# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_component (layouts, ABCs, tree model)."""

import dataclasses

import pytest

from access.config.parallel_component import ComponentLayout, GroupConstraint, LocalConstraint, ParallelComponent
from access.config.parallel_domain import Domain, DomainCartesianDecomposition


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(360, 300))


class _AlwaysTrueLocalConstraint(LocalConstraint):
    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return True


class _AlwaysTrueGroupConstraint(GroupConstraint):
    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_ranks: int) -> bool:
        return True


class TestComponentLayout:
    def test_basic(self, domain_2d: Domain) -> None:
        decomp = DomainCartesianDecomposition(domain_2d, grid=(2, 2))
        layout = ComponentLayout("comp", n_ranks=4, threads_per_rank=1, decomposition=decomp)
        assert layout.total_cores == 4

    def test_without_decomposition(self) -> None:
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

    def test_frozen(self) -> None:
        layout = ComponentLayout("c", n_ranks=4, threads_per_rank=1, decomposition=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            layout.n_ranks = 8  # type: ignore[misc]


class TestConstraintAbcs:
    def test_local_constraint_subclass_runs(self, domain_2d: Domain) -> None:
        layout = ComponentLayout(
            "c", n_ranks=4, threads_per_rank=1, decomposition=DomainCartesianDecomposition(domain_2d, (2, 2))
        )
        constraint = _AlwaysTrueLocalConstraint()
        assert constraint.is_satisfied(layout, total_ranks=4)

    def test_group_constraint_subclass_runs(self) -> None:
        layouts = (
            ComponentLayout("a", n_ranks=2, threads_per_rank=1, decomposition=None),
            ComponentLayout("b", n_ranks=2, threads_per_rank=1, decomposition=None),
        )
        constraint = _AlwaysTrueGroupConstraint()
        assert constraint.is_satisfied(layouts, total_ranks=4)


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
