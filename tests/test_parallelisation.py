# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallelisation (allocations, ParallelComponent)."""

import dataclasses

import pytest

from access.config.domain_parallelisation import Domain
from access.config.parallel_component import ParallelComponent
from access.config.parallelisation import AllocationStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(360, 300))


# ---------------------------------------------------------------------------
# RankAllocation types
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ParallelComponent
# ---------------------------------------------------------------------------


class TestParallelComponent:
    def test_minimal(self) -> None:
        c = ParallelComponent(name="atm")
        assert c.name == "atm"
        assert c.domain is None
        assert c.subcomponents == ()
        assert c.local_constraints == ()
        assert c.group_constraints == ()

    def test_with_domain(self, domain_2d: Domain) -> None:
        c = ParallelComponent(name="ocean", domain=domain_2d)
        assert c.domain is domain_2d

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

    def test_frozen(self) -> None:
        c = ParallelComponent("ice")
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.name = "sea_ice"  # type: ignore[misc]


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
        from access.config.constraints import FixedThreadsPerRankConstraint

        c = FixedThreadsPerRankConstraint(n_threads=1)
        spec = AllocationStrategy(local_constraints=(c,))
        assert spec.local_constraints == (c,)

    def test_group_constraints_default_empty(self) -> None:
        spec = AllocationStrategy()
        assert spec.group_constraints == ()

    def test_with_group_constraints(self) -> None:
        from access.config.constraints import RankRatioGroupConstraint

        gc = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=1.0)
        spec = AllocationStrategy(group_constraints=(gc,))
        assert spec.group_constraints == (gc,)

    def test_frozen(self) -> None:
        spec = AllocationStrategy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.n_ranks = 4  # type: ignore[misc]

    def test_hash_includes_structure(self) -> None:
        child = AllocationStrategy(n_ranks=2)
        spec_a = AllocationStrategy(subcomponents={"child": child})
        spec_b = AllocationStrategy(subcomponents={"child": child})
        assert hash(spec_a) == hash(spec_b)
