# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallelisation (allocations, ParallelComponent)."""

import dataclasses

import pytest

from access.config.domain_parallelisation import Domain
from access.config.parallelisation import (
    AllocationStrategy,
    FixedAllocation,
    FreeAllocation,
    ParallelComponent,
    RatioAllocation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def domain_2d() -> Domain:
    return Domain(shape=(360, 300))


# ---------------------------------------------------------------------------
# RankAllocation types
# ---------------------------------------------------------------------------


class TestFixedRanks:
    def test_valid(self) -> None:
        fr = FixedAllocation(n_ranks=16)
        assert fr.n_ranks == 16

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            FixedAllocation(n_ranks=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            FixedAllocation(n_ranks=-4)

    def test_frozen(self) -> None:
        fr = FixedAllocation(n_ranks=8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fr.n_ranks = 4  # type: ignore[misc]


class TestRatioAllocation:
    def test_valid(self) -> None:
        ra = RatioAllocation(weight=3)
        assert ra.weight == 3

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RatioAllocation(weight=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RatioAllocation(weight=-1)

    def test_frozen(self) -> None:
        ra = RatioAllocation(weight=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ra.weight = 5  # type: ignore[misc]


class TestFreeAllocation:
    def test_defaults(self) -> None:
        fa = FreeAllocation()
        assert fa.min_ranks == 1
        assert fa.max_ranks is None

    def test_explicit_bounds(self) -> None:
        fa = FreeAllocation(min_ranks=4, max_ranks=16)
        assert fa.min_ranks == 4
        assert fa.max_ranks == 16

    def test_equal_bounds(self) -> None:
        fa = FreeAllocation(min_ranks=8, max_ranks=8)
        assert fa.min_ranks == fa.max_ranks == 8

    def test_zero_min_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            FreeAllocation(min_ranks=0)

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ranks"):
            FreeAllocation(min_ranks=4, max_ranks=2)

    def test_frozen(self) -> None:
        fa = FreeAllocation(min_ranks=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fa.min_ranks = 5  # type: ignore[misc]


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
# AllocationSpec
# ---------------------------------------------------------------------------


class TestAllocationSpec:
    def test_leaf_fixed(self) -> None:
        spec = AllocationStrategy(FixedAllocation(12))
        assert spec.allocation == FixedAllocation(12)
        assert spec.subcomponents == {}

    def test_leaf_free(self) -> None:
        spec = AllocationStrategy(FreeAllocation())
        assert isinstance(spec.allocation, FreeAllocation)

    def test_leaf_ratio(self) -> None:
        spec = AllocationStrategy(RatioAllocation(weight=3))
        assert spec.allocation.weight == 3

    def test_with_subcomponents(self) -> None:
        child_a = AllocationStrategy(FreeAllocation())
        child_b = AllocationStrategy(FixedAllocation(12))
        parent = AllocationStrategy(FreeAllocation(), subcomponents={"a": child_a, "b": child_b})
        assert len(parent.subcomponents) == 2
        assert parent.subcomponents["b"].allocation == FixedAllocation(12)

    def test_with_local_constraints(self) -> None:
        from access.config.layouts import FixedThreadsPerRankConstraint

        c = FixedThreadsPerRankConstraint(n_threads=1)
        spec = AllocationStrategy(FreeAllocation(), local_constraints=(c,))
        assert spec.local_constraints == (c,)

    def test_group_constraints_default_empty(self) -> None:
        spec = AllocationStrategy(FreeAllocation())
        assert spec.group_constraints == ()

    def test_with_group_constraints(self) -> None:
        from access.config.layouts import RankRatioGroupConstraint

        gc = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=1.0)
        spec = AllocationStrategy(FreeAllocation(), group_constraints=(gc,))
        assert spec.group_constraints == (gc,)

    def test_frozen(self) -> None:
        spec = AllocationStrategy(FreeAllocation())
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.allocation = FixedAllocation(4)  # type: ignore[misc]

    def test_hash_includes_structure(self) -> None:
        child = AllocationStrategy(FixedAllocation(2))
        spec_a = AllocationStrategy(FreeAllocation(), subcomponents={"child": child})
        spec_b = AllocationStrategy(FreeAllocation(), subcomponents={"child": child})
        assert hash(spec_a) == hash(spec_b)
