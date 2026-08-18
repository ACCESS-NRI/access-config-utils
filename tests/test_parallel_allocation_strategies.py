# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for access.config.parallel_allocation_strategies (the allocation modes)."""

import dataclasses
import logging
import typing

import pytest

from access.config.parallel_allocation_strategies import (
    AllocationStrategy,
    FixedAllocation,
    FreeAllocation,
    RatioAllocation,
    RootAllocation,
    iter_core_splits,
    resolve_root_strategy,
)
from access.config.parallel_component import ParallelComponent

# ---------------------------------------------------------------------------
# Sibling-group scheduling
# ---------------------------------------------------------------------------


def _splits(strategy: AllocationStrategy, names: tuple[str, ...], parent_cores: int) -> list[tuple[int, ...]]:
    """Resolve *names* against *strategy* and split *parent_cores* among them."""
    return list(iter_core_splits(strategy.strategies_for(names), parent_cores, names))


class TestIterCoreSplits:
    def test_splits_are_positional_and_named(self) -> None:
        """Splits line up with the names given, whatever order the modes appear in."""
        strategy = FreeAllocation(
            subcomponents={
                "a": FixedAllocation(n_cores=2),
                "b": FreeAllocation(),
                "c": FreeAllocation(min_cores=1),
            }
        )
        splits = _splits(strategy, ("a", "b", "c"), 6)

        # "a" is pinned at 2; "b" and "c" divide what is left.
        assert all(split[0] == 2 for split in splits)
        assert all(sum(split) <= 6 for split in splits)
        assert (2, 1, 3) in splits

    def test_ratio_siblings_share_one_multiplier(self) -> None:
        strategy = FreeAllocation(subcomponents={"a": RatioAllocation(weight=3), "b": RatioAllocation(weight=2)})
        # k runs while k * (3 + 2) fits the budget, so 15 gives exactly three options.
        assert _splits(strategy, ("a", "b"), 15) == [(3, 2), (6, 4), (9, 6)]

    def test_no_split_when_fixed_allocations_overspend(self) -> None:
        strategy = FreeAllocation(subcomponents={"a": FixedAllocation(n_cores=8), "b": FixedAllocation(n_cores=8)})
        assert _splits(strategy, ("a", "b"), 4) == []


# ---------------------------------------------------------------------------
# AllocationStrategy
# ---------------------------------------------------------------------------


class TestAllocationStrategy:
    # --- The abstract base ---

    def test_base_class_cannot_be_instantiated(self) -> None:
        # The mode is the type, so there is no mode-less strategy to construct.
        with pytest.raises(TypeError, match="abstract"):
            AllocationStrategy()  # type: ignore[abstract]

    def test_modes_holding_the_same_number_are_distinct(self) -> None:
        # Both hold "4", but they mean different things and must not be conflated in the
        # enumerator's subtree memo. Inequality is what the memo relies on: two modes may
        # legitimately share a hash bucket, and __eq__ is what separates them there.
        fixed = FixedAllocation(n_cores=4)
        free = FreeAllocation(min_cores=4, max_cores=4)
        assert fixed != free

    def test_resolve_tpr_range_prefers_its_own(self) -> None:
        assert FreeAllocation(tpr_range=(2, 4)).resolve_tpr_range((1, 1)) == (2, 4)

    def test_resolve_tpr_range_falls_back_to_inherited(self) -> None:
        assert FreeAllocation().resolve_tpr_range((1, 8)) == (1, 8)

    def test_invalid_strategy_tpr_range_raises(self) -> None:
        with pytest.raises(ValueError, match="AllocationStrategy.tpr_range maximum"):
            FreeAllocation(tpr_range=(4, 2))

        with pytest.raises(ValueError, match="AllocationStrategy.tpr_range minimum"):
            FreeAllocation(tpr_range=(0, 2))

    def test_strategies_for_orders_by_name(self) -> None:
        spec = FreeAllocation(subcomponents={"ocn": FixedAllocation(n_cores=4), "atm": FreeAllocation()})
        atm, ocn = spec.strategies_for(("atm", "ocn"))
        assert atm == FreeAllocation()
        assert ocn == FixedAllocation(n_cores=4)

    def test_strategies_for_has_no_default_mode(self) -> None:
        # _validate_names has already ruled this out by the time the enumerator asks, so a
        # bare KeyError is the right answer rather than a silently invented free mode.
        spec = FreeAllocation(subcomponents={"ocn": FixedAllocation(n_cores=4)})
        with pytest.raises(KeyError, match="atm"):
            spec.strategies_for(("atm", "ocn"))

    # --- Fixed mode ---

    def test_fixed_valid(self) -> None:
        spec = FixedAllocation(n_cores=12)
        assert spec.n_cores == 12
        assert spec.subcomponents == {}

    def test_fixed_accepts_its_core_count_positionally(self) -> None:
        assert FixedAllocation(12) == FixedAllocation(n_cores=12)

    def test_fixed_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            FixedAllocation(n_cores=0)

    def test_fixed_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            FixedAllocation(n_cores=-4)

    # --- Ratio mode ---

    def test_ratio_valid(self) -> None:
        spec = RatioAllocation(weight=3)
        assert spec.weight == 3

    def test_ratio_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RatioAllocation(weight=0)

    def test_ratio_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RatioAllocation(weight=-1)

    # --- Free mode ---

    def test_free_core_range_is_capped_by_max_cores(self) -> None:
        assert FreeAllocation(min_cores=2, max_cores=4)._core_range(10) == range(2, 5)

    def test_free_core_range_is_capped_by_the_budget(self) -> None:
        assert FreeAllocation(min_cores=2, max_cores=9)._core_range(3) == range(2, 4)

    def test_free_core_range_is_unbounded_without_max_cores(self) -> None:
        assert FreeAllocation(min_cores=2)._core_range(5) == range(2, 6)

    def test_free_core_range_is_empty_when_min_cores_does_not_fit(self) -> None:
        assert list(FreeAllocation(min_cores=4)._core_range(2)) == []

    def test_free_tree_for_mirrors_the_component_tree(self) -> None:
        root = ParallelComponent("r", subcomponents=(ParallelComponent("a"), ParallelComponent("b")))
        tree = FreeAllocation._tree_for(root)
        assert set(tree.subcomponents) == {"a", "b"}
        assert tree.subcomponents["a"] == FreeAllocation()

    def test_free_defaults(self) -> None:
        spec = FreeAllocation()
        assert spec.min_cores == 1
        assert spec.max_cores is None

    def test_free_explicit_bounds(self) -> None:
        spec = FreeAllocation(min_cores=4, max_cores=16)
        assert spec.min_cores == 4
        assert spec.max_cores == 16

    def test_free_equal_bounds(self) -> None:
        spec = FreeAllocation(min_cores=8, max_cores=8)
        assert spec.min_cores == spec.max_cores == 8

    def test_free_zero_min_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            FreeAllocation(min_cores=0)

    def test_free_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_cores"):
            FreeAllocation(min_cores=4, max_cores=2)

    # --- Subcomponents and constraints ---

    def test_with_subcomponents(self) -> None:
        child_a = FreeAllocation()
        child_b = FixedAllocation(n_cores=12)
        parent = FreeAllocation(subcomponents={"a": child_a, "b": child_b})
        assert len(parent.subcomponents) == 2
        assert parent.subcomponents["b"].n_cores == 12

    def test_with_local_constraints(self) -> None:
        from access.config.parallel_constraints import FixedThreadsPerRankConstraint

        c = FixedThreadsPerRankConstraint(n_threads=1)
        spec = FreeAllocation(local_constraints=(c,))
        assert spec.local_constraints == (c,)

    def test_group_constraints_default_empty(self) -> None:
        spec = FreeAllocation()
        assert spec.group_constraints == ()

    def test_with_group_constraints(self) -> None:
        from access.config.parallel_constraints import RankRatioGroupConstraint

        gc = RankRatioGroupConstraint(name_a="a", name_b="b", min_ratio=1.0)
        spec = FreeAllocation(group_constraints=(gc,))
        assert spec.group_constraints == (gc,)

    def test_frozen(self) -> None:
        spec = FreeAllocation()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.n_cores = 4  # type: ignore[misc]

    def test_equal_strategies_hash_equal(self) -> None:
        # subcomponents is excluded from the hash (it is an unhashable mapping), so the
        # hash/eq contract is upheld by equality: equal strategies must hash equal.
        child = FixedAllocation(n_cores=2)
        spec_a = FreeAllocation(subcomponents={"child": child})
        spec_b = FreeAllocation(subcomponents={"child": child})
        assert spec_a == spec_b
        assert hash(spec_a) == hash(spec_b)

    def test_equality_still_compares_the_subtree(self) -> None:
        # The hash ignores subcomponents, so equality is the only thing keeping two
        # differently-shaped strategies apart in the enumerator's subtree memo.
        spec_a = FreeAllocation(subcomponents={"child": FixedAllocation(n_cores=2)})
        spec_b = FreeAllocation(subcomponents={"child": FixedAllocation(n_cores=3)})
        assert spec_a != spec_b

    def test_subcomponents_are_copied(self) -> None:
        # The mapping must be snapshotted, not proxied: mutating the caller's dict
        # afterwards must not change a frozen instance or what it compares equal to.
        subs = {"a": FixedAllocation(n_cores=4)}
        spec = FreeAllocation(subcomponents=subs)

        subs["b"] = FixedAllocation(n_cores=8)

        assert set(spec.subcomponents) == {"a"}
        assert spec == FreeAllocation(subcomponents={"a": FixedAllocation(n_cores=4)})

    def test_subcomponents_view_is_read_only(self) -> None:
        spec = FreeAllocation(subcomponents={"a": FixedAllocation(n_cores=4)})
        with pytest.raises(TypeError):
            spec.subcomponents["b"] = FixedAllocation(n_cores=8)  # type: ignore[index]


# ---------------------------------------------------------------------------
# The root
# ---------------------------------------------------------------------------


class TestRootAllocation:
    """The root cannot size itself, and no other mode can be the root."""

    @pytest.mark.parametrize("field", ["n_cores", "weight", "min_cores", "max_cores"])
    def test_it_cannot_state_a_size(self, field: str) -> None:
        # Not rejected at construction - unwritable. Every sizing field belongs to a mode
        # that takes a *share* of a parent's cores, which the root never does.
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{field}'"):
            RootAllocation(**{field: 4})  # type: ignore[arg-type]

    def test_it_keeps_the_fields_a_root_may_set(self) -> None:
        root = RootAllocation(tpr_range=(1, 4), subcomponents={"a": FreeAllocation()})
        assert root.tpr_range == (1, 4)
        assert root.strategies_for(("a",)) == (FreeAllocation(),)

    def test_it_refuses_to_take_a_share(self) -> None:
        # Unreachable through the scheduler, which trips on the missing _phase first, so
        # this is the answer to calling it directly.
        with pytest.raises(TypeError, match="cannot be nested under another strategy"):
            RootAllocation._iter_core_shares([RootAllocation()], 8)

    def test_nesting_one_fails_by_name(self) -> None:
        strategy = RootAllocation(subcomponents={"a": RootAllocation(), "b": FreeAllocation()})
        with pytest.raises(AttributeError, match="RootAllocation"):
            _splits(strategy, ("a", "b"), 4)


# ---------------------------------------------------------------------------
# resolve_root_strategy
# ---------------------------------------------------------------------------


class TestResolveRootStrategy:
    """The published entry point ``parallel_layouts`` uses to settle the root strategy."""

    def test_none_builds_an_all_free_tree_under_a_root(self) -> None:
        root = ParallelComponent("r", subcomponents=(ParallelComponent("a"), ParallelComponent("b")))
        resolved = resolve_root_strategy(root, None)
        assert resolved == RootAllocation(subcomponents={"a": FreeAllocation(), "b": FreeAllocation()})

    def test_a_valid_strategy_is_returned_unchanged(self) -> None:
        comp = ParallelComponent("r", subcomponents=(ParallelComponent("a"),))
        given = RootAllocation(subcomponents={"a": FixedAllocation(n_cores=4)})
        assert resolve_root_strategy(comp, given) is given

    @pytest.mark.parametrize(
        "allocations",
        [FixedAllocation(n_cores=4), RatioAllocation(weight=2), FreeAllocation(max_cores=8), FreeAllocation()],
    )
    def test_only_a_root_allocation_may_be_the_root(self, allocations: AllocationStrategy) -> None:
        # Even an unbounded FreeAllocation is refused: the type is the contract, so there
        # is no per-mode judgement about which fields happen to have been left alone.
        comp = ParallelComponent("r")
        with pytest.raises(TypeError, match="allocations must be a RootAllocation"):
            resolve_root_strategy(comp, allocations)

    def test_a_mismatched_tree_is_rejected(self) -> None:
        comp = ParallelComponent("r", subcomponents=(ParallelComponent("a"),))
        with pytest.raises(ValueError, match=r"no allocation given at root for \{'a'\}"):
            resolve_root_strategy(comp, RootAllocation())


# ---------------------------------------------------------------------------
# The scheduling protocol
# ---------------------------------------------------------------------------


class TestSchedulingProtocol:
    """The scheduler is mode-agnostic: it reads ``_phase`` and the three protocol hooks."""

    def test_phases_are_served_in_order_regardless_of_position(self) -> None:
        # Declaration order is free, ratio, fixed; the schedule is still fixed, ratio, free.
        strategy = FreeAllocation(
            subcomponents={
                "f": FreeAllocation(min_cores=1),
                "r": RatioAllocation(weight=2),
                "x": FixedAllocation(n_cores=3),
            }
        )
        splits = _splits(strategy, ("f", "r", "x"), 9)
        # "x" is pinned at 3 wherever it sits, and every split respects the budget.
        assert all(split[2] == 3 for split in splits)
        assert all(sum(split) <= 9 for split in splits)
        assert (1, 2, 3) in splits

    def test_a_middle_phase_reserves_nothing_by_default(self) -> None:
        # Fixed, then ratio, then free: computing what fixed must leave behind calls the
        # base-class _reserved_cores for the ratio group, which reserves nothing.
        strategy = FreeAllocation(
            subcomponents={
                "x": FixedAllocation(n_cores=2),
                "r": RatioAllocation(weight=1),
                "f": FreeAllocation(min_cores=1),
            }
        )
        splits = _splits(strategy, ("x", "r", "f"), 6)
        assert (2, 1, 1) in splits
        assert all(split[0] == 2 for split in splits)

    def test_a_mode_without_an_explanation_stays_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        # A free group that cannot fit yields nothing, and free mode offers no explanation,
        # so the scheduler logs nothing for it.
        strategy = FreeAllocation(subcomponents={"a": FreeAllocation(min_cores=99), "b": FreeAllocation()})
        with caplog.at_level(logging.DEBUG, logger="access.config.parallel_allocation_strategies"):
            assert _splits(strategy, ("a", "b"), 4) == []
        assert caplog.text == ""

    def test_a_mode_that_declares_no_phase_fails_by_name(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class WeirdAllocation(AllocationStrategy):
            @classmethod
            def _iter_core_shares(cls, allocs, budget):  # type: ignore[no-untyped-def]
                yield (1,) * len(allocs)

        strategy = FreeAllocation(subcomponents={"a": WeirdAllocation(), "b": FreeAllocation()})
        with pytest.raises(AttributeError, match="WeirdAllocation"):
            _splits(strategy, ("a", "b"), 4)

    def test_a_new_mode_joins_the_schedule_without_touching_it(self) -> None:
        # The whole point of the protocol: declare a phase, implement one method.
        @dataclasses.dataclass(frozen=True)
        class HalfAllocation(AllocationStrategy):
            """Takes exactly half of what it is offered, rounded down."""

            _phase: typing.ClassVar[int] = 1

            @classmethod
            def _iter_core_shares(cls, allocs, budget):  # type: ignore[no-untyped-def]
                share = budget // 2 // len(allocs)
                if share >= 1:
                    yield (share,) * len(allocs)

        strategy = FreeAllocation(subcomponents={"h": HalfAllocation(), "f": FreeAllocation(min_cores=1)})
        splits = _splits(strategy, ("h", "f"), 10)
        # HalfAllocation is phase 1, so it is served before the free sibling: (10-1)//2 = 4.
        assert splits == [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)]
