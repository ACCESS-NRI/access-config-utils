# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Rank-allocation strategy and layout enumeration for parallel component trees.

This module owns:

- :class:`AllocationStrategy`: rank-distribution specification for one node in a
    component tree.
- :func:`enumerate_layouts`: enumerate all valid :class:`ComponentLayout` trees for a
    component definition and a total core budget.

Concrete constraint implementations live in :mod:`access.config.parallel_constraints`.
See :mod:`access.config.parallel_component` for the component-tree model and
:mod:`access.config.parallel_domain` for domain/decomposition models.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from access.config import parallel_domain
from access.config.parallel_component import (
    ComponentLayout,
    GroupConstraint,
    LocalConstraint,
    ParallelComponent,
)

# ---------------------------------------------------------------------------
# AllocationStrategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationStrategy:
    """Rank-allocation assignment for one node in a component tree.

    An :class:`AllocationStrategy` tree is passed to
    :func:`enumerate_layouts` to describe how ranks are distributed among
    sub-components, independently of the :class:`ParallelComponent` structure
    (domains and constraints).

    Exactly one allocation mode is active, determined by the fields that are set:

    * **Fixed** (``n_ranks`` is not ``None``) – the component receives exactly
      ``n_ranks`` MPI ranks.
    * **Ratio** (``weight`` is not ``None``) – the component receives ranks in
      exact proportion to sibling weights.  There exists a shared integer
      multiplier ``k >= 1`` such that each ratio-allocated sibling receives
      ``k * weight`` ranks.  For example, weights (3, 2) can be allocated as
      (3, 2), (6, 4), (9, 6), and so on.
    * **Free** (default, both ``n_ranks`` and ``weight`` are ``None``) – the
      enumeration engine may assign any rank count in ``[min_ranks, max_ranks]``
      from the budget remaining after fixed and ratio allocations.

    Parameters
    ----------
    n_ranks : int | None
        Fixed allocation: the component must receive exactly this many MPI ranks.
        Must be >= 1 when set.  Cannot be combined with ``weight``,
        ``min_ranks`` != 1, or ``max_ranks`` != ``None``.
    weight : int | None
        Ratio allocation: relative weight among ratio-allocated siblings.
        Must be >= 1 when set.  Cannot be combined with ``n_ranks``,
        ``min_ranks`` != 1, or ``max_ranks`` != ``None``.
    min_ranks : int
        Free-mode lower bound on rank count.  Must be >= 1.  Defaults to 1.
        Ignored when ``n_ranks`` or ``weight`` is set.
    max_ranks : int | None
        Free-mode upper bound on rank count, or ``None`` for no upper bound.
        Must be >= ``min_ranks`` when provided.  Defaults to ``None``.
        Ignored when ``n_ranks`` or ``weight`` is set.
    subcomponents : dict[str, AllocationStrategy]
        Allocation strategies for direct sub-components, keyed by component name.
        Omit (or pass ``{}``) for leaf components.
    local_constraints : tuple[LocalConstraint, ...]
        Constraints applied only when this allocation strategy is active, checked
        alongside the component's own :attr:`~ParallelComponent.local_constraints`.
        Use this for strategy-specific filters (e.g. aspect-ratio limits) that
        should not be hard-coded into the component definition.
    group_constraints : tuple[GroupConstraint, ...]
        Group constraints applied only when this allocation strategy is active,
        checked alongside the parent component's own
        :attr:`~ParallelComponent.group_constraints`.
        Use this for strategy-specific cross-component constraints (e.g.
        rank-ratio limits) that should not be hard-coded into the component
        definition.

    Examples
    --------
    >>> AllocationStrategy(subcomponents={
    ...     "UM7":   AllocationStrategy(),
    ...     "MOM5":  AllocationStrategy(),
    ...     "CICE5": AllocationStrategy(n_ranks=12),
    ... })
    """

    n_ranks: int | None = None
    weight: int | None = None
    min_ranks: int = 1
    max_ranks: int | None = None
    subcomponents: dict[str, AllocationStrategy] = field(default_factory=dict)
    local_constraints: tuple[LocalConstraint, ...] = ()
    group_constraints: tuple[GroupConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.n_ranks is not None and self.weight is not None:
            raise ValueError("AllocationStrategy: n_ranks and weight cannot both be set.")
        if self.n_ranks is not None:
            if self.n_ranks < 1:
                raise ValueError(f"AllocationStrategy.n_ranks must be >= 1, got {self.n_ranks}.")
            if self.min_ranks != 1 or self.max_ranks is not None:
                raise ValueError("AllocationStrategy: min_ranks/max_ranks cannot be set alongside n_ranks.")
        elif self.weight is not None:
            if self.weight < 1:
                raise ValueError(f"AllocationStrategy.weight must be >= 1, got {self.weight}.")
            if self.min_ranks != 1 or self.max_ranks is not None:
                raise ValueError("AllocationStrategy: min_ranks/max_ranks cannot be set alongside weight.")
        else:
            if self.min_ranks < 1:
                raise ValueError(f"AllocationStrategy.min_ranks must be >= 1, got {self.min_ranks}.")
            if self.max_ranks is not None and self.max_ranks < self.min_ranks:
                raise ValueError(
                    f"AllocationStrategy.max_ranks ({self.max_ranks}) must be >= min_ranks ({self.min_ranks})."
                )
        object.__setattr__(self, "subcomponents", MappingProxyType(self.subcomponents))

    @property
    def allocation_mode(self) -> Literal["fixed", "ratio", "free"]:
        """The active allocation mode: ``"fixed"``, ``"ratio"``, or ``"free"``."""
        if self.n_ranks is not None:
            return "fixed"
        if self.weight is not None:
            return "ratio"
        return "free"

    def __hash__(self) -> int:
        return hash(
            (
                self.n_ranks,
                self.weight,
                self.min_ranks,
                self.max_ranks,
                tuple(sorted(self.subcomponents.items())),
                self.local_constraints,
                self.group_constraints,
            )
        )


# ---------------------------------------------------------------------------
# Enumeration internals
# ---------------------------------------------------------------------------


def _iter_free_assignments(
    free_allocs: list[AllocationStrategy],
    budget: int,
) -> Iterator[tuple[int, ...]]:
    """Yield all ways to assign up to *budget* ranks among *free_allocs*.

    Assigns ranks to each :class:`AllocationStrategy` with
    ``allocation_mode == "free"`` in order such that each allocation's
    ``min_ranks``/``max_ranks`` bounds are respected and the total assigned does
    not exceed *budget*. Any leftover budget (``budget - sum(assignment)``) is
    silently unused — callers may apply a
    :class:`MaxWastedRankFractionConstraint` on the parent component to
    restrict this slack.
    """
    if not free_allocs:
        if budget >= 0:
            yield ()
        return

    alloc = free_allocs[0]
    rest_allocs = free_allocs[1:]
    rest_min = sum(a.min_ranks for a in rest_allocs)

    lo = alloc.min_ranks
    hi = budget - rest_min
    if alloc.max_ranks is not None:
        hi = min(hi, alloc.max_ranks)

    for r in range(lo, hi + 1):
        for rest in _iter_free_assignments(rest_allocs, budget - r):
            yield (r, *rest)


def _iter_rank_splits(
    subcomponents: tuple[ParallelComponent, ...],
    parent_ranks: int,
    alloc_specs: tuple[AllocationStrategy, ...],
) -> Iterator[tuple[int, ...]]:
    """Yield all valid rank assignments to *subcomponents* that sum to at most *parent_ranks*.

    The allocation mode of each *alloc_spec* determines how ranks are distributed:

    * **fixed** (``n_ranks`` is set) – always receives exactly ``n_ranks``.
    * **ratio** (``weight`` is set) – receives ``k * weight`` ranks for an integer
      multiplier *k* >= 1; all ratio siblings share the same *k*.
    * **free** (default) – receives any rank count in ``[min_ranks, max_ranks]``
      from the budget remaining after fixed and ratio allocations.

    The total assigned across all sub-components may be less than *parent_ranks*;
    the unused ranks are left idle.  Use :class:`MaxWastedRankFractionConstraint`
    on the parent component to restrict idle cores.
    """
    allocs = list(alloc_specs)
    if len(subcomponents) != len(allocs):
        raise ValueError(
            "subcomponents and alloc_specs must have the same length: "
            f"got {len(subcomponents)} subcomponents and {len(allocs)} allocation specs"
        )
    n = len(allocs)

    fixed_indices = [i for i, a in enumerate(allocs) if a.n_ranks is not None]
    ratio_indices = [i for i, a in enumerate(allocs) if a.weight is not None]
    free_indices = [i for i, a in enumerate(allocs) if a.n_ranks is None and a.weight is None]

    fixed_allocs = [allocs[i] for i in fixed_indices]
    ratio_allocs = [allocs[i] for i in ratio_indices]
    free_allocs = [allocs[i] for i in free_indices]

    fixed_total = sum(alloc.n_ranks for alloc in fixed_allocs)  # type: ignore[misc]
    available = parent_ranks - fixed_total

    if available < 0:
        return  # fixed components alone exceed the budget

    free_min_total = sum(a.min_ranks for a in free_allocs)

    def _build_result(ratio_ranks: dict[int, int], free_assignment: tuple[int, ...]) -> tuple[int, ...]:
        result = [0] * n
        for i in fixed_indices:
            result[i] = fixed_allocs[fixed_indices.index(i)].n_ranks  # type: ignore[assignment]
        for i, r in ratio_ranks.items():
            result[i] = r
        for i, r in zip(free_indices, free_assignment, strict=True):
            result[i] = r
        return tuple(result)

    if ratio_indices:
        sum_ratio_weights = sum(alloc.weight for alloc in ratio_allocs)  # type: ignore[misc]
        max_k = (available - free_min_total) // sum_ratio_weights
        for k in range(1, max_k + 1):
            ratio_ranks = {i: k * alloc.weight for i, alloc in zip(ratio_indices, ratio_allocs, strict=True)}  # type: ignore[misc]
            remaining = available - k * sum_ratio_weights
            for free_assignment in _iter_free_assignments(free_allocs, remaining):
                yield _build_result(ratio_ranks, free_assignment)
    else:
        for free_assignment in _iter_free_assignments(free_allocs, available):
            yield _build_result({}, free_assignment)


def _default_alloc_spec(component: ParallelComponent) -> AllocationStrategy:
    """Return a default free-allocation strategy tree for *component*."""
    return AllocationStrategy(
        subcomponents={sub.name: _default_alloc_spec(sub) for sub in component.subcomponents},
    )


def _validate_alloc_spec_names(
    component: ParallelComponent, alloc_spec: AllocationStrategy, path: str = "root"
) -> None:
    """Validate that allocation-strategy child names match the component tree."""
    sub_names = {sub.name for sub in component.subcomponents}
    unknown = set(alloc_spec.subcomponents) - sub_names
    if unknown:
        raise ValueError(
            f"enumerate_layouts: unknown component names in allocations at {path}: {unknown}. Valid names: {sub_names}."
        )

    for sub in component.subcomponents:
        child_alloc_spec = alloc_spec.subcomponents.get(sub.name)
        if child_alloc_spec is not None:
            _validate_alloc_spec_names(sub, child_alloc_spec, f"{path}.{sub.name}")


def _enum_component(
    component: ParallelComponent,
    n_ranks: int,
    tpr: int,
    total_ranks: int,
    alloc_spec: AllocationStrategy,
) -> Iterator[ComponentLayout]:
    """Yield all valid :class:`ComponentLayout` objects for *component* with *n_ranks* and *tpr*."""
    decompositions: list[parallel_domain.DomainCartesianDecomposition | None]
    if component.domain is not None:
        decompositions = list(parallel_domain.iter_cartesian_decompositions(component.domain, n_ranks))
    else:
        decompositions = [None]

    all_constraints = component.local_constraints + alloc_spec.local_constraints
    for sub_layout_tuple in _iter_valid_sub_layouts(component, n_ranks, tpr, total_ranks, alloc_spec):
        for decomp in decompositions:
            candidate = ComponentLayout(
                name=component.name,
                n_ranks=n_ranks,
                threads_per_rank=tpr,
                decomposition=decomp,
                sub_layouts=sub_layout_tuple,
            )
            if all(c.is_satisfied(candidate, total_ranks) for c in all_constraints):
                yield candidate


def _iter_valid_sub_layouts(
    component: ParallelComponent,
    n_ranks: int,
    tpr: int,
    total_ranks: int,
    alloc_spec: AllocationStrategy,
) -> Iterator[tuple[ComponentLayout, ...]]:
    """Yield sub-layout tuples for *component*'s subcomponents that satisfy group constraints."""
    if not component.subcomponents:
        yield ()
        return

    # Resolve the name-keyed dict to a positional tuple aligned with component.subcomponents.
    # Missing names fall back to FreeAllocation.
    ordered_alloc_specs = tuple(
        alloc_spec.subcomponents.get(sub.name, AllocationStrategy()) for sub in component.subcomponents
    )

    for rank_split in _iter_rank_splits(component.subcomponents, n_ranks, ordered_alloc_specs):
        sub_options = [
            list(_enum_component(sub, r, tpr, total_ranks, spec))
            for sub, r, spec in zip(component.subcomponents, rank_split, ordered_alloc_specs, strict=True)
        ]
        for combo in itertools.product(*sub_options):
            all_group_constraints = component.group_constraints + alloc_spec.group_constraints
            if all(gc.is_satisfied(combo, total_ranks) for gc in all_group_constraints):
                yield combo


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def enumerate_layouts(
    component: ParallelComponent,
    total_cores: int,
    *,
    tpr_range: tuple[int, int] = (1, 1),
    allocations: AllocationStrategy | None = None,
) -> list[ComponentLayout]:
    """Return all valid :class:`ComponentLayout` trees for *component* given *total_cores*.

    For each ``tpr`` in ``tpr_range``:

    1. If ``total_cores`` is not divisible by ``tpr``, skip.
    2. Set ``total_ranks = total_cores // tpr``.
    3. Recursively enumerate all layouts for the component tree, distributing
       *total_ranks* among sub-components according to *allocations* and
       filtering with all registered constraints.

    Parameters
    ----------
    component : ParallelComponent
        Root of the component tree.  *total_cores* determines its rank/thread budget.
    total_cores : int
        Total number of CPU cores available.  Must be >= 1.
    tpr_range : tuple[int, int]
        Inclusive range ``(min_tpr, max_tpr)`` of threads-per-rank values to try.
        Defaults to ``(1, 1)`` (pure MPI, no OpenMP threading).
    allocations : AllocationStrategy | None
        Top-level :class:`AllocationStrategy` for *component*.
        Child allocation strategies are provided via its ``subcomponents`` mapping.
        Missing names at any level fall back to free mode (default
        :class:`AllocationStrategy` with no ``n_ranks`` or ``weight`` set).
        The root strategy in ``allocations`` is accepted for structural consistency,
        but its ``n_ranks`` field is not used because the root always receives
        ``total_ranks``.
        ``None`` (the default) assigns free-mode :class:`AllocationStrategy`
        to every node in the tree recursively.

    Returns
    -------
    list[ComponentLayout]
        All valid layout trees, in enumeration order.  Returns an empty list when
        no valid layout exists.

    Raises
    ------
    ValueError
        If *total_cores* < 1 or *tpr_range* is invalid.

    Examples
    --------
    >>> domain = Domain((12, 8))
    >>> comp = ParallelComponent("model", domain=domain)
    >>> layouts = enumerate_layouts(comp, total_cores=4)
    >>> len(layouts)
    3  # (1×4), (2×2), (4×1) decompositions
    """
    if total_cores < 1:
        raise ValueError(f"total_cores must be >= 1, got {total_cores}.")
    min_tpr, max_tpr = tpr_range
    if min_tpr < 1:
        raise ValueError(f"tpr_range minimum must be >= 1, got {min_tpr}.")
    if max_tpr < min_tpr:
        raise ValueError(f"tpr_range maximum ({max_tpr}) must be >= minimum ({min_tpr}).")

    if allocations is not None:
        _validate_alloc_spec_names(component, allocations)
        root_alloc_spec = allocations
    else:
        root_alloc_spec = _default_alloc_spec(component)

    results: list[ComponentLayout] = []
    for tpr in range(min_tpr, max_tpr + 1):
        if total_cores % tpr != 0:
            continue
        total_ranks = total_cores // tpr
        results.extend(_enum_component(component, total_ranks, tpr, total_ranks, root_alloc_spec))
    return results
