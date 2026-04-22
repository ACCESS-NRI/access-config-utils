# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Layout result types, constraints, and enumeration for parallel component trees.

This module builds on top of :mod:`access.config.parallelisation` and provides:

- :class:`ComponentLayout`: the resolved parallelisation assignment for one component.
- Constraint implementations for the four constraint categories:

  1. **Cartesian grid** - properties of the MPI process grid
     (:class:`ProcessGridDimEvenConstraint`, :class:`ProcessGridDimDivisibleConstraint`,
     :class:`ProcessGridAspectRatioConstraint`).
  2. **Distribution** - how ranks are divided between components
     (:class:`MaxRankFractionConstraint`, :class:`MaxWastedRankFractionConstraint`,
     :class:`RankRatioGroupConstraint`).
  3. **Domain layout** - properties of the local sub-domains
     (:class:`UniformSubdomainConstraint`, :class:`SubdomainSizeToleranceConstraint`,
     :class:`MinSubdomainSizeConstraint`, :class:`SubdomainAspectRatioConstraint`).
  4. **Threads** - OpenMP thread constraints
     (:class:`MaxThreadsPerRankConstraint`, :class:`FixedThreadsPerRankConstraint`,
     :class:`ThreadsDivisorConstraint`).

- :func:`iter_cartesian_decompositions`: enumerate all valid cartesian decompositions
  of a domain for a given rank count.
- :func:`enumerate_layouts`: enumerate all valid :class:`ComponentLayout` trees for a
  component definition and a total core budget.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from access.config.parallelisation import (
    AllocationSpec,
    CartesianDecomposition,
    Domain,
    FixedRanks,
    FreeAllocation,
    GroupConstraint,
    LocalConstraint,
    ParallelComponent,
    RatioAllocation,
)

# ---------------------------------------------------------------------------
# ComponentLayout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentLayout:
    """The resolved parallelisation assignment for one component.

    Parameters
    ----------
    name : str
        ParallelComponent name (mirrors :attr:`~access.config.parallelisation.ParallelComponent.name`).
    n_ranks : int
        MPI ranks assigned to this component.  Must be >= 1.
    threads_per_rank : int
        OpenMP threads per MPI rank.  Must be >= 1.
    decomposition : CartesianDecomposition | None
        How this component's domain is tiled across the MPI cartesian grid, or
        ``None`` when the component has no domain.
    sub_layouts : tuple[ComponentLayout, ...]
        Layouts for each direct sub-component, in the same order as
        :attr:`~access.config.parallelisation.ParallelComponent.subcomponents`.
    """

    name: str
    n_ranks: int
    threads_per_rank: int
    decomposition: CartesianDecomposition | None
    sub_layouts: tuple[ComponentLayout, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ComponentLayout.name must be non-empty.")
        if self.n_ranks < 1:
            raise ValueError(f"ComponentLayout.n_ranks must be >= 1, got {self.n_ranks}.")
        if self.threads_per_rank < 1:
            raise ValueError(f"ComponentLayout.threads_per_rank must be >= 1, got {self.threads_per_rank}.")

    @property
    def total_cores(self) -> int:
        """CPU cores consumed by this component (``n_ranks × threads_per_rank``)."""
        return self.n_ranks * self.threads_per_rank


# ---------------------------------------------------------------------------
# Category 1 — Cartesian grid constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessGridDimEvenConstraint(LocalConstraint):
    """``grid[dim]`` must be even.

    Parameters
    ----------
    dim : int
        Zero-based dimension index to check.
    """

    dim: int

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        return layout.decomposition.grid[self.dim] % 2 == 0


@dataclass(frozen=True)
class ProcessGridDimDivisibleConstraint(LocalConstraint):
    """``grid[dim]`` must be divisible by ``divisor``.

    Parameters
    ----------
    dim : int
        Zero-based dimension index to check.
    divisor : int
        Required divisor.  Must be >= 1.
    """

    dim: int
    divisor: int

    def __post_init__(self) -> None:
        if self.divisor < 1:
            raise ValueError(f"ProcessGridDimDivisibleConstraint.divisor must be >= 1, got {self.divisor}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        return layout.decomposition.grid[self.dim] % self.divisor == 0


@dataclass(frozen=True)
class ProcessGridAspectRatioConstraint(LocalConstraint):
    """``max(grid) / min(grid)`` must not exceed ``max_ratio``.

    This enforces near-square decompositions and helps balance communication costs.

    Parameters
    ----------
    max_ratio : float
        Maximum allowed ratio between the largest and smallest grid dimension.
        Must be >= 1.0.
    """

    max_ratio: float

    def __post_init__(self) -> None:
        if self.max_ratio < 1.0:
            raise ValueError(f"ProcessGridAspectRatioConstraint.max_ratio must be >= 1.0, got {self.max_ratio}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        g = layout.decomposition.grid
        return max(g) / min(g) <= self.max_ratio


# ---------------------------------------------------------------------------
# Category 2 — Distribution constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxRankFractionConstraint(LocalConstraint):
    """This component must not use more than ``max_fraction`` of the total MPI ranks.

    Parameters
    ----------
    max_fraction : float
        Maximum fraction of ``total_ranks`` allowed.  Must be in ``(0.0, 1.0]``.
    """

    max_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 < self.max_fraction <= 1.0):
            raise ValueError(f"MaxRankFractionConstraint.max_fraction must be in (0, 1], got {self.max_fraction}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return layout.n_ranks <= self.max_fraction * total_ranks


@dataclass(frozen=True)
class MaxWastedRankFractionConstraint(LocalConstraint):
    """The fraction of idle (unused) ranks must not exceed ``max_fraction``.

    "Wasted" ranks are those assigned to this component but not distributed to
    any sub-component: ``layout.n_ranks - sum(sub.n_ranks for sub in
    layout.sub_layouts)``.  For leaf components (no sub-components), this
    constraint is always satisfied.

    This constraint is typically placed on a *parent* component to limit how
    many of the available cores are left idle.  It mirrors the
    ``max_wasted_ncores_frac`` post-filter used in the legacy
    ``esm1p6_layout_input.py`` search utilities.

    Parameters
    ----------
    max_fraction : float
        Maximum allowed fraction of idle ranks.  Must be in ``[0.0, 1.0]``.
        Use ``0.0`` to require all ranks to be consumed exactly.
    """

    max_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_fraction <= 1.0):
            raise ValueError(
                f"MaxWastedRankFractionConstraint.max_fraction must be in [0.0, 1.0], got {self.max_fraction}."
            )

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if not layout.sub_layouts:
            return True  # leaf: no sub-components to waste anything
        used = sum(sub.n_ranks for sub in layout.sub_layouts)
        wasted = layout.n_ranks - used
        return wasted <= self.max_fraction * layout.n_ranks


@dataclass(frozen=True)
class RankRatioGroupConstraint(GroupConstraint):
    """``n_ranks(name_a) / n_ranks(name_b)`` must be >= ``min_ratio``.

    Place this on the *parent* component's :attr:`~ParallelComponent.group_constraints`.
    Uses sub-component names rather than positional indices, so the constraint
    remains correct regardless of subcomponent ordering.

    Parameters
    ----------
    name_a : str
        Name of the numerator sub-component.
    name_b : str
        Name of the denominator sub-component.
    min_ratio : float
        Minimum required ratio.  Must be > 0.
    """

    name_a: str
    name_b: str
    min_ratio: float

    def __post_init__(self) -> None:
        if self.min_ratio <= 0.0:
            raise ValueError(f"RankRatioGroupConstraint.min_ratio must be > 0, got {self.min_ratio}.")

    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_ranks: int) -> bool:
        a = next((lay for lay in sub_layouts if lay.name == self.name_a), None)
        b = next((lay for lay in sub_layouts if lay.name == self.name_b), None)
        if a is None or b is None:
            raise ValueError(
                f"RankRatioGroupConstraint: sub-component names {self.name_a!r} and/or "
                f"{self.name_b!r} not found in sub_layouts "
                f"(available: {[lay.name for lay in sub_layouts]})."
            )
        return a.n_ranks >= self.min_ratio * b.n_ranks


# ---------------------------------------------------------------------------
# Category 3 — Domain layout constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformSubdomainConstraint(LocalConstraint):
    """Every local sub-domain must have exactly the same size (exact integer division).

    Equivalent to requiring ``domain.shape[i] % grid[i] == 0`` for every dimension.
    """

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        d = layout.decomposition
        return all(dim % g == 0 for dim, g in zip(d.domain.shape, d.grid, strict=True))


@dataclass(frozen=True)
class SubdomainSizeToleranceConstraint(LocalConstraint):
    """The ratio of the largest to smallest local sub-domain must not exceed ``tolerance``.

    For each dimension ``i``, ``ceil(shape[i]/grid[i]) / floor(shape[i]/grid[i])``
    must be <= ``tolerance``.  This models load-imbalance tolerance.

    Parameters
    ----------
    tolerance : float
        Maximum allowed size ratio.  Must be >= 1.0.
    """

    tolerance: float

    def __post_init__(self) -> None:
        if self.tolerance < 1.0:
            raise ValueError(f"SubdomainSizeToleranceConstraint.tolerance must be >= 1.0, got {self.tolerance}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        d = layout.decomposition
        for dim_size, g in zip(d.domain.shape, d.grid, strict=True):
            lo = dim_size // g
            if lo == 0:
                return False
            hi = math.ceil(dim_size / g)
            if hi / lo > self.tolerance:
                return False
        return True


@dataclass(frozen=True)
class MinSubdomainSizeConstraint(LocalConstraint):
    """Each local sub-domain must have at least ``min_size`` grid points per dimension.

    Checks ``floor(shape[i] / grid[i]) >= min_size`` for every dimension.

    Parameters
    ----------
    min_size : int
        Minimum number of grid points per sub-domain dimension.  Must be >= 1.
    """

    min_size: int

    def __post_init__(self) -> None:
        if self.min_size < 1:
            raise ValueError(f"MinSubdomainSizeConstraint.min_size must be >= 1, got {self.min_size}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        d = layout.decomposition
        return all(dim // g >= self.min_size for dim, g in zip(d.domain.shape, d.grid, strict=True))


@dataclass(frozen=True)
class SubdomainAspectRatioConstraint(LocalConstraint):
    """``max(local_shape) / min(local_shape)`` must not exceed ``max_ratio``.

    Constrains the aspect ratio of the local subdomain in *physical* space
    (``domain.shape[i] / grid[i]`` per dimension), not the MPI process grid.
    Use this to ensure that the grid-point block held by each rank is near-square,
    which typically improves halo-exchange balance.

    For leaf components with no domain this constraint is always satisfied.

    Parameters
    ----------
    max_ratio : float
        Maximum allowed ratio between the longest and shortest local subdomain
        dimension.  Must be >= 1.0.
    """

    max_ratio: float

    def __post_init__(self) -> None:
        if self.max_ratio < 1.0:
            raise ValueError(f"SubdomainAspectRatioConstraint.max_ratio must be >= 1.0, got {self.max_ratio}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        if layout.decomposition is None:
            return True
        ls = layout.decomposition.local_shape
        return max(ls) / min(ls) <= self.max_ratio


# ---------------------------------------------------------------------------
# Category 4 — Thread constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxThreadsPerRankConstraint(LocalConstraint):
    """``threads_per_rank`` must not exceed ``max_threads``.

    Parameters
    ----------
    max_threads : int
        Maximum allowed threads per rank.  Must be >= 1.
    """

    max_threads: int

    def __post_init__(self) -> None:
        if self.max_threads < 1:
            raise ValueError(f"MaxThreadsPerRankConstraint.max_threads must be >= 1, got {self.max_threads}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return layout.threads_per_rank <= self.max_threads


@dataclass(frozen=True)
class FixedThreadsPerRankConstraint(LocalConstraint):
    """``threads_per_rank`` must equal exactly ``n_threads``.

    Parameters
    ----------
    n_threads : int
        Required thread count per rank.  Must be >= 1.
    """

    n_threads: int

    def __post_init__(self) -> None:
        if self.n_threads < 1:
            raise ValueError(f"FixedThreadsPerRankConstraint.n_threads must be >= 1, got {self.n_threads}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return layout.threads_per_rank == self.n_threads


@dataclass(frozen=True)
class ThreadsDivisorConstraint(LocalConstraint):
    """``divisor`` must be divisible by ``threads_per_rank``.

    Useful when a component's work tiles evenly only for specific thread counts
    (e.g., ``threads_per_rank`` must divide 8).

    Parameters
    ----------
    divisor : int
        Value that must be divisible by ``threads_per_rank``.  Must be >= 1.
    """

    divisor: int

    def __post_init__(self) -> None:
        if self.divisor < 1:
            raise ValueError(f"ThreadsDivisorConstraint.divisor must be >= 1, got {self.divisor}.")

    def is_satisfied(self, layout: ComponentLayout, total_ranks: int) -> bool:
        return self.divisor % layout.threads_per_rank == 0


# ---------------------------------------------------------------------------
# iter_cartesian_decompositions
# ---------------------------------------------------------------------------


def iter_cartesian_decompositions(domain: Domain, n_ranks: int) -> Iterator[CartesianDecomposition]:
    """Yield all :class:`CartesianDecomposition` objects for *domain* using *n_ranks* ranks.

    Enumerates every N-tuple ``(g0, g1, ..., g_{N-1})`` such that
    ``g0 * g1 * ... * g_{N-1} == n_ranks`` and ``gi >= 1`` for all ``i``.

    Parameters
    ----------
    domain : Domain
        The domain to be decomposed.
    n_ranks : int
        Total number of MPI ranks.  Must be >= 1.

    Yields
    ------
    CartesianDecomposition
        Each valid decomposition in lexicographic grid order.

    Examples
    --------
    >>> list(iter_cartesian_decompositions(Domain((10, 10)), 4))
    [CartesianDecomposition(domain=..., grid=(1, 4)),
     CartesianDecomposition(domain=..., grid=(2, 2)),
     CartesianDecomposition(domain=..., grid=(4, 1))]
    """
    for grid in _iter_grids(domain.ndim, n_ranks):
        yield CartesianDecomposition(domain, grid)


def _iter_grids(ndim: int, n_ranks: int) -> Iterator[tuple[int, ...]]:
    """Yield all ndim-tuples whose product equals n_ranks."""
    if ndim == 1:
        yield (n_ranks,)
        return
    for d0 in range(1, n_ranks + 1):
        if n_ranks % d0 == 0:
            for rest in _iter_grids(ndim - 1, n_ranks // d0):
                yield (d0, *rest)


# ---------------------------------------------------------------------------
# Enumeration internals
# ---------------------------------------------------------------------------


def _iter_free_assignments(
    free_allocs: list[FreeAllocation],
    budget: int,
) -> Iterator[tuple[int, ...]]:
    """Yield all ways to assign up to *budget* ranks among *free_allocs*.

    Assigns ranks to each :class:`FreeAllocation` in order such that each
    allocation's ``min_ranks``/``max_ranks`` bounds are respected and the total
    assigned does not exceed *budget*.  Any leftover budget (``budget -
    sum(assignment)``) is silently unused — callers may apply a
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
    alloc_specs: tuple[AllocationSpec, ...],
) -> Iterator[tuple[int, ...]]:
    """Yield all valid rank assignments to *subcomponents* that sum to at most *parent_ranks*.

    The three allocation types are read from *alloc_specs* (one per subcomponent):

    * :class:`FixedRanks` – always receives exactly ``n_ranks``.
    * :class:`RatioAllocation` – receives ``k * weight`` ranks for an integer
      multiplier *k* >= 1; all ratio siblings share the same *k*.
    * :class:`FreeAllocation` – receives any rank count in ``[min_ranks, max_ranks]``
      from the budget remaining after fixed and ratio allocations.

    The total assigned across all sub-components may be less than *parent_ranks*;
    the unused ranks are left idle.  Use :class:`MaxWastedRankFractionConstraint`
    on the parent component to restrict idle cores.
    """
    n = len(subcomponents)
    allocs = [spec.allocation for spec in alloc_specs]

    fixed_indices = [i for i, a in enumerate(allocs) if isinstance(a, FixedRanks)]
    ratio_indices = [i for i, a in enumerate(allocs) if isinstance(a, RatioAllocation)]
    free_indices = [i for i, a in enumerate(allocs) if isinstance(a, FreeAllocation)]

    fixed_total = sum(allocs[i].n_ranks for i in fixed_indices)
    available = parent_ranks - fixed_total

    if available < 0:
        return  # fixed components alone exceed the budget

    free_allocs = cast(list[FreeAllocation], [allocs[i] for i in free_indices])
    free_min_total = sum(a.min_ranks for a in free_allocs)

    def _build_result(ratio_ranks: dict[int, int], free_assignment: tuple[int, ...]) -> tuple[int, ...]:
        result = [0] * n
        for i in fixed_indices:
            result[i] = allocs[i].n_ranks
        for i, r in ratio_ranks.items():
            result[i] = r
        for i, r in zip(free_indices, free_assignment, strict=True):
            result[i] = r
        return tuple(result)

    if ratio_indices:
        sum_ratio_weights = sum(allocs[i].weight for i in ratio_indices)
        max_k = (available - free_min_total) // sum_ratio_weights
        for k in range(1, max_k + 1):
            ratio_ranks = {i: k * allocs[i].weight for i in ratio_indices}
            remaining = available - k * sum_ratio_weights
            for free_assignment in _iter_free_assignments(free_allocs, remaining):
                yield _build_result(ratio_ranks, free_assignment)
    else:
        for free_assignment in _iter_free_assignments(free_allocs, available):
            yield _build_result({}, free_assignment)


def _default_alloc_spec(component: ParallelComponent) -> AllocationSpec:
    """Return a default all-:class:`FreeAllocation` spec tree for *component*."""
    return AllocationSpec(
        FreeAllocation(),
        subcomponents={sub.name: _default_alloc_spec(sub) for sub in component.subcomponents},
    )


def _validate_alloc_spec_names(component: ParallelComponent, alloc_spec: AllocationSpec, path: str = "root") -> None:
    """Validate that allocation-spec child names match the component tree."""
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
    alloc_spec: AllocationSpec,
) -> Iterator[ComponentLayout]:
    """Yield all valid :class:`ComponentLayout` objects for *component* with *n_ranks* and *tpr*."""
    decompositions: list[CartesianDecomposition | None]
    if component.domain is not None:
        decompositions = list(iter_cartesian_decompositions(component.domain, n_ranks))
    else:
        decompositions = [None]

    all_constraints = component.local_constraints + alloc_spec.local_constraints
    for decomp in decompositions:
        for sub_layout_tuple in _iter_valid_sub_layouts(component, n_ranks, tpr, total_ranks, alloc_spec):
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
    alloc_spec: AllocationSpec,
) -> Iterator[tuple[ComponentLayout, ...]]:
    """Yield sub-layout tuples for *component*'s subcomponents that satisfy group constraints."""
    if not component.subcomponents:
        yield ()
        return

    # Resolve the name-keyed dict to a positional tuple aligned with component.subcomponents.
    # Missing names fall back to FreeAllocation.
    ordered_alloc_specs = tuple(
        alloc_spec.subcomponents.get(sub.name, AllocationSpec(FreeAllocation())) for sub in component.subcomponents
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
    allocations: AllocationSpec | None = None,
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
    allocations : AllocationSpec | None
        Top-level :class:`~access.config.parallelisation.AllocationSpec` for *component*.
        Child allocation specs are provided via its ``subcomponents`` mapping.
        Missing names at any level fall back to
        :class:`~access.config.parallelisation.FreeAllocation`.
        The root ``allocation`` field is accepted for structural consistency,
        but is not used because the root always receives ``total_ranks``.
        ``None`` (the default) assigns
        :class:`~access.config.parallelisation.FreeAllocation` to every node in
        the tree recursively.

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
