# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""A library of ready-made rules for filtering candidate parallel layouts.

Enumerating layouts turns up every structurally valid way to spend a core budget, which is
always far more than is wanted: process grids too elongated to exchange halos efficiently,
sub-domains too small to be worth a rank, one component handed a share of the machine out
of all proportion to its cost. A constraint is how such a candidate gets ruled out. This
module collects the rules that recur often enough to be worth naming, so that callers state
which ones apply instead of writing predicates.

Concepts
--------
constraint
    A yes-or-no test applied to a candidate layout. Returning ``False`` rejects that one
    candidate and means nothing more: the search moves on, so a rule that no layout can
    satisfy shows up as a search that found nothing, not as an error.
local and group
    A local constraint judges one component's layout on its own. A group constraint judges
    the layouts of a whole set of sibling sub-components together - a required ratio between
    two of them, say - and so is attached to their parent rather than to any one sibling.
applicability
    Not every rule can be evaluated against every layout: a grid rule needs a decomposition,
    a thread rule needs the thread count only a leaf has, and an idle-core rule needs a
    parent with sub-components to have left cores unspent. Asking anyway is a mistake in
    how the rule was attached, not a layout to reject, so these raise ``ValueError``
    rather than quietly answering ``True`` or ``False``. The ``_require_*`` helpers below
    hold those checks, one per precondition.

The rules fall into four families, in the order they appear below:

1. **Cartesian grid** - the shape of the grid of ranks a domain is split over: divisibility
   of a chosen dimension, and how far from square the grid as a whole may be.
2. **Core distribution** - how the budget is shared out: the largest fraction a single
   component may take, how much a parent may leave idle, and rank ratios between siblings.
3. **Domain layout** - the block of the domain a rank ends up holding: whether every rank
   holds exactly the same block, how small a block may get, and its aspect ratio.
4. **Threading** - the thread count a leaf runs on each of its ranks: a ceiling, or an
   exact required value.
"""

from __future__ import annotations

from dataclasses import dataclass

from access.config import parallel_domain
from access.config.parallel_component import (
    ComponentLayout,
    GroupConstraint,
    LocalConstraint,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_decomposition(
    constraint: LocalConstraint, layout: ComponentLayout
) -> parallel_domain.DomainDecompositionSpec:
    """Return *layout*'s decomposition, or raise if it has none.

    A constraint that inspects a domain decomposition cannot evaluate a component
    that has no domain, so this raises ``ValueError`` rather than reporting the
    layout as rejected.
    """
    if layout.decomposition is None:
        raise ValueError(
            f"{constraint.__class__.__name__} requires layout.decomposition, "
            f"but layout {layout.name!r} has no decomposition."
        )
    return layout.decomposition


def _require_threads_per_rank(constraint: LocalConstraint, layout: ComponentLayout) -> int:
    """Return *layout*'s thread count, or raise if it has none.

    Only a leaf runs ranks of its own, so only a leaf has a thread count. A parent's
    sub-components may each use a different one, which is why a thread constraint on a
    parent cannot be evaluated and must be attached to the leaf it describes.
    """
    if layout.threads_per_rank is None:
        raise ValueError(
            f"{constraint.__class__.__name__} requires layout.threads_per_rank, but layout "
            f"{layout.name!r} distributes its cores to sub-components and so has no thread count "
            "of its own. Attach this constraint to a leaf component."
        )
    return layout.threads_per_rank


def _require_grid_dim(constraint: LocalConstraint, layout: ComponentLayout, dim: int) -> int:
    """Return the number of ranks along *dim* of *layout*'s process grid.

    Raises ``ValueError`` if the layout has no decomposition, or if *dim* is out
    of range for its grid. The range cannot be checked when the constraint is
    constructed, because the grid is only known once a layout exists.
    """
    grid = _require_decomposition(constraint, layout).grid
    if dim >= grid.ndim:
        raise ValueError(
            f"{constraint.__class__.__name__}.dim={dim} is out of range for a {grid.ndim}-dimensional grid."
        )
    return grid[dim]


# ---------------------------------------------------------------------------
# Category 1 — Cartesian grid constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessGridDimEvenConstraint(LocalConstraint):
    """``grid[dim]`` must be even.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        dim (int): Zero-based dimension index to check.
    """

    dim: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"ProcessGridDimEvenConstraint.dim must be >= 0, got {self.dim}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        return _require_grid_dim(self, layout, self.dim) % 2 == 0


@dataclass(frozen=True)
class ProcessGridDimDivisibleConstraint(LocalConstraint):
    """``grid[dim]`` must be divisible by ``divisor``.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        dim (int): Zero-based dimension index to check.
        divisor (int): Required divisor. Must be >= 1.
    """

    dim: int
    divisor: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"ProcessGridDimDivisibleConstraint.dim must be >= 0, got {self.dim}.")
        if self.divisor < 1:
            raise ValueError(f"ProcessGridDimDivisibleConstraint.divisor must be >= 1, got {self.divisor}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        return _require_grid_dim(self, layout, self.dim) % self.divisor == 0


@dataclass(frozen=True)
class ProcessGridAspectRatioConstraint(LocalConstraint):
    """``max(grid) / min(grid)`` must not exceed ``max_ratio``.

    This enforces near-square decompositions and helps balance communication costs.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        max_ratio (float): Maximum allowed ratio between the largest and smallest grid
            dimension. Must be >= 1.0.
    """

    max_ratio: float

    def __post_init__(self) -> None:
        if self.max_ratio < 1.0:
            raise ValueError(f"ProcessGridAspectRatioConstraint.max_ratio must be >= 1.0, got {self.max_ratio}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        g = _require_decomposition(self, layout).grid
        return max(g) / min(g) <= self.max_ratio


# ---------------------------------------------------------------------------
# Category 2 — Core distribution constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxCoreFractionConstraint(LocalConstraint):
    """This component must not receive more than ``max_fraction`` of the whole budget.

    Measured in cores rather than ranks, because cores are what a layout distributes:
    two components on the same number of ranks can consume very different shares of the
    machine when their thread counts differ.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        max_fraction (float): Maximum fraction of ``total_cores`` allowed. Must be in
            ``(0.0, 1.0]``.
    """

    max_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 < self.max_fraction <= 1.0):
            raise ValueError(f"MaxCoreFractionConstraint.max_fraction must be in (0, 1], got {self.max_fraction}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        if total_cores < 1:
            raise ValueError(f"{self.__class__.__name__} requires total_cores >= 1, got {total_cores}.")
        # Compare the quotient, never `max_fraction * total_cores`: the product form
        # rounds the wrong way at the boundary (0.29 * 100 == 28.999999999999996, so
        # 29 of 100 cores would be rejected), whereas n / total rounds to the same
        # double as the fraction it is compared against.
        return layout.n_cores / total_cores <= self.max_fraction


@dataclass(frozen=True)
class MaxWastedCoreFractionConstraint(LocalConstraint):
    """The fraction of idle cores must not exceed ``max_fraction``.

    Idle cores are those allocated to this component but not handed to any
    sub-component: ``ComponentLayout.idle_cores``.

    This constraint must be placed on a *parent* component, to limit how many of the
    available cores are left unused. A leaf spends its cores exactly, by construction,
    so idle cores are not a meaningful quantity for it: applying this constraint to a
    leaf raises ``ValueError`` rather than silently reporting it as satisfied.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        max_fraction (float): Maximum allowed fraction of idle cores. Must be in
            ``[0.0, 1.0]``. Use ``0.0`` to require every core to be handed out.
    """

    max_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_fraction <= 1.0):
            raise ValueError(
                f"MaxWastedCoreFractionConstraint.max_fraction must be in [0.0, 1.0], got {self.max_fraction}."
            )

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        if layout.is_leaf:
            raise ValueError(
                f"{self.__class__.__name__} requires layout to have sub-layouts, "
                f"but layout {layout.name!r} has no sub-layouts."
            )
        # Quotient form, for the same reason as in MaxCoreFractionConstraint: comparing
        # against `max_fraction * n_cores` misjudges exact boundary cases.
        return layout.idle_cores / layout.n_cores <= self.max_fraction


@dataclass(frozen=True)
class RankRatioGroupConstraint(GroupConstraint):
    """``n_ranks(name_a) / n_ranks(name_b)`` must be >= ``min_ratio``.

    Place this on the *parent* component's ``ParallelComponent.group_constraints``.
    Uses sub-component names rather than positional indices, so the constraint
    remains correct regardless of subcomponent ordering.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        name_a (str): Name of the numerator sub-component.
        name_b (str): Name of the denominator sub-component.
        min_ratio (float): Minimum required ratio. Must be > 0.
    """

    name_a: str
    name_b: str
    min_ratio: float

    def __post_init__(self) -> None:
        if self.min_ratio <= 0.0:
            raise ValueError(f"RankRatioGroupConstraint.min_ratio must be > 0, got {self.min_ratio}.")

    def is_satisfied(self, sub_layouts: tuple[ComponentLayout, ...], total_cores: int) -> bool:
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

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.
    """

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        d = _require_decomposition(self, layout)
        return all(dim % g == 0 for dim, g in zip(d.domain.shape, d.grid, strict=True))


@dataclass(frozen=True)
class MinSubdomainSizeConstraint(LocalConstraint):
    """Each local sub-domain must have at least ``min_size`` grid points per dimension.

    Checks ``floor(shape[i] / grid[i]) >= min_size`` for every dimension.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        min_size (int): Minimum number of grid points per sub-domain dimension. Must be
            >= 1.
    """

    min_size: int

    def __post_init__(self) -> None:
        if self.min_size < 1:
            raise ValueError(f"MinSubdomainSizeConstraint.min_size must be >= 1, got {self.min_size}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        d = _require_decomposition(self, layout)
        return all(dim // g >= self.min_size for dim, g in zip(d.domain.shape, d.grid, strict=True))


@dataclass(frozen=True)
class SubdomainAspectRatioConstraint(LocalConstraint):
    """``max(mean_local_shape) / min(mean_local_shape)`` must not exceed ``max_ratio``.

    Constrains the aspect ratio of the *mean* subdomain in physical space
    (``domain.shape[i] / grid[i]`` per dimension), not the MPI process grid.
    Use this to ensure that the grid-point block held by a typical rank is near-square,
    which improves halo-exchange balance. Because it uses the mean, individual ranks may
    deviate when a dimension does not divide evenly.

    This constraint requires ``layout.decomposition`` to be present.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        max_ratio (float): Maximum allowed ratio between the longest and shortest local
            subdomain dimension. Must be >= 1.0.
    """

    max_ratio: float

    def __post_init__(self) -> None:
        if self.max_ratio < 1.0:
            raise ValueError(f"SubdomainAspectRatioConstraint.max_ratio must be >= 1.0, got {self.max_ratio}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        ls = _require_decomposition(self, layout).mean_local_shape
        return max(ls) / min(ls) <= self.max_ratio


# ---------------------------------------------------------------------------
# Category 4 — Thread constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxThreadsPerRankConstraint(LocalConstraint):
    """``threads_per_rank`` must not exceed ``max_threads``.

    Only a leaf has a thread count, so this must be attached to a leaf component;
    applying it to a parent raises ``ValueError``.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        max_threads (int): Maximum allowed threads per rank. Must be >= 1.
    """

    max_threads: int

    def __post_init__(self) -> None:
        if self.max_threads < 1:
            raise ValueError(f"MaxThreadsPerRankConstraint.max_threads must be >= 1, got {self.max_threads}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        return _require_threads_per_rank(self, layout) <= self.max_threads


@dataclass(frozen=True)
class FixedThreadsPerRankConstraint(LocalConstraint):
    """``threads_per_rank`` must equal exactly ``n_threads``.

    Only a leaf has a thread count, so this must be attached to a leaf component;
    applying it to a parent raises ``ValueError``.

    Frozen (``frozen=True``): an immutable, hashable value object, so one instance can be
    attached to several components without risk of it changing between checks.

    Args:
        n_threads (int): Required thread count per rank. Must be >= 1.
    """

    n_threads: int

    def __post_init__(self) -> None:
        if self.n_threads < 1:
            raise ValueError(f"FixedThreadsPerRankConstraint.n_threads must be >= 1, got {self.n_threads}.")

    def is_satisfied(self, layout: ComponentLayout, total_cores: int) -> bool:
        return _require_threads_per_rank(self, layout) == self.n_threads
