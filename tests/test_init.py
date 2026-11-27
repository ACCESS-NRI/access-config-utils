# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Tests for what the package itself exports."""

import inspect

import access.config
from access.config import parallel_allocation_strategies, parallel_constraints
from access.config.parallel_component import GroupConstraint, LocalConstraint


def test_every_name_listed_is_exported() -> None:
    """Test that ``__all__`` names only what the package has."""
    assert all(hasattr(access.config, name) for name in access.config.__all__)


def test_the_allocation_strategies_are_exported() -> None:
    """Test that every strategy a caller writes is importable from the package.

    The abstract base is not: a strategy tree is written with the root and the modes.
    """
    modes = set(parallel_allocation_strategies.__all__) - {"AllocationStrategy"}

    assert modes <= set(access.config.__all__)
    assert all(getattr(access.config, name) is getattr(parallel_allocation_strategies, name) for name in modes)


def test_the_constraints_are_exported() -> None:
    """Test that every constraint of ``parallel_constraints`` is exported by the package."""
    constraints = {
        name
        for name, value in vars(parallel_constraints).items()
        if inspect.isclass(value)
        and value.__module__ == parallel_constraints.__name__
        and issubclass(value, (LocalConstraint, GroupConstraint))
    }

    assert len(constraints) == 10
    assert constraints <= set(access.config.__all__)
