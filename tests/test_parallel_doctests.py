# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
"""Runs the ``Examples`` blocks of the parallel_* modules as doctests.

The ``Examples:`` sections in these modules are written as interactive sessions, so
they can be executed. Doing so keeps them honest: an example that stops matching the
code it documents fails here instead of quietly misleading a reader.

Only the ``parallel_*`` modules are covered. The rest of the package documents by
prose and inline code rather than by interactive sessions, so it has nothing to run.
"""

import doctest
from types import ModuleType

import pytest

from access.config import (
    parallel_allocation_strategies,
    parallel_component,
    parallel_constraints,
    parallel_domain,
    parallel_layouts,
    parallel_mpi_grid,
)

PARALLEL_MODULES = [
    parallel_domain,
    parallel_mpi_grid,
    parallel_component,
    parallel_constraints,
    parallel_allocation_strategies,
    parallel_layouts,
]


@pytest.mark.parametrize("module", PARALLEL_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_module_doctests(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False, report=True)
    assert results.failed == 0, f"{results.failed} of {results.attempted} doctests failed in {module.__name__}"
