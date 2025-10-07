# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.config.layout_config import convert_num_nodes_to_ncores, find_layouts_with_maxncore, return_layout_tuple


@pytest.fixture(scope="module")
def layout_tuple():
    return return_layout_tuple()


def test_layout_tuple(layout_tuple):
    # Test that the tuple has the correct field names and types
    atm_nx, atm_ny = 6, 4
    mom_nx, mom_ny = 5, 4
    ice_ncores = 2
    ncores_used = atm_nx * atm_ny * mom_nx * mom_ny * ice_ncores
    layout = layout_tuple(ncores_used, atm_nx, atm_ny, mom_nx, mom_ny, ice_ncores)
    assert isinstance(layout.ncores_used, int), f"Expected int, got {type(layout.ncores_used)}"
    assert isinstance(layout.atm_nx, int), f"Expected int, got {type(layout.atm_nx)}"
    assert isinstance(layout.atm_ny, int), f"Expected int, got {type(layout.atm_ny)}"
    assert isinstance(layout.mom_nx, int), f"Expected int, got {type(layout.mom_nx)}"
    assert isinstance(layout.mom_ny, int), f"Expected int, got {type(layout.mom_ny)}"
    assert isinstance(layout.ice_ncores, int), f"Expected int, got {type(layout.ice_ncores)}"

    assert layout.ncores_used == ncores_used, f"Expected {ncores_used}, got {layout.ncores_used}"
    assert layout.atm_nx == atm_nx, f"Expected {atm_nx}, got {layout.atm_nx}"
    assert layout.atm_ny == atm_ny, f"Expected {atm_ny}, got {layout.atm_ny}"
    assert layout.mom_nx == mom_nx, f"Expected {mom_nx}, got {layout.mom_nx}"
    assert layout.mom_ny == mom_ny, f"Expected {mom_ny}, got {layout.mom_ny}"
    assert layout.ice_ncores == ice_ncores, f"Expected {ice_ncores}, got {layout.ice_ncores}"


def test_find_layouts_with_maxncore():
    maxncores = 20
    layouts = find_layouts_with_maxncore(maxncores)
    assert isinstance(layouts, list), f"Expected list, got {type(layouts)}"
    assert all(isinstance(layout, tuple) for layout in layouts), "All items in the list should be tuples"
    assert all(len(layout) == 2 for layout in layouts), "All tuples should have length 2"
    assert all(isinstance(n, int) for layout in layouts for n in layout), (
        "All elements in the tuples should be integers"
    )
    assert all(layout[0] * layout[1] <= maxncores for layout in layouts), (
        f"All layouts should have nx * ny <= {maxncores}"
    )
    assert (5, 4) in layouts, f"(5, 4) should be in the layouts for maxncores={maxncores}"
    assert (4, 5) in layouts, f"(4, 5) should be in the layouts for maxncores={maxncores}"
    assert (4, 4) not in layouts, f"(4, 4) should *not* be in the layouts for maxncores={maxncores}"
    assert (4, 1) not in layouts, f"(4, 1) should *not* be in the layouts for maxncores={maxncores}"

    layouts_even_nx = find_layouts_with_maxncore(maxncores, even_nx=True)
    assert (5, 4) not in layouts_even_nx, (
        "(5, 4) should *not* be in the layouts for maxncores={maxncores} with even_nx=True"
    )
    assert all(layout[0] % 2 == 0 for layout in layouts_even_nx), "All nx should be even when even_nx=True"

    layouts_nx_ge_ny = find_layouts_with_maxncore(maxncores, prefer_nx_greater_than_ny=True)
    assert all(layout[0] >= layout[1] for layout in layouts_nx_ge_ny), (
        "All layouts should have nx >= ny when prefer_nx_greater_than_ny=True"
    )

    layouts_both = find_layouts_with_maxncore(maxncores, even_nx=True, prefer_nx_greater_than_ny=True)
    assert all(layout[0] % 2 == 0 and layout[0] >= layout[1] for layout in layouts_both), (
        "All layouts should have even nx and nx >= ny when both options are set"
    )

    layout_none = find_layouts_with_maxncore(1, even_nx=True)
    assert layout_none == [], "No layouts should be found for maxncores=1 with even_nx=True"

    layout_abs_maxdiff = find_layouts_with_maxncore(maxncores, abs_maxdiff_nx_ny=1)
    assert all(abs(layout[0] - layout[1]) <= 1 for layout in layout_abs_maxdiff), (
        "All layouts should have abs(nx - ny) <= 1 when abs_maxdiff_nx_ny=1"
    )

    layout_abs_maxdiff = find_layouts_with_maxncore(16, abs_maxdiff_nx_ny=0)
    assert len(layout_abs_maxdiff) == 1 and layout_abs_maxdiff[0] == (4, 4), (
        "Only (4, 4) should be found for maxncores=16 with abs_maxdiff_nx_ny=0"
    )

    with pytest.raises(ValueError):
        find_layouts_with_maxncore(0)
    with pytest.raises(ValueError):
        find_layouts_with_maxncore(-4)
    with pytest.raises(ValueError):
        find_layouts_with_maxncore(16, abs_maxdiff_nx_ny=-1)
    with pytest.raises(ValueError):
        find_layouts_with_maxncore(-16, even_nx=True)
    with pytest.raises(ValueError):
        find_layouts_with_maxncore(-16, prefer_nx_greater_than_ny=True)
    with pytest.raises(ValueError):
        find_layouts_with_maxncore(-16, even_nx=True, prefer_nx_greater_than_ny=True)


def test_convert_num_nodes_to_ncores():
    with pytest.raises(ValueError):
        convert_num_nodes_to_ncores(2.5, queue="broadwell")
    with pytest.raises(ValueError):
        convert_num_nodes_to_ncores(2.5, queue="unknown_queue")

    assert convert_num_nodes_to_ncores(2, queue="normalsr") == 208, (
        f"Expected 208, got {convert_num_nodes_to_ncores(2, queue='normalsr')}"
    )
    assert convert_num_nodes_to_ncores(2.0, queue="normalsr") == 208, (
        f"Expected 208, got {convert_num_nodes_to_ncores(2.0, queue='normalsr')}"
    )
    assert convert_num_nodes_to_ncores(1, queue="normalsr") == 104, (
        f"Expected 104, got {convert_num_nodes_to_ncores(1, queue='normalsr')}"
    )
    assert convert_num_nodes_to_ncores(1.0, queue="normalsr") == 104, (
        f"Expected 104, got {convert_num_nodes_to_ncores(1.0, queue='normalsr')}"
    )
    assert convert_num_nodes_to_ncores(0.5) == 52, f"Expected 52, got {convert_num_nodes_to_ncores(0.5)}"
    assert convert_num_nodes_to_ncores(0.5, queue="normal") == 24, (
        f"Expected 24, got {convert_num_nodes_to_ncores(0.5, queue='normal')}"
    )
    assert convert_num_nodes_to_ncores(3, queue="normal") == 144, (
        f"Expected 144, got {convert_num_nodes_to_ncores(3, queue='normal')}"
    )
    assert convert_num_nodes_to_ncores(3.0, queue="normal") == 144, (
        f"Expected 144, got {convert_num_nodes_to_ncores(3.0, queue='normal')}"
    )
    with pytest.raises(ValueError):
        convert_num_nodes_to_ncores([2])
