# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.config.esm1p6_layout_input import (
    _generate_esm1p6_layout_from_core_counts,
    generate_esm1p6_core_layouts_from_node_count,
    generate_esm1p6_perturb_block,
    return_layout_tuple,
)


@pytest.fixture(scope="module")
def layout_tuple():
    return return_layout_tuple()


@pytest.fixture(scope="module")
def esm1p6_ctrl_layout():
    layout_tuple = return_layout_tuple()
    return layout_tuple(ncores_used=416, atm_nx=16, atm_ny=13, mom_nx=14, mom_ny=14, ice_ncores=12)  # Example layout


def test_generate_esm1p6_layout_from_core_counts(layout_tuple):
    # Test the the validation works
    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=120,
            max_atm_ncores=96,
            ice_ncores=6,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
        )
    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=1,
            max_atm_ncores=96,
            ice_ncores=6,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
        )
    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=96,
            max_atm_ncores=120,
            atm_ncore_delta=0,
            ice_ncores=6,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
        )
    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=96,
            max_atm_ncores=120,
            ice_ncores=0,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
        )

    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=96,
            max_atm_ncores=120,
            ice_ncores=6,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
            mom_ncores_over_atm_ncores_range=(-1.0, 1.0),
        )
    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=96,
            max_atm_ncores=120,
            ice_ncores=6,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
            mom_ncores_over_atm_ncores_range=(1.0, -1.0),
        )
    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=96,
            max_atm_ncores=120,
            ice_ncores=6,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
            mom_ncores_over_atm_ncores_range=(1.1, 0.9),
        )

    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=96,
            max_atm_ncores=120,
            ice_ncores=6,
            ncores_for_atm_and_ocn=208 - 2 * 6,
            min_ncores_needed=208,
        )

    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=2,
            max_atm_ncores=2,
            ice_ncores=6,
            ncores_for_atm_and_ocn=2,
            min_ncores_needed=1,
        )

    # Test with a valid core count
    core_count = 208
    max_atm_ncores = 120
    min_atm_ncores = 96
    ice_ncores = 6
    ncores_for_atm_and_ocn = core_count - ice_ncores
    min_ncores_needed = core_count - 1  # Allow for some unused cores

    layouts = _generate_esm1p6_layout_from_core_counts(
        max_atm_ncores=max_atm_ncores,
        min_atm_ncores=min_atm_ncores,
        ice_ncores=ice_ncores,
        ncores_for_atm_and_ocn=ncores_for_atm_and_ocn,
        min_ncores_needed=min_ncores_needed,
    )
    assert all(layout.ncores_used >= min_ncores_needed for layout in layouts), (
        f"Some layouts have ncores_used < min_ncores_needed. Min ncores needed: {min_ncores_needed}, "
        f"Min ncores used: {min([x.ncores_used for x in layouts])}"
    )
    assert all(layout.ncores_used <= (ncores_for_atm_and_ocn + layout.ice_ncores) for layout in layouts), (
        f"Some layouts have ncores_used > ncores_for_atm_and_ocn + ice_ncores. Max ncores for "
        f"atm and ocn: {ncores_for_atm_and_ocn}, Max ncores used: {max([x.ncores_used for x in layouts])}"
    )

    # Test that setting min_ncores_needed less than ncores_for_atm_and_ocn produces larger number of layouts
    layouts_without_min_ncores = _generate_esm1p6_layout_from_core_counts(
        max_atm_ncores=max_atm_ncores,
        min_atm_ncores=min_atm_ncores,
        ice_ncores=ice_ncores,
        ncores_for_atm_and_ocn=ncores_for_atm_and_ocn,
        min_ncores_needed=ncores_for_atm_and_ocn - 1,
    )

    assert len(layouts_without_min_ncores) >= len(layouts), (
        f"Expected more layouts when min_ncores_needed is less than "
        f"ncores_for_atm_and_ocn. Got {len(layouts_without_min_ncores)} vs {len(layouts)}"
    )
    assert all(x in layouts_without_min_ncores for x in layouts), (
        "All layouts from the first call should be in the second call"
    )

    # Test that the continue statement in the loop works by setting abs_maxdiff_nx_ny to 0
    min_atm_ncores = 98
    max_atm_ncores = 102
    ice_ncores = 6
    ncores_for_atm_and_ocn = 10 * 10 + (10 * 10 - 1)
    min_ncores_needed = 1
    abs_maxdiff_nx_ny = 0
    layouts = _generate_esm1p6_layout_from_core_counts(
        max_atm_ncores=max_atm_ncores,
        min_atm_ncores=min_atm_ncores,
        ice_ncores=ice_ncores,
        ncores_for_atm_and_ocn=ncores_for_atm_and_ocn,
        min_ncores_needed=min_ncores_needed,
        abs_maxdiff_nx_ny=abs_maxdiff_nx_ny,
    )
    assert layouts == [], f"Expected *no* layouts to be returned. Got layouts = {layouts}"

    # Test with zero cores
    core_count = 0
    with pytest.raises(ValueError):
        layouts = _generate_esm1p6_layout_from_core_counts(
            max_atm_ncores=max_atm_ncores,
            min_atm_ncores=min_atm_ncores,
            ice_ncores=ice_ncores,
            ncores_for_atm_and_ocn=0,
            min_ncores_needed=ice_ncores,
        )

    # Test that the layouts are returned with ncores_used <= ncores_for_atm_and_ocn
    assert all(x.ncores_used <= ncores_for_atm_and_ocn for x in layouts), (
        f"Some layouts have ncores_used > ncores_for_atm_and_ocn. "
        f"Max. ncores used : {max([x.ncores_used for x in layouts])}"
    )

    # Test that the cores_used are sorted in descending order
    assert all(layouts[i].ncores_used >= layouts[i + 1].ncores_used for i in range(len(layouts) - 1)), (
        "Layouts are not sorted in descending order of ncores_used"
    )


def test_generate_esm1p6_core_layouts_from_node_count(esm1p6_ctrl_layout):
    # Test the the validation works
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, tol_around_ctrl_ratio=-0.1)

    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(
            4, tol_around_ctrl_ratio=None, mom_ncores_over_atm_ncores_range=None
        )

    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, tol_around_ctrl_ratio=-0.1)

    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=(None,))
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=None)
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=(None, None))
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=(1.0, None))
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=(None, 1.0))
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=(1.2, 0.8))
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=(0.8, -1.0))
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, mom_ncores_over_atm_ncores_range=(-0.8, 1.0))

    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, atm_ncore_delta=0)
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, atm_ncore_delta=-1)

    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, abs_maxdiff_nx_ny=-1)

    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, max_wasted_ncores_frac=-0.1)
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(4, max_wasted_ncores_frac=1.01)

    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count([4, "abcd"])

    # Test with negative nodes
    node_count = -3
    with pytest.raises(ValueError):
        generate_esm1p6_core_layouts_from_node_count(node_count)

    # Test that with a very low node count, no layouts are returned (i.e. empty list of an empty list)
    layouts = generate_esm1p6_core_layouts_from_node_count([0.2], max_wasted_ncores_frac=0.2)
    assert layouts != [[]], f"Expected layouts to be returned even with small node fraction. Got layouts = {layouts}"

    layouts = generate_esm1p6_core_layouts_from_node_count([0.001], max_wasted_ncores_frac=0.5)
    assert layouts == [[]], f"Expected no layouts to be returned for nearly zero nodes. Got layouts = {layouts}"

    # Test with a valid node count that should return the control layout
    node_count = 4
    layouts = generate_esm1p6_core_layouts_from_node_count(node_count, tol_around_ctrl_ratio=0.0)[0]
    assert len(layouts) == 1, f"Expected *exactly* one layout to be returned. Got layouts = {layouts}"
    layouts = layouts[0]
    assert esm1p6_ctrl_layout == layouts, f"Control config layout={esm1p6_ctrl_layout} not found in solved {layouts}"

    # Test with a valid node count as a float that should return the control layout
    node_count = 4.0
    layouts = generate_esm1p6_core_layouts_from_node_count(node_count, tol_around_ctrl_ratio=0.0)[0]
    assert len(layouts) == 1, f"Expected *exactly* one layout to be returned. Got layouts = {layouts}"
    layouts = layouts[0]
    assert esm1p6_ctrl_layout == layouts, f"Control config layout={esm1p6_ctrl_layout} not found in solved {layouts}"

    # Test with zero nodes
    node_count = 0
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(node_count)

    # Test with non-integer nodes
    node_count = 2.5
    layouts = generate_esm1p6_core_layouts_from_node_count(node_count)
    assert layouts != [[]], f"Expected layouts to be returned for non-integer nodes. Got layouts = {layouts}"

    # Test that specifying mom_ncores_over_atm_ncores_range works
    node_count = 4
    mom_ncores_over_atm_ncores_range = (0.8, 1.2)
    layouts = generate_esm1p6_core_layouts_from_node_count(
        node_count, mom_ncores_over_atm_ncores_range=mom_ncores_over_atm_ncores_range
    )
    assert layouts != [[]], f"Expected layouts to be returned for non-integer nodes. Got layouts = {layouts}"

    # Test that allocating remaining cores to ICE works
    from access.config.layout_config import convert_num_nodes_to_ncores

    node_count, queue = 4, "normalsr"
    totncores = convert_num_nodes_to_ncores(node_count, queue=queue)
    mom_ncores_over_atm_ncores_range = (0.8, 1.2)
    layouts = generate_esm1p6_core_layouts_from_node_count(
        node_count,
        mom_ncores_over_atm_ncores_range=mom_ncores_over_atm_ncores_range,
        allocate_unused_cores_to_ice=True,
        queue=queue,
    )
    assert layouts != [[]], f"Expected layouts to be returned for non-integer nodes. Got layouts = {layouts}"
    assert all(layout.ice_ncores >= esm1p6_ctrl_layout.ice_ncores for layout in layouts[0]), (
        f"Expected ice_ncores to be >= {esm1p6_ctrl_layout.ice_ncores}. Got layout = {layouts[0]}"
    )
    assert all(layout.ncores_used == totncores for layout in layouts[0]), (
        f"Expected ncores used to be *exactly* equal to {totncores}. Got layout = {layouts[0]}"
    )


def test_generate_esm1p6_perturb_block(esm1p6_ctrl_layout):
    # Test that the validation works
    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(num_nodes=None, layouts=esm1p6_ctrl_layout, branch_name_prefix="test_block")
    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(num_nodes=-1, layouts=esm1p6_ctrl_layout, branch_name_prefix="test_block")

    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(num_nodes=4, layouts=esm1p6_ctrl_layout, branch_name_prefix=None)

    # Test with invalid layout
    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(num_nodes=4, layouts=None, branch_name_prefix="test_block")

    # Test with empty layout
    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(num_nodes=4, layouts=[[]], branch_name_prefix="test_block")

    # Test that the validation works for layouts with missing fields
    with pytest.raises(ValueError):
        missing_ice_ncores_layout = [[416, 16, 13, 14, 14]]
        generate_esm1p6_perturb_block(num_nodes=4, layouts=missing_ice_ncores_layout, branch_name_prefix="test_block")

    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(
            num_nodes=4, layouts=esm1p6_ctrl_layout, branch_name_prefix="test_block", start_blocknum=-1
        )

    # Test with valid parameters
    branch_name_prefix = "test_block"
    perturb_block, _ = generate_esm1p6_perturb_block(
        num_nodes=4, layouts=esm1p6_ctrl_layout, branch_name_prefix=branch_name_prefix
    )
    assert isinstance(perturb_block, str), f"Expected perturb block to be a string, but got: {type(perturb_block)}"
    assert branch_name_prefix in perturb_block, (
        f"Expected branch name prefix '{branch_name_prefix}' to be in perturb block, but got: {perturb_block}"
    )
