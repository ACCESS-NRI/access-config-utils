# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.config.esm1p6_layout_input import (
    LayoutSearchConfig,
    _generate_esm1p6_layout_from_core_counts,
    generate_esm1p6_core_layouts_from_node_count,
    generate_esm1p6_perturb_block,
)
from access.config.layout_config import LayoutTuple


@pytest.fixture(scope="module")
def layout_tuple():
    return LayoutTuple


@pytest.fixture(scope="module")
def esm1p6_ctrl_layout(layout_tuple):
    return layout_tuple(atm_nx=16, atm_ny=13, mom_nx=14, mom_ny=14, ice_ncores=12)  # Example layout


@pytest.fixture(scope="module")
def layout_search_config():
    return LayoutSearchConfig


def test_set_ice_ncores():
    from access.config.esm1p6_layout_input import set_ice_ncores

    # Test that the validation works
    with pytest.raises(ValueError):
        set_ice_ncores(min_ice_ncores=-1, max_ice_ncores=10)

    with pytest.raises(ValueError):
        set_ice_ncores(min_ice_ncores=10, max_ice_ncores=5)

    with pytest.raises(ValueError):
        set_ice_ncores(min_ice_ncores=0, max_ice_ncores=0)

    with pytest.raises(ValueError):
        set_ice_ncores(min_ice_ncores=361, max_ice_ncores=400, blocksize=360)

    with pytest.raises(ValueError):
        set_ice_ncores(min_ice_ncores=10, max_ice_ncores=20, blocksize=0)

    with pytest.raises(ValueError):
        ice_ncores = set_ice_ncores(min_ice_ncores=13, max_ice_ncores=14)

    # Test with valid parameters
    ice_ncores = set_ice_ncores(min_ice_ncores=5, max_ice_ncores=12)
    assert ice_ncores == 5, f"Expected ice_ncores to be *exactly* 5. Got ice_ncores = {ice_ncores}"
    assert 360 % ice_ncores == 0, f"Expected ice_ncores to be a factor of 360. Got ice_ncores = {ice_ncores}"

    ice_ncores = set_ice_ncores(min_ice_ncores=5, max_ice_ncores=12, smallest_factor=False)
    assert ice_ncores == 12, f"Expected ice_ncores to be *exactly* 12. Got ice_ncores = {ice_ncores}"
    assert 360 % ice_ncores == 0, f"Expected ice_ncores to be a factor of 360. Got ice_ncores = {ice_ncores}"

    ice_ncores = set_ice_ncores(min_ice_ncores=1, max_ice_ncores=1)
    assert ice_ncores == 1, f"Expected ice_ncores to be 1. Got ice_ncores = {ice_ncores}"


def test_layout_search_config(layout_search_config):
    # # Test valid initialization
    config = layout_search_config()
    assert config is not None, "Config should not be None"
    assert isinstance(config, LayoutSearchConfig), f"Expected LayoutSearchConfig, got {type(config)}"
    assert config.frac_mom_ncores_over_atm_ncores[0] == 0.75, (
        f"Expected 0.75, got {config.frac_mom_ncores_over_atm_ncores[0]}"
    )
    assert config.frac_mom_ncores_over_atm_ncores[1] == 1.25, (
        f"Expected 1.25, got {config.frac_mom_ncores_over_atm_ncores[1]}"
    )
    assert config.tol_around_ctrl_ratio is None, f"Expected None, got {config.tol_around_ctrl_ratio}"
    assert config.atm_ncore_stepsize == 2, f"Expected 2, got {config.atm_ncore_stepsize}"
    assert config.abs_maxdiff_nx_ny == 4, f"Expected 4, got {config.abs_maxdiff_nx_ny}"
    assert config.max_wasted_ncores_frac == 0.02, f"Expected 0.02, got {config.max_wasted_ncores_frac}"
    assert config.allocate_unused_cores_to_ice is False, f"Expected False, got {config.allocate_unused_cores_to_ice}"

    # Test invalid min/max frac
    with pytest.raises(ValueError):
        layout_search_config(frac_mom_ncores_over_atm_ncores=(-0.5, 1.25))
    with pytest.raises(ValueError):
        layout_search_config(frac_mom_ncores_over_atm_ncores=(0.75, -1.0))
    with pytest.raises(ValueError):
        layout_search_config(frac_mom_ncores_over_atm_ncores=(1.5, 1.0))
    with pytest.raises(TypeError):
        layout_search_config(frac_mom_ncores_over_atm_ncores="low")
    with pytest.raises(TypeError):
        layout_search_config(frac_mom_ncores_over_atm_ncores=(0.1, 0.2, "string"))
    with pytest.raises(TypeError):
        layout_search_config(frac_mom_ncores_over_atm_ncores=("low", "hi"))

    # Test invalid tol_around_ctrl_ratio
    with pytest.raises(ValueError):
        layout_search_config(tol_around_ctrl_ratio=-0.1)
    with pytest.raises(ValueError):
        layout_search_config(tol_around_ctrl_ratio=1.5)
    with pytest.raises(TypeError):
        layout_search_config(tol_around_ctrl_ratio="medium")

    # Test invalid atm_ncore_stepsize
    with pytest.raises(TypeError):
        layout_search_config(atm_ncore_stepsize=2.5)
    with pytest.raises(ValueError):
        layout_search_config(atm_ncore_stepsize=0)

    # Test that when tol_around_ctrl_ratio is set, max_frac and min_frac are set to None
    config = layout_search_config(
        tol_around_ctrl_ratio=0.1,
        frac_mom_ncores_over_atm_ncores=(0.5, 1.0),
    )
    assert config.tol_around_ctrl_ratio == 0.1, f"Expected 0.1, got {config.tol_around_ctrl_ratio}"
    assert config.frac_mom_ncores_over_atm_ncores == (0.8480769230769231, 1.0), (
        f"Expected (0.8480769230769231, 1.0), got {config.frac_mom_ncores_over_atm_ncores}"
    )

    # Test that ValueError is raised if both tol_around_ctrl_ratio and frac_mom_ncores_over_atm_ncores are None
    with pytest.raises(ValueError):
        layout_search_config(
            tol_around_ctrl_ratio=None,
            frac_mom_ncores_over_atm_ncores=None,
        )

    # Test that setting the min. and max. frac. directly works
    config = layout_search_config(frac_mom_ncores_over_atm_ncores=(0.5, 1.5))
    assert config.frac_mom_ncores_over_atm_ncores == (0.5, 1.5), (
        f"Expected (0.5, 1.5), got {config.frac_mom_ncores_over_atm_ncores}"
    )
    assert config.tol_around_ctrl_ratio is None, f"Expected None, got {config.tol_around_ctrl_ratio}"

    # Test the invalid abs_maxdiff_nx_ny
    with pytest.raises(ValueError):
        layout_search_config(abs_maxdiff_nx_ny=-1)
    with pytest.raises(TypeError):
        layout_search_config(abs_maxdiff_nx_ny=2.5)

    # Test the invalid max_wasted_ncores_frac
    with pytest.raises(ValueError):
        layout_search_config(max_wasted_ncores_frac=-0.1)
    with pytest.raises(ValueError):
        layout_search_config(max_wasted_ncores_frac=1.5)
    with pytest.raises(TypeError):
        layout_search_config(max_wasted_ncores_frac="low")

    config = layout_search_config(max_wasted_ncores_frac=0.5)
    assert config.max_wasted_ncores_frac == 0.5, f"Expected 0.5, got {config.max_wasted_ncores_frac}"

    # Test that the boolean attributes are set correctly
    bool_attr_names = [
        "prefer_atm_nx_greater_than_ny",
        "prefer_mom_nx_greater_than_ny",
        "prefer_atm_ncores_greater_than_mom_ncores",
        "allocate_unused_cores_to_ice",
    ]
    for attr_name in bool_attr_names:
        with pytest.raises(TypeError):
            layout_search_config(**{attr_name: "yes"})

        with pytest.raises(TypeError):
            layout_search_config(**{attr_name: 2.5})

        config = layout_search_config(**{attr_name: True})
        assert getattr(config, attr_name) is True, f"Expected True, got {getattr(config, attr_name)}"

        config = layout_search_config(**{attr_name: False})
        assert getattr(config, attr_name) is False, f"Expected False, got {getattr(config, attr_name)}"


def test_generate_esm1p6_layout_from_core_counts(layout_tuple, layout_search_config):
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
            ice_ncores=0,
            ncores_for_atm_and_ocn=208 - 6,
            min_ncores_needed=0,
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
        layout_search_config=layout_search_config(
            abs_maxdiff_nx_ny=abs_maxdiff_nx_ny,
        ),
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


def test_generate_esm1p6_core_layouts_from_node_count(esm1p6_ctrl_layout, layout_search_config):
    # Test that the validation works
    with pytest.raises(TypeError):
        layouts = generate_esm1p6_core_layouts_from_node_count("abcd", cores_per_node=104)

    # Test with negative nodes
    node_count = -3
    with pytest.raises(ValueError):
        generate_esm1p6_core_layouts_from_node_count(node_count, cores_per_node=104)

    # Test with negative cores per node
    with pytest.raises(ValueError):
        generate_esm1p6_core_layouts_from_node_count(1, cores_per_node=-1)

    # Test that with a very low node count, no layouts are returned (i.e. empty list of an empty list)
    layouts = generate_esm1p6_core_layouts_from_node_count(
        0.2, cores_per_node=104, layout_search_config=layout_search_config(max_wasted_ncores_frac=0.2)
    )
    assert layouts != [], f"Expected layouts to be returned even with small node fraction. Got layouts = {layouts}"

    # Test that no layouts are returned with nearly zero nodes
    layouts = generate_esm1p6_core_layouts_from_node_count(
        0.001, cores_per_node=104, layout_search_config=layout_search_config(max_wasted_ncores_frac=0.5)
    )
    assert layouts == [], f"Expected no layouts to be returned for nearly zero nodes. Got layouts = {layouts}"

    # Test with a valid node count that should return the control layout
    node_count = 4
    layouts = generate_esm1p6_core_layouts_from_node_count(
        node_count, cores_per_node=104, layout_search_config=layout_search_config(tol_around_ctrl_ratio=0.0)
    )
    assert len(layouts) == 1, f"Expected *exactly* one layout to be returned. Got layouts = {layouts}"
    layouts = layouts[0]
    assert esm1p6_ctrl_layout == layouts, f"Control config layout={esm1p6_ctrl_layout} not found in solved {layouts}"

    # Test with a valid node count as a float that should return the control layout
    node_count = 4.0
    layouts = generate_esm1p6_core_layouts_from_node_count(
        node_count, cores_per_node=104, layout_search_config=layout_search_config(tol_around_ctrl_ratio=0.0)
    )
    assert len(layouts) == 1, f"Expected *exactly* one layout to be returned. Got layouts = {layouts}"
    layouts = layouts[0]
    assert esm1p6_ctrl_layout == layouts, f"Control config layout={esm1p6_ctrl_layout} not found in solved {layouts}"

    # Test with zero nodes
    node_count = 0
    with pytest.raises(ValueError):
        layouts = generate_esm1p6_core_layouts_from_node_count(node_count, cores_per_node=104)

    # Test with non-integer nodes
    node_count = 2.5
    layouts = generate_esm1p6_core_layouts_from_node_count(node_count, cores_per_node=104)
    assert layouts != [], f"Expected layouts to be returned for non-integer nodes. Got layouts = {layouts}"

    # Test that specifying frac_mom_ncores_over_atm_ncores works
    node_count = 4
    frac_mom_ncores_over_atm_ncores = (0.8, 1.2)
    layouts = generate_esm1p6_core_layouts_from_node_count(
        node_count,
        cores_per_node=104,
        layout_search_config=layout_search_config(frac_mom_ncores_over_atm_ncores=frac_mom_ncores_over_atm_ncores),
    )
    assert layouts != [], f"Expected layouts to be returned for non-integer nodes. Got layouts = {layouts}"

    # Test that the layouts are all unique
    node_count = 4
    layouts = generate_esm1p6_core_layouts_from_node_count(node_count, cores_per_node=104)
    assert len(layouts) == len(set(layouts)), f"Got duplicate elements in layouts. {layouts = }"

    # Test that allocating remaining cores to ICE works
    node_count = 4
    totncores = node_count * 104
    frac_mom_ncores_over_atm_ncores = (0.8, 1.2)
    layouts = generate_esm1p6_core_layouts_from_node_count(
        node_count,
        cores_per_node=104,
        layout_search_config=layout_search_config(
            frac_mom_ncores_over_atm_ncores=frac_mom_ncores_over_atm_ncores, allocate_unused_cores_to_ice=True
        ),
    )
    assert layouts != [], f"Expected layouts to be returned for non-integer nodes. Got layouts = {layouts}"
    assert all(layout.ice_ncores >= esm1p6_ctrl_layout.ice_ncores for layout in layouts), (
        f"Expected ice_ncores to be >= {esm1p6_ctrl_layout.ice_ncores}. Got layout = {layouts}"
    )
    assert all(layout.ncores_used <= totncores for layout in layouts), (
        f"Expected ncores used to be at most equal to {totncores}. Got layout = {layouts}"
    )


def test_generate_esm1p6_perturb_block(esm1p6_ctrl_layout):
    # Test that the validation works
    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(layout=esm1p6_ctrl_layout, branch_name_prefix=None)

    # Test with invalid layout
    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(layout=None, branch_name_prefix="test_block")

    # Test with empty layout
    with pytest.raises(ValueError):
        generate_esm1p6_perturb_block(layout=[], branch_name_prefix="test_block")

    # Test that the validation works for layouts with missing fields
    with pytest.raises(ValueError):
        missing_ice_ncores_layout = [416, 16, 13, 14, 14]
        generate_esm1p6_perturb_block(layout=missing_ice_ncores_layout, branch_name_prefix="test_block")

    # Test with valid parameters
    branch_name_prefix = "test_block"
    perturb_block = generate_esm1p6_perturb_block(layout=esm1p6_ctrl_layout, branch_name_prefix=branch_name_prefix)
    assert isinstance(perturb_block, dict), f"Expected perturb block to be a dict, but got: {type(perturb_block)}"
    assert len(perturb_block["branches"]) == 1, (
        f"Expected one branch in the perturb block, but got: {len(perturb_block['branches'])}"
    )
    # Check that the branch name prefix is correctly included in the name of the first (and only) branch
    assert branch_name_prefix in perturb_block["branches"][0], (
        f"Expected branch name prefix '{branch_name_prefix}' to be in perturb block, but got: {perturb_block}"
    )
