from access.config.layout_config import (
    layout_tuple,
    convert_num_nodes_to_ncores,
    find_layouts_with_maxncore,
)
import sys  
import logging
logger = logging.getLogger(__name__)


def _get_esm1p6_layout_from_core_counts(
    min_atm_ncores: int,
    max_atm_ncores: int,
    atm_ncore_delta: int,
    prefer_atm_nx_greater_than_ny: bool,
    prefer_atm_ncores_greater_than_mom_ncores: bool,
    ncores_left: int,
    ice_ncores: int,
    min_ncores_needed: int,
    abs_maxdiff_nx_ny: int,
    mom_ncores_over_atm_ncores_range: (float, float),
) -> list:
    if min_atm_ncores < 2 or max_atm_ncores < 2 or min_atm_ncores > max_atm_ncores:
        raise ValueError("Invalid ATM ncores range")

    if atm_ncore_delta <= 0:
        raise ValueError("atm_ncore_delta must be a positive integer")

    if ncores_left < 3:
        raise ValueError("ncores_left must be at least 3 (2 for atm and 1 for mom)")

    if ice_ncores < 1:
        raise ValueError("ice_ncores must be at least 1")

    if min_ncores_needed < (3 + ice_ncores):
        raise ValueError(
            f"Min. number of cores must be at least {3 + ice_ncores} (2 for atm, 1 for mom and {ice_ncores} for ice)"
        )

    if (
        mom_ncores_over_atm_ncores_range[0] <= 0.0
        or mom_ncores_over_atm_ncores_range[1] <= 0.0
        or mom_ncores_over_atm_ncores_range[0] > mom_ncores_over_atm_ncores_range[1]
    ):
        raise ValueError(f"Invalid MOM ncores over ATM ncores fractions. Got {mom_ncores_over_atm_ncores_range} instead")

    all_layouts = []
    for atm_ncores in range(min_atm_ncores, max_atm_ncores + 1, atm_ncore_delta):
        atm_layout = find_layouts_with_maxncore(
            atm_ncores,
            abs_maxdiff_nx_ny=abs_maxdiff_nx_ny,
            even_nx=True,
            prefer_nx_greater_than_ny=prefer_atm_nx_greater_than_ny,
        )
        if not atm_layout:
            continue

        min_mom_ncores = int(atm_ncores * mom_ncores_over_atm_ncores_range[0])
        max_mom_ncores = int(atm_ncores * mom_ncores_over_atm_ncores_range[1])
        for atm in atm_layout:
            atm_nx, atm_ny = atm

            mom_ncores = ncores_left - atm_nx * atm_ny
            if mom_ncores < min_mom_ncores or mom_ncores > max_mom_ncores:
                continue

            mom_layout = find_layouts_with_maxncore(
                mom_ncores, abs_maxdiff_nx_ny=abs_maxdiff_nx_ny
            )
            if not mom_layout:
                continue

            # filter mom_layout to only include layouts with ncores in the range [min_mom_ncores, max_mom_ncores]
            layout = []
            for mom_nx, mom_ny in mom_layout:
                mom_ncores = mom_nx * mom_ny
                if mom_ncores < min_mom_ncores or mom_ncores > max_mom_ncores:
                    logger.debug(f"Skipping mom layout {mom_nx}x{mom_ny} with {mom_ncores} ncores not in the range [{min_mom_ncores}, {max_mom_ncores}]")
                    continue

                if prefer_atm_ncores_greater_than_mom_ncores and (
                    atm_nx * atm_ny < mom_ncores
                ):
                    logger.debug(f"Skipping mom layout {mom_nx}x{mom_ny} with {mom_ncores} ncores not less than atm ncores {atm_nx * atm_ny}")
                    continue

                ncores_used = mom_nx * mom_ny + atm_nx * atm_ny + ice_ncores
                if ncores_used < min_ncores_needed:
                    logger.debug(f"Skipping layout atm {atm_nx}x{atm_ny} mom {mom_nx}x{mom_ny} ice {ice_ncores} with {ncores_used} ncores used < min_ncores_needed = {min_ncores_needed}")
                    continue

                layout.append(
                    layout_tuple(ncores_used, atm_nx, atm_ny, mom_nx, mom_ny, ice_ncores)
                )

            # create a set of layouts to avoid duplicates
            all_layouts.extend(set(layout))

    return all_layouts


def generate_esm1p6_core_layouts_from_node_count(
    num_nodes_list: float,
    *,
    queue="normalsr",
    tol_around_ctrl_ratio=None,  # if set, keep the min and max fractions of MOM ncores over ATM ncores close to the control ratio
    min_frac_mom_ncores_over_atm_ncores=0.75,
    max_frac_mom_ncores_over_atm_ncores=1.25,
    atm_ncores_nsteps=100,
    prefer_atm_nx_greater_than_ny=True,
    prefer_atm_ncores_greater_than_mom_ncores=True,
    abs_maxdiff_nx_ny=4,
    max_wasted_ncores_frac=0.01,
) -> list:
    """
    Given a list of target number of nodes to use, this function generates
    possible core layouts for the Atmosphere and Ocean for the ESM 1.6 PI config.

    Parameters
    ----------
    num_nodes_list : scalar or a list of integer/floats, required
        A positive number or a list of positive numbers representing the number of nodes to use.

    queue : str, optional
        Queue name on ``gadi``. Allowed values are "normalsr" and "normal".
        Default is "normalsr".

    tol_around_ctrl_ratio : float, optional
        If set, the min and max fractions of MOM ncores over ATM ncores will be set to (at most) within
        (1 ± tol_around_ctrl_ratio) of the released PI config. Must be in the range [0.0, 1.0].
        If not set, the min and max fractions of MOM ncores over ATM ncores are used.
        Default is None.

    min_frac_mom_ncores_over_atm_ncores : float, optional
        Minimum fraction of MOM ncores over ATM ncores to consider when generating layouts.
        Must be greater than 0. Ignored if ``tol_around_ctrl_ratio`` is set.
        Default is 0.75.

    max_frac_mom_ncores_over_atm_ncores : float, optional
        Maximum fraction of MOM ncores over ATM ncores to consider when generating layouts.
        Must be greater than 0. Ignored if ``tol_around_ctrl_ratio`` is set.
        Default is 1.25.

    atm_ncores_nsteps : int, optional
        Number of steps to take between the min. and max. ATM ncores when generating layouts.
        Must be a positive integer.
        Default is 100.

    prefer_atm_nx_greater_than_ny : bool, optional
        If True, only consider ATM layouts with nx >= ny.
        Default is True.

    prefer_atm_ncores_greater_than_mom_ncores : bool, optional
        If True, only consider layouts with ATM ncores >= MOM ncores.
        Default is True.

    abs_maxdiff_nx_ny : int, optional
        Absolute max. of the difference between nx and ny (in the solved layout) to 
        consider when generating layouts. Must be a non-negative integer.
        Default is 4.

    max_wasted_ncores_frac : float, optional
        Maximum fraction of wasted cores (i.e. not used by atm, mom or ice) to allow when generating layouts.
        Must be in the range [0.0, 1.0].
        Default is 0.01.

    Returns
    -------
    list
        A list of lists of layout_tuples. Each inner list corresponds to the layouts for the respective
        number of nodes in ``num_nodes_list``. Each layout_tuple has the following fields:
        - ncores_used : int
        - atm_nx : int
        - atm_ny : int
        - mom_nx : int
        - mom_ny : int
        - ice_ncores : int

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.

    Notes
    -----
    - atm requires nx to be even -> atm requires min 2 ncores (2x1 layout),
      mom requires min 1 ncore (1x1 layout), ice requires min 1 ncore
    - The released configuration used is:
        - atm: 16x13 (208 cores)
        - ocn: 14x14 (196 cores)
        - ice: 12    (12 cores)
        - queue: normalsr
        - num_nodes: 4 (416 cores)
    """


    if tol_around_ctrl_ratio:
        if tol_around_ctrl_ratio < 0.0 or tol_around_ctrl_ratio > 1.0:
            raise ValueError(
                f"The tolerance fraction for setting the MOM to ATM core ratio to the control ratio must be in [0.0, 1.0]. {tol_around_ctrl_ratio} was provided instead"
            )
        min_frac_mom_ncores_over_atm_ncores = None
        max_frac_mom_ncores_over_atm_ncores = None

    if min_frac_mom_ncores_over_atm_ncores and min_frac_mom_ncores_over_atm_ncores <= 0:
        raise ValueError(f"The minimum fraction of MOM ncores over ATM ncores must be greater than 0. Got {min_frac_mom_ncores_over_atm_ncores} instead")

    if max_frac_mom_ncores_over_atm_ncores and max_frac_mom_ncores_over_atm_ncores <= 0:
        raise ValueError(f"The maximum fraction of MOM ncores over ATM ncores must be greater than 0. Got {max_frac_mom_ncores_over_atm_ncores} instead")

    if min_frac_mom_ncores_over_atm_ncores and max_frac_mom_ncores_over_atm_ncores and
        min_frac_mom_ncores_over_atm_ncores > max_frac_mom_ncores_over_atm_ncores
        raise ValueError(f"Invalid MOM ncores over ATM ncores fractions - min. must be <= max."
                         f" Got min={min_frac_mom_ncores_over_atm_ncores} and max={max_frac_mom_ncores_over_atm_ncores} instead")

    if atm_ncores_nsteps <= 0:
        raise ValueError(f"The number of steps to take between the min. and max. ATM ncores must be a positive integer. Got {atm_ncores_nsteps} instead")

    if abs_maxdiff_nx_ny < 0:
        raise ValueError("The absolute max. of the difference between nx and ny (in the solved layout) must be a non-negative integer. Got {abs_maxdiff_nx_ny} instead")

    if max_wasted_ncores_frac < 0 or max_wasted_ncores_frac > 1.0:
        raise ValueError("The max. fraction of wasted cores must be in the range [0.0, 1.0]. Got {max_wasted_ncores_frac} instead")

    if not isinstance(num_nodes_list, list):
        num_nodes_list = [num_nodes_list]

    if any(n <= 0 for n in num_nodes_list) or any(
        not isinstance(n, (int, float)) for n in num_nodes_list
    ):
        raise ValueError(
            "num_nodes must be a positive number or a list of positive numbers"
        )

    # atm requires nx to be even -> atm requires min 2 ncores 
    # (2x1 layout), mom requires min 1 ncore (1x1 layout), ice requires min 1 ncore
    min_cores_required = 2 + 1 + 1
    if any(convert_num_nodes_to_ncores(n, queue=queue) < min_cores_required
           for n in num_nodes_list):
        logger.warning(f"Warning: Some of the provided num_nodes values are too low to run the model. The minimum number of cores required is {min_cores_required} for the {queue} queue")

    ctrl_num_nodes, ctrl_config, ctrl_queue = 4, {"atm": (16, 13), "ocn": (14, 14), "ice": (12)}, "normalsr"
    ctrl_ratio_mom_over_atm = (
        ctrl_config["ocn"][0]
        * ctrl_config["ocn"][1]
        / (ctrl_config["atm"][0] * ctrl_config["atm"][1])
    )

    if tol_around_ctrl_ratio:
        logger.debug(
            f"The min and max fractions of MOM ncores over ATM ncores will be set to (at most) within (1 \u00b1 {tol_around_ctrl_ratio})"
            f" of the control ratio={ctrl_ratio_mom_over_atm:0.3g}"
        )
    final_layouts = []
    for num_nodes in num_nodes_list:
        # cast num_nodes to int if it is an integer value
        if isinstance(num_nodes, float) and num_nodes.is_integer():
            num_nodes = int(num_nodes)

        totncores = convert_num_nodes_to_ncores(num_nodes, queue=ctrl_queue)
        if totncores < min_cores_required:
            logger.warning(
                f"Total ncores = {totncores} is less than the min. number of cores required = {min_cores_required}. Skipping"
            )
            final_layouts.append([])
            continue

        logger.debug(f"Generating layouts for {num_nodes = } nodes")
        full_layouts = []
        ctrl_totncores = convert_num_nodes_to_ncores(ctrl_num_nodes, queue=queue)
        ctrl_ice_ncores = ctrl_config["ice"]
        ice_ncores = max(1, int(ctrl_ice_ncores / ctrl_totncores * totncores))

        ncores_left = totncores - ice_ncores
        max_wasted_ncores = int(totncores * max_wasted_ncores_frac)
        min_ncores_needed = ncores_left - max_wasted_ncores

        # atm + mom = totncores - cice_ncores
        # => 1 + mom/atm = (totncores - cice_ncores)/atm
        # => 1 + min_frac_mom_ncores_over_atm_ncores = (totncores - cice_ncores)/max_atm_ncores
        # => max_atm_ncores = (totncores - cice_ncores)/(1 + min_frac_mom_ncores_over_atm_ncores)
        if not tol_around_ctrl_ratio:
            max_atm_ncores = max(
                2, int(ncores_left / (1.0 + min_frac_mom_ncores_over_atm_ncores))
            )
            min_atm_ncores = max(
                2, int(ncores_left / (1.0 + max_frac_mom_ncores_over_atm_ncores))
            )
            mom_ncores_over_atm_ncores_range = (
                min_frac_mom_ncores_over_atm_ncores,
                max_frac_mom_ncores_over_atm_ncores,
            )
        else:
            target_atm_ncores = ncores_left / (
                1.0 + ctrl_ratio_mom_over_atm
            )  # intentionally not converted to int here -> done in the next lines
            max_atm_ncores = max(
                2, int(target_atm_ncores * (1.0 + tol_around_ctrl_ratio))
            )  # allow some variation around the control ratio
            min_atm_ncores = max(
                2, int(target_atm_ncores * (1.0 - tol_around_ctrl_ratio))
            )  # allow some variation around the control ratio
            mom_ncores_over_atm_ncores_range = (
                ctrl_ratio_mom_over_atm * (1.0 - tol_around_ctrl_ratio),
                ctrl_ratio_mom_over_atm * (1.0 + tol_around_ctrl_ratio),
            )

        if prefer_atm_ncores_greater_than_mom_ncores:
            mom_ncores_over_atm_ncores_range = (mom_ncores_over_atm_ncores_range[0], 1.0)

        atm_ncore_delta = max(1, int((max_atm_ncores - min_atm_ncores) / atm_ncores_nsteps))

        # convert max and min atm ncores to be even
        if max_atm_ncores % 2 != 0:
            max_atm_ncores -= 1
        if min_atm_ncores % 2 != 0:
            min_atm_ncores += 1

        logger.debug(
            f"ATM ncores range, steps = ({min_atm_ncores}, {max_atm_ncores}, {atm_ncore_delta})"
        )
        logger.debug(
            f"MOM ncores range = ({ncores_left - max_atm_ncores}, {ncores_left - min_atm_ncores})"
        )
        full_layouts.extend(
            _get_esm1p6_layout_from_core_counts(
                min_atm_ncores=min_atm_ncores,
                max_atm_ncores=max_atm_ncores,
                atm_ncore_delta=atm_ncore_delta,
                prefer_atm_nx_greater_than_ny=prefer_atm_nx_greater_than_ny,
                prefer_atm_ncores_greater_than_mom_ncores=prefer_atm_ncores_greater_than_mom_ncores,
                ncores_left=ncores_left,
                ice_ncores=ice_ncores,
                min_ncores_needed=min_ncores_needed,
                abs_maxdiff_nx_ny=abs_maxdiff_nx_ny,
                mom_ncores_over_atm_ncores_range=mom_ncores_over_atm_ncores_range,
            )
        )

        full_layouts = set(full_layouts)
        # Sort on ncores used, descending and then the sum of (abs(atm_nx - atm_ny) + abs(mom_nx - mom_ny)), ascending
        # I am using the negative of ncores used to sort in descending order
        sorted_layouts = sorted(
            full_layouts, key=lambda x: (-x[0], abs(x[1] - x[2]) + abs(x[3] - x[4]))
        )
        logger.debug(
            f"Found {len(sorted_layouts)} layouts for {num_nodes = } nodes"
        )
        final_layouts.append(sorted_layouts)

    return final_layouts


def _generate_esm1p6_perturb_block(
    num_nodes: (float | int),
    layouts: list,
    queue: str,
    branch_name_prefix: str,
    start_blocknum: int,
) -> str:
    if num_nodes is None:
        raise ValueError("num_nodes must be provided.}")

    if not isinstance(num_nodes, (int, float)) or num_nodes <= 0:
        raise ValueError(
            "num_nodes must be a positive number or a list of positive numbers. Got {} instead".format(
                num_nodes
            )
        )

    # cast num_nodes to int if it is an integer value
    if isinstance(num_nodes, float) and num_nodes.is_integer():
        num_nodes = int(num_nodes)

    if branch_name_prefix is None:
        raise ValueError("branch_name_prefix must be provided")

    if not layouts:
        raise ValueError("No layouts provided")

    if not start_blocknum or start_blocknum < 1:
        raise ValueError("start_blocknum must be a positive integer greater than 0")

    totncores = convert_num_nodes_to_ncores(num_nodes, queue=queue)
    blocknum = start_blocknum
    block = ""
    for _, atm_nx, atm_ny, mom_nx, mom_ny, ice_ncores in layouts:
        atm_ncores = atm_nx * atm_ny
        mom_ncores = mom_nx * mom_ny
        branch_name = f"{branch_name_prefix}_atm_{atm_nx}x{atm_ny}_mom_{mom_nx}x{mom_ny}_ice_{ice_ncores}x1"
        block += f"""
  Scaling_numnodes_{num_nodes}_totncores_{totncores}_ncores_used_{atm_ncores + mom_ncores + ice_ncores}_seqnum_{blocknum}:
    branches:
      - {branch_name}
    config.yaml:
      submodels:
        - - ncpus: # atmosphere
              - {atm_ncores} # ncores for atmosphere
          - ncpus: # ocean
              - {mom_ncores} # ncores for ocean
          - ncpus: # ice
              - {ice_ncores} # ncores for ice

    atmosphere/um_env.yaml:
      UM_ATM_NPROCX: {atm_nx}
      UM_ATM_NPROCY: {atm_ny}
      UM_NPES: {atm_ncores}

    ocean/input.nml:
        ocean_model_nml:
            layout:
                - {mom_nx},{mom_ny}

    ice/cice_in.nml:
          domain_nml:
                - {ice_ncores}
    """
        blocknum += 1

    return block, blocknum


if __name__ == "__main__":
    from typing import NamedTuple

    scaling_config_tup = NamedTuple(
        "scaling_config_tup",
        [
            ("num_nodes", float),
            ("min_frac_mom_ncores_over_atm_ncores", float),
            ("max_frac_mom_ncores_over_atm_ncores", float),
            ("tol_around_ctrl_ratio", (float | None)),
            ("atm_ncores_nsteps", int),
            ("abs_maxdiff_nx_ny", int),
            ("max_wasted_ncores_frac", float),
            (
                "allocate_unused_cores_to_ice",
                bool,
            ),  # if True, allocate all unused cores to ice
        ],
    )

    scaling_configs = [
        scaling_config_tup(
            num_nodes=0.25,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.10,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.050,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=0.50,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.10,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.050,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=1.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.05,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.020,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=2.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.02,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.010,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=3.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.02,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.010,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=4.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.02,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.010,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=5.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.02,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=2,
            max_wasted_ncores_frac=0.005,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=6.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.02,
            atm_ncores_nsteps=100,
            abs_maxdiff_nx_ny=2,
            max_wasted_ncores_frac=0.005,
            allocate_unused_cores_to_ice=False,
        ),
    ]

    model_type = "access-esm1.6"
    test_path = f"tests/{model_type}/test_scaling_layouts"
    repository_directory = f"{model_type}-PI-config"
    generator_config_prefix = f"""
model_type: {model_type}
repository_url: git@github.com:ACCESS-NRI/{model_type}-configs.git
start_point: "1ebd393" # the commit hash that access-bot refers to when committing the new checksums

test_path: {test_path}
repository_directory: {repository_directory}

control_branch_name: ctrl

Control_Experiment:
  config.yaml:
    walltime: 10:0:0
    modules:
      use:
        - /g/data/vk83/modules
      load:
        - access-esm1p6/2025.09.002

    manifest:
      reproduce:
        exe: false    # cice5 has to be manually compiled because the runtime core counts is set at compile time

    repeat: True
    runspersub: 10
    """
    blocknum = 1
    queue = "normalsr"
    branch_name_prefix = "esm1p6-layout"
    for i, config in enumerate(scaling_configs):
        num_nodes = config.num_nodes
        layout = generate_esm1p6_core_layouts_from_node_count(
            num_nodes,
            queue=queue,
            min_frac_mom_ncores_over_atm_ncores=config.min_frac_mom_ncores_over_atm_ncores,
            max_frac_mom_ncores_over_atm_ncores=config.max_frac_mom_ncores_over_atm_ncores,
            tol_around_ctrl_ratio=config.tol_around_ctrl_ratio,
            atm_ncores_nsteps=config.atm_ncores_nsteps,
            abs_maxdiff_nx_ny=config.abs_maxdiff_nx_ny,
            max_wasted_ncores_frac=config.max_wasted_ncores_frac,
        )
        if not layout:
            print(f"No layouts found for {num_nodes} nodes", file=sys.stderr)
            continue

        layout = layout[0]

        if config.allocate_unused_cores_to_ice:
            totncores = convert_num_nodes_to_ncores(num_nodes, queue=queue)
            print(
                f"# Adding another layout with allocating unused ncores to ice for {num_nodes} nodes",
                file=sys.stderr,
            )
            new_layouts = [
                layout_tuple(
                    totncores,
                    atm_nx,
                    atm_ny,
                    mom_nx,
                    mom_ny,
                    ice_ncores + totncores - ncores_used,
                )
                for ncores_used, atm_nx, atm_ny, mom_nx, mom_ny, ice_ncores in layout
                if ncores_used < totncores
            ]
            layout.extend(new_layouts)
            layout = sorted(
                layout, key=lambda x: (-x[0], abs(x[1] - x[2]) + abs(x[3] - x[4]))
            )

        block, blocknum = _generate_esm1p6_perturb_block(
            num_nodes, layout, queue, branch_name_prefix, blocknum
        )
        blocknum += 1
        print(block)
