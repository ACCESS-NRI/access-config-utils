import logging
import sys

from access.config.layout_config import (
    convert_num_nodes_to_ncores,
    find_layouts_with_maxncore,
    return_layout_tuple,
)

logger = logging.getLogger(__name__)


def _generate_esm1p6_layout_from_core_counts(
    min_atm_ncores: int,
    max_atm_ncores: int,
    ncores_for_atm_and_ocn: int,
    ice_ncores: int,
    min_ncores_needed: int,
    *,
    atm_ncore_delta: int = 2,
    abs_maxdiff_nx_ny: int = 4,
    prefer_atm_nx_greater_than_ny: bool = True,
    prefer_mom_nx_greater_than_ny: bool = True,
    prefer_atm_ncores_greater_than_mom_ncores: bool = True,
    mom_ncores_over_atm_ncores_range: (float, float) = (0.75, 1.25),
) -> list:
    """
    Returns a list of possible core layouts for the Atmosphere and Ocean for the ESM 1.6 PI config

    Parameters
    ----------

    min_atm_ncores : int, required
        Minimum number of ATM cores to consider when generating layouts. 
        Must be at least 2 and less than or equal to max_atm_ncores.

    max_atm_ncores : int, required
        Maximum number of ATM cores to consider when generating layouts. 
        Must be at least 2 and greater than or equal to min_atm_ncores.

    ncores_for_atm_and_ocn : int, required
        Total number of cores available for ATM and MOM. 
        Must be at least 3 (2 for atm and 1 for mom).

    ice_ncores : int, required
        Number of cores allocated to ICE. Must be at least 1.

    min_ncores_needed : int, required
        Minimum number of cores that must be used by ATM, MOM and ICE combined. 
        Must be at least 3 + ice_ncores (2 for ATM, 1 for MOM and ``ice_ncores`` for ice).
        Layouts using fewer cores will be discarded.

    atm_ncore_delta : int, optional, default=2
        Step size to use when iterating between min_atm_ncores and max_atm_ncores. 
        Must be a non-zero and positive integer.

    abs_maxdiff_nx_ny : int, optional, default=4
        Absolute max. of the difference between nx and ny (in the solved layout) to consider 
        when generating layouts. Must be a non-negative integer. 

        Setting to 0 will return only square layouts. Applies to both ATM and MOM layouts.

    prefer_atm_nx_greater_than_ny : bool, optional, default=True
        If True, only consider ATM layouts with nx >= ny.

    prefer_mom_nx_greater_than_ny : bool, optional, default=True
        If True, only consider MOM layouts with nx >= ny.

    prefer_atm_ncores_greater_than_mom_ncores : bool, optional, default=True
        If True, only consider layouts with ATM ncores >= MOM ncores.

    mom_ncores_over_atm_ncores_range : tuple of float, optional, default=(0.75, 1.25)
        A tuple of two floats representing the minimum and maximum fractions of MOM ncores 
        over ATM ncores to consider when generating layouts.
    """

    min_atm_and_mom_ncores = 3  # atm requires min 2 ncores (2x1 layout), mom requires min 1 ncore (1x1 layout)

    if min_atm_ncores < 2 or max_atm_ncores < 2 or min_atm_ncores > max_atm_ncores:
        raise ValueError(f"Invalid ATM ncores range. Got ({min_atm_ncores}, {max_atm_ncores}) instead")

    if atm_ncore_delta <= 0:
        raise ValueError(
            "Stepsize in core counts to cover min. and max. ATM ncores must be a positive integer. "\
            f"Got {atm_ncore_delta} instead"
        )

    if ncores_for_atm_and_ocn < min_atm_and_mom_ncores:
        raise ValueError(
            "Number of cores available for ATM and OCN must be at least {min_atm_and_mom_ncores} "\
            f"(2 for atm and 1 for mom). Got {ncores_for_atm_and_ocn} instead"
        )

    if ice_ncores < 1:
        raise ValueError(f"ice_ncores must be at least 1. Got {ice_ncores} instead")

    if min_ncores_needed > (ncores_for_atm_and_ocn + ice_ncores):
        raise ValueError(
            f"Min. number of cores needed ({min_ncores_needed}) cannot be greater than the total "\
            f"number of available cores ({ncores_for_atm_and_ocn + ice_ncores})"
        )

    if min_ncores_needed < ncores_for_atm_and_ocn:
        logger.warning(
            f"Min. total cores required for a valid config ({min_ncores_needed}) should be greater "\
            f"than the number of ATM + OCN cores ({ncores_for_atm_and_ocn}). "
            f"Currently, any config that satisfies the ATM + OCN core requirements will also satisfy "\
            "the requirement for the min. total cores"
        )

    if (
        mom_ncores_over_atm_ncores_range[0] <= 0.0
        or mom_ncores_over_atm_ncores_range[1] <= 0.0
        or mom_ncores_over_atm_ncores_range[0] > mom_ncores_over_atm_ncores_range[1]
    ):
        raise ValueError(
            f"Invalid MOM ncores over ATM ncores fractions. Got {mom_ncores_over_atm_ncores_range} instead"
        )

    layout_tuple = return_layout_tuple()
    all_layouts = []
    logger.debug(
        f"Generating layouts with {min_atm_ncores=}, {max_atm_ncores=}, {atm_ncore_delta=}, "\
        f"{ncores_for_atm_and_ocn=}, {ice_ncores=}, {min_ncores_needed=}, "\
        f"{mom_ncores_over_atm_ncores_range=}, {abs_maxdiff_nx_ny=}, "\
        f"{prefer_atm_nx_greater_than_ny=}, {prefer_mom_nx_greater_than_ny=}, "\
        f"{prefer_atm_ncores_greater_than_mom_ncores=}"
    )
    for atm_ncores in range(min_atm_ncores, max_atm_ncores + 1, atm_ncore_delta):
        logger.debug(f"Trying atm_ncores = {atm_ncores}")
        atm_layout = find_layouts_with_maxncore(
            atm_ncores,
            abs_maxdiff_nx_ny=abs_maxdiff_nx_ny,
            even_nx=True,
            prefer_nx_greater_than_ny=prefer_atm_nx_greater_than_ny,
        )
        if not atm_layout:
            continue

        logger.debug(f"  Found {len(atm_layout)} atm layouts for atm_ncores = {atm_ncores}: {atm_layout}")

        min_mom_ncores = int(atm_ncores * mom_ncores_over_atm_ncores_range[0])
        max_mom_ncores = int(atm_ncores * mom_ncores_over_atm_ncores_range[1])
        for atm in atm_layout:
            atm_nx, atm_ny = atm

            mom_ncores = ncores_for_atm_and_ocn - atm_nx * atm_ny
            logger.debug(f"  Trying atm layout {atm_nx}x{atm_ny} with {atm_nx * atm_ny} ncores")
            mom_layout = find_layouts_with_maxncore(
                mom_ncores,
                abs_maxdiff_nx_ny=abs_maxdiff_nx_ny,
                prefer_nx_greater_than_ny=prefer_mom_nx_greater_than_ny,
            )
            if not mom_layout:
                continue

            # filter mom_layout to only include layouts with ncores in the range [min_mom_ncores, max_mom_ncores]
            layout = []
            for mom_nx, mom_ny in mom_layout:
                mom_ncores = mom_nx * mom_ny
                if mom_ncores < min_mom_ncores or mom_ncores > max_mom_ncores:
                    logger.debug(
                        f"Skipping mom layout {mom_nx}x{mom_ny} with {mom_ncores} ncores "\
                        f"not in the range [{min_mom_ncores}, {max_mom_ncores}]"
                    )
                    continue

                if prefer_atm_ncores_greater_than_mom_ncores and (atm_nx * atm_ny < mom_ncores):
                    logger.debug(
                        f"Skipping mom layout since mom ncores = {mom_nx}x{mom_ny} is not less "\
                        f"than atm ncores = {atm_nx * atm_ny}"
                    )
                    continue

                ncores_used = mom_nx * mom_ny + atm_nx * atm_ny + ice_ncores
                if ncores_used < min_ncores_needed:
                    logger.debug(
                        f"Skipping layout atm {atm_nx}x{atm_ny} mom {mom_nx}x{mom_ny} ice {ice_ncores}, "\
                        f"with {ncores_used=} is less than {min_ncores_needed=}"
                    )
                    continue

                logger.debug(
                    f"Adding layout atm {atm_nx}x{atm_ny} mom {mom_nx}x{mom_ny} ice {ice_ncores} with {ncores_used=}"
                )
                layout.append(layout_tuple(ncores_used, atm_nx, atm_ny, mom_nx, mom_ny, ice_ncores))

            # create a set of layouts to avoid duplicates
            all_layouts.extend(set(layout))

    if all_layouts:
        # sort the layouts by ncores_used (descending, fewer wasted cores first), and then
        # the sum of the absolute differences between nx and ny for atm and mom (ascending,
        # i.e., more square layouts first)
        all_layouts = sorted(
            all_layouts, key=lambda x: (-x.ncores_used, abs(x.atm_nx - x.atm_ny) + abs(x.mom_nx - x.mom_ny))
        )

    return all_layouts


def generate_esm1p6_core_layouts_from_node_count(
    num_nodes_list: float,
    *,
    queue: str = "normalsr",
    tol_around_ctrl_ratio: float = None,
    atm_ncore_delta: int = 2,
    prefer_atm_nx_greater_than_ny: bool = True,
    prefer_mom_nx_greater_than_ny: bool = True,
    prefer_atm_ncores_greater_than_mom_ncores: bool = True,
    abs_maxdiff_nx_ny: int = 4,
    mom_ncores_over_atm_ncores_range: (float, float) = (0.75, 1.25),
    max_wasted_ncores_frac: float = 0.01,
    allocate_unused_cores_to_ice: bool = False,
) -> list:
    """
    Given a list of target number of nodes to use, this function generates
    possible core layouts for the Atmosphere and Ocean for the ESM 1.6 PI config.

    Parameters
    ----------
    num_nodes_list : scalar or a list of integer/floats, required
        A positive number or a list of positive numbers representing the number of nodes to use.

    queue : str, optional, default="normalsr"
        Queue name on ``gadi``. Allowed values are "normalsr" and "normal".

    tol_around_ctrl_ratio : float, optional, default=None
        If set, the min and max fractions of MOM ncores over ATM ncores will be set to (at most) within
        (1 ± tol_around_ctrl_ratio) of the released PI config. Must be in the range [0.0, 1.0].
        If not set, the min and max fractions of MOM ncores over ATM ncores are used from the
        ``mom_ncores_over_atm_ncores_range`` parameter.

    mom_ncores_over_atm_ncores_range : tuple of float, optional, default=(0.75, 1.25)
        A tuple of two floats representing the minimum and maximum fractions of MOM ncores over ATM
        ncores to consider when generating layouts. Must be greater than 0.0, and the second
        value (i.e, the max.) must be at least equal to the first value (i.e., the min.)
        Layouts with MOM ncores over ATM ncores outside this range will be discarded.

        *Note*: This parameter is ignored if ``tol_around_ctrl_ratio`` is set.

    atm_ncore_delta : int, optional, default=100
        Number of steps to take between the min. and max. ATM ncores when generating layouts.
        Must be a positive integer.

    prefer_atm_nx_greater_than_ny : bool, optional, default=True
        If True, only consider ATM layouts with nx >= ny.

    prefer_mom_nx_greater_than_ny : bool, optional, default=True
        If True, only consider MOM layouts with nx >= ny.

    prefer_atm_ncores_greater_than_mom_ncores : bool, optional, default=True
        If True, only consider layouts with ATM ncores >= MOM ncores.

    abs_maxdiff_nx_ny : int, optional, default=4
        Absolute max. of the difference between nx and ny (in the solved layout) to
        consider when generating layouts. Must be a non-negative integer.

    allocate_unused_cores_to_ice : bool, optional, default=False
        If True, any unused cores (i.e., total cores - atm cores - mom cores) will be allocated to ice.

    max_wasted_ncores_frac : float, optional, default=0.01
        Maximum fraction of wasted cores (i.e. not used by atm, mom or ice) to allow when generating layouts.
        Must be in the range [0.0, 1.0].

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

        An empty list is returned for a given number of nodes if no valid layouts could be generated.

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

    if tol_around_ctrl_ratio is None and mom_ncores_over_atm_ncores_range is None:
        raise ValueError("Either tol_around_ctrl_ratio or mom_ncores_over_atm_ncores_range must be provided")

    if tol_around_ctrl_ratio is not None:
        if tol_around_ctrl_ratio < 0.0:
            raise ValueError(
                f"The tolerance fraction for setting the MOM to ATM core ratio to the control ratio must "\
                f"be in >= 0.0. Got {tol_around_ctrl_ratio} instead"
            )
    else:  # tol_around_ctrl_ratio is None -> use mom_ncores_over_atm_ncores_range
        if not isinstance(mom_ncores_over_atm_ncores_range, tuple) or len(mom_ncores_over_atm_ncores_range) != 2:
            raise ValueError(
                f"The min. and max. fraction of MOM ncores over ATM ncores must be a tuple of two floats. "\
                f"Got {mom_ncores_over_atm_ncores_range} instead"
            )

        min_frac_mom_ncores_over_atm_ncores, max_frac_mom_ncores_over_atm_ncores = mom_ncores_over_atm_ncores_range
        if not min_frac_mom_ncores_over_atm_ncores or not max_frac_mom_ncores_over_atm_ncores:
            raise ValueError(
                f"The min. and max. fraction of MOM ncores over ATM ncores must be a valid float. Got "\
                f"min={min_frac_mom_ncores_over_atm_ncores} and max={max_frac_mom_ncores_over_atm_ncores} instead"
            )

        if min_frac_mom_ncores_over_atm_ncores <= 0:
            raise ValueError(
                "The minimum fraction of MOM ncores over ATM ncores must be greater than 0. "\
                f"Got {min_frac_mom_ncores_over_atm_ncores} instead"
            )

        if max_frac_mom_ncores_over_atm_ncores <= 0:
            raise ValueError(
                f"The maximum fraction of MOM ncores over ATM ncores must be greater than 0. "\
                f"Got {max_frac_mom_ncores_over_atm_ncores} instead"
            )

        if min_frac_mom_ncores_over_atm_ncores > max_frac_mom_ncores_over_atm_ncores:
            raise ValueError(
                f"Invalid MOM ncores over ATM ncores fractions - min. must be <= max."
                f" Got min={min_frac_mom_ncores_over_atm_ncores} and "\
                f"max={max_frac_mom_ncores_over_atm_ncores} instead"
            )

    if atm_ncore_delta <= 0:
        raise ValueError(
            f"The stepsize in core counts to take between the min. and max. ATM ncores must "\
            f"be a positive integer. Got {atm_ncore_delta} instead"
        )

    if abs_maxdiff_nx_ny < 0:
        raise ValueError(
            "The absolute max. diff. between nx and ny (in the solved layout) must be a non-zero integer. "\
            f"Got {abs_maxdiff_nx_ny} instead"
        )

    if max_wasted_ncores_frac < 0 or max_wasted_ncores_frac > 1.0:
        raise ValueError(
            "The max. fraction of wasted cores must be in the range [0.0, 1.0]. Got {max_wasted_ncores_frac} instead"
        )

    if not isinstance(num_nodes_list, list):
        num_nodes_list = [num_nodes_list]

    if any(not isinstance(n, (int, float)) for n in num_nodes_list):
        raise ValueError(f"Number of nodes must be a float or an integer. Got {num_nodes_list} instead")

    if any(n <= 0 for n in num_nodes_list):
        raise ValueError(f"Number of nodes must be > 0. Got {num_nodes_list} instead")

    # atm requires nx to be even -> atm requires min 2 ncores
    # (2x1 layout), mom requires min 1 ncore (1x1 layout), ice requires min 1 ncore
    min_cores_required = 2 + 1 + 1

    ctrl_num_nodes, ctrl_config, ctrl_queue = 4, {"atm": (16, 13), "ocn": (14, 14), "ice": (12)}, "normalsr"
    ctrl_ratio_mom_over_atm = (
        ctrl_config["ocn"][0] * ctrl_config["ocn"][1] / (ctrl_config["atm"][0] * ctrl_config["atm"][1])
    )
    ctrl_totncores = convert_num_nodes_to_ncores(ctrl_num_nodes, queue=ctrl_queue)
    ctrl_ice_ncores = ctrl_config["ice"]

    if tol_around_ctrl_ratio is not None:
        logger.debug(
            "The min and max fractions of MOM ncores over ATM ncores will be set to "
            f"(at most) within (1 \u00b1 {tol_around_ctrl_ratio})"
            f" of the control ratio={ctrl_ratio_mom_over_atm:0.3g}"
        )

    # create the named tuple for holding the layouts
    layout_tuple = return_layout_tuple()
    final_layouts = []
    for num_nodes in num_nodes_list:
        # cast num_nodes to int if it is an integer value
        if isinstance(num_nodes, float) and num_nodes.is_integer():
            num_nodes = int(num_nodes)

        totncores = convert_num_nodes_to_ncores(num_nodes, queue=queue)
        if totncores < min_cores_required:
            logger.warning(
                f"Total ncores = {totncores} is less than the min. ncores required = {min_cores_required}. Skipping"
            )
            final_layouts.append([])
            continue

        logger.debug(f"Generating layouts for {num_nodes = } nodes")
        ice_ncores = max(1, int(ctrl_ice_ncores / ctrl_totncores * totncores))

        ncores_left = totncores - ice_ncores
        max_wasted_ncores = int(totncores * max_wasted_ncores_frac)
        min_ncores_needed = totncores - max_wasted_ncores

        # atm + mom = totncores - cice_ncores
        # => 1 + mom/atm = (totncores - cice_ncores)/atm
        # => 1 + min_frac_mom_ncores_over_atm_ncores = (totncores - cice_ncores)/max_atm_ncores
        # => max_atm_ncores = (totncores - cice_ncores)/(1 + min_frac_mom_ncores_over_atm_ncores)
        if tol_around_ctrl_ratio is not None:
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
            logger.debug(
                f"The min. and max. frac. of MOM ncores over ATM ncores are set to {mom_ncores_over_atm_ncores_range} "
                f"based on the control ratio={ctrl_ratio_mom_over_atm:0.3g} and {tol_around_ctrl_ratio=}"
            )
        else:
            max_atm_ncores = max(2, int(ncores_left / (1.0 + min_frac_mom_ncores_over_atm_ncores)))
            min_atm_ncores = max(2, int(ncores_left / (1.0 + max_frac_mom_ncores_over_atm_ncores)))
            mom_ncores_over_atm_ncores_range = (
                min_frac_mom_ncores_over_atm_ncores,
                max_frac_mom_ncores_over_atm_ncores,
            )
            logger.debug(
                f"The min. and max. frac. of MOM ncores over ATM ncores are set to {mom_ncores_over_atm_ncores_range} "
                f"based on the provided mom_ncores_over_atm_ncores_range={mom_ncores_over_atm_ncores_range}"
            )

        # If we want ATM ncores to be >= MOM ncores, then the max. fraction of MOM ncores over ATM ncores must be <= 1.0
        if prefer_atm_ncores_greater_than_mom_ncores:
            mom_ncores_over_atm_ncores_range = (mom_ncores_over_atm_ncores_range[0], 1.0)

        logger.debug(f"ATM ncores range, steps = ({min_atm_ncores}, {max_atm_ncores}, {atm_ncore_delta})")
        logger.debug(f"MOM ncores range = ({ncores_left - max_atm_ncores}, {ncores_left - min_atm_ncores})")
        layout = _generate_esm1p6_layout_from_core_counts(
            min_atm_ncores=min_atm_ncores,
            max_atm_ncores=max_atm_ncores,
            atm_ncore_delta=atm_ncore_delta,
            prefer_atm_nx_greater_than_ny=prefer_atm_nx_greater_than_ny,
            prefer_mom_nx_greater_than_ny=prefer_mom_nx_greater_than_ny,
            prefer_atm_ncores_greater_than_mom_ncores=prefer_atm_ncores_greater_than_mom_ncores,
            mom_ncores_over_atm_ncores_range=mom_ncores_over_atm_ncores_range,
            abs_maxdiff_nx_ny=abs_maxdiff_nx_ny,
            ncores_for_atm_and_ocn=ncores_left,
            ice_ncores=ice_ncores,
            min_ncores_needed=min_ncores_needed,
        )

        if allocate_unused_cores_to_ice and layout:
            # update the ice_ncores in each layout to include any unused cores
            updated_layouts = [
                layout_tuple(
                    totncores,
                    x.atm_nx,
                    x.atm_ny,
                    x.mom_nx,
                    x.mom_ny,
                    x.ice_ncores + (totncores - x.ncores_used),
                )
                for x in layout
            ]
            layout = list(set(updated_layouts))
            # sort the layouts by ncores_used (descending, fewer wasted cores first), and then
            # the sum of the absolute differences between nx and ny for atm and mom (ascending, i.e.,
            # more square layouts first)
            layout = sorted(layout, key=lambda x: (-x.ncores_used, abs(x.atm_nx - x.atm_ny) + abs(x.mom_nx - x.mom_ny)))

        final_layouts.append(layout)

    logger.info(f"Generated a total of {len(final_layouts)} layouts for {num_nodes_list} nodes")
    return final_layouts


def generate_esm1p6_perturb_block(
    num_nodes: (float | int),
    layouts: list,
    branch_name_prefix: str,
    *,
    queue: str = "normalsr",
    start_blocknum: int = 1,
) -> str:
    """

    Generates a block for "perturbation" experiments in the ESM 1.6 PI config.

    Parameters
    ----------
    num_nodes : float or int, required
        A positive number representing the number of nodes to use.

    layouts : list, required
        A list of layout_tuples as returned by ``generate_esm1p6_core_layouts_from_node_count``.
        Each layout_tuple has the following fields:
        - ncores_used : int
        - atm_nx : int
        - atm_ny : int
        - mom_nx : int
        - mom_ny : int
        - ice_ncores : int

        The layouts will be used in the order they appear in the list.

    branch_name_prefix : str, required
        Prefix to use for the branch names in the generated block.

    queue : str, optional, default="normalsr"
        Queue name on ``gadi``. Allowed values are "normalsr" and "normal".

    start_blocknum : int, optional, default=1
        The starting block number to use in the generated block. Must be a positive integer greater than 0.

    Returns
    -------
    str
        A string representing the generated block.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.

    """

    if num_nodes is None:
        raise ValueError("num_nodes must be provided.}")

    if not isinstance(num_nodes, (int, float)) or num_nodes <= 0:
        raise ValueError(
            f"Number of nodes must be a positive number or a list of positive numbers. Got {num_nodes} instead"
        )

    if branch_name_prefix is None:
        raise ValueError("branch_name_prefix must be provided")

    if not layouts:
        raise ValueError("No layouts provided")

    if not isinstance(layouts, list):
        layouts = [layouts]

    if any(len(x) != 6 for x in layouts):
        raise ValueError(f"Invalid layouts provided. Layouts = {layouts}, {len(layouts[0])=} instead of 6")

    if any(x.ncores_used > convert_num_nodes_to_ncores(num_nodes, queue=queue) for x in layouts):
        raise ValueError("One or more layouts require more cores than available")

    if not start_blocknum or start_blocknum < 1:
        raise ValueError("start_blocknum must be a positive integer greater than 0")

    totncores = convert_num_nodes_to_ncores(num_nodes, queue=queue)
    blocknum = start_blocknum
    block = ""
    for layout in layouts:
        atm_nx, atm_ny = layout.atm_nx, layout.atm_ny
        mom_nx, mom_ny = layout.mom_nx, layout.mom_ny
        ice_ncores = layout.ice_ncores
        atm_ncores = atm_nx * atm_ny
        mom_ncores = mom_nx * mom_ny
        branch_name = f"{branch_name_prefix}_atm_{atm_nx}x{atm_ny}_mom_{mom_nx}x{mom_ny}_ice_{ice_ncores}x1"
        ncores_used = atm_ncores + mom_ncores + ice_ncores
        block += f"""
  Scaling_numnodes_{num_nodes}_totncores_{totncores}_ncores_used_{ncores_used}_seqnum_{blocknum}:
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


def example_esmp16_layouts():
    """
    Example usage of the functions in this module to generate and print out
    possible layouts for the ESM 1.6 PI config for different node counts.
    """

    import logging
    from typing import NamedTuple

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    scaling_config_tup = NamedTuple(
        "scaling_config_tup",
        [
            ("num_nodes", float),
            ("min_frac_mom_ncores_over_atm_ncores", float),
            ("max_frac_mom_ncores_over_atm_ncores", float),
            ("tol_around_ctrl_ratio", (float | None)),
            ("atm_ncore_delta", int),
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
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.050,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=0.50,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.10,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.050,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=1.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.020,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=2.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.010,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=3.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.010,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=4.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.010,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=4.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=0.02,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=3,
            max_wasted_ncores_frac=0.010,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=5.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=2,
            max_wasted_ncores_frac=0.05,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=6.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=2,
            max_wasted_ncores_frac=0.05,
            allocate_unused_cores_to_ice=False,
        ),
        scaling_config_tup(
            num_nodes=6.00,
            min_frac_mom_ncores_over_atm_ncores=0.75,
            max_frac_mom_ncores_over_atm_ncores=1.25,
            tol_around_ctrl_ratio=None,
            atm_ncore_delta=2,
            abs_maxdiff_nx_ny=2,
            max_wasted_ncores_frac=0.05,
            allocate_unused_cores_to_ice=True,
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
    walltime: [walltime]
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
    entire_block = generator_config_prefix
    for config in scaling_configs:
        num_nodes = config.num_nodes
        logger.info(f"\nGenerating layouts for {num_nodes} nodes. Type num_nodes = {type(num_nodes)} ...")
        layout = generate_esm1p6_core_layouts_from_node_count(
            num_nodes,
            queue=queue,
            tol_around_ctrl_ratio=config.tol_around_ctrl_ratio,
            mom_ncores_over_atm_ncores_range=(
                config.min_frac_mom_ncores_over_atm_ncores,
                config.max_frac_mom_ncores_over_atm_ncores,
            ),
            atm_ncore_delta=config.atm_ncore_delta,
            abs_maxdiff_nx_ny=config.abs_maxdiff_nx_ny,
            max_wasted_ncores_frac=config.max_wasted_ncores_frac,
            allocate_unused_cores_to_ice=config.allocate_unused_cores_to_ice,
        )[0]
        logger.info(f"\nGenerating layouts for {num_nodes} nodes. Type num_nodes = {type(num_nodes)} ...done")
        if not layout:
            print(f"No layouts found for {num_nodes} nodes", file=sys.stderr)
            continue

        logger.debug(f"Generated {len(layout)} layouts for {num_nodes} nodes. Layouts: {layout}")

        block, blocknum = generate_esm1p6_perturb_block(
            num_nodes, layout, branch_name_prefix, queue=queue, start_blocknum=blocknum
        )
        blocknum += 1
        entire_block += block

    walltime_hrs = blocknum * 1.0  # assuming 1 hr walltime per experiment
    entire_block = entire_block.replace("[walltime]", f"{int(walltime_hrs):0d}:00:00")
    print(entire_block)


if __name__ == "__main__":
    example_esmp16_layouts()
