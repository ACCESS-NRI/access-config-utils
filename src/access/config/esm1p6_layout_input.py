import logging

from access.config.layout_config import (
    LayoutTuple,
    find_layouts_with_maxncore,
    get_ctrl_layout,
)

logger = logging.getLogger(__name__)


class LayoutSearchConfig:
    """
    Configuration class for setting up searching for core layouts with different model components.

    Parameters
    ----------
    tol_around_ctrl_ratio : float or None, optional, default=None
        Tolerance around the control ratio for core allocation to ATM and MOM.
        If set, the min and max fractions of MOM ncores over ATM ncores will be set to (at most) within
        (1 ± tol_around_ctrl_ratio) of the released PI config. Must be in the range [0.0, 1.0].
        If not set, the min and max fractions of MOM ncores over ATM ncores are used from the
        ``mom_ncores_over_atm_ncores_range`` parameter.

        When set to 0.0, the ratio of MOM ncores to ATM ncores needs to *exactly* match the
        ratio in the control layout. This is guaranteed to at least generate the control
        layout for the control num_nodes, but may not generate any layouts for other node counts.

        *Note*: This parameter takes precedence over ``mom_ncores_over_atm_ncores_range`` if both are set.

    frac_mom_ncores_over_atm_ncores : (float, float), optional, default=(0.75, 1.25)
        Fraction of ocean model cores over atmosphere model cores.
        A tuple of two floats representing the minimum and maximum fractions of MOM ncores over ATM
        ncores to consider when generating layouts. Must be greater than 0.0, and the second
        value (i.e, the max.) must be at least equal to the first value (i.e., the min.)
        Layouts with MOM ncores over ATM ncores outside this range will be discarded.

        *Note*: This parameter is set automatically if ``tol_around_ctrl_ratio`` is set.

    atm_ncore_stepsize : int, optional, default=2
        The step size to cover the range of allowed number of atmosphere model cores.
        Must be a non-zero and positive integer.

    abs_maxdiff_nx_ny : int, optional, default=4
        Absolute max. of the difference between nx and ny (in the solved layout) to
        consider when generating layouts. Must be a non-negative integer.

        *Note*: Setting to 0 will return only square layouts. Applies to both ATM and MOM layouts.

    max_wasted_ncores_frac : float, optional, default=0.01
        Maximum fraction of wasted cores (i.e. not used by atm, mom or ice) to allow when generating layouts.
        Must be in the range [0.0, 1.0].

    prefer_atm_nx_greater_than_ny : bool, optional, default=True
        If True, only consider ATM layouts with nx >= ny.

    prefer_mom_nx_greater_than_ny : bool, optional, default=True
        If True, only consider MOM layouts with nx >= ny.

    prefer_atm_ncores_greater_than_mom_ncores : bool, optional, default=True
        If True, only consider layouts with ATM ncores >= MOM ncores.

    allocate_unused_cores_to_ice : bool, optional, default=False
        If True, allocate unused cores to the ice model.

    ctrl_ratio : float
        Ratio of MOM to ATM cores in the control layout. Set automatically
        when the class is instantiated.

    """

    frac_mom_ncores_over_atm_ncores: tuple[float, float]
    tol_around_ctrl_ratio: float | None
    atm_ncore_stepsize: int
    abs_maxdiff_nx_ny: int
    max_wasted_ncores_frac: float
    prefer_atm_nx_greater_than_ny: bool = True
    prefer_mom_nx_greater_than_ny: bool = True
    prefer_atm_ncores_greater_than_mom_ncores: bool = True
    allocate_unused_cores_to_ice: bool
    ctrl_ratio: float  # Ratio of MOM to ATM cores in the control PI configuration

    def __init__(
        self,
        *,
        frac_mom_ncores_over_atm_ncores=(0.75, 1.25),
        tol_around_ctrl_ratio=None,
        atm_ncore_stepsize=2,
        abs_maxdiff_nx_ny=4,
        max_wasted_ncores_frac=0.02,
        prefer_atm_nx_greater_than_ny: bool = True,
        prefer_mom_nx_greater_than_ny: bool = True,
        prefer_atm_ncores_greater_than_mom_ncores: bool = True,
        allocate_unused_cores_to_ice=False,
        validate_on_init: bool = True,
    ):
        self.frac_mom_ncores_over_atm_ncores = frac_mom_ncores_over_atm_ncores
        self.tol_around_ctrl_ratio = tol_around_ctrl_ratio
        self.atm_ncore_stepsize = atm_ncore_stepsize
        self.abs_maxdiff_nx_ny = abs_maxdiff_nx_ny
        self.max_wasted_ncores_frac = max_wasted_ncores_frac
        self.prefer_atm_nx_greater_than_ny = prefer_atm_nx_greater_than_ny
        self.prefer_mom_nx_greater_than_ny = prefer_mom_nx_greater_than_ny
        self.prefer_atm_ncores_greater_than_mom_ncores = prefer_atm_ncores_greater_than_mom_ncores
        self.allocate_unused_cores_to_ice = allocate_unused_cores_to_ice

        ctrl_layout = get_ctrl_layout()["layout"]
        self.ctrl_ratio = (ctrl_layout.mom_nx * ctrl_layout.mom_ny) / (ctrl_layout.atm_nx * ctrl_layout.atm_ny)

        if validate_on_init:
            self.validate()

        if self.tol_around_ctrl_ratio is not None:
            self._set_frac_mom_ncores_from_tol()

    # ruff complains that this function is too complex (C901) -> however, I do not see
    # another way of doing this validation
    def validate(self):  # noqa: C901
        if self.tol_around_ctrl_ratio is None and self.frac_mom_ncores_over_atm_ncores is None:
            raise ValueError(
                "Either tolerance around control ratio or the range for the fraction of MOM ncores "
                "over ATM ncores must be provided. Currently, both are None."
            )

        # Validate tol_around_ctrl_ratio
        if self.tol_around_ctrl_ratio is not None:
            if not isinstance(self.tol_around_ctrl_ratio, (int, float)):
                raise TypeError(
                    "Tolerance around control ratio must be either None or a number (int or float).\n"
                    f"Got {type(self.tol_around_ctrl_ratio)} instead."
                )

            if self.tol_around_ctrl_ratio < 0.0 or self.tol_around_ctrl_ratio > 1.0:
                raise ValueError(
                    "Tolerance around control ratio must be in the inclusive range [0.0, 1.0].\n"
                    f"Got {self.tol_around_ctrl_ratio} instead."
                )

        # Validate frac_mom_ncores_over_atm_ncores
        if (
            not isinstance(self.frac_mom_ncores_over_atm_ncores, tuple)
            or len(self.frac_mom_ncores_over_atm_ncores) != 2
        ):
            raise TypeError(
                "frac_mom_ncores_over_atm_ncores must be a tuple of two numbers (min, max).\n"
                f"Got {type(self.frac_mom_ncores_over_atm_ncores)} instead."
            )
        if any(not isinstance(x, (int, float)) for x in self.frac_mom_ncores_over_atm_ncores):
            raise TypeError(
                "Min. and max. frac. of MOM ncores over ATM ncores must be numbers (int or float).\n"
                f"Got {self.frac_mom_ncores_over_atm_ncores} instead."
            )
        if (
            self.frac_mom_ncores_over_atm_ncores[0] <= 0
            or self.frac_mom_ncores_over_atm_ncores[1] <= 0
            or self.frac_mom_ncores_over_atm_ncores[0] > self.frac_mom_ncores_over_atm_ncores[1]
        ):
            raise ValueError(
                "Min. and max. frac. of MOM ncores over ATM ncores must be positive numbers, "
                "and min must be less than or equal to max.\n"
                f"Got min={self.frac_mom_ncores_over_atm_ncores[0]} and "
                f"max={self.frac_mom_ncores_over_atm_ncores[1]} instead."
            )

        # Validate atm_ncore_stepsize
        if not isinstance(self.atm_ncore_stepsize, int):
            raise TypeError(
                f"Stepsize for spanning the range of ATM ncore values must be an integer.\n"
                f"Got {type(self.atm_ncore_stepsize)} instead."
            )
        if self.atm_ncore_stepsize < 1:
            raise ValueError(
                f"Stepsize for spanning the range of ATM ncore values must be >= 1.\n"
                f"Got {self.atm_ncore_stepsize} instead."
            )

        # Validate abs_maxdiff_nx_ny
        if not isinstance(self.abs_maxdiff_nx_ny, int):
            raise TypeError(
                f"Absolute max. difference between nx and ny in the layout must be an integer.\n"
                f"Got {type(self.abs_maxdiff_nx_ny)} instead."
            )
        if self.abs_maxdiff_nx_ny < 0:
            raise ValueError(
                f"Absolute max. difference between nx and ny in the layout must be a non-negative integer.\n"
                f"Got {self.abs_maxdiff_nx_ny} instead."
            )

        # Validate max_wasted_ncores_frac
        if not isinstance(self.max_wasted_ncores_frac, (int, float)):
            raise TypeError(
                f"Max. fraction of wasted cores must be a number (int or float).\n"
                f"Got {type(self.max_wasted_ncores_frac)} instead."
            )
        if self.max_wasted_ncores_frac < 0.0 or self.max_wasted_ncores_frac > 1.0:
            raise ValueError(
                f"Max. fraction of wasted cores must be in the inclusive range [0.0, 1.0].\n"
                f"Got {self.max_wasted_ncores_frac} instead."
            )

        # Validate the relevant boolean attributes
        boolean_attr_names = [
            "prefer_atm_nx_greater_than_ny",
            "prefer_mom_nx_greater_than_ny",
            "prefer_atm_ncores_greater_than_mom_ncores",
            "allocate_unused_cores_to_ice",
        ]
        for attr_name in boolean_attr_names:
            attr = getattr(self, attr_name)
            if not isinstance(attr, bool):
                raise TypeError(
                    f"Got invalid type for `{attr_name}`: must be a boolean (True or False).\nGot {type(attr)} instead."
                )

    def _set_frac_mom_ncores_from_tol(self):
        assert self.tol_around_ctrl_ratio is not None, "Unexpected value of None tolerance around control ratio"
        min_frac = (1.0 - self.tol_around_ctrl_ratio) * self.ctrl_ratio
        max_frac = (
            1.0
            if self.prefer_atm_ncores_greater_than_mom_ncores
            else (1.0 + self.tol_around_ctrl_ratio) * self.ctrl_ratio
        )
        logger.debug(
            f"Setting frac_mom_ncores_over_atm_ncores from tolerance around control ratio: "
            f"min={min_frac}, max={max_frac}"
        )
        self.frac_mom_ncores_over_atm_ncores = (min_frac, max_frac)


def set_ice_ncores(min_ice_ncores: int, max_ice_ncores: int, blocksize: int = 360, smallest_factor: bool = True) -> int:
    """
    Sets the number of cores for ICE such that it divides the given blocksize, and is the largest
    possible value within the given min and max range.

    Parameters
    ----------
    min_ice_ncores : int, required
        Minimum number of ICE cores to consider. Must be at least 1 and less than or equal to max_ice_ncores.

    max_ice_ncores : int, required
        Maximum number of ICE cores to consider. Must be at least 1 and greater than or equal to min_ice_ncores.

    blocksize : int, optional, default=360 (the blocksize used in the released ESM 1.6 PI config)
        Blocksize to use for determining the ICE core layout. Must be at least 1.

    smallest_factor : bool, optional, default=True
        If True, the smallest factor of the blocksize that is within the min and max ICE ncores will be used.
        If False, the largest factor will be used.

    """
    if min_ice_ncores < 1 or max_ice_ncores < 1 or max_ice_ncores < min_ice_ncores:
        raise ValueError("Invalid min/max ice ncores. Expected both to be >= 1 and max >= min.")

    if blocksize < 1 or blocksize < min_ice_ncores:
        raise ValueError(
            "Blocksize must be at least 1 and greater than or equal to min. "
            f"ice ncores = {min_ice_ncores}. Got {blocksize} instead."
        )

    factors = [f for f in range(min_ice_ncores, max_ice_ncores + 1) if blocksize % f == 0]
    if not factors:
        raise ValueError(
            f"No valid cores found for ice ncores in the range ({min_ice_ncores}, {max_ice_ncores}). "
            "Please increase the min. and max. range of ICE ncores to use"
        )
    logger.debug(f"Factors of blocksize {blocksize} in the range ({min_ice_ncores}, {max_ice_ncores}): {factors}")
    ice_ncores = min(factors) if smallest_factor else max(factors)
    logger.debug(f"Selected ice ncores = {ice_ncores}")
    return ice_ncores


# The noqa comment is to suppress the complexity warning from ruff/flake8
# The complexity of this function is high due to the nested loops and multiple conditionals. Some day
# I or someone else will refactor it to reduce the complexity. - MS 7th Oct, 2025
def _generate_esm1p6_layout_from_core_counts(  # noqa: C901
    min_atm_ncores: int,
    max_atm_ncores: int,
    ncores_for_atm_and_ocn: int,
    ice_ncores: int,
    min_ncores_needed: int,
    *,
    layout_search_config: LayoutSearchConfig | None = None,
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

    layout_search_config : LayoutSearchConfig, optional, default=None
        An instance of the LayoutSearchConfig class containing configuration parameters for layout generation.
        Please refer to the class documentation for the parameters and their descriptions.
        If None, default values will be used.

    """

    min_atm_and_mom_ncores = 3  # atm requires min 2 ncores (2x1 layout), mom requires min 1 ncore (1x1 layout)

    if min_atm_ncores < 2 or max_atm_ncores < 2 or min_atm_ncores > max_atm_ncores:
        raise ValueError(f"Invalid ATM ncores range. Got ({min_atm_ncores}, {max_atm_ncores}) instead")

    if ice_ncores < 1:
        raise ValueError(f"Number of cores for ICE must be at least 1. Got {ice_ncores} instead")

    if ncores_for_atm_and_ocn < min_atm_and_mom_ncores:
        raise ValueError(
            "Number of cores available for ATM and OCN must be at least {min_atm_and_mom_ncores} "
            f"(2 for atm and 1 for mom). Got {ncores_for_atm_and_ocn} instead"
        )

    if min_ncores_needed > (ncores_for_atm_and_ocn + ice_ncores):
        raise ValueError(
            f"Min. number of cores needed ({min_ncores_needed}) cannot be greater than the total "
            f"number of available cores ({ncores_for_atm_and_ocn + ice_ncores})"
        )
    if min_ncores_needed < ncores_for_atm_and_ocn:
        logger.warning(
            f"Min. total cores required for a valid config ({min_ncores_needed}) should be greater "
            f"than the number of ATM + OCN cores ({ncores_for_atm_and_ocn}). "
            f"Currently, any config that satisfies the ATM + OCN core requirements will also satisfy "
            "the requirement for the min. total cores"
        )

    if layout_search_config is not None:
        layout_search_config.validate()
    else:
        layout_search_config = LayoutSearchConfig()

    all_layouts = []
    logger.debug(
        f"Generating layouts with {min_atm_ncores=}, {max_atm_ncores=}, {layout_search_config.atm_ncore_stepsize=}, "
        f"{ncores_for_atm_and_ocn=}, {ice_ncores=}, {min_ncores_needed=}, "
        f"{layout_search_config.frac_mom_ncores_over_atm_ncores=}, "
        f"{layout_search_config.abs_maxdiff_nx_ny=}, "
        f"{layout_search_config.prefer_atm_nx_greater_than_ny=}, "
        f"{layout_search_config.prefer_mom_nx_greater_than_ny=}, "
        f"{layout_search_config.prefer_atm_ncores_greater_than_mom_ncores=}"
    )
    for atm_ncores in range(min_atm_ncores, max_atm_ncores + 1, layout_search_config.atm_ncore_stepsize):
        logger.debug(f"Trying atm_ncores = {atm_ncores}")
        atm_layout = find_layouts_with_maxncore(
            atm_ncores,
            abs_maxdiff_nx_ny=layout_search_config.abs_maxdiff_nx_ny,
            even_nx=True,
            prefer_nx_greater_than_ny=layout_search_config.prefer_atm_nx_greater_than_ny,
        )
        if not atm_layout:
            continue

        logger.debug(f"  Found {len(atm_layout)} atm layouts for atm_ncores = {atm_ncores}: {atm_layout}")

        min_mom_ncores = int(atm_ncores * layout_search_config.frac_mom_ncores_over_atm_ncores[0])
        max_mom_ncores = int(atm_ncores * layout_search_config.frac_mom_ncores_over_atm_ncores[1])
        for atm in atm_layout:
            atm_nx, atm_ny = atm

            mom_ncores = ncores_for_atm_and_ocn - atm_nx * atm_ny
            logger.debug(f"  Trying atm layout {atm_nx}x{atm_ny} with {atm_nx * atm_ny} ncores")
            mom_layout = find_layouts_with_maxncore(
                mom_ncores,
                abs_maxdiff_nx_ny=layout_search_config.abs_maxdiff_nx_ny,
                prefer_nx_greater_than_ny=layout_search_config.prefer_mom_nx_greater_than_ny,
            )
            if not mom_layout:
                continue

            # filter mom_layout to only include layouts with ncores in the range [min_mom_ncores, max_mom_ncores]
            layout = []
            for mom_nx, mom_ny in mom_layout:
                mom_ncores = mom_nx * mom_ny
                if mom_ncores < min_mom_ncores or mom_ncores > max_mom_ncores:
                    logger.debug(
                        f"Skipping mom layout {mom_nx}x{mom_ny} with {mom_ncores} ncores "
                        f"not in the range [{min_mom_ncores}, {max_mom_ncores}]"
                    )
                    continue

                if layout_search_config.prefer_atm_ncores_greater_than_mom_ncores and (atm_nx * atm_ny < mom_ncores):
                    logger.debug(
                        f"Skipping mom layout since mom ncores = {mom_nx}x{mom_ny} is not less "
                        f"than atm ncores = {atm_nx * atm_ny}"
                    )
                    continue

                ncores_used = mom_nx * mom_ny + atm_nx * atm_ny + ice_ncores
                if ncores_used < min_ncores_needed:
                    logger.debug(
                        f"Skipping layout atm {atm_nx}x{atm_ny} mom {mom_nx}x{mom_ny} ice {ice_ncores}, "
                        f"with {ncores_used=} is less than {min_ncores_needed=}"
                    )
                    continue

                logger.debug(
                    f"Adding layout atm {atm_nx}x{atm_ny} mom {mom_nx}x{mom_ny} ice {ice_ncores} with {ncores_used=}"
                )
                layout.append(LayoutTuple(atm_nx, atm_ny, mom_nx, mom_ny, ice_ncores))

            # create a set of layouts to avoid duplicates
            all_layouts.extend(set(layout))

    # The following works even if all_layouts == [] (i.e., no layouts found)
    # sort the layouts by ncores_used (descending, fewer wasted cores first), and then
    # the sum of the absolute differences between nx and ny for atm and mom (ascending,
    # i.e., more square layouts first)
    all_layouts = sorted(
        all_layouts, key=lambda x: (-x.ncores_used, abs(x.atm_nx - x.atm_ny) + abs(x.mom_nx - x.mom_ny))
    )

    return all_layouts


# The noqa comment is to suppress the complexity warning from ruff/flake8
# The complexity of this function is high due to the nested loops and multiple conditionals. Some day
# I or someone else will refactor it to reduce the complexity. - MS 7th Oct, 2025
def generate_esm1p6_core_layouts_from_node_count(
    num_nodes: float,
    cores_per_node: int,
    *,
    layout_search_config: LayoutSearchConfig | None = None,
) -> list:
    """
    Given a target number of nodes to use, this function generates possible core layouts for the Atmosphere and Ocean
    for the ESM 1.6 PI config.

    Parameters
    ----------
    num_nodes : float or integer, required
        A positive number representing the number of nodes to use.

    cores_per_node : int, required
        Number of cores available per node. Must be a positive integer.

    layout_search_config : LayoutSearchConfig, optional, default=None
        An instance of the LayoutSearchConfig class containing configuration parameters for layout generation.
        Please refer to the class documentation for the parameters and their descriptions.
        If None, default values will be used.

    Returns
    -------
    list
        A list containing instances of the class LayoutTuple, corresponds to the layouts for the respective number of
        nodes. Each instance has the following fields:
        - atm_nx : int
        - atm_ny : int
        - mom_nx : int
        - mom_ny : int
        - ice_ncores : int
        - ncores_used : int (computed property := atm_nx * atm_ny + mom_nx * mom_ny + ice_ncores)

        An empty list is returned if no valid layouts could be generated.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.

    Notes
    -----
    - atm requires nx to be even -> atm requires min 2 cores (2x1 layout),
      mom requires min 1 core (1x1 layout), ice requires min 1 core
    - The released PI configuration used is:
        - atm: 16x13 (208 cores)
        - ocn: 14x14 (196 cores)
        - ice: 12x1  (12 cores)
        - queue: normalsr
        - num_nodes: 4
        - totncores: 416 cores
        - ncores_used: 416 cores
    """

    if num_nodes <= 0:
        raise ValueError(f"Number of nodes must be > 0. Got {num_nodes} instead")

    if not isinstance(cores_per_node, int) or cores_per_node <= 0:
        raise ValueError(f"Cores per node must be a positive integer. Got {cores_per_node} instead")

    if layout_search_config is not None:
        layout_search_config.validate()  # check that the provided config is valid
    else:
        layout_search_config = LayoutSearchConfig()

    # atm requires nx to be even -> atm requires min 2 ncores
    # (2x1 layout), mom requires min 1 ncore (1x1 layout), ice requires min 1 ncore
    min_cores_required = 2 + 1 + 1

    ctrl_layout_config = get_ctrl_layout()
    ctrl_totncores = ctrl_layout_config["totncores"]
    ctrl_ice_ncores = ctrl_layout_config["layout"].ice_ncores

    totncores = int(num_nodes * cores_per_node)
    if totncores < min_cores_required:
        logger.warning(
            f"Total ncores = {totncores} is less than the min. ncores required = {min_cores_required}."
            "Returning an empty list"
        )
        return []

    logger.debug(f"Generating layouts for {num_nodes = } nodes")
    target_ice_ncores = max(1, int(ctrl_ice_ncores / ctrl_totncores * totncores))
    # Allow a 20% increase in ice ncores over the target to find a suitable factor of blocksize (360 for ESM1.6)
    ice_ncores = set_ice_ncores(min_ice_ncores=target_ice_ncores, max_ice_ncores=int(target_ice_ncores * 1.2))

    ncores_left = totncores - ice_ncores
    max_wasted_ncores = int(totncores * layout_search_config.max_wasted_ncores_frac)
    min_ncores_needed = totncores - max_wasted_ncores

    min_frac_mom_ncores_over_atm_ncores, max_frac_mom_ncores_over_atm_ncores = (
        layout_search_config.frac_mom_ncores_over_atm_ncores
    )
    max_atm_ncores = max(2, int(ncores_left / (1.0 + min_frac_mom_ncores_over_atm_ncores)))
    min_atm_ncores = max(2, int(ncores_left / (1.0 + max_frac_mom_ncores_over_atm_ncores)))

    logger.debug(
        f"ATM ncores range, stepsize = ({min_atm_ncores}, {max_atm_ncores}, {layout_search_config.atm_ncore_stepsize})"
    )
    logger.debug(f"OCN ncores range = ({ncores_left - max_atm_ncores}, {ncores_left - min_atm_ncores})")
    layout = _generate_esm1p6_layout_from_core_counts(
        min_atm_ncores=min_atm_ncores,
        max_atm_ncores=max_atm_ncores,
        ncores_for_atm_and_ocn=ncores_left,
        ice_ncores=ice_ncores,
        min_ncores_needed=min_ncores_needed,
        layout_search_config=layout_search_config,
    )

    if layout_search_config.allocate_unused_cores_to_ice:
        # update the ice_ncores in each layout to include any unused cores
        # This will recreate the existing layout if the total cores used
        # is equal to the total available cores

        # This works even if layout == [] (i.e., no layouts found). In that case,
        # layout will remain an empty list
        layout = [
            LayoutTuple(
                x.atm_nx,
                x.atm_ny,
                x.mom_nx,
                x.mom_ny,
                set_ice_ncores(
                    min_ice_ncores=x.ice_ncores,
                    max_ice_ncores=x.ice_ncores + (totncores - x.ncores_used),
                    smallest_factor=False,
                ),
            )
            if x
            else None
            for x in layout
            # ruff insists that this line by line breaking up is the correct formatting
            # even though IMO that's less readable - MS 14th Oct, 2025
        ]

    layout = list(set(layout))  # remove duplicates

    # sort the layouts by ncores_used (descending, fewer wasted cores first), and then
    # the sum of the absolute differences between nx and ny for atm and mom (ascending, i.e.,
    # more square layouts first)
    # Still works even if layout == [] (i.e., no layouts found)
    layout = sorted(layout, key=lambda x: (-x.ncores_used, abs(x.atm_nx - x.atm_ny) + abs(x.mom_nx - x.mom_ny)))

    logger.info(f"Generated a total of {len(layout)} layouts for {num_nodes} nodes")

    return layout


def generate_esm1p6_perturb_block(layout: LayoutTuple, branch_name_prefix: str) -> dict:
    """Generates a block for "perturbation" experiments in the ESM 1.6 PI config.

    Parameters
    ----------
    layout : ``LayoutTuple`` tuple, required
        An instances of the class `LayoutTuple` as returned by ``generate_esm1p6_core_layouts_from_node_count``.
        Each instance of `LayoutTuple` has the following fields:
        - atm_nx : int
        - atm_ny : int
        - mom_nx : int
        - mom_ny : int
        - ice_ncores : int
        - ncores_used : int (computed property := atm_nx * atm_ny + mom_nx * mom_ny + ice_ncores)

    branch_name_prefix : str, required
        Prefix to use for the branch names in the generated block.

    Returns
    -------
    dict
        A dict representing the generated block.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.

    """
    if not layout:
        raise ValueError("No layout provided")

    if branch_name_prefix is None:
        raise ValueError("The prefix for the branch name must be provided")

    if not isinstance(layout, LayoutTuple):
        raise ValueError(f"Invalid layout provided. Layout = {layout} must be of type LayoutTuple")

    atm_nx, atm_ny = layout.atm_nx, layout.atm_ny
    mom_nx, mom_ny = layout.mom_nx, layout.mom_ny
    ice_ncores = layout.ice_ncores
    atm_ncores = atm_nx * atm_ny
    mom_ncores = mom_nx * mom_ny
    branch_name = f"{branch_name_prefix}_atm_{atm_nx}x{atm_ny}_mom_{mom_nx}x{mom_ny}_ice_{ice_ncores}x1"
    block = {
        "branches": [branch_name],
        "config.yaml": {
            "submodels": [
                [
                    {"ncpus": atm_ncores},  # ncores for atmosphere
                    {"ncpus": mom_ncores},  # ncores for ocean
                    {
                        "ncpus": ice_ncores,  # ncores for ice
                        "exe": [f"cice_access-esm1.6_360x300_{ice_ncores}x1_{ice_ncores}p.exe"],
                    },
                ]
            ]
        },
        "atmosphere/um_env.yaml": {
            "UM_ATM_NPROCX": str(atm_nx),
            "UM_ATM_NPROCY": str(atm_ny),
            "UM_NPES": str(atm_ncores),
        },
        "ocean/input.nml": {
            "ocean_model_nml": {
                "layout": [f"{mom_nx},{mom_ny}"],
            }
        },
        "ice/cice_in.nml": {
            "domain_nml": {
                "nprocs": [f"{ice_ncores}"],
            }
        },
    }

    return block
