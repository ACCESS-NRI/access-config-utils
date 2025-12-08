import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class LayoutTuple(NamedTuple):
    """
    Defines a class with a namedtuple to hold layout information.
    Returns
    -------
    object
        - atm_nx (int): Number of cores in the x-direction for the atmosphere model.
        - atm_ny (int): Number of cores in the y-direction for the atmosphere model.
        - mom_nx (int): Number of cores in the x-direction for the ocean model.
        - mom_ny (int): Number of cores in the y-direction for the ocean model.
        - ice_ncores (int): Number of cores used for the ice model.
        - ncores_used (int): Total number of cores used. Computed property.
    """

    atm_nx: int
    atm_ny: int
    mom_nx: int
    mom_ny: int
    ice_ncores: int

    @property
    def ncores_used(self) -> int:
        return self.atm_nx * self.atm_ny + self.mom_nx * self.mom_ny + self.ice_ncores


def get_ctrl_layout(model: str = "ESM 1.6 PI config") -> {}:
    """
    Get the control layout used in the current PI configuration.

    Parameters
    ----------
    model : str, optional
        Model name. Currently, only "ESM 1.6 PI config" is supported

    Returns
    -------
    dict
        A dictionary containing the control layout for ESM1.6 PI configuration:
        - layout (LayoutTuple): Named tuple with layout information.
        - queue (str): Queue name = "normalsr"
        - totncores (int): Total number of cores available = 416
        - num_nodes (int): Number of nodes used = 4

    LayoutTuple
        A named tuple containing the control layout information:
        - atm_nx (int): Number of cores in the x-direction for the atmosphere model = 16.
        - atm_ny (int): Number of cores in the y-direction for the atmosphere model = 13.
        - mom_nx (int): Number of cores in the x-direction for the ocean model = 14.
        - mom_ny (int): Number of cores in the y-direction for the ocean model = 14.
        - ice_ncores (int): Number of cores used for the ice model = 12.
        - ncores_used (int): Total number of cores used.
          Computed property := (atm_nx * atm_ny + mom_nx * mom_ny + ice_ncores) = 416 (i.e., same as ``totncores``).

    """
    if not isinstance(model, str):
        raise TypeError(f"Model name must be a string. Got {type(model)} instead")

    valid_models = ["ESM 1.6 PI config"]
    if model not in valid_models:
        raise ValueError(f"Model = {model} not allowed. Allowed values are {valid_models}")

    ctrl_layout_config = {}
    ctrl_layout_config["layout"] = LayoutTuple(atm_nx=16, atm_ny=13, mom_nx=14, mom_ny=14, ice_ncores=12)
    ctrl_layout_config["queue"] = "normalsr"
    ctrl_layout_config["totncores"] = 416
    ctrl_layout_config["num_nodes"] = 4
    return ctrl_layout_config


def find_layouts_with_maxncore(
    maxncore: int,
    *,  # keyword-only arguments follow
    abs_maxdiff_nx_ny: int = 4,
    even_nx: bool = False,
    prefer_nx_greater_than_ny: bool = False,
) -> list:
    """
    Find possible (nx, ny) layouts for a given maximum number of cores (maxncore).

    The function returns a list of tuples (nx, ny) such that ``nx * ny <= maxncore``.
    The function tries to find layouts with nx and ny as close as possible to
    sqrt(maxncore).

    Parameters
    ----------
    maxncore : int, required
        Maximum number of cores to use.

    abs_maxdiff_nx_ny : int, optional
        Maximum absolute difference between nx and ny in the layout. Default is 4.

    even_nx : bool, optional
        If True, only layouts with even nx are returned. Default is False.

    prefer_nx_greater_than_ny : bool, optional
        If True, only layouts with nx >= ny are returned. Default is False.

    Returns
    -------
    list of tuples
        List of (nx, ny) tuples representing the unique layouts found.
        If no layouts are found, an empty list is returned.

    Raises
    ------
    ValueError
        If maxncore is not a positive integer or if abs_maxdiff_nx_ny is negative.
    """
    import math

    if maxncore < 1:
        raise ValueError(f"Max. number of cores to use must be a positive integer. Got {maxncore} instead")
    if abs_maxdiff_nx_ny < 0:
        raise ValueError(
            "The max. absolute difference between nx and ny in the layout "
            f" must be a non-negative integer. Got {abs_maxdiff_nx_ny} instead"
        )

    if maxncore < 2 and even_nx:
        return []

    best_ncore = int(math.sqrt(maxncore))
    layouts = []
    start = max(1, best_ncore - abs_maxdiff_nx_ny)
    if prefer_nx_greater_than_ny:
        start = best_ncore

    for nx in range(start, best_ncore + abs_maxdiff_nx_ny + 1):
        if even_nx and nx % 2 != 0:
            continue
        ny = maxncore // nx
        if abs(nx - ny) > abs_maxdiff_nx_ny:
            continue
        if prefer_nx_greater_than_ny and nx < ny:
            continue

        layouts.append((nx, ny))

    return layouts
