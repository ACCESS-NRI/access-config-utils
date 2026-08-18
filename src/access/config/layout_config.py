import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class LayoutTuple(NamedTuple):
    """Named tuple holding the core layout of an ESM1.6 configuration.

    The total core count is not stored: it is the computed property ``ncores_used``.

    Args:
        atm_nx (int): Number of cores in the x-direction for the atmosphere model.
        atm_ny (int): Number of cores in the y-direction for the atmosphere model.
        mom_nx (int): Number of cores in the x-direction for the ocean model.
        mom_ny (int): Number of cores in the y-direction for the ocean model.
        ice_ncores (int): Number of cores used for the ice model.
    """

    atm_nx: int
    atm_ny: int
    mom_nx: int
    mom_ny: int
    ice_ncores: int

    @property
    def ncores_used(self) -> int:
        return self.atm_nx * self.atm_ny + self.mom_nx * self.mom_ny + self.ice_ncores


def get_ctrl_layout(model: str = "ESM 1.6 PI config") -> dict:
    """Get the control layout used in the current PI configuration.

    For ``"ESM 1.6 PI config"`` the layout is ``atm_nx=16``, ``atm_ny=13``, ``mom_nx=14``,
    ``mom_ny=14`` and ``ice_ncores=12``, on the ``normalsr`` queue over 4 nodes. Those
    counts spend the whole allocation: ``layout.ncores_used`` equals the reported
    ``totncores`` of 416.

    Args:
        model (str): Model name. Currently, only ``"ESM 1.6 PI config"`` is supported.

    Returns:
        dict: The control layout, with keys ``layout`` (a ``LayoutTuple``), ``queue``
            (str), ``totncores`` (int) and ``num_nodes`` (int).
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
    """Find possible ``(nx, ny)`` layouts for a given maximum number of cores.

    The function returns a list of tuples ``(nx, ny)`` such that ``nx * ny <= maxncore``.
    The function tries to find layouts with ``nx`` and ``ny`` as close as possible to
    ``sqrt(maxncore)``.

    Args:
        maxncore (int): Maximum number of cores to use.
        abs_maxdiff_nx_ny (int): Maximum absolute difference between ``nx`` and ``ny``
            in the layout. Defaults to 4.
        even_nx (bool): If ``True``, only layouts with even ``nx`` are returned.
            Defaults to ``False``.
        prefer_nx_greater_than_ny (bool): If ``True``, only layouts with ``nx >= ny``
            are returned. Defaults to ``False``.

    Returns:
        list[tuple[int, int]]: The unique ``(nx, ny)`` layouts found, or an empty list
            if there are none.

    Raises:
        ValueError: If *maxncore* is not a positive integer, or if *abs_maxdiff_nx_ny*
            is negative.
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
