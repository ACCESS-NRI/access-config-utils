from typing import NamedTuple


def return_layout_tuple() -> NamedTuple:
    """
    Define a named tuple to hold layout information.
    Returns
    -------
    NamedTuple
        A named tuple with fields:
        - ncores_used (int): Total number of cores used.
        - atm_nx (int): Number of cores in the x-direction for the atmosphere model.
        - atm_ny (int): Number of cores in the y-direction for the atmosphere model.
        - mom_nx (int): Number of cores in the x-direction for the ocean model.
        - mom_ny (int): Number of cores in the y-direction for the ocean model.
        - ice_ncores (int): Number of cores used for the ice model.
    """
    # The noqa comment is to suppress the "convert to class" warning from ruff/flake8
    layout_tuple = NamedTuple(  # noqa: UP014
        "layout_tuple",
        [
            ("ncores_used", int),  # This can be derived from the other fields -> perhaps remove it? MS: 3rd Oct, 2025
            ("atm_nx", int),
            ("atm_ny", int),
            ("mom_nx", int),
            ("mom_ny", int),
            ("ice_ncores", int),
        ],
    )
    return layout_tuple


def convert_num_nodes_to_ncores(num_nodes: (int | float), queue: str = "normalsr") -> int:
    """
    Convert number of nodes to number of cores based on queue properties.

    Parameters
    ----------
    num_nodes : int or float, required
        Number of nodes to convert. Must be a positive number.
    queue : str, optional
        Queue name. Allowed values are "normalsr" and "normal".
        Default is "normalsr".
    Returns
    -------
    int
        Total number of cores corresponding to the given number of nodes.

    Raises
    ------
    ValueError
        If the queue name is not recognized or if num_nodes is not a positive number.

    """
    queue_properties = {
        "normalsr": {"ncores_per_node": 104},
        "normal": {"ncores_per_node": 48},
    }
    if queue not in list(queue_properties.keys()):
        raise ValueError(f"Queue = {queue} not allowed. Allowed values are {list(queue_properties.keys())}")

    if not isinstance(num_nodes, (int, float)) or num_nodes <= 0:
        raise ValueError("Number of nodes must be a positive number (integer or float).")

    return int(num_nodes * queue_properties[queue]["ncores_per_node"])


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
