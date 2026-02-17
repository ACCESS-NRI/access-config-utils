"""
This module provides a small, reusable workflow to,
- (1) create valid processor layouts,
- (2) provide a ready-to-run `experiment-generator` yaml input file.

This module is designed to work across multiple model components, including
MOM6, CICE, mediator (MED), and WW3.

Note:
- The output of `generate_experiment_generator_yaml_input` can be directly used as an
    `experiment-generator` yaml input file for ACCESS-OM3 configurations.
- `generate_experiment_generator_yaml_input` injects only:
    - branches, ncpus, mem, walltime, and enable Control_Experiment metadata.
    - Other yaml fragments such as PELAYOUT_attributes, ESMF trace env are built via
        standalone helper functions and merged by the caller.
"""

import io
import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

ryaml = YAML()
ryaml.indent(mapping=2, sequence=4, offset=2)


def flow_seq(lst: list) -> CommentedSeq:
    """
    Render a Python list as an inline yaml list
    """
    cs = CommentedSeq(lst)
    cs.fa.set_flow_style()
    return cs


@dataclass
class QueueConfig:
    """Configuration for different pbs job queues."""
    queue: str
    nodesize: int
    nodemem: int

    @classmethod
    def from_queue(cls, queue: str) -> "QueueConfig":
        """Creates a QueueConfig instance based on the queue name."""
        mapping: dict[str, tuple(int, int)] = {
            "normalsr": (104, 512),  # Sapphire Rapids
            "expresssr": (104, 512),  # Sapphire Rapids (express)
            "normal": (48, 192),  # Cascade lake
            "express": (48, 192),  # Cascade lake (express)
            "normalbw": (28, 128),  # broadwell
            "expressbw": (28, 128),  # broadwell (express)
            "normalsl": (32, 192),  # Skylake
        }
        if queue not in mapping:
            raise ValueError(f"Unknown queue name: {queue}")
        nodesize, nodemem = mapping[queue]
        return cls(queue=queue, nodesize=nodesize, nodemem=nodemem)


@dataclass
class ConfigLayout:
    """Configuration layout for ACCESS-OM3.
    Parameters:
        ncpus (int): Total number of CPUs.
        pool_ntasks (dict[str, int]): Number of tasks per pool.
        pool_rootpe (dict[str, int]): Root PE for each pool.
    """
    ncpus: int
    pool_ntasks: dict[str, int]
    pool_rootpe: dict[str, int]


def flatten_layouts(layouts_by_nodes: dict[int, list[ConfigLayout]]) -> list[ConfigLayout]:
    """Flatten layouts_by_nodes into a deterministic list."""
    all_layouts: list[ConfigLayout] = []
    for node in layouts_by_nodes:
        all_layouts.extend(layouts_by_nodes[node])
    return all_layouts


def merge_dicts(base_dict: dict, override_dict: dict) -> dict:
    """Recursively merge override_dict to base_dict."""
    res = copy.deepcopy(base_dict)
    for key, value in override_dict.items():
        if isinstance(value, dict) and isinstance(res.get(key), dict):
            res[key] = merge_dicts(res[key], value)
        else:
            res[key] = value
    return res


class ACCESSOM3LayoutGenerator:
    """Generates ACCESS-OM3 configuration layouts from pool mappings and constraints."""

    def __init__(
        self,
        queue_config: QueueConfig,
        blocks_per_node: int = 8,
        baseline_pool: str = "shared",
        eps: float = 1e-6,
    ) -> None:
        """
        Parameters:
            queue_config (QueueConfig): Configuration for the job queue.
            blocks_per_node (int): Number of allocation blocks per node, eg 8 blocks of 13 cpus for 104 cpus/node.
            baseline_pool (str): Baseline pool for ratio constraints (usually "shared").
            eps (float): Small tolerance to avoid floating point issues.
        """
        if queue_config.nodesize % blocks_per_node != 0:
            raise ValueError(
                f"nodesize {queue_config.nodesize} must be divisible by blocks_per_node {blocks_per_node}."
            )

        self.queue_config = queue_config
        self.nodesize = queue_config.nodesize
        self.blocks_per_node = blocks_per_node
        self.baseline_pool = baseline_pool
        self.eps = eps

    def generator(
        self,
        nodes: list[int],
        pool_map: dict[str, str],
        max_ratio_to_baseline: dict[str, float] | list[dict[str, float]] | None = None,
        min_ratio_to_baseline: dict[str, float] | list[dict[str, float]] | None = None,
        min_blocks: dict[str, int] | None = None,
        pool_order: list[str] | None = None,
        blocks_per_node: int | list[int] | None = None,
    ) -> dict[int, list[ConfigLayout]]:
        """Generates all valid layouts for each node count.

        Parameters:
            nodes (list[int]): List of available node counts.
            pool_map (dict[str, str]): Mapping of submodels to their respective pools.
            max_ratio_to_baseline : dict | list[dict] | None: Maximum allowed ratios to the baseline pool.
            min_ratio_to_baseline : dict | list[dict] | None: Minimum required ratios to the baseline pool.
            min_blocks (dict[str, int] | None): Minimum required blocks for each pool.
            pool_order (list[str] | None): Optional order of pools.
            blocks_per_node (int | list[int] | None): Block constraints per node.

        Returns:
            dict[int, list[ConfigLayout]]: A dictionary mapping node counts to lists of valid ConfigLayout instances.
        """

        # pool names in use
        pools = self._check_pools(pool_map, pool_order=pool_order)

        if self.baseline_pool not in pools:
            raise ValueError(
                f"Baseline pool '{self.baseline_pool}' not in pools={pools}."
                "Check pool_map argument or baseline_pool parameter."
            )

        # normalise ratio constraints per node
        max_ratio_list = self._normalise_ratio_per_node(len(nodes), max_ratio_to_baseline, "max_ratio_to_baseline")
        min_ratio_list = self._normalise_ratio_per_node(len(nodes), min_ratio_to_baseline, "min_ratio_to_baseline")

        # normalise blocks per node constraints
        blocks_per_node_list = self._normalise_blocks_per_node(
            num_nodes=len(nodes),
            blocks_per_node=blocks_per_node,
            default=self.blocks_per_node,
            name="blocks_per_node",
        )

        # store layouts by node count
        layout_by_node: dict[int, list[ConfigLayout]] = {}

        for i, node in enumerate(nodes):
            blocks_per_node = blocks_per_node_list[i]

            if self.nodesize % blocks_per_node != 0:
                raise ValueError(
                    f"nodesize {self.nodesize} must be divisible by blocks_per_node {blocks_per_node}."
                )

            # re-compute block size for each node count
            block_size = self.nodesize // blocks_per_node

            layouts: list[ConfigLayout] = []

            total_blocks = [node * blocks_per_node]

            for total_block in total_blocks:
                for alloc_blocks in self._enumerate_block_allocations(total_block, list(pools), min_blocks):
                    if not self._pass_ratio_constraints(
                        alloc_blocks,
                        max_ratio_to_baseline=max_ratio_list[i],
                        min_ratio_to_baseline=min_ratio_list[i],
                    ):
                        continue

                    # convert blocks to ntasks
                    pool_ntasks = {pool: alloc_blocks[pool] * block_size for pool in pools}

                    # assign rootpes
                    pool_rootpe = self._assign_rootpes(pools, pool_ntasks)

                    layouts.append(
                        ConfigLayout(
                            ncpus=sum(pool_ntasks.values()),
                            pool_ntasks=pool_ntasks,
                            pool_rootpe=pool_rootpe,
                        )
                    )

            # store layouts for this node count
            layout_by_node[node] = layouts

        return layout_by_node

    def _check_pools(
        self,
        pool_map: dict[str, str],
        pool_order: list[str] | None = None,
    ) -> list[str]:
        """Checks and returns the list and ordering of unique pools from the pool map.

        Parameters:
            pool_map (dict[str, str]): Mapping of submodels to their respective pools.
            pool_order (list[str] | None): Optional order of pools.
        """
        pools = list(pool_map.values())

        if pool_order is not None:
            missing = set(pools) - set(pool_order)
            if missing:
                raise ValueError(f"pool_order {pool_order} is missing pools: {missing}")
            return [pool for pool in pool_order if pool in pools]

        preferred_order = ["shared", "ocn", "wav", "ice", "atm", "rof", "cpl"]
        ordered_pools = [pool for pool in preferred_order if pool in pools]
        return ordered_pools

    def _normalise_blocks_per_node(
        self,
        num_nodes: int,
        blocks_per_node: int | list[int] | None,
        default: int,
        name: str,
    ) -> list[int]:
        """Normalises block constraints per node.

        Parameters:
            num_nodes (int): Number of nodes.
            blocks_per_node (dict[str, int] | list[dict[str, int]] | None): Block constraints per node.
            default (int): Default blocks per node
            name (str): Name of the block constraint for error messages.

        Returns:
            list[dict[str, int]]: List of block constraints normalised for total nodes.
        """
        if blocks_per_node is None:
            return [default] * num_nodes

        if isinstance(blocks_per_node, int):
            return [blocks_per_node] * num_nodes

        if isinstance(blocks_per_node, list):
            if len(blocks_per_node) != num_nodes:
                raise ValueError(
                    f"Length of {name} ({len(blocks_per_node)}) does not match number of nodes ({num_nodes})."
                )
            if not all(isinstance(b, int) for b in blocks_per_node):
                raise TypeError(f"All elements in {name} list must be integers.")
            return blocks_per_node

        raise TypeError(f"{name} must be an int or a list of ints.")

    def _normalise_ratio_per_node(
        self,
        num_nodes: int,
        ratio_per_node: dict[str, float] | list[dict[str, float]] | None,
        name: str,
    ) -> list[dict[str, float]]:
        """Normalises ratio constraints per node.

        Parameters:
            num_nodes (int): Number of nodes.
            ratio_per_node (dict[str, float] | list[dict[str, float]] | None): Ratio constraints per node.
            name (str): Name of the ratio constraint for error messages.

        Returns:
            list[dict[str, float]]: List of ratio constraints normalised for total nodes.
        """
        if ratio_per_node is None:
            return [{} for _ in range(num_nodes)]

        if isinstance(ratio_per_node, dict):
            return [dict(ratio_per_node) for _ in range(num_nodes)]

        if isinstance(ratio_per_node, list):
            if len(ratio_per_node) != num_nodes:
                raise ValueError(
                    f"Length of {name} ({len(ratio_per_node)}) does not match number of nodes ({num_nodes})."
                )
            return ratio_per_node

        raise TypeError(f"{name} must be a dict or a list of dicts.")

    def _pass_ratio_constraints(
        self,
        alloc: dict[str, int],
        max_ratio_to_baseline: dict[str, float] | list[dict[str, float]] | None,
        min_ratio_to_baseline: dict[str, float] | list[dict[str, float]] | None,
    ) -> bool:
        """Checks if the allocation passes the ratio constraints.

        Parameters:
            alloc (dict[str, int]): Allocation of blocks to pools.
            max_ratio_to_baseline (dict[str, float]): Maximum allowed ratios to the baseline pool.

        Returns:
            bool: True if allocation passes constraints, False otherwise.
        """
        max_ratio_to_baseline = max_ratio_to_baseline or {}
        min_ratio_to_baseline = min_ratio_to_baseline or {}

        base = alloc[self.baseline_pool]

        # For max ratio
        for pool, max_ratio in max_ratio_to_baseline.items():
            if pool == self.baseline_pool or pool not in alloc:
                continue
            if alloc[pool] / base > max_ratio + self.eps:
                return False

        # For min ratio
        for pool, min_ratio in min_ratio_to_baseline.items():
            if pool == self.baseline_pool or pool not in alloc:
                continue
            if alloc[pool] / base < min_ratio - self.eps:
                return False

        return True

    def _enumerate_block_allocations(
        self,
        total_blocks: int,
        pools: list[str],
        min_blocks: dict[str, int] | None = None,
    ) -> Iterable[dict[str, int]]:
        """Enumerates all possible block allocations for given pools and total blocks.

        Parameters:
            total_blocks (int): Total number of blocks available.
            pools (list[str]): List of pool names.
            min_blocks (dict[str, int] | None): Minimum required blocks for each pool. Defaults to None.

        Returns:
            list[dict[str, int]]: List of block allocation dictionaries for each pool.
        """
        min_blocks = min_blocks or {}

        # ensure every pool has a minimum
        mins = {pool: int(min_blocks.get(pool, 1)) for pool in pools}

        def allocate_blocks(
            i: int,
            remaining: int,
            current_alloc: dict[str, int],
        ):
            """Recursively allocate blocks to pools.

            Parameters:
                i (int): Current pool index.
                remaining (int): Remaining blocks to allocate.
                current_alloc (dict[str, int]): Current allocation of blocks to pools.

            Yields:
                dict[str, int]: A valid block allocation dict.
            """
            if i == len(pools) - 1:
                pool = pools[i]
                if remaining >= mins[pool]:
                    out = dict(current_alloc)
                    out[pool] = remaining
                    yield out
                return

            pool = pools[i]
            rest_min = sum(mins[p] for p in pools[i + 1 :])
            low = mins[pool]  # minimum blocks for this pool
            high = remaining - rest_min  # maximum blocks for this pool

            for blocks in range(low, high + 1):
                current_alloc[pool] = blocks
                yield from allocate_blocks(i + 1, remaining - blocks, current_alloc)
            current_alloc.pop(pool, None)

        yield from allocate_blocks(0, total_blocks, {})

    def _assign_rootpes(
        self,
        pools: list[str],
        pool_ntasks: dict[str, int],
    ) -> dict[str, int]:
        """Assigns root PEs for each pool based on their task counts.

        Parameters:
            pools (list[str]): List of pool names.
            pool_ntasks (dict[str, int]): Number of tasks per pool.
        """
        rootpe = {}
        current_pe = 0
        for pool in pools:
            rootpe[pool] = current_pe
            current_pe += pool_ntasks[pool]
        return rootpe


def build_scheduler_resources(
    layouts_by_nodes: dict[int, list[ConfigLayout]],
    queue_config: QueueConfig,
    walltime: list[str],
) -> dict[str, list]:
    """
    Returns fields that can be used in the experiment generator for runtime resources like ncpus, mem, walltime.
    """
    all_layouts = flatten_layouts(layouts_by_nodes)

    if not all_layouts:
        raise ValueError("No layouts generated to create runtime resources.")

    if queue_config is None:
        raise ValueError("queue_config is required to create runtime resources.")

    if walltime is None:
        raise ValueError("walltime is required to create runtime resources.")

    ncpus = flow_seq([layout.ncpus for layout in all_layouts])
    mem = flow_seq(
        [
            f"{math.ceil(layout.ncpus / queue_config.nodesize) * queue_config.nodemem}GB"
            for layout in all_layouts
        ]
    )
    walltime_seq = flow_seq(walltime)

    return {
        "ncpus": ncpus,
        "mem": mem,
        "walltime": walltime_seq,
    }


def build_pelayout_attributes(
    layouts_by_nodes: dict[int, list[ConfigLayout]],
    pool_map: dict[str, str],
    include_rootpe: bool = True,
) -> dict[str, list]:
    """
    Build PE layout attributes for the experiment generator yaml input.
    """
    all_layouts = flatten_layouts(layouts_by_nodes)

    if not all_layouts:
        raise ValueError("No layouts generated to create PE layout attributes.")

    pelayout: dict[str, list] = {}
    submodels = list(pool_map.keys())

    for sub in submodels:
        pool = pool_map[sub]
        pelayout[f"{sub}_ntasks"] = flow_seq([layout.pool_ntasks[pool] for layout in all_layouts])
        if include_rootpe:
            pelayout[f"{sub}_rootpe"] = flow_seq([layout.pool_rootpe[pool] for layout in all_layouts])

    return pelayout


def build_esmf_trace_env(
    esmf_trace_analysis: bool,
    trace_pets: str | int | list[int] | None,
    n_layouts: int,
) -> dict[str, str]:
    """
    Build ESMF runtime tracing env vars.

    trace_pets:
        - "all": trace all pets and output SUMMARY and BINARY profiles.
        - list[list[int]]: per-layout PETLIST, each starting with 0, e.g. [[0,12], [0, 24, 48], ...]
    """
    if not esmf_trace_analysis:
        return {}

    if trace_pets is None:
        raise ValueError(
            "trace_pets must be provided when esmf_trace_analysis is True."
            "Use 'all', or a list of PET lists per layout (e.g. [[0, 12], [0, 24, 48]])."
            )

    env = {
        "ESMF_RUNTIME_PROFILE": DoubleQuotedScalarString("on"),
        "ESMF_RUNTIME_TRACE": DoubleQuotedScalarString("on"),
    }

    if trace_pets == "all":
        env["ESMF_RUNTIME_PROFILE_OUTPUT"] = DoubleQuotedScalarString("SUMMARY,BINARY")
        return env

    if not isinstance(trace_pets, list) or len(trace_pets) != n_layouts:
        raise ValueError(
            f"trace_pets must be a list of length n_layouts={n_layouts}"
        )

    petlist_strings = []
    for i, pets in enumerate(trace_pets):
        if not isinstance(pets, list) or not pets:
            raise ValueError(
                f"trace_pets[{i}] must be a non-empty list of ints)."
            )
        if pets[0] != 0:
            raise ValueError("trace_pets must include 0 as the first element (e.g. [0, 12]).")
        petlist_str = " ".join(str(idx) for idx in pets)
        petlist_strings.append(DoubleQuotedScalarString(petlist_str))

    env["ESMF_RUNTIME_PROFILE_OUTPUT"] = DoubleQuotedScalarString("SUMMARY")
    env["ESMF_RUNTIME_TRACE_PETLIST"] = flow_seq(petlist_strings)
    return env


def make_branch_names(
    layouts_by_nodes: dict[int, list],
    pool_map: dict[str, str],
    branch_name_prefix: str,
    queue_config: QueueConfig,
    pool_order: list[str] | None = None,
):
    """Generates branch names for each layout.

    Each layout becomes one branch entry.
    ntasks/rootpe for each submodel is derived from pool_map[submodel] -> pool.
    """
    pools = set(pool_map.values())

    if pool_order is not None:
        pools = [pool for pool in pool_order if pool in pools]

    branches = []

    for node in layouts_by_nodes:
        for layout in layouts_by_nodes[node]:
            tmp = [
                branch_name_prefix,
                f"node_{node}",
                f"queue_{queue_config.queue}",
            ]
            for pool in pools:
                ntasks = layout.pool_ntasks.get(pool)
                tmp.append(f"{pool}_{ntasks}")
            branches.append("_".join(tmp))

    return branches


def generate_experiment_generator_yaml_input(
    layouts_by_nodes: dict[int, list],
    pool_map: dict[str, str],
    branch_name_prefix: str,
    block_name: str,
    queue_config: QueueConfig,
    walltime: list[str],
    user_dict: dict | None = None,
    pool_order: list[str] | None = None,
) -> str:
    """Generates an experiment generator yaml input file for ACCESS-OM3 configurations."""

    all_layouts = flatten_layouts(layouts_by_nodes)
    if not all_layouts:
        raise ValueError("No layouts generated.")

    scheduler_basic = build_scheduler_resources(
        layouts_by_nodes=layouts_by_nodes,
        queue_config=queue_config,
        walltime=walltime,
    )

    branches = make_branch_names(
        layouts_by_nodes=layouts_by_nodes,
        pool_map=pool_map,
        branch_name_prefix=branch_name_prefix,
        queue_config=queue_config,
        pool_order=pool_order,
    )

    # yaml output
    base_output = {
        # always ensures metadata enabled for control and hence perturbation experiments
        "Control_Experiment": {
            "config.yaml": {
                "metadata": {"enable": True},
            },
        }
        # "Perturbation_Experiment": {
        #     block_name: {
        #         "branches": branches,
        #         # "MOM_input": {
        #         #     "AUTO_MASKTABLE": flow_seq(["REMOVE"] * n_layouts),
        #         # },
        #         # "ice_in": {
        #         #     "domain_nml": {"max_blocks": -1},
        #         # },
        #         "config.yaml": {
        #             "env": env,
        #             "ncpus": ncpus,
        #             "mem": mem,
        #             "walltime": walltime,
        #             "queue": queue_config.queue,
        #             "platform": {
        #                 "nodesize": queue_config.nodesize,
        #                 "nodemem": queue_config.nodemem,
        #             },
        #         },
        #         "nuopc.runconfig": {
        #             "PELAYOUT_attributes": pelayout,
        #             "CLOCK_attributes": {
        #                 "restart_n": restart_n,
        #                 "restart_option": restart_option,
        #                 "stop_n": stop_n,
        #                 "stop_option": stop_option,
        #             },
        #         },
        #     }
        # }
    }

    dict_output = merge_dicts(base_output, user_dict)

    try:
        block = dict_output["Perturbation_Experiment"][block_name]
    except KeyError as e:
        raise KeyError(
            f"user_dict must define Perturbation_Experiment -> {block_name}."
        ) from e

    block["branches"] = branches
    config = block.setdefault("config.yaml", {})
    config["queue"] = queue_config.queue
    config["ncpus"] = scheduler_basic["ncpus"]
    config["mem"] = scheduler_basic["mem"]
    config["walltime"] = scheduler_basic["walltime"]

    buf = io.StringIO()
    ryaml.dump(dict_output, buf)

    return buf.getvalue()


if __name__ == "__main__":
    pool_map = {
        "atm": "shared",
        "cpl": "shared",
        "ice": "shared",
        "rof": "shared",
        "ocn": "ocn",
        # "wav": "wav",
    }

    # for layout generation (optional)
    pool_order = [
        "shared",
        "ocn",
        # "wav"
    ]

    queue_config = QueueConfig.from_queue("normalsr")
    blocks_per_node = 8  # divided into 8 blocks of 13 cpus each (int or list[int])
    baseline_pool = "shared"  # usually "shared"
    eps = 1e-6  # floating point tolerance

    # define nodes to generate layouts for
    nodes = [1, 2, 3, 4, 5]
    n = len(nodes)

    layout_generator = ACCESSOM3LayoutGenerator(
        queue_config=queue_config,
        blocks_per_node=blocks_per_node,
        baseline_pool=baseline_pool,
        eps=eps,
    )

    max_ratio_to_baseline = [
        {"ocn": 8.0},  # for 1 node
        *([{"ocn": 6.0}] * (n - 1)),  # for other nodes
    ]

    min_ratio_to_baseline = [
        {"ocn": 1.0},
        *([{"ocn": 2.0}] * (n - 1)),
    ]

    # generate layouts for each node count
    layouts_by_nodes = layout_generator.generator(
        nodes=nodes,
        pool_map=pool_map,
        max_ratio_to_baseline=max_ratio_to_baseline,
        min_ratio_to_baseline=min_ratio_to_baseline,
        pool_order=pool_order,
        blocks_per_node=blocks_per_node,
    )

    all_layouts = flatten_layouts(layouts_by_nodes)
    n_layouts = len(all_layouts)

    # for perturbation block generation
    branch_name_prefix = "MC-100km-ryf"
    experiment_generator_block_name = "Parameter_block_test"
    walltime = ["05:00:00"]

    pelayout = build_pelayout_attributes(
        layouts_by_nodes=layouts_by_nodes,
        pool_map=pool_map,
        include_rootpe=True,
    )

    trace_pets = [[0, layout.pool_rootpe["ocn"]] for layout in all_layouts]
    trace_env = build_esmf_trace_env(
        esmf_trace_analysis=True,
        trace_pets=trace_pets,
        n_layouts=n_layouts,
    )

    universal_config_setup = {
        "model_type": "access-om3",
        "repository_url": "https://github.com/ACCESS-NRI/access-om3-configs.git",
        "start_point": "e8f7559",
        "test_path": "om3_scalings",
        "repository_directory": "Scaling_MC-100km-ryf",
    }

    clock_attributes = {
        "restart_n": 10,
        "restart_option": "ndays",
        "stop_n": 10,
        "stop_option": "ndays",
    }

    user_dict = {
        **universal_config_setup,
        "Perturbation_Experiment": {
            experiment_generator_block_name: {
                "MOM_input": {
                    "AUTO_MASKTABLE": flow_seq(["REMOVE"] * n_layouts),
                },
                "config.yaml": {
                    "metadata": {"enable": True},
                    "queue": queue_config.queue,  # ok to include; generator will overwrite consistently
                    "platform": {"nodesize": queue_config.nodesize, "nodemem": queue_config.nodemem},
                    "env": trace_env,
                },
                "nuopc.runconfig": {
                    "PELAYOUT_attributes": pelayout,
                    "CLOCK_attributes": clock_attributes,
                },
            }
        }
    }

    # generate perturbation block yaml text
    yaml_input = generate_experiment_generator_yaml_input(
        layouts_by_nodes=layouts_by_nodes,
        pool_map=pool_map,
        pool_order=pool_order,
        branch_name_prefix=branch_name_prefix,
        block_name=experiment_generator_block_name,
        queue_config=queue_config,
        walltime=walltime,
        user_dict=user_dict,
    )
    print(yaml_input)
