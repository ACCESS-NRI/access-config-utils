"""
This module provides a small, reusable workflow to,
- (1) create valid processor layouts,
- (2) provide a ready-to-run `experiment-generator` yaml input file.

This module is designed to work across multiple model components, including
MOM6, CICE, mediator (MED), and WW3.

Note:
- The output of `generate_experiment_generator_yaml_input` can be directly used as an
    `experiment-generator` yaml input file for ACCESS-OM3 configurations.
"""

import io
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
    """Configuration for different job queues."""

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
        self.cpus_per_node = queue_config.nodesize
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

            if self.cpus_per_node % blocks_per_node != 0:
                raise ValueError(
                    f"cpus_per_node {self.cpus_per_node} must be divisible by blocks_per_node {blocks_per_node}."
                )

            # re-compute block size for each node count
            block_size = self.cpus_per_node // blocks_per_node

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
        pools = sorted(set(pool_map.values()))

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
            return [ratio_per_node for _ in range(num_nodes)]

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


def generate_experiment_generator_yaml_input(
    layouts_by_nodes: dict[int, list],
    pool_map: dict[str, str],
    branch_name_prefix: str,
    block_name: str,
    queue_config: QueueConfig,
    pool_order: list[str] | None = None,
    submodels: list[str] | None = None,
    start_block_id: int = 1,
    petlist_submodel: str | None = None,
    include_rootpe: bool = True,
    # config lists
    mem: list[str] | None = None,
    walltime: list[str] | None = None,
    # others
    restart_n: int = 10,
    restart_option: str = "days",
    stop_n: int = 10,
    stop_option: str = "days",
    model_type: str = "access-om3",
    repository_url: str | None = None,
    start_point: str | None = None,
    test_path: str | None = None,
    repository_directory: str | None = None,
    control_branch_name: str = "ctrl",
) -> str:
    """Generates an experiment generator yaml input file for ACCESS-OM3 configurations."""

    all_layouts: list[ConfigLayout] = []
    for node in layouts_by_nodes:
        all_layouts.extend(layouts_by_nodes[node])

    if not all_layouts:
        raise ValueError("No layouts generated to create perturbation block.")

    n_layouts = len(all_layouts)

    # ncpus
    ncpus = flow_seq([layout.ncpus for layout in all_layouts])

    # mem and walltime
    if mem is None:
        if queue_config is None:
            # REMOVED is a special keyword in experiment-generator to remove mem keyword
            # Then let the scheduler decide the memory allocation based on queue type.
            mem = "REMOVED"
        else:
            mem = flow_seq(
                [
                    f"{math.ceil(layout.ncpus / queue_config.nodesize) * queue_config.nodemem}GB"
                    for layout in all_layouts
                ]
            )
    if walltime is None:
        walltime = ["05:00:00"]

    branches = _make_branch_names(
        layouts_by_nodes=layouts_by_nodes,
        pool_map=pool_map,
        branch_name_prefix=branch_name_prefix,
        queue_config=queue_config,
        pool_order=pool_order,
    )

    # for pelayout_attributes
    pelayout: dict[str, list] = {}

    # submodels
    if submodels is None:
        submodels = ["ocn", "atm", "cpl", "ice", "rof"]

    for sub in submodels:
        pool = pool_map[sub]
        pelayout[f"{sub}_ntasks"] = flow_seq([layout.pool_ntasks[pool] for layout in all_layouts])
        if include_rootpe:
            pelayout[f"{sub}_rootpe"] = flow_seq([layout.pool_rootpe[pool] for layout in all_layouts])

    if petlist_submodel is None:
        petlist_submodel = "ocn" if "ocn" in submodels else submodels[0]

    pet_pool = pool_map[petlist_submodel]
    pet_rootpes = [layout.pool_rootpe[pet_pool] for layout in all_layouts]
    petlist = flow_seq([DoubleQuotedScalarString(f"0 {pe}") for pe in pet_rootpes])

    print(f"Generated {n_layouts} layouts for perturbation block '{block_name}'!")

    # yaml output
    yaml_output = {
        "model_type": model_type,
        "repository_url": repository_url,
        "start_point": start_point,
        "test_path": test_path,
        "repository_directory": repository_directory,
        "control_branch_name": control_branch_name,
        "Control_Experiment": {
            "config.yaml": {
                "metadata": {"enable": True},
            },
        },
        "Perturbation_Experiment": {
            block_name: {
                "branches": branches,
                "MOM_input": {
                    "AUTO_MASKTABLE": flow_seq(["REMOVE"] + ["PRESERVE"] * (n_layouts - 1)),
                },
                "ice_in": {
                    "domain_nml": {"max_blocks": -1},
                },
                "config.yaml": {
                    "env": {
                        "ESMF_RUNTIME_PROFILE": "on",
                        "ESMF_RUNTIME_TRACE": "on",
                        "ESMF_RUNTIME_TRACE_PETLIST": petlist,
                        "ESMF_RUNTIME_PROFILE_OUTPUT": "SUMMARY",
                    },
                    "ncpus": ncpus,
                    "mem": mem,
                    "walltime": walltime,
                    "metadata": {"enable": True},
                    "queue": queue_config.queue,
                    "platform": {
                        "nodesize": queue_config.nodesize,
                        "nodemem": queue_config.nodemem,
                    },
                },
                "nuopc.runconfig": {
                    "PELAYOUT_attributes": pelayout,
                    "CLOCK_attributes": {
                        "restart_n": restart_n,
                        "restart_option": restart_option,
                        "stop_n": stop_n,
                        "stop_option": stop_option,
                    },
                },
            }
        },
    }

    buf = io.StringIO()
    ryaml.dump(yaml_output, buf)

    return buf.getvalue()


def _make_branch_names(
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
    pools = sorted(set(pool_map.values()))

    if pool_order is not None:
        pools = [pool for pool in pool_order if pool in pools]

    branches = []

    for node in sorted(layouts_by_nodes):
        for layout in layouts_by_nodes[node]:
            tmp = [
                branch_name_prefix,
                f"node_{node}",
                f"queue_{queue_config.queue}",
            ]
            for pool in pools:
                ntasks = layout.pool_ntasks.get(pool, 0)
                tmp.append(f"{pool}_{ntasks}")
            branches.append("_".join(tmp))

    return branches


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

    # for perturbation block generation
    branch_name_prefix = "MC-100km-ryf"
    experiment_generator_block_name = "Parameter_block_test"
    petlist_submodel = "ocn"
    include_rootpe = True

    model_type = "access-om3"
    repository_url = "https://github.com/ACCESS-NRI/access-om3-configs.git"
    start_point = "e8f7559"
    test_path = "om3_scalings"
    repository_directory = "Scaling_MC-100km-ryf"

    # generate perturbation block yaml text
    yaml_input = generate_experiment_generator_yaml_input(
        layouts_by_nodes=layouts_by_nodes,
        pool_map=pool_map,
        pool_order=pool_order,
        branch_name_prefix=branch_name_prefix,
        block_name=experiment_generator_block_name,
        queue_config=queue_config,
        petlist_submodel=petlist_submodel,
        include_rootpe=include_rootpe,
        model_type=model_type,
        repository_url=repository_url,
        start_point=start_point,
        test_path=test_path,
        repository_directory=repository_directory,
    )
    print(yaml_input)
