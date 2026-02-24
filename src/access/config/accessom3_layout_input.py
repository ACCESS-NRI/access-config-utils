"""
ACCESS-OM3 layout generation and perturbation block builder.

This module provides functionality to generate valid core layouts for ACCESS-OM3 based on the number of nodes,
cores per node, and user-defined constraints.

It also includes a function to generate perturbation blocks for experiment generation.

 - OM3LayoutSearchConfig: user-supplied configuration for layout search
 - OM3ConfigLayout: generated layout (pool_ntasks + pool_rootpe + ncpus)
 - generate_om3_core_layouts_from_node_count: generate layouts for a given node count
 - generate_om3_perturb_block: generate perturbation block for a given layout
"""

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQString

DEFAULT_POOL_ORDER = ["shared", "ocn", "ice", "cpl", "wav", "atm", "rof"]


def merge_blocks(base_dict: dict, override_dict: dict) -> dict:
    """Recursively merges two dicts, with override_dict taking precedence over base_dict."""
    res = copy.deepcopy(base_dict)
    for key, value in override_dict.items():
        if isinstance(value, dict) and isinstance(res.get(key), dict):
            res[key] = merge_blocks(res[key], value)
        else:
            res[key] = value
    return res


@dataclass(frozen=True)
class QueueConfig:
    """Scheduler-dependent hardware configuration."""

    scheduler: str  # e.g. "pbs" or "slurm", currently only "pbs" is supported
    queue: str  # pbs queue name for Gadi or slurm partition name for Setonix
    nodesize: int  # number of cpu cores per node
    nodemem: int  # memory per node in GB

    @classmethod
    def from_scheduler(cls, scheduler: str, queue: str) -> "QueueConfig":
        """Creates a QueueConfig instance based on the scheduler and queue name."""
        scheduler = scheduler.lower()

        if scheduler == "pbs":
            return cls._from_pbs(queue)
        elif scheduler == "slurm":
            raise NotImplementedError("Slurm scheduler is not yet supported.")

        raise ValueError(f"Unsupported scheduler: {scheduler}")

    @classmethod
    def _from_pbs(cls, queue: str) -> "QueueConfig":
        """Creates a QueueConfig instance based on the pbs queue name."""
        mapping: dict[str, tuple[int, int]] = {
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
        return cls(scheduler="pbs", queue=queue, nodesize=nodesize, nodemem=nodemem)


@dataclass
class OM3ConfigLayout:
    """Configuration layout for ACCESS-OM3.
    Parameters:
        ncpus (int): Total number of CPUs.
        pool_ntasks (dict[str, int]): Number of tasks per pool.
        pool_rootpe (dict[str, int]): Root PE for each pool.
    """

    ncpus: int
    pool_ntasks: dict[str, int]
    pool_rootpe: dict[str, int]


@dataclass
class OM3LayoutSearchConfig:
    """
    Search config for generating OM3 layouts

    pool_map:   dict mapping submodel to pool. Used to build pelayout attributes for nuopc.runconfig
                e.g. {"ocn": "shared", "ice": "shared", "atm": "ocn", "rof": "ocn"}
    pool_order: pool ordering for layout generation.
    blocks_per_node: number of blocks per node.
                e.g. Default is 8, which corresponds to 8 blocks of 13 cpus for 104 cpus/node.
    baseline_pool_name: pool name to use as baseline for ratio constraints. Default is "shared".

    Constraints:
    - Ratio constraints: max_ratio_to_baseline and min_ratio_to_baseline specify the maximum and minimum
                         allowed ratios of blocks in each pool to the baseline pool (usually "shared").
                         These can be used to enforce that certain pools have more or fewer resources
                         relative to the baseline. Eg max_ratio_to_baseline={"ocn": 1.0} would enforce
                         that ocn can have at most the same number of blocks as "shared".
    - Minimum blocks:    optional specifies the minimum number of blocks that must be allocated to each pool.
                         Eg min_blocks={"ocn": 2} would enforce that ocn must have at least 2 blocks allocated.
    - enable_esmf_trace: if True, adds ESMF trace env variables to the config.yaml.
    - trace_pets:        if enable_esmf_trace is True,
                         - "all": for all PETs
                         - list[int]: for specified PET list (must start with 0)
    """

    scheduler: str
    queue: str
    pool_map: dict[str, str]

    blocks_per_node: int = 8
    pool_order: list[str] | None = None

    baseline_pool_name: str = "shared"
    eps: float = 1e-6
    max_ratio_to_baseline: dict[str, float] | None = None
    min_ratio_to_baseline: dict[str, float] | None = None
    min_blocks: dict[str, int] | None = None

    enable_esmf_trace: bool = False
    trace_pets: str | list[int] | None = None  # "all" or per-layout petlist


def _ordered_pools(pool_map: dict[str, str], pool_order: list[str] | None) -> list[str]:
    """
    The list of unique pools in pool_map ordered according to pool_order.

    If pool_order is None, fall back to a default ordering.
    """
    pools_in_use = set(pool_map.values())

    if pool_order is None:
        return [pool for pool in DEFAULT_POOL_ORDER if pool in pools_in_use]

    missing = pools_in_use - set(pool_order)
    if missing:
        raise ValueError(f"pool_order missing pools: {sorted(missing)}")

    return [pool for pool in pool_order if pool in pools_in_use]


def _assign_rootpes(pools: list[str], pool_ntasks: dict[str, int]) -> dict[str, int]:
    """Assign root PEs for each pool based on their task counts."""
    rootpe = {}
    current_pe = 0
    for pool in pools:
        rootpe[pool] = current_pe
        current_pe += pool_ntasks[pool]
    return rootpe


def _pass_ratio_constraints(
    alloc: dict[str, int],
    baseline_pool_name: str,
    eps: float,
    max_ratio: dict[str, float] | None,
    min_ratio: dict[str, float] | None,
) -> bool:
    """
    Check alloc[pool] / alloc[baseline_pool_name] against max_ratio and min_ratio constraints for each pool.
    """
    max_ratio = max_ratio or {}
    min_ratio = min_ratio or {}

    base = alloc.get(baseline_pool_name, 0)
    if base == 0:
        return False  # cannot have any blocks if baseline is zero

    for pool, max_r in max_ratio.items():
        if pool == baseline_pool_name or pool not in alloc:
            continue
        if alloc[pool] / base > max_r + eps:
            return False

    for pool, min_r in min_ratio.items():
        if pool == baseline_pool_name or pool not in alloc:
            continue
        if alloc[pool] / base < min_r - eps:
            return False

    return True


def _enumerate_block_allocations(
    total_blocks: int,
    pools: list[str],
    min_blocks: dict[str, int] | None = None,
) -> Iterable[dict[str, int]]:
    """
    Generate all possible ways to distribute total_blocks across pools.
    Each pool must receive at least min_blocks[pool] blocks (default = 1).
    """
    min_blocks = min_blocks or {}
    mins = {pool: int(min_blocks.get(pool, 1)) for pool in pools}

    def _allocate_blocks(
        i: int,
        remaining: int,
        current_alloc: dict[str, int],
    ):
        """
        Recursively allocate blocks to pools.

        It performs a depth-first search over all integer compositions of total_blocks,
        subject to the minimum block constraints for each pool.
        The recursion iterates through the pools in order, allocating blocks to the current pool,
        and then recursing to allocate the remaining blocks to the next pools. When it reaches the last pool,
        it checks if the remaining blocks meet the minimum requirement and yields a complete allocation if valid.

        i: current pool index
        remaining: number of blocks still available among the current and remaining pools
        current_alloc: current allocation of blocks to pools

        It returns a complete allocation mapping each pool to a block count that
            - sum(blocks) = total_blocks
            - blocks[pool] >= min_blocks[pool] for all pools
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
        low = mins[pool]
        high = remaining - rest_min

        for alloc in range(low, high + 1):
            current_alloc[pool] = alloc
            yield from _allocate_blocks(i + 1, remaining - alloc, current_alloc)
        current_alloc.pop(pool, None)

    yield from _allocate_blocks(0, total_blocks, {})


def _build_esmf_env(layout: OM3ConfigLayout, config: OM3LayoutSearchConfig) -> dict[str, str]:
    """
    Construct ESMF runtime profiling environment variables.

    trace_pets:
        - "all" -> trace all pets, output SUMMARY and BINARY
        - list[int] -> explicit petlist, must start with 0, output SUMMARY only
        - None -> trace enabled but no petlist specified, default to SUMMARY output
    """
    if not config.enable_esmf_trace:
        return {}

    env = {
        "ESMF_RUNTIME_PROFILE": "on",
        "ESMF_RUNTIME_TRACE": "on",
    }

    if config.trace_pets == "all":
        env["ESMF_RUNTIME_PROFILE_OUTPUT"] = "SUMMARY,BINARY"
        return env

    if config.trace_pets is None:
        # This allows overrides to set trace_pets later
        env["ESMF_RUNTIME_PROFILE_OUTPUT"] = "SUMMARY"
        return env

    if isinstance(config.trace_pets, list):
        if not config.trace_pets:
            raise ValueError("trace_pets list cannot be empty")

        if config.trace_pets[0] != 0:
            raise ValueError("trace_pets must start with 0 (eg, [0, 12])")

        env["ESMF_RUNTIME_PROFILE_OUTPUT"] = "SUMMARY"
        env["ESMF_RUNTIME_TRACE_PETLIST"] = " ".join(str(pet) for pet in config.trace_pets)

        return env

    raise TypeError("trace_pets must be either 'all', list[int], or None")


def _resolve_runtime_context(layout_search_config: OM3LayoutSearchConfig) -> tuple[QueueConfig, list[str]]:
    """Resolve runtime context variables based on the layout search config and node count."""
    queue_config = QueueConfig.from_scheduler(
        layout_search_config.scheduler,
        layout_search_config.queue,
    )
    pools = _ordered_pools(layout_search_config.pool_map, layout_search_config.pool_order)

    return queue_config, pools


def generate_om3_core_layouts_from_node_count(
    num_nodes: float,
    cores_per_node: int,
    layout_search_config: OM3LayoutSearchConfig,
) -> list[OM3ConfigLayout]:
    """
    Generates valid core layouts for ACCESS-OM3.

    Ordering: within a node count, layouts are ordered by baseline pool ntasks ascending
              eg shared_13, shared_26, shared_39, ...
    """
    queue_config, pools = _resolve_runtime_context(layout_search_config)

    if queue_config.nodesize != cores_per_node:
        raise ValueError("cores_per_node does not match queue nodesize")

    if queue_config.nodesize % layout_search_config.blocks_per_node != 0:
        raise ValueError("blocks_per_node must divide nodesize")

    block_size = queue_config.nodesize // layout_search_config.blocks_per_node
    total_blocks = int(num_nodes * layout_search_config.blocks_per_node)

    layouts: list[OM3ConfigLayout] = []
    for alloc_blocks in _enumerate_block_allocations(total_blocks, pools, min_blocks=layout_search_config.min_blocks):
        if not _pass_ratio_constraints(
            alloc_blocks,
            baseline_pool_name=layout_search_config.baseline_pool_name,
            eps=layout_search_config.eps,
            max_ratio=layout_search_config.max_ratio_to_baseline,
            min_ratio=layout_search_config.min_ratio_to_baseline,
        ):
            continue

        pool_ntasks = {pool: alloc_blocks[pool] * block_size for pool in pools}
        pool_rootpe = _assign_rootpes(pools, pool_ntasks)

        layouts.append(
            OM3ConfigLayout(
                ncpus=sum(pool_ntasks.values()),
                pool_ntasks=pool_ntasks,
                pool_rootpe=pool_rootpe,
            )
        )

    layouts.sort(key=lambda x: x.pool_ntasks[layout_search_config.baseline_pool_name])

    return layouts


def quote_env_for_yaml(env_in: dict[str, str]) -> dict[str, str]:
    # Quote env variable values to ensure they are treated as strings in yaml
    return {key: DQString(value) for key, value in env_in.items()}


def generate_om3_perturb_block(
    layout: OM3ConfigLayout,
    num_nodes: float,
    layout_search_config: OM3LayoutSearchConfig,
    branch_name_prefix: str,
    walltime_hrs: float,
    block_overrides: dict | None = None,
) -> dict:
    """
    Generates a perturbation block for a given layout.

    block_overrides can be used to override any part of the generated block (e.g. to add additional env variables).
    """
    queue_config, pools = _resolve_runtime_context(layout_search_config)

    branch_parts = [branch_name_prefix, f"node_{int(num_nodes)}", f"queue_{queue_config.queue}"]
    for pool in pools:
        branch_parts.append(f"{pool}_{layout.pool_ntasks[pool]}")
    branch_name = "_".join(branch_parts)

    nodes_used = math.ceil(layout.ncpus / queue_config.nodesize)
    mem = f"{nodes_used * queue_config.nodemem}GB"

    pelayout = {}
    for submodel, pool in layout_search_config.pool_map.items():
        pelayout[f"{submodel}_ntasks"] = layout.pool_ntasks[pool]
        pelayout[f"{submodel}_rootpe"] = layout.pool_rootpe[pool]

    env = _build_esmf_env(layout, layout_search_config)
    if env:
        env = quote_env_for_yaml(env)

    block = {
        "branches": [branch_name],
        "config.yaml": {
            "metadata": {"enable": True},  # always enabled for perturbation blocks
            "queue": queue_config.queue,
            "ncpus": layout.ncpus,
            "mem": mem,
            "walltime": str(timedelta(hours=walltime_hrs)),
            "platform": {
                "nodesize": queue_config.nodesize,
                "nodemem": queue_config.nodemem,
            },
            **({"env": env} if env else {}),
        },
        "nuopc.runconfig": {
            "PELAYOUT_attributes": pelayout,
        },
    }

    if block_overrides:
        block = merge_blocks(block, block_overrides)

    return block
