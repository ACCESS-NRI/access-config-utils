"""access-config-utils package."""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

__version__ = "unknown"
with suppress(PackageNotFoundError):
    __version__ = version("access-config-utils")

from access.config.config_dict import Config, ConfigBlockList, ConfigList, ConfigRecord
from access.config.field_table import FieldTableParser
from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_values import UnsupportedEntryError
from access.config.json_config import JSONParser
from access.config.mom6_input import MOM6InputParser
from access.config.nuopc_config import NUOPCParser
from access.config.nuopc_runseq import NUOPCRunSeqParser
from access.config.parallel_allocation_strategies import (
    FixedAllocation,
    FreeAllocation,
    RootAllocation,
    WeightedAllocation,
    WholeRangeAllocation,
)
from access.config.parallel_constraints import (
    DomainDivisibleByRanksConstraint,
    EqualCoresGroupConstraint,
    FixedThreadsPerRankConstraint,
    MaxThreadsPerRankConstraint,
    MaxWastedCoreFractionConstraint,
    MinSubdomainSizeConstraint,
    ProcessGridDimDivisibleConstraint,
    ProcessGridDimEvenConstraint,
    RankRatioGroupConstraint,
    SubdomainAspectRatioConstraint,
)
from access.config.parser import ConfigParser
from access.config.rose_conf import RoseConfParser
from access.config.yaml_config import YAMLParser

__all__ = [
    "Config",
    "ConfigBlockList",
    "ConfigList",
    "ConfigParser",
    "ConfigRecord",
    "FieldTableParser",
    "FortranNMLParser",
    "JSONParser",
    "MOM6InputParser",
    "UnsupportedEntryError",
    "YAMLParser",
    "NUOPCParser",
    "NUOPCRunSeqParser",
    "RoseConfParser",
    # The allocation strategies and the constraints that a study of the layout search writes
    "RootAllocation",
    "FixedAllocation",
    "WeightedAllocation",
    "FreeAllocation",
    "WholeRangeAllocation",
    "DomainDivisibleByRanksConstraint",
    "EqualCoresGroupConstraint",
    "FixedThreadsPerRankConstraint",
    "MaxThreadsPerRankConstraint",
    "MaxWastedCoreFractionConstraint",
    "MinSubdomainSizeConstraint",
    "ProcessGridDimDivisibleConstraint",
    "ProcessGridDimEvenConstraint",
    "RankRatioGroupConstraint",
    "SubdomainAspectRatioConstraint",
]
