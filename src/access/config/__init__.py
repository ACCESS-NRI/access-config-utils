"""access-config-utils package."""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

__version__ = "unknown"
with suppress(PackageNotFoundError):
    __version__ = version("access-config-utils")

from access.config.fortran_nml import FortranNMLParser
from access.config.mom6_input import MOM6InputParser
from access.config.nuopc_config import NUOPCParser
from access.config.parser import Config, ConfigList, ConfigParser
from access.config.parser_types import UnsupportedEntryError
from access.config.yaml_config import YAMLParser

__all__ = [
    "Config",
    "ConfigList",
    "ConfigParser",
    "FortranNMLParser",
    "MOM6InputParser",
    "UnsupportedEntryError",
    "YAMLParser",
    "NUOPCParser",
]
