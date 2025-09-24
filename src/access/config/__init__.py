"""
access-parsers package.
"""

__version__ = "0.1.0"
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("access-parsers")
except PackageNotFoundError:
    # package is not installed
    pass

from access.config.parser import ConfigParser
from access.config.fortran_nml import FortranNMLParser
from access.config.mom6_input import MOM6InputParser
from access.config.yaml_config import YAMLParser
from access.config.nuopc_config import NUOPCParser

__all__ = [
    "ConfigParser",
    "FortranNMLParser",
    "MOM6InputParser",
    "YAMLParser",
    "NUOPCParser",
]
