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

from .config import ConfigParser
from .fortran_nml import FortranNMLParser
from .mom6_input import MOM6InputParser
from .yaml_config import YAMLParser
from .nuopc_config import NUOPCParser
from .profiling import ProfilingParser
from .fms_profiling import FMSProfilingParser

__all__ = [
    "ConfigParser",
    "FortranNMLParser",
    "MOM6InputParser",
    "YAMLParser",
    "NUOPCParser",
    "ProfilingParser",
    "FMSProfilingParser",
]
