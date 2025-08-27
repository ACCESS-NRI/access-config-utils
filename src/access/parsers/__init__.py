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
