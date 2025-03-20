"""Utilities to handle payu configuration files.

Configuration for payu experiments are stored using YAML. Documentation about these files can be found here:

    https://payu.readthedocs.io/en/latest/config.html

Round-trip parsing is supported by using the ruamel.yaml parser.
"""

from pathlib import Path
from ruamel.yaml import YAML, CommentedMap


def read_payu_config_yaml(file_name: str) -> CommentedMap:
    """Read a payu configuration file.

    This function uses ruamel to parse the YAML file, so that we can do round-trip parsing.

    Args:
        file_name: Name of file to read.

    Returns:
        dict: Payu configuration.
    """
    fname = Path(file_name)
    if not fname.is_file():
        raise FileNotFoundError(f"File not found: {fname.as_posix()}")

    config = YAML().load(fname)

    return config


def write_payu_config_yaml(config: [dict | CommentedMap], file: Path):
    """Write a Payu configuration to a file.

    Args:
        config (dict| CommentedMap): Payu configuration.
        file(Path): File to write to.
    """
    YAML().dump(config, file)
