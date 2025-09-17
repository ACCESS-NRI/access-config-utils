# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Classes and utilities to build profiling parsers for reading profiling data."""

from abc import ABC, abstractmethod
from typing import Any


class ProfilingParser(ABC):
    """Abstract parser of profiling data.

    The main purpose of a parser of profiling data is to read said data from a file or directory and return it in a
    standard format.

    Once parsed, the profiling data should be stored in a dict in the following way:

    {
        'region': ['region1', 'region2', ...],
        'metric a': [val1a, val2a, ...],
        'metric b': [val1b, val2b, ...],
        ...
    }

    The 'region' values correspond to the labels of the profile regions. Then, for each metric, there is a list of
    values, one for each profiling region. Therefore, 'val1a', is the value for metric a of region 1.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def metrics(self) -> list:
        """list: Metrics available when using this parser."""

    @abstractmethod
    def read(self, stream: str) -> dict:
        """Parse the given text.

        Args:
            stream (str): text to parse.

        Returns:
            dict: profiling data.
        """


def _convert_from_string(value: str) -> Any:
    """Tries to convert a string to the most appropriate numeric type. Leaves it unchanged if conversion does not succeed.

    Args:
        value (str): string to convert.

    Returns:
        Any: the converted string or the original string.
    """
    for type_conversion in (int, float):
        try:
            return type_conversion(value)
        except:
            continue
    return value
