# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for CICE5 profiling data.
The data to be parsed is written in the following form, where bloc stats are discarded:

Timer   1:     Total    8133.37 seconds
  Timer stats (node): min =     8133.36 seconds
                      max =     8133.37 seconds
                      mean=     8133.36 seconds
  Timer stats(block): min =        0.00 seconds
                      max =        0.00 seconds
                      mean=        0.00 seconds
Timer   2:  TimeLoop    8133.00 seconds
  Timer stats (node): min =     8132.99 seconds
                      max =     8133.00 seconds
                      mean=     8132.99 seconds
  Timer stats(block): min =        0.00 seconds
                      max =        0.00 seconds
                      mean=        0.00 seconds
"""

from access.parsers.profiling import ProfilingParser
import re


class CICE5ProfilingParser(ProfilingParser):
    """CICE5 profiling output parser."""

    def __init__(self):
        super().__init__()
        self._metrics = ["min", "max", "mean"]

    @property
    def metrics(self) -> list:
        """Implements "metrics" abstract method/property.

        Returns:
            list: the metric names captured by this parser.
        """
        return self._metrics

    def read(self, stream: str) -> dict:
        """Implements "read" abstract method to parse profiling data in CICE5 log output.

        Args:
            stream (str): String containing the CICE5 log to be parsed.

        Returns:
            dict: Parsed timing information.
        """
        # Initialize result dictionary
        result = {"region": [], "min": [], "max": [], "mean": []}

        # Regex pattern to match timer blocks
        # This captures the region name and the three node timing values
        pattern = r"Timer\s+\d+:\s+(\w+)\s+[\d.]+\s+seconds\s+Timer stats \(node\): min =\s+([\d.]+) seconds\s+max =\s+([\d.]+) seconds\s+mean=\s+([\d.]+) seconds"

        # Find all matches
        matches = re.findall(pattern, stream, re.MULTILINE | re.DOTALL)

        # Extract data from matches
        for match in matches:
            region, min_time, max_time, mean_time = match
            result["region"].append(region)
            result["min"].append(float(min_time))
            result["max"].append(float(max_time))
            result["mean"].append(float(mean_time))

        return result
