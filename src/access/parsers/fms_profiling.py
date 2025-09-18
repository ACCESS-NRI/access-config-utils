# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for FMS profiling data, such as output by MOM5 and MOM6.
The data to be parsed is written in the following form:

                                    hits          tmin          tmax          tavg          tstd  tfrac grain pemin pemax
Total runtime                         1    138.600364    138.600366    138.600365      0.000001  1.000     0     0    11
Ocean Initialization                  2      2.344926      2.345701      2.345388      0.000198  0.017    11     0    11
Ocean                                23     86.869466     86.871652     86.870450      0.000744  0.627     1     0    11
Ocean dynamics                       96     43.721019     44.391032     43.957944      0.244785  0.317    11     0    11
Ocean thermodynamics and tracers     72     27.377185     33.281659     29.950144      1.792324  0.216    11     0    11
 MPP_STACK high water mark=           0
"""

from access.parsers.profiling import ProfilingParser, _convert_from_string
import re


class FMSProfilingParser(ProfilingParser):
    """FMS profiling output parser."""

    def __init__(self, has_hits: bool = True):
        """Instantiate FMS profiling parser.

        Args:
            has_hits (bool): whether FMS timings contains "hits" column.
        """
        super().__init__()

        # FMS provides the following metrics:
        if has_hits:
            self._metrics = ["hits"]
        else:
            self._metrics = []
        self._metrics += ["tmin", "tmax", "tavg", "tstd", "tfrac", "grain", "pemin", "pemax"]

    @property
    def metrics(self) -> list:
        return self._metrics

    def read(self, stream: str) -> dict:

        # Regular expression to extract the profiling section from the file
        header = r"\s*" + r"\s*".join(self._metrics) + r"\s*"
        footer = r" MPP_STACK high water mark=\s*\d*"
        profiling_section_p = re.compile(header + r"(.*)" + footer, re.DOTALL)

        # Regular expression to parse the data for each region
        profile_line = r"^\s*(?P<region>[a-zA-Z:()_/\-*&\s]+(?<!\s))"
        for metric in self.metrics:
            profile_line += r"\s+(?P<" + metric + r">[0-9.]+)"
        profile_line += r"$"
        profiling_region_p = re.compile(profile_line, re.MULTILINE)

        # Parse data
        stats = {"region": []}
        stats.update({m: [] for m in self.metrics})
        profiling_section = profiling_section_p.search(stream).group(1)
        for line in profiling_region_p.finditer(profiling_section):
            stats["region"].append(line.group("region"))
            for metric in self.metrics:
                stats[str(metric)].append(_convert_from_string(line.group(metric)))

        return stats
