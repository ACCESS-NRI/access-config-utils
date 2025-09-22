# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for UM profiling data.
This routine parses the inclusive timers from the UM output log
(e.g. ``atm.fort6.pe0`` for UM7) and returns a dictionary of the
profiling data. Since UM7 and UM13 provides multiple sections with timer
output - we have chosen to use the 'Wallclock times' sub-section
within the Inclusive Timer Summary section.

The profiling data is assumed to have the following
format:

```
 MPP : Inclusive timer summary

 WALLCLOCK  TIMES
 <N>   ROUTINE                   MEAN   MEDIAN       SD   % of mean      MAX   (PE)      MIN   (PE)
  1 AS3 Atmos_Phys2        1308.30  1308.30     0.02       0.00%  1308.33 ( 118)  1308.26 ( 221)
  2 AP2 Boundary Layer      956.50   956.13     3.26       0.34%   981.27 ( 136)   953.28 (  43)
  3 AS5-8 Updates           884.62   885.52     2.89       0.33%   889.49 (  48)   879.36 ( 212)

...

         CPU TIMES (sorted by wallclock times)
 <N>    ROUTINE                   MEAN   MEDIAN       SD   % of mean      MAX   (PE)      MIN   (PE)
...

 ```

All columns in the first sub-section, except for the numeric index and the `% of mean`, are parsed and returned.
For UM versions 13.x, there is an extra 'N' column name that appears to the left of 'ROUTINE'; this 'N' is
not present in the output from UM v7.x .

"""

from access.parsers.profiling import ProfilingParser, _convert_from_string
import re
import logging

logger = logging.getLogger(__name__)


class UMProfilingParser(ProfilingParser):
    """UM profiling output parser."""

    def __init__(self):
        """Instantiate the UM profiling parser."""
        super().__init__()

        # The parsed column names that will be kept. The order needs to match
        # the order of the column names in the input data (defined as ``raw_headers``
        # in the ``read``` method), after discarding the ignored columns.
        self._metrics = ["tavg", "tmed", "tstd", "tmax", "pemax", "tmin", "pemin"]

    @property
    def metrics(self) -> list:
        return self._metrics

    def get_um_version(self, stream: str) -> str:
        """Extract UM version from the input stream.

        Args:
            stream (str): input string to parse.
        Returns:
            str: A standardised UM version string.

        >>> data='**************** Based upon UM release vn13.1             *****************'
        >>> get_um_version(data)
        '13.1'

        >>> data='UM Version No         703'
        >>> get_um_version(data)
        '7.3'

        """
        version_p = re.compile(r"Based upon UM release vn(?P<version>[\d\.]+)")
        version_m = version_p.search(stream)
        if version_m:
            return version_m.group("version")

        # This works for UM7.3 (possibly 7.x)
        version_p = re.compile(r"UM Version No\s*(?P<version>[\d\.]+)")
        version_m = version_p.search(stream)
        if version_m:
            ver = version_m.group("version")
            if len(ver) == 3:
                # Convert to a more standard format
                ver = f"{ver[0]}.{ver[2:]}"
            return ver

        return None

    def _match_um_header(self, stream: str, raw_headers: str) -> (str, re.Match):
        header = r"MPP : Inclusive timer summary\s+WALLCLOCK  TIMES\s*" + r"\s*".join(raw_headers) + r"\s*"
        header_pattern = re.compile(header, re.MULTILINE)
        return header, header_pattern.search(stream)

    def read(self, stream: str) -> dict:
        """Parse UM profiling data from a string.

        Args:
            stream (str): input string to parse.

        Returns:
            stats (dict): dictionary of parsed profiling data.
                    Ignores two columns ``N``, and ``% over mean`` columns.

                    To keep consistent column names across all parsers, the following
                    mapping is used:
                        ==================  ==================
                        UM column name      Standard column name
                        ==================  ==================
                        N                   ignored
                        MEAN                tavg
                        MEDIAN              tmed
                        SD                  tstd
                        % of mean           ignored
                        MAX                 tmax
                        (PE)                pemax
                        MIN                 tmin
                        (PE)                pemin
                        ==================  ==================
                    Each key returns a list of values, one for each region. For
                    example, if there are 20 regions, ``stats['tavg']`` will
                    return a list with 20 values, one each for each of the regions.

                    The assumption is that people will want to look at the same metric
                    for *all* regions at a time; if you want to look at all metrics for
                    a single region, then you will have to first find the index for the
                    ``region``, and then extract that index from *each* of the 'metric'
                    lists.

        Raises:
            ValueError: If the UM version number can not be found in the input string data.
            ValueError: If a match for any of header, footer or section is not found.
            AssertionError: If the expected format is not found in *all* of the lines within the
                            profiling section.
        """

        # First create the local variable with the metrics list
        metrics = self.metrics

        raw_headers_dict = {
            "7": ["ROUTINE", "MEAN", "MEDIAN", "SD", r"\% of mean", "MAX", r"\(PE\)", "MIN", r"\(PE\)"],
            "13": ["N", "ROUTINE", "MEAN", "MEDIAN", "SD", r"\% of mean", "MAX", r"\(PE\)", "MIN", r"\(PE\)"],
        }

        # Need to check UM version to determine the correct headers
        um_version = self.get_um_version(stream)
        if not um_version:
            logger.debug("Could not determine UM version from input stream.")
            logger.debug("Input stream: %s", stream)
            # UM version was not there in the input stream - let's try to check if
            # there are any of the known matching headers
            for test_um_ver, raw_headers in raw_headers_dict.items():
                _, h_match = self._match_um_header(stream, raw_headers)
                if h_match:
                    um_version = test_um_ver
                    break

        logger.debug("Detected UM version: %s", um_version)
        um_version_major = um_version.split(".")[0]
        if um_version_major not in raw_headers_dict.keys():
            raise ValueError(
                f"Full UM version = {um_version} detected. UM major version = "
                f"{um_version_major} is invalid. Valid versions are "
                f"{list(raw_headers_dict.keys())}"
            )
        raw_headers = raw_headers_dict[um_version_major]

        header, header_match = self._match_um_header(stream, raw_headers)
        if not header_match:
            logger.debug("Header pattern: %s", header)
            logger.debug("Input string: %s", stream)
            raise ValueError("No matching header found.")
        logger.debug("Found header: %s", header_match.group(0))

        footer = r"CPU TIMES \(sorted by wallclock times\)\s*"
        footer_pattern = re.compile(footer, re.MULTILINE)
        footer_match = footer_pattern.search(stream)
        if not footer_match:
            logger.debug("Footer pattern: %s", footer)
            logger.debug("Input string: %s", stream)
            raise ValueError("No matching footer found.")
        logger.debug("Found footer: %s", footer_match.group(0))

        # Match *everything* between the header and footer
        profiling_section_p = re.compile(header + r"(.*)" + footer, re.MULTILINE | re.DOTALL)
        profiling_section = profiling_section_p.search(stream)

        profiling_section = profiling_section.group(1)
        logger.debug("Found section: %s", profiling_section)

        # This is regex dark arts - seems to work, I roughly understood when I
        # was refining this named capture group, but I might not be able to in
        # the future. Made heavy use of the regex debugger at regex101.com :) - MS 19/9/2025
        profile_line = r"^\s*\d+\s+(?P<region>[a-zA-Z:()_/\-*&0-9\s\.]+(?<!\s))"
        for metric in metrics:
            logger.debug(f"Adding {metric=}")
            if metric in ["pemax", "pemin"]:
                # the pemax and pemin values are enclosed within brackets '()',
                # so we need to ignore both the opening and closing brackets
                add_pattern = r"\s+\(\s*(?P<" + metric + r">[0-9.]+)\s*\)"
            elif metric == "tstd":
                add_pattern = (
                    r"\s+(?P<" + metric + r">[0-9.]+)\s+[\S]+"
                )  # SD is followed by % of mean -> ignore that column
            else:
                add_pattern = (
                    r"\s+(?P<" + metric + r">[0-9.]+)"
                )  # standard white-space followed by a sequence of digits or '.'

            logger.debug(f"{add_pattern=} for {metric=}")
            profile_line += add_pattern
            logger.debug(f"{profile_line=} after {metric=}")

        profile_line += r"$"  # the regexp should match till the end of line.
        profiling_region_p = re.compile(profile_line, re.MULTILINE)

        stats = {"region": []}
        stats.update(dict(zip(metrics, [[] for _ in metrics])))
        for line in profiling_region_p.finditer(profiling_section):
            logger.debug(f"Matched line: {line.group(0)}")
            stats["region"].append(line.group("region"))
            for metric in metrics:
                stats[str(metric)].append(_convert_from_string(line.group(metric)))

        # Parsing is done - let's run some checks
        num_lines = len(profiling_section.strip().split("\n"))
        logger.debug(f"Found {num_lines} lines in profiling section")
        if len(stats["region"]) != num_lines:
            raise AssertionError(f"Expected {num_lines} regions, found {len(stats['region'])}.")

        for metric in metrics:
            if len(stats[metric]) != num_lines:
                raise AssertionError(f"Expected {num_lines} entries for {metric}, found {len(stats[metric])}")

        logger.info(f"Found {len(stats['region'])} regions with profiling info")
        return stats
