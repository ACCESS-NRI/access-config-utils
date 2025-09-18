# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.parsers.cice5_profiling import CICE5ProfilingParser


@pytest.fixture(scope="module")
def cice5_required_metrics():
    return ("min", "max", "mean")


@pytest.fixture(scope="module")
def cice5_parser():
    """Fixture instantiating the CICE5 parser."""
    return CICE5ProfilingParser()


@pytest.fixture(scope="module")
def cice5_profiling():
    """Fixture returning a dict holding the parsed CICE5 timing content."""
    return {
        "region": ["Total", "TimeLoop"],
        "min": [16197.42, 16197.14],
        "max": [16197.47, 16197.19],
        "mean": [16197.44, 16197.16],
    }


@pytest.fixture(scope="module")
def cice5_log_file():
    """Fixture returning the CICE5 timing content."""
    return """ --------------------------------
   CICE model diagnostic output  
 --------------------------------
  
  Document ice_in namelist parameters:
  ==================================== 
  
  runtype                   = continue
 Restart read/written        17520   58159382400.0000     
  0.000000000000000E+000

Timing information:

Timer   1:     Total   16197.47 seconds
  Timer stats (node): min =    16197.42 seconds
                      max =    16197.47 seconds
                      mean=    16197.44 seconds
  Timer stats(block): min =        0.00 seconds
                      max =        0.00 seconds
                      mean=        0.00 seconds
Timer   2:  TimeLoop   16197.19 seconds
  Timer stats (node): min =    16197.14 seconds
                      max =    16197.19 seconds
                      mean=    16197.16 seconds
  Timer stats(block): min =        0.00 seconds
                      max =        0.00 seconds
                      mean=        0.00 seconds
"""


def test_cice5_profiling(cice5_required_metrics, cice5_parser, cice5_log_file, cice5_profiling):
    """Test the correct parsing of CICE5 timing information."""
    parsed_log = cice5_parser.read(cice5_log_file)

    # check metrics are present in parser and parsed output
    for metric in cice5_required_metrics:
        assert metric in cice5_parser.metrics, f"{metric} metric not found in CICE5 parser metrics."
        assert metric in parsed_log, f"{metric} metric not found in CICE5 parsed log."

    # check content for each metric is correct
    for idx, region in enumerate(cice5_profiling["region"]):
        assert region in parsed_log["region"], f"{region} not found in CICE5 parsed log"
        for metric in cice5_required_metrics:
            assert (
                cice5_profiling[metric][idx] == parsed_log[metric][idx]
            ), f"Incorrect {metric} for region {region} (idx: {idx})."
