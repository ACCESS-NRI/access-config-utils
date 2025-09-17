# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.parsers.fms_profiling import FMSProfilingParser


@pytest.fixture(scope="module")
def fms_hits_parser():
    """Fixture instantiating the FMS parser where hits column is present."""
    return FMSProfilingParser()


@pytest.fixture(scope="module")
def fms_nohits_parser():
    """Fixture instantiating the FMS parser where hits column is not present."""
    return FMSProfilingParser(has_hits=False)


@pytest.fixture(scope="module")
def fms_nohits_profiling():
    """Fixture returning a dict holding the parsed FMS timing content without hits."""
    return {
        "region": [
            "Total runtime",
            "Ocean",
            "(Ocean initialization)",
            "(Ocean ODA)",
            "(Red Sea/Gulf Bay salinity fix)",
            "OASIS init",
            "oasis_recv",
            "oasis_send",
        ],
        "tmin": [16282.797785, 15969.542784, 4.288529, 0.0, 0.024143, 0.231678, 168.797136, 2.468914],
        "tmax": [16282.797792, 16000.704550, 4.296586, 0.0, 0.077235, 0.232671, 171.648384, 2.756777],
        "tavg": [16282.797789, 15986.765795, 4.291991, 0.0, 0.040902, 0.232397, 170.460762, 2.593809],
        "tstd": [0.000001, 8.643639, 0.001470, 0.0, 0.013836, 0.000242, 0.650894, 0.079459],
    }


@pytest.fixture(scope="module")
def fms_nohits_log_file():
    """Fixture returning the FMS timing content without hits column."""
    return """ MPP_DOMAINS_STACK high water mark=      747000

Tabulating mpp_clock statistics across     49 PEs...

                                          tmin          tmax          tavg          tstd  tfrac grain pemin pemax
Total runtime                     16282.797785  16282.797792  16282.797789      0.000001  1.000     0     0    48
Ocean                             15969.542784  16000.704550  15986.765795      8.643639  0.982     1     0    48
(Ocean initialization)                4.288529      4.296586      4.291991      0.001470  0.000    11     0    48
(Ocean ODA)                           0.000000      0.000000      0.000000      0.000000  0.000    11     0    48
(Red Sea/Gulf Bay salinity fix)       0.024143      0.077235      0.040902      0.013836  0.000    31     0    48
OASIS init                            0.231678      0.232671      0.232397      0.000242  0.000     1     0    48
oasis_recv                          168.797136    171.648384    170.460762      0.650894  0.010    31     0    48
oasis_send                            2.468914      2.756777      2.593809      0.079459  0.000    31     0    48
 MPP_STACK high water mark=           0
 MOM5: --- completed ---
"""


@pytest.fixture(scope="module")
def fms_hits_profiling():
    """Fixture returning a dict holding the parsed FMS timing content with hits."""
    return {
        "region": [
            "Total runtime",
            "Initialization",
            "Main loop",
            "Termination",
            "Ocean Initialization",
            "Ocean",
            "Ocean dynamics",
            "Ocean thermodynamics and tracers",
            "Ocean grid generation and remapp",
            "Ocean Other",
            "(Ocean tracer advection)",
        ],
        "hits": [1, 1, 1, 1, 2, 24, 192, 72, 0, 192, 48],
        "tmin": [
            100.641190,
            0.987726,
            98.930085,
            0.718969,
            1.529830,
            98.279247,
            84.799971,
            11.512013,
            0.0,
            1.710326,
            4.427230,
        ],
        "tmax": [
            100.641190,
            0.987726,
            98.930085,
            0.718969,
            1.529830,
            98.279247,
            84.799971,
            11.512013,
            0.0,
            1.710326,
            4.427230,
        ],
        "tavg": [
            100.641190,
            0.987726,
            98.930085,
            0.718969,
            1.529830,
            98.279247,
            84.799971,
            11.512013,
            0.000000,
            1.710326,
            4.427230,
        ],
        "tstd": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


@pytest.fixture(scope="module")
def fms_hits_log_file():
    """Fixture returning a dict holding the parsed FMS timing content wiht hits."""
    return """ MPP_DOMAINS_STACK high water mark=      380512

Tabulating mpp_clock statistics across      1 PEs...

                                      hits          tmin          tmax          tavg          tstd  tfrac grain pemin pemax
Total runtime                            1    100.641190    100.641190    100.641190      0.000000  1.000     0     0     0
Initialization                           1      0.987726      0.987726      0.987726      0.000000  0.010     0     0     0
Main loop                                1     98.930085     98.930085     98.930085      0.000000  0.983     0     0     0
Termination                              1      0.718969      0.718969      0.718969      0.000000  0.007     0     0     0
Ocean Initialization                     2      1.529830      1.529830      1.529830      0.000000  0.015    11     0     0
Ocean                                   24     98.279247     98.279247     98.279247      0.000000  0.977     1     0     0
Ocean dynamics                         192     84.799971     84.799971     84.799971      0.000000  0.843    11     0     0
Ocean thermodynamics and tracers        72     11.512013     11.512013     11.512013      0.000000  0.114    11     0     0
Ocean grid generation and remapp         0      0.000000      0.000000      0.000000      0.000000  0.000    11     0     0
Ocean Other                            192      1.710326      1.710326      1.710326      0.000000  0.017    11     0     0
(Ocean tracer advection)                48      4.427230      4.427230      4.427230      0.000000  0.044    21     0     0
 MPP_STACK high water mark=           0
"""


def test_fms_nohits_profiling(fms_nohits_parser, fms_nohits_log_file, fms_nohits_profiling):
    """Test the correct parsing of FMS timing information without hits column."""
    parsed_log = fms_nohits_parser.read(fms_nohits_log_file)
    for idx, region in enumerate(fms_nohits_profiling.keys()):
        assert region in parsed_log, f"{region} not found in mom5 parsed log"
        for metric in ("tmin", "tmax", "tavg", "tstd"):
            assert (
                fms_nohits_profiling[metric][idx] == parsed_log[metric][idx]
            ), f"Incorrect {metric} for region {region} (idx: {idx})."


def test_mom6_profiling(fms_hits_parser, fms_hits_log_file, fms_hits_profiling):
    """Test the correct parsing of FMS timing information with hits column."""
    parsed_log = fms_hits_parser.read(fms_hits_log_file)
    for idx, region in enumerate(fms_hits_profiling.keys()):
        assert region in parsed_log, f"{region} not found in mom6 parsed log"
        for metric in ("hits", "tmin", "tmax", "tavg", "tstd"):
            assert (
                fms_hits_profiling[metric][idx] == parsed_log[metric][idx]
            ), f"Incorrect {metric} for region {region} (idx: {idx})."
