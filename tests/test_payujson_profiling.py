# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest
import datetime

from access.parsers.payujson_profiling import PayuJSONProfilingParser


@pytest.fixture(scope="module")
def payujson_parser():
    """Fixture instantiating the Payu JSON parser."""
    return PayuJSONProfilingParser()


@pytest.fixture(scope="module")
def payujson_profiling():
    """Fixture returning a dict holding the parsed FMS timing content without hits."""
    return {
        "regions": [
            "payu_setup_duration_seconds",
            "payu_model_run_duration_seconds",
            "payu_run_duration_seconds",
            "payu_archive_duration_seconds",
            "payu_total_duration_seconds",
        ],
        "walltime": [47.73822930175811, 6776.044810215011, 6779.385873348918, 8.063649574294686, 6838.225644],
    }


@pytest.fixture(scope="module")
def payujson_log_file():
    """Fixture returning the FMS timing content without hits column."""
    return """{
    "scheduler_job_id": "149764665.gadi-pbs",
    "timings": {
        "payu_start_time": "2025-09-16T08:52:50.748807",
        "payu_setup_duration_seconds": 47.73822930175811,
        "payu_model_run_duration_seconds": 6776.044810215011,
        "payu_run_duration_seconds": 6779.385873348918,
        "payu_archive_duration_seconds": 8.063649574294686,
        "payu_finish_time": "2025-09-16T10:46:48.974451",
        "payu_total_duration_seconds": 6838.225644
    },
    "payu_run_id": "5c9027104cc39a5d39814624537c21440b68beb7",
    "payu_model_run_status": 0,
    "model_finish_time": "1844-01-01T00:00:00",
    "model_start_time": "1843-01-01T00:00:00",
    "model_calendar": "proleptic_gregorian",
    "payu_run_status": 0
}
"""


def test_payujson_profiling(payujson_parser, payujson_log_file, payujson_profiling):
    """Test the correct parsing of Payu JSON timing information."""
    parsed_log = payujson_parser.read(payujson_log_file)
    for idx, region in enumerate(payujson_profiling.keys()):
        assert region in parsed_log, f"{region} not found in Payu JSON parsed log"
        assert (
            payujson_profiling["walltime"][idx] == parsed_log["walltime"][idx]
        ), f"Incorrect walltime for region {region} (idx: {idx})."
