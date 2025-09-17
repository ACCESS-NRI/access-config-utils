# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.parsers.profiling import ProfilingParser, _convert_from_string


class MockProfilingParser(ProfilingParser):
    """A Mock concrete Profiling Parser."""

    def __init__(self, data: dict):
        self._metrics = ["hits", "tmin", "tmax", "tavg"]
        self._data = data

    @property
    def metrics(self) -> list:
        return self._metrics

    def read(self, stream: str) -> dict:
        return self._data[stream]


@pytest.fixture(scope="module")
def profiling_data():
    """Fixture instantiating fake parsed profiling data."""
    return {
        "1cpu_stream": {
            "regions": ["Total runtime", "Ocean Initialization"],
            "hits": [1, 2],
            "tmin": [138.600364, 2.344926],
            "tmax": [138.600366, 2.345701],
            "tavg": [600365, 2.345388],
        },
        "2cpu_stream": {
            "regions": ["Total runtime", "Ocean Initialization"],
            "hits": [3, 4],
            "tmin": [69.300182, 1.162463],
            "tmax": [49.300182, 1.162463],
            "tavg": [300182.5, 1.172694],
        },
    }


def test_base_parser(profiling_data):
    """Tests methods and properties of abstract base class, ProfilingParser."""

    parser = MockProfilingParser(profiling_data)

    assert parser.metrics == ["hits", "tmin", "tmax", "tavg"], "Incorrect metrics returned in MockProfilingParser!"
    for stream in ("1cpu_stream", "2cpu_stream"):
        assert parser.read(stream) == profiling_data[stream], f'Incorrect profiling stats returned for "{stream}"'


def test_str2num():
    """Tests conversion of numbers to most appropriate type."""
    str2int = _convert_from_string("42")
    assert type(str2int) == int
    assert str2int == 42
    str2float = _convert_from_string("-1.23")
    assert type(str2float) == float
    assert str2float == -1.23
    str2float = _convert_from_string("0.00000")
    assert str2float == 0.0
    str2str = _convert_from_string("somestr")
    assert type(str2str) == str
    assert str2str == "somestr"
