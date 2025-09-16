# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path

from access.parsers.profiling import _convert_from_string


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
