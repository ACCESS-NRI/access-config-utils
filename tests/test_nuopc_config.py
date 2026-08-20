# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF

from access.config.nuopc_config import NUOPCParser


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser."""
    return NUOPCParser()


@pytest.fixture(scope="module")
def nuopc_config():
    """Fixture returning a dict holding the parsed content of a NUOPC config file."""
    return {
        "DRIVER_attributes": {
            "Verbosity": "off",
            "cime_model": "cesm",
            "logFilePostFix": Path(".log"),
            "pio_blocksize": -1,
            "pio_rearr_comm_enable_hs_comp2io": True,
            "pio_rearr_comm_enable_hs_io2comp": False,
            "reprosum_diffmax": -1.0e-8,
            "wv_sat_table_spacing": 1.0,
            "wv_sat_transition_start": 20.0,
        },
        "COMPONENTS": ["atm", "ocn"],
        "TEST": "On",
        "ALLCOMP_attributes": {
            "ATM_model": "datm",
            "GLC_model": "sglc",
            "OCN_model": "mom",
            "ocn2glc_levels": [1, 10, 19, 26, 30, 33, 35],
        },
    }


@pytest.fixture(scope="module")
def nuopc_config_file():
    """Fixture returning the content of a NUOPC config file."""
    return """DRIVER_attributes:: # Comment 1

  Verbosity = off
  cime_model = cesm # Comment 2

  logFilePostFix = .log
  pio_blocksize = -1

  # Comment 3

  pio_rearr_comm_enable_hs_comp2io = .true.
  pio_rearr_comm_enable_hs_io2comp = .false.
  reprosum_diffmax = -1.000000D-08
  wv_sat_table_spacing = 1.000000D+00
  wv_sat_transition_start = 2.000000D+01
::

TEST: On

  # Comment 4
# Comment 5

COMPONENTS: atm ocn   # Comment 6

ALLCOMP_attributes::

  ATM_model = datm
  GLC_model = sglc
  OCN_model = mom
  ocn2glc_levels = 1:10:19:26:30:33:35

::

"""


@pytest.fixture(scope="module")
def modified_nuopc_config_file():
    """Fixture returning the previous NUOPC config file, with some modifications."""
    return """DRIVER_attributes:: # Comment 1

  Verbosity = off
  cime_model = cesm # Comment 2

  logFilePostFix = .log
  pio_blocksize = -1

  # Comment 3

  pio_rearr_comm_enable_hs_comp2io = .true.
  pio_rearr_comm_enable_hs_io2comp = .false.
  reprosum_diffmax = -1.000000D-08
  wv_sat_table_spacing = 1.000000D+00
  wv_sat_transition_start = 2.000000D+01
::

TEST: Off

  # Comment 4
# Comment 5

COMPONENTS: atm um   # Comment 6

ALLCOMP_attributes::

  ATM_model = um
  GLC_model = sglc
  OCN_model = mom
  ocn2glc_levels = 1:10:19:26:30:33:36

::

"""


def test_valid_nuopc_config(parser):
    """Test the basic grammar constructs"""
    assert dict(parser.parse("TEST: a")) == {"TEST": "a"}
    assert dict(parser.parse(" TEST:a")) == {"TEST": "a"}
    assert dict(parser.parse("TEST: a b")) == {"TEST": ["a", "b"]}
    assert dict(parser.parse("TEST1: a\n TEST2: b")) == {"TEST1": "a", "TEST2": "b"}
    assert dict(parser.parse("TEST1: a \nTEST2: b")) == {"TEST1": "a", "TEST2": "b"}
    assert dict(parser.parse("TEST1: a b \n TEST2: c")) == {"TEST1": ["a", "b"], "TEST2": "c"}
    assert dict(parser.parse("TEST::\na=1\n::")) == {"TEST": {"a": 1}}
    assert dict(parser.parse("TEST::\na=1:2:3\n::")) == {"TEST": {"a": [1, 2, 3]}}


def test_invalid_nuopc_config(parser):
    """Test checking that the parser catches malformed expressions"""
    with pytest.raises(UnexpectedCharacters):
        parser.parse("TEST::\n cime_model - cesm")

    with pytest.raises(UnexpectedCharacters):
        parser.parse("TEST:\n cime_model = cesm\n::")

    with pytest.raises(UnexpectedEOF):
        parser.parse("TEST::\n cime_model = cesm")

    with pytest.raises(UnexpectedCharacters):
        parser.parse("TEST::\n cime_model = cesm ATM_model = datm")


def test_nuopc_config_parse(parser, nuopc_config, nuopc_config_file):
    """Test parsing of a file."""
    config = parser.parse(nuopc_config_file)
    assert dict(config) == nuopc_config


def test_nuopc_config_roundtrip(parser, nuopc_config_file):
    """Test round-trip parsing."""
    config = parser.parse(nuopc_config_file)

    assert str(config) == nuopc_config_file


@pytest.mark.parametrize(
    "text",
    [
        # A list of two is the shape a top-level list can take as well.
        "TEST::\n a = 1:2\n::\n",
        "TEST::\n a = 1.5:2.5\n::\n",
        "TEST::\n a = x:y\n::\n",
        # With no whitespace around the "=", so is a scalar, and a list of any length.
        "TEST::\n a=1\n::\n",
        "TEST::\n a=1:2:3\n::\n",
        # The top-level syntax has to keep writing itself, too.
        "TEST: 1 2\n",
        "TEST:1 2\n",
        "TEST:1\n",
    ],
)
def test_nuopc_config_table_syntax_roundtrip(parser, text):
    """Test that an entry is written back in the syntax of the level it sits at.

    The two levels of the format alias their entries to the same rule names, so what tells
    them apart when writing out is the ``=`` that a table entry keeps in the parse tree.
    Without it a table entry can be written in the top-level syntax -- a file that does not
    parse -- whenever the two levels admit the same sequence of children.
    """
    assert str(parser.parse(text)) == text


def test_nuopc_config_roundtrip_with_mutation(parser, nuopc_config_file, modified_nuopc_config_file):
    """Test round-trip parsing with mutation of the config."""
    config = parser.parse(nuopc_config_file)

    # Scalar
    config["TEST"] = "Off"
    # List
    config["COMPONENTS"] = ["atm", "um"]
    # Scalar in table
    config["ALLCOMP_attributes"]["ATM_model"] = "um"
    # List in table
    config["ALLCOMP_attributes"]["ocn2glc_levels"] = [1, 10, 19, 26, 30, 33, 36]

    assert str(config) == modified_nuopc_config_file
