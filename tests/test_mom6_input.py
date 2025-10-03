# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest
from lark.exceptions import UnexpectedCharacters

from access.config.mom6_input import MOM6InputParser


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser."""
    return MOM6InputParser()


@pytest.fixture(scope="module")
def mom6_input():
    """Fixture returning a dict holding the parsed content of a mom6_input file."""
    return {
        "VAR1": "test",
        "BLOCK": {"BVAR": 4, "BLIST": [1, 2, 3]},
        "Var2": 2,
        "List": ["a", "b", "c"],
        "FLOAT1": 1800.0,
        "FLOAT2": 1e-10,
        "BOOL": True,
    }


@pytest.fixture(scope="module")
def mom6_input_file():
    """Fixture returning the content of a mom6_input file."""
    return """
BOOL = True ! This is a comment
FLOAT1 = 1800.0

  ! This is another comment

FLOAT2 = 1e-10
VAR1="test"
Var2 = 2
BLOCK%
BVAR = 4
 ! A comment inside a block
BLIST = 1,2,3
%BLOCK ! Yet another comment

List = 'a','b','c'
"""


@pytest.fixture(scope="module")
def modified_mom6_input_file():
    """Fixture returning the content of the previous mom6_input file, but with some modifications."""
    return """
BOOL = True ! This is a comment
FLOAT1 = 900.0

  ! This is another comment

FLOAT2 = 1e-10
VAR1="replaced"
Var2 = 2
BLOCK%
BVAR = 32
 ! A comment inside a block
BLIST = 1,2,3
%BLOCK ! Yet another comment

List = 'a','b','c'
"""


def test_valid_mom6_input(parser):
    """Test the basic grammar constructs"""
    assert dict(parser.parse("TEST = 'a'")) == {"TEST": "a"}
    assert dict(parser.parse("TEST='a'")) == {"TEST": "a"}
    assert dict(parser.parse('TEST = "a"')) == {"TEST": "a"}
    assert dict(parser.parse("TEST = True")) == {"TEST": True}
    assert dict(parser.parse("TEST = False")) == {"TEST": False}
    assert dict(parser.parse("TEST = 'a','b'")) == {"TEST": ["a", "b"]}
    assert dict(parser.parse("TEST = 1,2")) == {"TEST": [1, 2]}
    assert dict(parser.parse("TEST%\na=1\n%TEST")) == {"TEST": {"a": 1}}
    assert dict(parser.parse("TEST = 'a' ! Comment\n ! Comment\n")) == {"TEST": "a"}
    assert dict(parser.parse("TEST%\na=1\n ! Comment\n%TEST")) == {"TEST": {"a": 1}}


def test_invalid_mom6_input(parser):
    """Test checking that the parser catches malformed expressions"""
    with pytest.raises(UnexpectedCharacters):
        parser.parse(" TEST = a")

    with pytest.raises(UnexpectedCharacters):
        parser.parse("TEST = a")

    with pytest.raises(UnexpectedCharacters):
        parser.parse("TEST : a")

    with pytest.raises(UnexpectedCharacters):
        parser.parse("TEST = true")

    with pytest.raises(UnexpectedCharacters):
        parser.parse("TEST = false")

    with pytest.raises(UnexpectedCharacters):
        dict(parser.parse("%TEST\na=1\n%TEST"))

    with pytest.raises(UnexpectedCharacters):
        dict(parser.parse("TEST%\na=1\nTEST%"))

    with pytest.raises(UnexpectedCharacters):
        parser.parse("BLOCK%\n TEST ='a'")


def test_mom6_input_parse(parser, mom6_input, mom6_input_file):
    """Test parsing of a file."""
    config = parser.parse(mom6_input_file)
    assert dict(config) == mom6_input


def test_mom6_input_roundtrip(parser, mom6_input_file):
    """Test round-trip parsing."""
    config = parser.parse(mom6_input_file)

    assert str(config) == mom6_input_file


def test_mom6_input_roundtrip_with_mutation(parser, mom6_input_file, modified_mom6_input_file):
    """Test round-trip parsing with mutation of the config."""
    config = parser.parse(mom6_input_file)

    config["VAR1"] = "replaced"
    config["FLOAT1"] = 900.0
    config["BLOCK"]["BVAR"] = 32

    assert str(config) == modified_mom6_input_file
