# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF

from access.parsers.fortran_nml import FortranNMLParser


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser."""
    return FortranNMLParser()


@pytest.fixture()
def fortran_nml():
    """Fixture returning a dict holding the parsed content of a Fortran namelist file."""
    return {
        "LIST_A": {
            "VAR": 4,
            "LIST": [1, 2, 3],
            "FLOAT": 1800.0,
            "COMPLEX": 3.0 + 4.0j,
            "NOVALUE": None,
        },
        "LIST_B": {
            "BOOL": True,
            "LIST": ["a", "b", "c"],
            "DOUBLE": 1.0e-10,
            "STRING": "a string",
            "ANOTHER_LIST": [1, 2, 3, 4, 5, 6],
        },
        "LIST_C": {},
    }


@pytest.fixture()
def fortran_nml_file():
    """Fixture returning the content of a Fortran namelist file."""
    return """
&LIST_A ! This is a comment
  Var = 4 , ! This is a comment after an assignment
  LIST = 1, 2, 3

! This is another comment

  Float = 1800.0  ,
  COMPLEX = (3.0, 4.0),

  NOVALUE =
/

! Yet another comment
 This is some random text 

&LIST_B
  Bool = .true.
  LIST = "a", "b", "c", DOUBLE = 1d-10
  STRING="a string"
  ANOTHER_LIST = 1, 2,
                 3,   ! Comment in line break
                  4, 5, 6
/

&LIST_C
/
"""


@pytest.fixture()
def modified_fortran_nml_file():
    """Fixture returning the content of the previous Fortran namelist file, but with some modifications."""
    return """
&LIST_A ! This is a comment
  Var = 6 , ! This is a comment after an assignment
  LIST = 1, 2, 3

! This is another comment

  Float = 900.0  ,
  COMPLEX = (3.0, 4.0),

  NOVALUE =
/

! Yet another comment
 This is some random text 

&LIST_B
  Bool = .false.
  LIST = "a", "b", "c", DOUBLE = 1d-10
  STRING="another string"
  ANOTHER_LIST = 1, 2,
                 3,   ! Comment in line break
                  4, 5, 6
/

&LIST_C
/
"""


def test_valid_fortran_nml(parser):
    """Test the basic grammar constructs"""
    assert dict(parser.parse("&LIST TEST='a'/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST='a'/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST='a'\n/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST='a'\n&end")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST='a'\n&End")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse(" &LIST\nTEST='a'/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST = 'a' /")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST = 'a'/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST= 'a'/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST='a',\n/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST='a' , \n/")) == {"LIST": {"TEST": "a"}}
    assert dict(parser.parse("&LIST\nTEST = .true.\n/")) == {"LIST": {"TEST": True}}
    assert dict(parser.parse("&LIST\nTEST = .false.\n/")) == {"LIST": {"TEST": False}}
    assert dict(parser.parse("&LIST\nTEST='a', 'b'\n/")) == {"LIST": {"TEST": ["a", "b"]}}
    assert dict(parser.parse("&LIST\nTEST='a','b'\n/")) == {"LIST": {"TEST": ["a", "b"]}}
    assert dict(parser.parse("&LIST\nTEST='a','b',\n/")) == {"LIST": {"TEST": ["a", "b"]}}
    assert dict(parser.parse("&LIST\nTEST='a', \n'b', \n'c'\n/")) == {"LIST": {"TEST": ["a", "b", "c"]}}
    assert dict(parser.parse("&LIST\nVAR1=1, VAR2=2\n/")) == {"LIST": {"VAR1": 1, "VAR2": 2}}
    assert dict(parser.parse("&LIST\nVAR1=1, VAR2=2,\n/")) == {"LIST": {"VAR1": 1, "VAR2": 2}}
    assert dict(parser.parse("&LIST\nVAR1=1, 2, VAR2=3\n/")) == {"LIST": {"VAR1": [1, 2], "VAR2": 3}}
    assert dict(parser.parse("&LIST\nVAR1=1, VAR2=2, 3\n/")) == {"LIST": {"VAR1": 1, "VAR2": [2, 3]}}
    assert dict(parser.parse("&LIST\nVAR1=1, 2, VAR2=3, 4\n/")) == {"LIST": {"VAR1": [1, 2], "VAR2": [3, 4]}}
    assert dict(parser.parse("&LIST\nVAR1=1, VAR2=2, VAR3=3\n/")) == {"LIST": {"VAR1": 1, "VAR2": 2, "VAR3": 3}}
    assert dict(parser.parse("&LIST\nVAR1=1, VAR2=2, VAR3=3,\n/")) == {"LIST": {"VAR1": 1, "VAR2": 2, "VAR3": 3}}
    assert dict(parser.parse(" &LIST\nTEST = \n /")) == {"LIST": {"TEST": None}}


def test_invalid_fortran_nml(parser):
    """Test checking that the parser catches malformed expressions"""
    with pytest.raises(UnexpectedEOF):
        parser.parse("&LIST\nTEST : 'a'\n/")

    with pytest.raises(UnexpectedEOF):
        parser.parse("&LIST\nTEST = true\n/")

    with pytest.raises(UnexpectedEOF):
        parser.parse("&LIST\nTEST = false\n/")

    with pytest.raises(UnexpectedEOF):
        parser.parse("%TEST\na=1\n%TEST")

    with pytest.raises(UnexpectedEOF):
        parser.parse("TEST%\na=1\nTEST%")

    with pytest.raises(UnexpectedEOF):
        parser.parse("&TEST\na=1 2\n/")

    with pytest.raises(UnexpectedEOF):
        parser.parse("&TEST\na=1 \n2\n/")

    with pytest.raises(UnexpectedEOF):
        parser.parse("BLOCK\n TEST ='a'/")

    with pytest.raises(UnexpectedEOF):
        parser.parse("&BLOCK\n VAR1=1\n&e")


def test_fortran_nml_parse(parser, fortran_nml, fortran_nml_file):
    """Test parsing a file."""
    config = parser.parse(fortran_nml_file)
    assert dict(config) == fortran_nml


def test_fortran_nml_roundtrip(parser, fortran_nml_file):
    """Test round-trip parsing."""
    config = parser.parse(fortran_nml_file)

    assert str(config) == fortran_nml_file


def test_fortran_nml_roundtrip_with_mutation(parser, fortran_nml_file, modified_fortran_nml_file):
    """Test round-trip parsing with mutation of the config."""
    config = parser.parse(fortran_nml_file)

    config["LIST_A"]["VAR"] = 6
    config["LIST_A"]["FLOAT"] = 900.0
    config["LIST_B"]["BOOL"] = False
    config["LIST_B"]["STRING"] = "another string"

    assert str(config) == modified_fortran_nml_file
