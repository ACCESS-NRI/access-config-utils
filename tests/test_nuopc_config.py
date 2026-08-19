# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from conftest import assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF

from access.config.grammar_compiled import compile_grammar
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


@pytest.mark.parametrize(
    "text",
    [
        "BLK::\n Verbosity = off\n::\n",
        "BLK::\n Verbosity= off\n::\n",
        "BLK::\n Verbosity =off\n::\n",
        "BLK::\n Verbosity=off\n::\n",
        "BLK::\n Verbosity  =  off\n::\n",
    ],
)
def test_nuopc_config_roundtrip_assignment_spacing(parser, text) -> None:
    """Test that a table assignment keeps its "=" and the whitespace around it.

    The "=" is an anonymous literal, so Lark filters it out of the parse tree. With a bare
    "ws*" on each side, the tree records that a whitespace run exists but not which side of
    the "=" it fell on. Here that also let the reconstructor confuse this rule with the
    resource-file rule aliased to the same name, and write "Verbosity: off" instead.
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


@pytest.fixture()
def extended_nuopc_config_file():
    """Fixture returning the content of a NUOPC config file with new keys added."""
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
  NEWB = datm
  NEWBLIST = 1:2:3
  NEWD = 1e-10

::
NEWTOP: off
NEWTOPLIST: a b
NEWPATH: x.nc
NEWBOOL: .true.
NEWTABLE::
  X = y
::

"""


def test_nuopc_config_roundtrip_with_additions(parser, nuopc_config_file, extended_nuopc_config_file):
    """Test adding new keys at both levels of the format.

    This file uses two assignment syntaxes -- ``:`` at the top level and ``=`` inside a
    ``::`` table -- and two list separators, a space at the top level and ``:`` in a table.
    Which one a new entry uses follows from the container it goes into.

    A new table is created empty, so it has no entry of its own to copy a style from: its
    contents are indented like those of the tables already in the file.
    """
    config = parser.parse(nuopc_config_file)

    config["ALLCOMP_attributes"]["NEWB"] = "datm"
    config["ALLCOMP_attributes"]["NEWBLIST"] = [1, 2, 3]
    config["ALLCOMP_attributes"]["NEWD"] = 1.0e-10

    config["NEWTOP"] = "off"
    config["NEWTOPLIST"] = ["a", "b"]
    config["NEWPATH"] = Path("x.nc")
    config["NEWBOOL"] = True
    config["NEWTABLE"] = {"X": "y"}

    assert str(config) == extended_nuopc_config_file
    assert dict(parser.parse(str(config))) == dict(config)


def test_nuopc_config_addition_two_element_list_in_table(parser):
    """Test that a new list of two in a table uses the table's list separator.

    A list of two is the length at which a table entry and a top-level one hold the same
    children, so it is the case where the entry could be written with the top-level
    separator instead.
    """
    config = parser.parse("TEST::\n a = 1:2:3\n::\n")

    config["TEST"]["b"] = [4, 5]

    assert str(config) == "TEST::\n a = 1:2:3\n b = 4:5\n::\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_nuopc_config_addition_after_comment_run(parser):
    """Test that a new key lands between comments rather than inside one.

    A comment on the entry's own line is held by that entry, so the position one past it is
    the second line of the comment. The run is stepped over instead, as far as the first
    blank line -- which is what leaves the new key between two comment blocks.
    """
    # A trailing comment carried on over the line below it.
    config = parser.parse("A: 1  # what A is\n# ...continued\n")
    config["NEW"] = 2
    assert str(config) == "A: 1  # what A is\n# ...continued\nNEW: 2\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # A blank line ends the run, so the new key goes between the two blocks.
    config = parser.parse("A: 1\n# about A\n\n# a separate block\n")
    config["NEW"] = 2
    assert str(config) == "A: 1\n# about A\nNEW: 2\n\n# a separate block\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # In a table, the run is stepped over but the closing "::" is not.
    config = parser.parse("T::\n a = 1  # what a is\n # ...continued\n::\n")
    config["T"]["b"] = 2
    assert str(config) == "T::\n a = 1  # what a is\n # ...continued\n b = 2\n::\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # A file of nothing but comments: the header keeps its place at the top.
    config = parser.parse("# header\n# second line\n")
    config["NEW"] = 2
    assert str(config) == "# header\n# second line\nNEW: 2\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # A blank line is not a comment, so nothing is stepped over and it still follows.
    config = parser.parse("A: 1\n\n# a block after a blank line\n")
    config["NEW"] = 2
    assert str(config) == "A: 1\nNEW: 2\n\n# a block after a blank line\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_nuopc_config_addition_errors(parser):
    """Test additions a NUOPC config file cannot express."""
    config = parser.parse("A: 1\n")

    # This grammar has no quoted-string value type, so a string that is not a bare word
    # cannot be written at all.
    for value in ("a string", "1", "a-b"):
        with pytest.raises(TypeError):
            config["Z"] = value

    # "foo" would be written as a bare word, which reads back as a string rather than a
    # path, so the candidate is rejected instead of storing a value of the wrong type.
    with pytest.raises(TypeError):
        config["Z"] = Path("foo")

    # No valueless assignment, and no complex value type.
    with pytest.raises(TypeError):
        config["Z"] = None
    with pytest.raises(TypeError):
        config["Z"] = 1 + 2j

    assert str(config) == "A: 1\n"


def test_nuopc_config_addition_multi_component_path(parser):
    """Test that a path of more than one component is stored as a path."""
    config = parser.parse("A: 1\n")

    config["Z"] = Path("./bar/baz")

    assert config["Z"] == Path("bar/baz")
    assert str(config) == "A: 1\nZ: bar/baz\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_nuopc_config_delete_removes_every_entry_that_wrote_the_key(parser):
    """Test that a label assigned more than once is deleted from every line assigning it."""
    config = parser.parse("A: 1\nA: 2\nB: 3\n")
    assert dict(config) == {"A": 2, "B": 3}

    del config["A"]

    assert str(config) == "B: 3\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_nuopc_config_new_table_body_is_indented(parser):
    """Test that a brand-new table indents its body, as the format conventionally does.

    Nothing in the grammar asks for this -- ``block`` accepts ``ws*`` -- so the text derived
    from the grammar alone would open the body in column 0. The parser declares a template
    to supply the convention for the case where no existing table can demonstrate it.
    """
    config = parser.parse("A: 1\n")

    config["tab"] = {"x": 2, "y": [3, 4]}

    assert str(config) == "A: 1\ntab::\n x = 2\n y = 3:4\n::\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_nuopc_config_new_table_copies_a_sibling_indent(parser):
    """Test that a sibling table's indentation beats the declared template.

    The template says one space, but matching the file wins: a declared convention only
    decides the case where there is nothing to copy.
    """
    config = parser.parse("old::\n   p = 1\n::\n")

    config["new"] = {"x": 2}

    assert str(config) == "old::\n   p = 1\n::\nnew::\n   x = 2\n::\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_nuopc_config_added_key_matches_its_own_table(parser):
    """Test that a key added to an existing table is spaced like the entries beside it.

    The regression the template must not cause: an entry one line below a three-space
    neighbour has to be indented three spaces too, not the one the template declares.
    """
    config = parser.parse("old::\n   p = 1\n::\n")

    config["old"]["q"] = 9

    assert str(config) == "old::\n   p = 1\n   q = 9\n::\n"
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("nuopc_config"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was generated for. These rows are the
    statement of what this grammar can express and how it spells it.
    """
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected)
