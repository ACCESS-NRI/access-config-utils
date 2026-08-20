# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from conftest import assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedCharacters

from access.config.grammar_compiled import compile_grammar
from access.config.mom6_input import MOM6InputParser
from access.config.tree_edits import _parse_values
from access.config.tree_navigation import is_value_node


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
    """Fixture returning the previous mom6_input file, with some modifications."""
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


@pytest.mark.parametrize("text", ["A = 1\n", "A= 1\n", "A =1\n", "A=1\n", "A  =  1\n"])
def test_mom6_input_roundtrip_assignment_spacing(parser, text) -> None:
    """Test that whitespace either side of "=" is written back on the side it came from.

    The "=" is an anonymous literal, so Lark filters it out of the parse tree. With a bare
    "ws*" on each side, the tree records that a whitespace run exists but not which side of
    the "=" it fell on, and the reconstructor guesses -- writing "A= 1" back as "A =1".
    """
    assert str(parser.parse(text)) == text


def test_mom6_input_roundtrip_with_mutation(parser, mom6_input_file, modified_mom6_input_file):
    """Test round-trip parsing with mutation of the config."""
    config = parser.parse(mom6_input_file)

    config["VAR1"] = "replaced"
    config["FLOAT1"] = 900.0
    config["BLOCK"]["BVAR"] = 32

    assert str(config) == modified_mom6_input_file


@pytest.fixture()
def extended_mom6_input_file():
    """Fixture returning the content of a MOM6 input file with new keys added."""
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
BNEW = 2
%BLOCK ! Yet another comment

List = 'a','b','c'
NEW1 = 1
NEWLIST = 1,2
NEWSTR = "a"
NEWBOOL = True
NEWBLOCK%
X = 1
%NEWBLOCK
"""


def test_mom6_input_roundtrip_with_additions(parser, mom6_input_file, extended_mom6_input_file):
    """Test adding new keys of every kind the format supports."""
    config = parser.parse(mom6_input_file)

    config["BLOCK"]["BNEW"] = 2
    config["NEW1"] = 1
    config["NEWLIST"] = [1, 2]
    config["NEWSTR"] = "a"
    # MOM6 uses Python-style booleans rather than Fortran logicals.
    config["NEWBOOL"] = True
    config["NEWBLOCK"] = {"X": 1}

    assert str(config) == extended_mom6_input_file
    assert dict(parser.parse(str(config))) == dict(config)


def test_mom6_input_addition_no_indentation(parser):
    """Test that a new key is not indented, since the grammar does not allow it.

    The file ends with a whitespace-only line, which is the trap: copying the indentation
    from that line would produce a line the grammar cannot parse.
    """
    config = parser.parse("BOOL = True\n  \n")

    config["NEW"] = 1

    # The new entry follows the last assignment, so it precedes the trailing blank line, and
    # it carries no indentation.
    assert str(config) == "BOOL = True\nNEW = 1\n  \n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_mom6_input_addition_block_style(parser):
    """Test that a new block copies the assignment spacing of the block above it.

    MOM6 cannot indent an assignment at all, so the indentation a new block would inherit is
    always empty here. The spacing around "=" is what carries over instead, and this file
    writes it tight.
    """
    config = parser.parse("BLK%\nX=1\n%BLK\n")

    config["NEWBLK"] = {"Y": 2}

    assert str(config) == "BLK%\nX=1\n%BLK\nNEWBLK%\nY=2\n%NEWBLK\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_mom6_input_addition_after_comment_run(parser):
    """Test that a new key lands after a parameter's documentation, not inside it.

    This is the format where it matters most: a MOM6 parameter file documents nearly every
    parameter in a comment that starts on the parameter's own line and carries on below it.
    That trailing comment belongs to the entry, so the position one past the entry is the
    second line of its documentation.
    """
    config = parser.parse("DT = 900.0  ! [s]\n            ! The dynamics time step.\n")

    config["NEW"] = 1

    assert str(config) == "DT = 900.0  ! [s]\n            ! The dynamics time step.\nNEW = 1\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # The same inside a block, where the run is stepped over but "%BLK" is not.
    config = parser.parse("BLK%\nA = 1  ! what A is\n ! ...continued\n%BLK\n")

    config["BLK"]["B"] = 2

    assert str(config) == "BLK%\nA = 1  ! what A is\n ! ...continued\nB = 2\n%BLK\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_mom6_input_addition_errors(parser):
    """Test additions a MOM6 input file cannot express."""
    config = parser.parse("A = 1\n")

    # The grammar has no valueless assignment, and no complex or path value type.
    with pytest.raises(TypeError):
        config["Z"] = None
    with pytest.raises(TypeError):
        config["Z"] = 1 + 2j
    with pytest.raises(TypeError):
        config["Z"] = Path("/x/y")

    # Blocks do not nest.
    config = parser.parse("B%\nX = 1\n%B\n")
    with pytest.raises(TypeError):
        config["B"]["N"] = {"Y": 1}

    assert str(config) == "B%\nX = 1\n%B\n"


def test_mom6_input_delete_everything_keeps_comments(parser):
    """Test that deleting every key still writes out the text that is not an entry.

    The comments and blank lines are genuinely part of the file, so a configuration with no
    entries left is not the same thing as an empty file.
    """
    config = parser.parse("A = 1\n\n  ! keep me\n\nB = 2\n")

    for key in list(config):
        del config[key]

    assert str(config) == "\n  ! keep me\n\n"
    assert dict(parser.parse(str(config))) == {}

    # Adding a key back does not resurrect anything: the text was never hidden.
    config["C"] = 3
    assert str(config) == "C = 3\n\n  ! keep me\n\n"


def test_mom6_input_delete_removes_every_entry_that_wrote_the_key(parser):
    """Test that a key assigned more than once is deleted from every line assigning it.

    Only the last assignment survives in the configuration, so removing only the entry it
    came from left the earlier ones in the file, to reappear the next time it was read.
    """
    config = parser.parse("A = 1\nA = 2\nB = 3\n")
    assert dict(config) == {"A": 2, "B": 3}

    del config["A"]

    assert str(config) == "B = 3\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_mom6_input_value_snippet_parses_for_a_line_ending_rule(parser):
    """Test that values can be re-parsed for a format whose entry rule ends with its line.

    Rewriting an entry's values hands the text back to Lark with an entry rule as the start
    symbol. MOM6's ``key_value`` ends with ``line_end``, so a snippet without a trailing
    newline does not parse in it, where Fortran's does not accept one *with*. Only Fortran
    has repeat counts, so nothing reaches this through the public API yet -- which is why it
    is pinned here rather than left as a trap for the next grammar.
    """
    nodes = _parse_values(compile_grammar(parser.grammar).lark, ["1", "2"])

    assert [node.data for node in nodes if is_value_node(node)] == ["integer", "integer"]


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("mom6_input"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was generated for. These rows are the
    statement of what this grammar can express and how it spells it.
    """
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected)
