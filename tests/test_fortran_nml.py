# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from conftest import assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedEOF

from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import compile_grammar


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
    """Fixture returning the previous Fortran namelist file, with some modifications."""
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


@pytest.mark.parametrize(
    "text",
    [
        "&L\nA = 1\n/\n",
        "&L\nA= 1\n/\n",
        "&L\nA =1\n/\n",
        "&L\nA=1\n/\n",
        "&L\nA  =  1\n/\n",
        "&L\nA=  0 , 1 ,\n/\n",
    ],
)
def test_fortran_nml_roundtrip_assignment_spacing(parser, text) -> None:
    """Test that whitespace either side of "=" is written back on the side it came from.

    The "=" is an anonymous literal, so Lark filters it out of the parse tree. With a bare
    "ws*" on each side, the tree records that a whitespace run exists but not which side of
    the "=" it fell on, and the reconstructor guesses -- writing "A= 1" back as "A =1". The
    "eq" rule binds the leading whitespace to the "=" so the position survives.
    """
    assert str(parser.parse(text)) == text


def test_fortran_nml_roundtrip_with_mutation(parser, fortran_nml_file, modified_fortran_nml_file):
    """Test round-trip parsing with mutation of the config."""
    config = parser.parse(fortran_nml_file)

    config["LIST_A"]["VAR"] = 6
    config["LIST_A"]["FLOAT"] = 900.0
    config["LIST_B"]["BOOL"] = False
    config["LIST_B"]["STRING"] = "another string"

    assert str(config) == modified_fortran_nml_file


@pytest.fixture()
def extended_fortran_nml_file():
    """Fixture returning the content of a Fortran namelist file with new keys added."""
    return """
&LIST_A ! This is a comment
  Var = 4 , ! This is a comment after an assignment
  LIST = 1, 2, 3

! This is another comment

  Float = 1800.0  ,
  COMPLEX = (3.0, 4.0),

  NOVALUE =
  NEWVAR = 7
  NEWNULL =
  NEWSTR = "a string"
  NEWBOOL = .true.
  NEWCOMPLEX = (3.0, 4.0)
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
  NEWLIST = 1, 2, 3
  NEWDOUBLE = 1e-10
/

&LIST_C
NEWVAR3 = 9
/
&NEWNML
A = 1
/
"""


def test_fortran_nml_roundtrip_with_additions(parser, fortran_nml_file, extended_fortran_nml_file):
    """Test adding new keys of every kind the format supports.

    One byte-exact expectation covers indentation, where each new entry lands relative to
    trailing blank lines and the namelist terminator, and the notation chosen per value.
    """
    config = parser.parse(fortran_nml_file)

    # Each new key goes after the last assignment of its namelist and before the closing
    # "/", indented to match its neighbours.
    config["LIST_A"]["NEWVAR"] = 7
    config["LIST_A"]["NEWNULL"] = None
    config["LIST_A"]["NEWSTR"] = "a string"
    config["LIST_A"]["NEWBOOL"] = True
    config["LIST_A"]["NEWCOMPLEX"] = 3.0 + 4.0j

    # LIST_B ends with a list continued over three physical lines; the new keys follow it.
    config["LIST_B"]["NEWLIST"] = [1, 2, 3]
    config["LIST_B"]["NEWDOUBLE"] = 1.0e-10

    # LIST_C is empty, so there is no entry to copy a style from: the layout is canonical.
    config["LIST_C"]["NEWVAR3"] = 9

    # A whole new namelist group, created empty and then filled. Its contents copy the
    # nearest group above it, which is the LIST_C just filled canonically, so they are not
    # indented either. See test_fortran_nml_addition_block_indentation.
    config["NEWNML"] = {"A": 1}

    assert str(config) == extended_fortran_nml_file
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_addition_block_indentation(parser):
    """Test that the contents of a new namelist group follow the groups already there.

    A group created by an assignment is empty, so it holds no entry of its own to copy a
    style from and its contents would otherwise be written flush left, in a file whose other
    groups are indented.
    """
    config = parser.parse("&L\n  X = 1\n/\n")
    config["NEW"] = {"A": 1}

    assert str(config) == "&L\n  X = 1\n/\n&NEW\n  A = 1\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # Nothing is invented: a file that indents nothing gets an unindented group.
    config = parser.parse("&L\nX = 1\n/\n")
    config["NEW"] = {"A": 1}

    assert str(config) == "&L\nX = 1\n/\n&NEW\nA = 1\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # The style reaches keys added later, not just those in the dict the block was made
    # from, since the block keeps it until it has an entry of its own to copy.
    config = parser.parse("&L\n  X = 1\n/\n")
    config["NEW"] = {}
    config["NEW"]["A"] = 1

    assert str(config) == "&L\n  X = 1\n/\n&NEW\n  A = 1\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_addition_after_comment_run(parser):
    """Test that a new key lands after a comment run inside a namelist group.

    The comment on the assignment's own line is held by the ``nml_line`` the index is
    measured past, so the position one past it is the second line of that comment.
    """
    config = parser.parse("&L\n  X = 1  ! what X is\n  ! ...continued\n/\n")

    config["L"]["NEW"] = 2

    assert str(config) == "&L\n  X = 1  ! what X is\n  ! ...continued\n  NEW = 2\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # Between namelist groups the comment is not a node of its own: the grammar lumps every
    # line there into one "random_text" blob, so a new group still goes above it. That is a
    # known limitation of this format, not of the placement rule.
    config = parser.parse("&L\n  X = 1\n/\n! a top-level comment\n")

    config["NEW"] = {"A": 1}

    assert str(config) == "&L\n  X = 1\n/\n&NEW\n  A = 1\n/\n! a top-level comment\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_addition_values(parser):
    """Test the notation a namelist uses for values written into new keys."""
    config = parser.parse("&L\n  X = 1\n/\n")

    config["L"]["B"] = True
    config["L"]["S"] = "text"
    config["L"]["N"] = None

    # Fortran logicals, and a quoted string since the grammar has no bare-word value type.
    assert str(config) == '&L\n  X = 1\n  B = .true.\n  S = "text"\n  N =\n/\n'
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_addition_case_insensitive(parser):
    """Test that a new key is written as given while the dict entry is upper-cased."""
    config = parser.parse("&L\n  Var = 4\n/\n")

    config["L"]["newvar"] = 1

    assert "NEWVAR" in dict(config["L"])
    assert config["L"]["NewVar"] == 1
    assert str(config) == "&L\n  Var = 4\n  newvar = 1\n/\n"

    # Assigning again finds the existing key rather than adding a second entry.
    config["L"]["NEWVAR"] = 2
    assert str(config) == "&L\n  Var = 4\n  newvar = 2\n/\n"


def test_fortran_nml_addition_errors(parser):
    """Test additions a Fortran namelist cannot express."""
    config = parser.parse("&L\n  X = 1\n/\n")

    # Assignments exist only inside a namelist group.
    with pytest.raises(TypeError):
        config["TOPLEVEL"] = 1

    # Namelists do not nest.
    with pytest.raises(TypeError):
        config["L"]["NESTED"] = {"A": 1}

    # The grammar has no value type for a path.
    with pytest.raises(TypeError):
        config["L"]["P"] = Path("/x/y")

    assert str(config) == "&L\n  X = 1\n/\n"


def test_fortran_nml_delete_namelist(parser, fortran_nml_file):
    """Test that a whole namelist group can be removed and the file written out again.

    A namelist and the text around it alternate in the grammar, so removing one leaves two
    runs of text side by side. They are rejoined on deletion, which keeps the tree in a
    shape the grammar derives while preserving every comment.
    """
    config = parser.parse(fortran_nml_file)

    del config["LIST_B"]

    output = str(config)
    assert "LIST_B" not in output
    # The comment and the free text that surrounded the deleted group are still there.
    assert "! Yet another comment" in output
    assert "This is some random text" in output
    assert dict(parser.parse(output)) == dict(config)

    # The remaining groups are untouched, and the rest can be deleted too. A namelist file
    # needs at least one group, so once they are all gone there is no text left to write.
    assert list(config) == ["LIST_A", "LIST_C"]
    del config["LIST_A"]
    del config["LIST_C"]
    assert str(config) == ""


@pytest.mark.parametrize(
    ("donor", "added"),
    [("  x =1", "  N =5"), ("  x= 1", "  N= 5"), ("  x = 1", "  N = 5"), ("  x=1", "  N=5")],
)
def test_fortran_nml_addition_copies_spacing_on_the_right_side(parser, donor, added) -> None:
    """Test that a new entry puts the whitespace on the side of "=" its neighbour did.

    A single run of whitespace is ambiguous -- ``x =1`` and ``x= 1`` both have one -- so the
    style has to record which side of the assignment it fell on.
    """
    config = parser.parse(f"&L\n{donor}\n/\n")

    config["L"]["N"] = 5

    assert str(config) == f"&L\n{donor}\n{added}\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("fortran_nml"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was generated for. These rows are the
    statement of what this grammar can express and how it spells it.
    """
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected)
