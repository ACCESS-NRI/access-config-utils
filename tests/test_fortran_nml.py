# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from conftest import assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedEOF

from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import compile_grammar
from access.config.grammar_values import UnsupportedEntryError


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

    # "true" and "false" are logicals, not stray words -- see the logical spelling test.
    with pytest.raises(UnexpectedEOF):
        parser.parse("&LIST\nTEST = maybe\n/")

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


def test_fortran_nml_delete_entry_removes_its_line(parser):
    """Test that deleting an entry alone on its line removes the line.

    A namelist line is an assignment followed by an optional comment, so the line rule
    cannot derive what is left once the assignment goes. The wrapper is removed with it,
    taking the trailing comment, which described the line that has gone. A line holding two
    assignments keeps the one that remains.
    """
    config = parser.parse("&L\n  X = 1  ! note\n  Y = 2\n/\n")
    del config["L"]["X"]
    assert str(config) == "&L\n  Y = 2\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)

    shared = parser.parse("&L\n X = 1, Y = 2\n/\n")
    del shared["L"]["X"]
    assert dict(shared["L"]) == {"Y": 2}
    assert dict(parser.parse(str(shared))) == dict(shared)


def test_fortran_nml_delete_removes_every_entry_that_wrote_the_key(parser):
    """Test that a key written more than once is deleted from every line that writes it.

    Only the last assignment survives in the configuration, so removing only that one left
    the key behind in the file, to reappear the next time it was read.
    """
    config = parser.parse("&L\n  x = 1\n  x = 2\n/\n")

    del config["L"]["X"]

    assert str(config) == "&L\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("&L\nA%B = 1\n/\n", {"A": {"B": 1}}),
        ("&L\nA%B%C = 2\n/\n", {"A": {"B": {"C": 2}}}),
        ("&L\nA%B = 1, 2, 3\n/\n", {"A": {"B": [1, 2, 3]}}),
        ("&L\nA%B =\n/\n", {"A": {"B": None}}),
        ("&L\n  filename%met = 'x'\n/\n", {"FILENAME": {"MET": "x"}}),
    ],
)
def test_fortran_nml_derived_type(parser, text, expected) -> None:
    """Test that a derived-type component assignment reads as a nested block.

    ``a%b = 1`` is the block ``a`` holding ``b = 1``, so it goes through the same key_block
    machinery as a namelist group and nests to any depth for free.
    """
    config = parser.parse(text)

    assert dict(config)["L"] == expected
    assert str(config) == text


def test_fortran_nml_derived_type_merges_across_lines(parser):
    """Test that the components of one derived type merge, wherever their lines are.

    A derived type spells one component per line, and the lines need not be adjacent. Each
    is a separate entry in the tree under the same key, so without merging only the last
    would survive -- losing the others with no error.
    """
    text = "&L\n  filename%met = 'x'\n  z = 1\n  filename%out = 'y'\n/\n"
    config = parser.parse(text)

    assert dict(config["L"]) == {"FILENAME": {"MET": "x", "OUT": "y"}, "Z": 1}
    assert str(config) == text


def test_fortran_nml_repeated_group_merges(parser):
    """Test that a namelist group named twice in one file merges rather than overwriting."""
    text = "&L\n V = 1\n/\n&L\n W = 2\n/\n"
    config = parser.parse(text)

    assert dict(config["L"]) == {"V": 1, "W": 2}
    assert str(config) == text


def test_fortran_nml_merged_block_edit_and_delete(parser):
    """Test that a merged block is editable in place and deletes every line it occupies.

    The merged block is a node the parse tree does not contain, but its children are the
    real ones, so writing a value rewrites just that line and leaves the rest -- including a
    comment on a sibling line -- exactly as it was.
    """
    text = "&L\n  filename%met = 'x.nc'   ! met file\n  z = 1\n  filename%out = 'y.nc'\n/\n"
    config = parser.parse(text)

    config["L"]["FILENAME"]["OUT"] = "z.nc"
    assert str(config) == text.replace("'y.nc'", "'z.nc'")
    assert dict(parser.parse(str(config))) == dict(config)

    del config["L"]["FILENAME"]
    assert str(config) == "&L\n  z = 1\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_derived_type_refuses_additions(parser):
    """Test that a key cannot be added to a block with no body to add it to.

    A derived type has no delimited body: it is written one component per line, so there is
    nowhere inside it for an entry to go, and a merged block's root is not in the parse tree
    at all. Splicing into either would change nothing that gets written out, so both refuse
    rather than silently dropping the addition. The namelist group around them still
    accepts new keys.
    """
    config = parser.parse("&L\n A%B = 1\n A%C = 2\n/\n")

    with pytest.raises(UnsupportedEntryError):
        config["L"]["A"]["D"] = 3

    # A single, unmerged derived type has no body to add to either.
    single = parser.parse("&L\n A%B = 1\n/\n")
    with pytest.raises(UnsupportedEntryError):
        single["L"]["A"]["C"] = 2

    # The namelist group is a real block, so it still takes new keys.
    single["L"]["NEW"] = 2
    assert str(single) == "&L\n A%B = 1\n NEW = 2\n/\n"
    assert dict(parser.parse(str(single))) == dict(single)


def test_fortran_nml_delete_derived_type_component(parser):
    """Test that a derived type survives losing one component, and goes when it loses all.

    A derived type is written a component per line, so removing the last one leaves nothing
    to write: the key goes from the file, and has to go from the configuration with it.
    """
    config = parser.parse("&L\n  grid%nx = 96\n  grid%ny = 72\n  z = 1\n/\n")

    del config["L"]["GRID"]["NX"]
    assert str(config) == "&L\n  grid%ny = 72\n  z = 1\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)

    del config["L"]["GRID"]["NY"]
    assert str(config) == "&L\n  z = 1\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # An empty namelist group is still a namelist group, so that one stays.
    del config["L"]["Z"]
    assert str(config) == "&L\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_key_used_as_both_value_and_block(parser):
    """Test that a key assigned a value and also used as a block is refused.

    Merging the two would build a block around the value node and drop the value with no
    error, and deleting the key afterwards left the configuration disagreeing with the file.
    """
    with pytest.raises(ValueError):
        parser.parse("&L\n  a = 1\n  a%b = 2\n/\n")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (".true.", True),
        (".TRUE.", True),
        (".false.", False),
        (".t.", True),
        (".f.", False),
        (".t", True),
        (".f", False),
        ("true", True),
        ("FALSE", False),
        ("T", True),
        ("f", False),
    ],
)
def test_fortran_nml_logical_spellings(parser, text, expected) -> None:
    """Test the spellings Fortran accepts for a logical.

    A logical is an optional dot, T or F, and an optional rest of the word, in either case.
    All of these mean the same thing, and all appear in real namelists -- the WW3 ones use a
    bare "T".
    """
    source = f"&L\nA = {text}\n/\n"
    config = parser.parse(source)

    assert config["L"]["A"] is expected
    assert str(config) == source


@pytest.mark.parametrize(
    ("text", "rewritten"),
    [(".true.", ".false."), (".TRUE.", ".FALSE."), ("T", "F"), ("t", "f")],
)
def test_fortran_nml_logical_keeps_notation(parser, text, rewritten) -> None:
    """Test that rewriting a logical keeps the notation and case it was written in.

    Fortran spells a logical several ways, so a file written with bare logicals should not
    acquire dotted ones just because a value changed.
    """
    config = parser.parse(f"&L\nA = {text}\n/\n")

    config["L"]["A"] = False

    assert str(config) == f"&L\nA = {rewritten}\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("'plain'", "plain"),
        ("''", ""),
        ("'it''s'", "it's"),
        ('"a ""quoted"" word"', 'a "quoted" word'),
        ("'a!b'", "a!b"),
        ("'a/b'", "a/b"),
    ],
)
def test_fortran_nml_string_quoting(parser, text, expected) -> None:
    """Test that a quote inside a string is written twice rather than escaped.

    That is Fortran's rule, and it differs from the backslash escaping the shared "string"
    rule assumes. A "!" or a "/" inside a string is data, not a comment or a terminator.
    """
    source = f"&L\nA = {text}\n/\n"
    config = parser.parse(source)

    assert config["L"]["A"] == expected
    assert str(config) == source


def test_fortran_nml_string_holding_both_quotes(parser):
    """Test that a string containing both quote characters can be written.

    Formats that wrap a value verbatim have to refuse this, since neither quote can delimit
    it. Fortran doubles the quote instead, so nothing is out of reach.
    """
    config = parser.parse("&L\nA = 'x'\n/\n")

    config["L"]["A"] = """he said "no" and it's fine"""

    assert str(config) == "&L\nA = 'he said \"no\" and it''s fine'\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize("text", ["(1,2)", "(1.0, 2)", "(3.0, 4.0)", "(-1, +2)"])
def test_fortran_nml_complex_components(parser, text) -> None:
    """Test that either component of a complex may be written as a plain integer."""
    source = f"&L\nA = {text}\n/\n"
    config = parser.parse(source)

    assert isinstance(config["L"]["A"], complex)
    assert str(config) == source


@pytest.mark.parametrize(
    ("written", "value", "rewritten"),
    [
        (".true.", False, ".false."),
        (".TRUE.", False, ".FALSE."),
        ("T", False, "F"),
        ("true", False, "false"),
        ("TRUE", False, "FALSE"),
        (".t.", False, ".f."),
        (".t", False, ".f"),
        ("f", True, "t"),
    ],
)
def test_fortran_nml_logical_keeps_its_spelling(parser, written, value, rewritten) -> None:
    """Test that rewriting a logical keeps the dots, the case and the length of the word."""
    config = parser.parse(f"&L\n  b = {written}\n/\n")

    config["L"]["B"] = value

    assert str(config) == f"&L\n  b = {rewritten}\n/\n"
    assert parser.parse(str(config))["L"]["B"] is value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2*2.0", [2.0, 2.0]),
        ("3*", [None, None, None]),
        ("4*.true.", [True, True, True, True]),
        ("3*'ab'", ["ab", "ab", "ab"]),
        ("3*1, 2, 3", [1, 1, 1, 2, 3]),
        ("3*1, 2*2, 3", [1, 1, 1, 2, 2, 3]),
        ("1, 3*, 3, 4", [1, None, None, None, 3, 4]),
        ("2*(1.0, 2.0)", [1 + 2j, 1 + 2j]),
    ],
)
def test_fortran_nml_repeat_count(parser, text, expected) -> None:
    """Test that "n*v" reads as v repeated n times, and a bare "n*" as n nulls.

    A repeat is written where one value goes but means several, so even "x = 3*1" is a list.
    CABLE's parameter files are written almost entirely this way.
    """
    source = f"&L\nX = {text}\n/\n"
    config = parser.parse(source)

    assert config["L"]["X"] == expected
    assert str(config) == source


def test_fortran_nml_repeat_reflow_on_write(parser):
    """Test that writing one element of a repeat splits the run and keeps the notation.

    The six values of "6*9.0" are backed by a single node, so writing one of them has to
    rewrite the run. Neighbouring equal values are regrouped, which keeps the line in the
    notation it was written in instead of expanding it, and the rest of the line is left
    alone -- including the comment.
    """
    config = parser.parse("&L\n  X = 6*9.0, 4.0   ! keep me\n/\n")

    config["L"]["X"][2] = 8.0

    assert config["L"]["X"] == [9.0, 9.0, 8.0, 9.0, 9.0, 9.0, 4.0]
    assert str(config) == "&L\n  X = 2*9.0, 8.0, 3*9.0, 4.0   ! keep me\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(
    ("values", "written"),
    [([5, 5, 5], "3*5"), ([1, 2, 3], "1, 2, 3"), ([1, 1, 2], "2*1, 2")],
)
def test_fortran_nml_repeat_whole_list_replacement(parser, values, written) -> None:
    """Test replacing a whole list that was written as a repeat.

    Holding a different number of values than before makes it a different kind of entry --
    one value is a scalar assignment and several are a list one -- so the entry rule itself
    changes with the values.
    """
    config = parser.parse("&L\n  X = 3*1\n/\n")

    config["L"]["X"] = values

    assert str(config) == f"&L\n  X = {written}\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_repeat_type_errors(parser):
    """Test that a repeat is as strict about types as any other value.

    A repeat of nulls names no type at all, so there is no notation to write a value in.
    """
    nulls = parser.parse("&L\n  X = 3*\n/\n")
    with pytest.raises(TypeError):
        nulls["L"]["X"][0] = 1

    integers = parser.parse("&L\n  X = 3*1\n/\n")
    with pytest.raises(TypeError):
        integers["L"]["X"][0] = "a"

    assert str(integers) == "&L\n  X = 3*1\n/\n"


def test_fortran_nml_slice_assignment_from_an_iterator(parser):
    """Test that a slice can be assigned from a one-shot iterable.

    The values were taken twice: once to write the file and once to store, and the second
    pass got nothing, leaving the list shorter than the nodes behind it.
    """
    config = parser.parse("&L\n  x = 1.0, 2.0, 3.0, 4.0\n/\n")
    values = config["L"]["X"]

    values[1:3] = (value for value in [8.0, 9.0])

    assert list(values) == [1.0, 8.0, 9.0, 4.0]
    assert str(config) == "&L\n  x = 1.0, 8.0, 9.0, 4.0\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_list_handle_stays_valid_after_replacement(parser):
    """Test that a list handed out earlier still writes to the file after replacement.

    Replacing a list that was written as a repeat replaces the nodes behind it, and the list
    object the caller was holding went on writing through the ones no longer in the tree.
    """
    config = parser.parse("&L\n  x = 3*1\n/\n")
    values = config["L"]["X"]

    config["L"]["X"] = [1, 2, 3]
    values[0] = 99

    assert str(config) == "&L\n  x = 99, 2, 3\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


def test_fortran_nml_null_repeat_does_not_block_writing(parser):
    """Test that a list holding a null repeat can still have its real values written.

    A run of nulls names no type, so it has no notation to write a value in -- but that is
    only true of the run itself, and demanding one for every run made the whole list
    unwritable.
    """
    config = parser.parse("&L\n  x = 2*, 1\n/\n")
    assert config["L"]["X"] == [None, None, 1]

    config["L"]["X"][2] = 5
    assert str(config) == "&L\n  x = 2*, 5\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)

    # Writing *into* a null is still refused: there is no value there to change.
    with pytest.raises(TypeError):
        config["L"]["X"][0] = 5


def test_fortran_nml_repeat_rewrite_keeps_what_is_between_the_values(parser):
    """Test that splitting a repeat leaves the rest of the entry alone.

    Only the node being split is replaced, so a comment or a line break between the entry's
    other values survives.
    """
    config = parser.parse("&L\n  x = 2*9.0, ! keep me\n      1.0\n/\n")

    config["L"]["X"][0] = 8.0

    assert str(config) == "&L\n  x = 8.0, 9.0, ! keep me\n      1.0\n/\n"
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("fortran_nml"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was generated for. These rows are the
    statement of what this grammar can express and how it spells it.
    """
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected)
