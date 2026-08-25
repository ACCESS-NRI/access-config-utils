# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MOM6 parameter file format.

Two halves. The first has a test per rule of the grammar, or per assumption its comments
state -- why one whitespace run belongs to the ``eq`` rule, why a block cannot nest.
Grouping them by rule is what makes an untested construct visible as an empty spot rather
than an absence nobody notices.

The second half covers the round trip: reading a file, changing it, and writing it back with
that change and nothing else. Almost every test asserts byte-exact output as well as the
value read, because for this package the text is half the result.

What is worth naming about this format: keys are case-sensitive, a block is delimited
``NAME% … %NAME``, a value is never valueless, and the booleans are ``True`` and ``False``.
"""

from pathlib import Path

import pytest
from conftest import assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedInput

from access.config.grammar_compiled import compile_grammar
from access.config.mom6_input import MOM6InputParser
from access.config.tree_edits import _parse_values
from access.config.tree_navigation import is_value_node


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser."""
    return MOM6InputParser()


# =============================================================================
# The grammar
# =============================================================================


class TestAssignmentForms:
    """``key_value`` and ``key_list``, the two ways a key is given a value."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("TEST = 'a'", "a"), ("TEST='a'", "a"), ('TEST = "a"', "a"), ("TEST = 1", 1)],
    )
    def test_a_single_value_is_a_scalar(self, parser, text, expected) -> None:
        """Test ``key_value``: one value reads as the value, not as a list of one."""
        config = parser.parse(text)

        assert config["TEST"] == expected
        assert str(config) == text

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("TEST = 'a','b'", ["a", "b"]),
            ("TEST = 1,2", [1, 2]),
            ("TEST = 1, 2, 3", [1, 2, 3]),
            ("TEST = 1.0, 2.0", [1.0, 2.0]),
        ],
    )
    def test_two_or_more_values_are_a_list(self, parser, text, expected) -> None:
        """Test ``key_list``, whose values are parted by commas.

        The rule requires two: one value is a ``key_value``, which is what makes a list and
        a scalar different kinds of entry rather than the same one of different lengths.
        """
        config = parser.parse(text)

        assert config["TEST"] == expected
        assert str(config) == text

    def test_an_assignment_may_end_the_file_without_a_newline(self, parser) -> None:
        """Test that ``line_end`` is satisfied by the end of the file."""
        assert dict(parser.parse("TEST = 1")) == {"TEST": 1}


class TestBlocks:
    """``key_block``: ``NAME%`` opens a block and ``%NAME`` closes it."""

    def test_a_block_holds_its_own_entries(self, parser) -> None:
        """Test that the entries between the delimiters belong to the block."""
        source = "TEST%\na=1\nb = 2, 3\n%TEST\n"
        config = parser.parse(source)

        assert dict(config["TEST"]) == {"a": 1, "b": [2, 3]}
        assert str(config) == source

    def test_a_block_may_be_empty(self, parser) -> None:
        """Test that a block with no entries is still a block."""
        source = "B%\n%B\n"
        config = parser.parse(source)

        assert dict(config) == {"B": {}}
        assert str(config) == source

    def test_a_block_may_hold_a_comment(self, parser) -> None:
        """Test that ``block`` admits an ``empty_line``, which is what carries a comment."""
        source = "TEST%\na=1\n ! Comment\n%TEST\n"
        config = parser.parse(source)

        assert dict(config["TEST"]) == {"a": 1}
        assert str(config) == source

    def test_entries_may_follow_a_block(self, parser) -> None:
        """Test that a block does not swallow what comes after it."""
        source = "B%\nx=1\n%B\nY = 2\n"
        config = parser.parse(source)

        assert dict(config) == {"B": {"x": 1}, "Y": 2}
        assert str(config) == source

    def test_a_block_does_not_nest(self, parser) -> None:
        """Test that ``block`` admits only assignments and blank lines, never a block.

        MOM6 has one level of grouping, so a block inside a block is refused rather than
        read as something else.
        """
        with pytest.raises(UnexpectedInput):
            parser.parse("B%\nC%\nx=1\n%C\n%B\n")

    def test_the_closing_name_is_not_checked_against_the_opening_one(self, parser) -> None:
        """Test the leniency in ``key_block``: the closing ``key`` is unconstrained.

        The rule is ``key "%" line_end block "%" key line_end`` -- two independent ``key``
        nodes -- so ``B% … %C`` parses and the block takes the opening name. MOM6 itself
        requires the two to match, so this is leniency rather than a decision; nothing is
        lost, since the text round-trips and no entry goes missing.
        """
        source = "B%\nx=1\n%C\n"
        config = parser.parse(source)

        assert dict(config) == {"B": {"x": 1}}
        assert str(config) == source


class TestValueTypes:
    """The four alternatives of ``?value``: bool, integer, float, string."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("True", True),
            ("False", False),
            ("1", 1),
            ("-5", -5),
            ("1800.0", 1800.0),
            ("1e-10", 1e-10),
            ("1.5e-3", 0.0015),
            ("'a'", "a"),
            ('"a"', "a"),
            ("'a b'", "a b"),
            ("'2*3'", "2*3"),
        ],
    )
    def test_each_value_type_reads_and_writes_back_unchanged(self, parser, text, expected) -> None:
        """Test one case per value type.

        The booleans are ``True`` and ``False`` only, and a string may hold a ``*`` -- both
        spellings a format with logicals and repeat counts would read as something else.
        """
        source = f"A = {text}\n"
        config = parser.parse(source)

        assert config["A"] == expected
        assert str(config) == source


class TestKeyCase:
    """``case_sensitive_keys`` is True, so two spellings of a key are two entries."""

    def test_two_spellings_of_a_key_are_two_entries(self, parser) -> None:
        """Test that case is significant, so ``A`` and ``a`` do not collide."""
        source = "A = 1\na = 2\n"
        config = parser.parse(source)

        assert dict(config) == {"A": 1, "a": 2}
        assert str(config) == source

    def test_a_key_is_matched_exactly(self, parser) -> None:
        """Test that looking a key up in the wrong case does not find it."""
        config = parser.parse("Var2 = 2\n")

        assert config["Var2"] == 2
        assert "VAR2" not in config


class TestCommentsAndBlankLines:
    """``fortran_comment``, ``empty_line`` and the ``line_end`` that carries them."""

    @pytest.mark.parametrize(
        "source",
        [
            "A = 1 ! after an assignment\n",
            "A = 1! with no space before it\n",
            "! on a line of its own\nA = 1\n",
            "\n\nA = 1\n\n",
            "  ! an indented comment\nA = 1\n",
            "! just a comment\n",
            "",
        ],
    )
    def test_a_comment_or_a_blank_line_survives_untouched(self, parser, source) -> None:
        """Test every position a comment may take, and a file that is nothing but comment.

        A blank line and a comment-only line are the same rule, told apart by whether a
        comment node hangs below it, so a file holding neither entry nor text is valid.
        """
        assert str(parser.parse(source)) == source


class TestRejected:
    """Input this format refuses, rather than reading as something else."""

    @pytest.mark.parametrize(
        "source",
        [
            " TEST = a",
            "TEST = a",
            "TEST : a",
            "TEST = true",
            "TEST = false",
            "%TEST\na=1\n%TEST",
            "TEST%\na=1\nTEST%",
            "BLOCK%\n TEST ='a'",
            "#override TEST = 1\n",
            "TEST = 1 /* a C-style comment */\n",
        ],
    )
    def test_malformed_input_raises(self, parser, source) -> None:
        """Test that each of these raises rather than parsing.

        Notable cases: an assignment cannot be indented, a bare word is not a value, and the
        booleans are capitalised. The last two are documented limitations rather than
        malformed files -- the ``#override`` directive is not implemented, and the C-style
        comments the module docstring reports in CESM's files are not part of the format.
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(source)


# =============================================================================
# The round trip
# =============================================================================


class TestRoundTripFidelity:
    """Reading a file and writing it back out unchanged, byte for byte."""

    @pytest.mark.parametrize(
        ("name", "source"),
        [
            ("crlf line endings", "A = 1\r\n"),
            ("no trailing newline", "A = 1"),
            ("tab whitespace", "A\t=\t1\n"),
            ("trailing blank line", "A = 1\n\n"),
            ("whitespace-only line", "A = 1\n  \n"),
            ("empty file", ""),
        ],
    )
    def test_the_shape_of_the_file_survives(self, parser, name, source) -> None:
        """Test the line and whitespace forms the terminals admit.

        ``NEWLINE`` is ``CR? LF``, so a file written on Windows has to come back with its
        carriage returns. ``parse`` appends a newline to simplify the grammar, so a file
        without one is the case where that has to be undone on the way out.
        """
        assert str(parser.parse(source)) == source


class TestAssignmentSpacing:
    """The ``eq`` rule: whitespace is written back on the side of "=" it came from."""

    @pytest.mark.parametrize("source", ["A = 1\n", "A= 1\n", "A =1\n", "A=1\n", "A  =  1\n"])
    def test_whitespace_stays_on_its_own_side(self, parser, source) -> None:
        """Test that the position of a single whitespace run survives.

        The "=" is an anonymous literal, so Lark filters it out of the parse tree. With a
        bare "ws*" on each side, the tree records that a run exists but not which side of
        the "=" it fell on, and the reconstructor guesses -- writing "A= 1" back as "A =1".
        """
        assert str(parser.parse(source)) == source


class TestEditingAValue:
    """Writing a value back into the file, changing that and nothing else."""

    def test_a_scalar_rewrites_only_its_own_token(self, parser) -> None:
        """Test that the rest of the line, including a comment, is left alone."""
        config = parser.parse("A = 1  ! note\nB = 2\n")

        config["A"] = 7

        assert str(config) == "A = 7  ! note\nB = 2\n"

    def test_a_value_inside_a_block_reaches_its_own_line(self, parser) -> None:
        """Test that editing through a block writes to the entry, not to the block."""
        source = "B%\nX = 1\nY = 2\n%B\n"
        config = parser.parse(source)

        config["B"]["Y"] = 9

        assert str(config) == "B%\nX = 1\nY = 9\n%B\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_whole_list_may_be_replaced(self, parser) -> None:
        """Test that a list is rewritten through the nodes backing its elements."""
        config = parser.parse("A = 1,2,3\n")

        config["A"] = [4, 5, 6]

        assert str(config) == "A = 4,5,6\n"
        assert dict(parser.parse(str(config))) == dict(config)


class TestAddingAnEntry:
    """Writing a key the file does not have, in the format's own syntax."""

    def test_a_new_key_is_never_indented(self, parser) -> None:
        """Test that the indentation of a blank line is not copied.

        The file ends with a whitespace-only line, which is the trap: copying that line's
        indentation would produce a line the grammar cannot parse.
        """
        config = parser.parse("BOOL = True\n  \n")

        config["NEW"] = 1

        # The new entry follows the last assignment, so it precedes the trailing blank line.
        assert str(config) == "BOOL = True\nNEW = 1\n  \n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_block_copies_the_spacing_of_the_block_above(self, parser) -> None:
        """Test what carries over to a block created by an assignment.

        MOM6 cannot indent an assignment at all, so the indentation a new block inherits
        is always empty. The spacing around "=" is what carries over instead, and this file
        writes it tight.
        """
        config = parser.parse("BLK%\nX=1\n%BLK\n")

        config["NEWBLK"] = {"Y": 2}

        assert str(config) == "BLK%\nX=1\n%BLK\nNEWBLK%\nY=2\n%NEWBLK\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_lands_after_a_parameter_s_documentation(self, parser) -> None:
        """Test that a new key does not land inside a comment run.

        This is the format where it matters most: a MOM6 parameter file documents nearly
        every parameter in a comment that starts on the parameter's own line and carries on
        below it. That trailing comment belongs to the entry, so the position one past the
        entry is the second line of its documentation.
        """
        config = parser.parse("DT = 900.0  ! [s]\n            ! The dynamics time step.\n")

        config["NEW"] = 1

        assert str(config) == "DT = 900.0  ! [s]\n            ! The dynamics time step.\nNEW = 1\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_in_a_block_stops_at_the_closing_delimiter(self, parser) -> None:
        """Test that the comment run is stepped over but ``%BLK`` is not."""
        config = parser.parse("BLK%\nA = 1  ! what A is\n ! ...continued\n%BLK\n")

        config["BLK"]["B"] = 2

        assert str(config) == "BLK%\nA = 1  ! what A is\n ! ...continued\nB = 2\n%BLK\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_what_a_mom6_input_file_cannot_express(self, parser) -> None:
        """Test the additions the format has no syntax for."""
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

    def test_a_value_snippet_parses_for_a_line_ending_rule(self, parser) -> None:
        """Test re-parsing values for a format whose entry rule ends with its line.

        Rewriting an entry's values hands the text back to Lark with an entry rule as the
        start symbol. This format's ``key_value`` ends with ``line_end``, so a snippet
        without a trailing newline does not parse in it, while a grammar whose entry rule
        stops short of the newline will not accept one *with*. Nothing reaches this
        through the public API here, so it is pinned rather than left as a trap.
        """
        nodes = _parse_values(compile_grammar(parser.grammar).lark, ["1", "2"])

        assert [node.data for node in nodes if is_value_node(node)] == ["integer", "integer"]


class TestDeleting:
    """Removing an entry, and the line or lines it occupied."""

    def test_deleting_everything_keeps_the_text_that_is_not_an_entry(self, parser) -> None:
        """Test that comments and blank lines are genuinely part of the file.

        A configuration with no entries left is not the same thing as an empty file.
        """
        config = parser.parse("A = 1\n\n  ! keep me\n\nB = 2\n")

        for key in list(config):
            del config[key]

        assert str(config) == "\n  ! keep me\n\n"
        assert dict(parser.parse(str(config))) == {}

        # Adding a key back does not resurrect anything: the text was never hidden.
        config["C"] = 3
        assert str(config) == "C = 3\n\n  ! keep me\n\n"

    def test_every_entry_that_wrote_a_key_is_removed(self, parser) -> None:
        """Test that a key assigned twice does not reappear the next time it is read.

        Only the last assignment survives in the configuration, so removing just the entry
        it came from left the earlier ones in the file.
        """
        config = parser.parse("A = 1\nA = 2\nB = 3\n")
        assert dict(config) == {"A": 2, "B": 3}

        del config["A"]

        assert str(config) == "B = 3\n"
        assert dict(parser.parse(str(config))) == dict(config)


# =============================================================================
# End to end
# =============================================================================

SAMPLE = """
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

SAMPLE_VALUES = {
    "BOOL": True,
    "FLOAT1": 1800.0,
    "FLOAT2": 1e-10,
    "VAR1": "test",
    "Var2": 2,
    "BLOCK": {"BVAR": 4, "BLIST": [1, 2, 3]},
    "List": ["a", "b", "c"],
}

SAMPLE_EDITED = """
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
BNEW = 2
%BLOCK ! Yet another comment

List = 'a','b','c'
NEW1 = 1
NEWLIST = 1,2
NEWBOOL = True
NEWBLOCK%
X = 1
%NEWBLOCK
"""


def test_a_whole_file_parses_edits_and_round_trips(parser):
    """Test that the rules above compose, over a file carrying all of them at once.

    Each focused test pins one construct in isolation. This one is the anchor: comments,
    blank lines, a block with a comment inside it, and keys differing only in case, all in
    one file that is read, changed, added to and written back with one byte-exact
    expectation.

    What it covers that no focused test can: where each new entry lands relative to the
    block that precedes it and the trailing text, and that editing one entry leaves every
    other line alone.
    """
    config = parser.parse(SAMPLE)
    assert dict(config) == SAMPLE_VALUES

    config["VAR1"] = "replaced"
    config["FLOAT1"] = 900.0
    config["BLOCK"]["BVAR"] = 32

    config["BLOCK"]["BNEW"] = 2
    config["NEW1"] = 1
    config["NEWLIST"] = [1, 2]
    # This format spells its booleans ``True`` and ``False``.
    config["NEWBOOL"] = True
    config["NEWBLOCK"] = {"X": 1}

    assert str(config) == SAMPLE_EDITED
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("mom6_input"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was generated for. These rows are the
    statement of what this grammar can express and how it spells it.
    """
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected)
