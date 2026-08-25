# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NUOPC config format.

Two halves. The first has a test per rule of the grammar, or per assumption its comments
state. The second covers the round trip: reading a file,
changing it, and writing it back with that change and nothing else. Almost every test
asserts byte-exact output as well as the value read, because the text is half the result.

What makes this format the richest of the three is that it has **two syntaxes aliased to the
same entry categories**. At the resource-file level an entry is ``key: value`` and a list
is parted by spaces; in a ``::`` table it is ``key = value`` and a list is parted by colons.
Both alias to ``key_value`` and ``key_list``, so the only thing telling them apart when
writing out is the ``eq`` rule -- which is what ``TestTheTwoSyntaxes`` exists for.
"""

from pathlib import Path

import pytest
from conftest import assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedInput

from access.config.grammar_compiled import compile_grammar
from access.config.nuopc_config import NUOPCParser


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser."""
    return NUOPCParser()


# =============================================================================
# The grammar
# =============================================================================


class TestResourceFileEntries:
    """``rfile_key_value`` and ``rfile_key_list``: the top level, where ``:`` assigns."""

    @pytest.mark.parametrize("source", ["TEST: a", "TEST:a", " TEST:a", "  TEST:   a"])
    def test_a_colon_assigns_one_value(self, parser, source) -> None:
        """Test ``rfile_key_value``, and the leading ``ws*`` that lets a key be indented."""
        config = parser.parse(source)

        assert config["TEST"] == "a"
        assert str(config) == source

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("TEST: a b", ["a", "b"]),
            ("TEST: a b c", ["a", "b", "c"]),
            ("TEST:1 2", [1, 2]),
            ("TEST: 1   2", [1, 2]),
        ],
    )
    def test_a_list_is_parted_by_spaces(self, parser, source, expected) -> None:
        """Test ``rfile_key_list``: at this level whitespace alone separates the values.

        This is the opposite of a table entry, where the separator is a colon and whitespace
        around it is not admitted at all.
        """
        config = parser.parse(source)

        assert config["TEST"] == expected
        assert str(config) == source

    @pytest.mark.parametrize(
        "source",
        ["TEST1: a\n TEST2: b", "TEST1: a \nTEST2: b", "TEST1: a b \n TEST2: c"],
    )
    def test_one_entry_does_not_swallow_the_next(self, parser, source) -> None:
        """Test that a space-parted list stops at the end of its line.

        A list runs on until ``line_end``, so the risk is the key on the next line being
        taken for one more value.
        """
        config = parser.parse(source)

        assert len(config) == 2
        assert str(config) == source


class TestTables:
    """``rfile_key_block``: ``NAME::`` opens a table and a bare ``::`` closes it."""

    def test_a_table_holds_its_own_entries(self, parser) -> None:
        """Test that the entries between the delimiters belong to the table."""
        source = "TEST::\na=1\nb = x\n::\n"
        config = parser.parse(source)

        assert dict(config["TEST"]) == {"a": 1, "b": "x"}
        assert str(config) == source

    def test_the_closing_delimiter_carries_no_name(self, parser) -> None:
        """Test that a table closes with ``::`` alone, not by repeating the name."""
        source = "TEST::\na=1\n::\n"

        assert dict(parser.parse(source)) == {"TEST": {"a": 1}}
        assert str(parser.parse(source)) == source

    def test_a_table_may_be_empty(self, parser) -> None:
        """Test that ``block: block_line*`` admits no lines at all."""
        source = "T::\n::\n"
        config = parser.parse(source)

        assert dict(config) == {"T": {}}
        assert str(config) == source

    def test_a_table_may_hold_a_blank_line_or_a_comment(self, parser) -> None:
        """Test the ``empty_line`` alternative of ``block_line``."""
        source = "T::\n a=1\n\n # a comment\n::\n"
        config = parser.parse(source)

        assert dict(config["T"]) == {"a": 1}
        assert str(config) == source

    def test_entries_may_follow_a_table(self, parser) -> None:
        """Test that a table does not swallow what comes after it."""
        source = "T::\n a=1\n::\nB: 2\n"
        config = parser.parse(source)

        assert dict(config) == {"T": {"a": 1}, "B": 2}
        assert str(config) == source

    def test_a_table_does_not_nest(self, parser) -> None:
        """Test that ``block_line`` admits only assignments and blank lines.

        This format has one level of grouping, so a table inside a table is refused rather
        than read as something else. Written flush left, so that the only thing refusing it
        is the missing alternative in ``block_line`` -- an indented inner table would also
        fail on its closing delimiter, which is a different rule.
        """
        with pytest.raises(UnexpectedInput):
            parser.parse("T::\nU::\na=1\n::\n::\n")

    def test_only_the_opening_delimiter_may_be_indented(self, parser) -> None:
        """Test the asymmetry in ``rfile_key_block``.

        The rule is ``ws* key "::" line_end block "::" line_end``: there is a ``ws*`` before
        the key that opens the table and none before the ``::`` that closes it. Every table
        in a real file closes flush left, so the grammar has never had to admit otherwise.
        """
        source = " T::\n a = 1\n::\n"
        config = parser.parse(source)

        assert dict(config["T"]) == {"a": 1}
        assert str(config) == source

        with pytest.raises(UnexpectedInput):
            parser.parse("T::\n a = 1\n ::\n")


class TestTableEntries:
    """``block_key_value`` and ``block_key_list``: inside a table, where ``=`` assigns."""

    @pytest.mark.parametrize("source", ["T::\na=1\n::\n", "T::\n a = 1\n::\n", "T::\n a=x\n::\n"])
    def test_an_equals_assigns_one_value(self, parser, source) -> None:
        """Test ``block_key_value``, the table-level scalar assignment."""
        config = parser.parse(source)

        assert len(config["T"]) == 1
        assert str(config) == source

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1:2", [1, 2]),
            ("1:2:3", [1, 2, 3]),
            ("1:10:19:26:30:33:35", [1, 10, 19, 26, 30, 33, 35]),
            ("1.5:2.5", [1.5, 2.5]),
            ("x:y", ["x", "y"]),
        ],
    )
    def test_a_list_is_parted_by_colons(self, parser, text, expected) -> None:
        """Test ``block_key_list``: at this level a colon separates the values.

        The rule is ``value (":" value)+`` with no ``ws`` in it, so the colons are tight
        against the values. ``ocn2glc_levels`` in a real ``nuopc.runconfig`` is written this
        way.
        """
        source = f"T::\n a = {text}\n::\n"
        config = parser.parse(source)

        assert config["T"]["a"] == expected
        assert str(config) == source


class TestTheTwoSyntaxes:
    """The point of the ``eq`` rule: two levels aliased to one pair of category names."""

    @pytest.mark.parametrize(
        "source",
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
    def test_an_entry_is_written_in_the_syntax_of_its_own_level(self, parser, source) -> None:
        """Test that a table entry is not written back in the resource-file syntax.

        The two levels alias their entries to the same rule names, so what tells them apart
        when writing out is the ``=`` a table entry keeps in the parse tree. Without it a
        table entry can be written as ``a: 1`` -- a file that does not parse -- whenever the
        two levels admit the same sequence of children.
        """
        assert str(parser.parse(source)) == source


class TestValueTypes:
    """The six ``?value`` alternatives: logical, integer, float, double, identifier and
    path."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (".true.", True),
            (".false.", False),
            ("-1", -1),
            ("1.0", 1.0),
            ("-1.000000D-08", -1.0e-8),
            ("2.000000D+01", 20.0),
            ("cesm", "cesm"),
            ("off", "off"),
        ],
    )
    def test_each_value_type_reads_and_writes_back_unchanged(self, parser, text, expected) -> None:
        """Test one case per value type, in the spellings a ``nuopc.runconfig`` uses.

        The ``D`` exponent is the notable one: it is the double-precision spelling the ESMF
        documentation this grammar was written from uses throughout.
        """
        source = f"A: {text}\n"
        config = parser.parse(source)

        assert config["A"] == expected
        assert str(config) == source

    @pytest.mark.parametrize("text", [".log", "x.nc", "/a/b/c.nc", "dir/file.nc"])
    def test_a_path_is_read_as_a_path(self, parser, text) -> None:
        """Test the ``path`` value type, which no other format in this package has."""
        source = f"A: {text}\n"
        config = parser.parse(source)

        assert config["A"] == Path(text)
        assert str(config) == source


class TestKeyCase:
    """``case_sensitive_keys`` is True, so two spellings of a key are two entries."""

    def test_two_spellings_of_a_key_are_two_entries(self, parser) -> None:
        """Test that case is significant, so ``A`` and ``a`` do not collide."""
        source = "A: 1\na: 2\n"
        config = parser.parse(source)

        assert dict(config) == {"A": 1, "a": 2}
        assert str(config) == source

    def test_a_key_is_matched_exactly(self, parser) -> None:
        """Test that looking a key up in the wrong case does not find it."""
        config = parser.parse("Verbosity: off\n")

        assert config["Verbosity"] == "off"
        assert "VERBOSITY" not in config


class TestCommentsAndBlankLines:
    """``comment``, ``empty_line`` and the ``line_end`` that carries them."""

    @pytest.mark.parametrize(
        "source",
        [
            "# on a line of its own\nA: 1\n",
            "A: 1 # after an entry\n",
            "A: 1# with no space before it\n",
            "  # an indented comment\nA: 1\n",
            "\n\nA: 1\n\n",
            "T:: # after the table header\n a = 1\n::\n",
            "T::\n a = 1 # after a table entry\n::\n",
            "# just a comment\n",
            "",
        ],
    )
    def test_a_comment_or_a_blank_line_survives_untouched(self, parser, source) -> None:
        """Test every position a comment may take, at both levels of the format.

        This format's comment character is ``#``.
        """
        assert str(parser.parse(source)) == source


class TestRejected:
    """Input this format refuses, rather than reading as something else."""

    @pytest.mark.parametrize(
        "source",
        [
            "TEST::\n cime_model - cesm",
            "TEST:\n cime_model = cesm\n::",
            "TEST::\n cime_model = cesm",
            "TEST::\n cime_model = cesm ATM_model = datm",
            "T::\n U::\n a=1\n ::\n::\n",
            "A: 1 ! an exclamation mark, which is not this format's comment\n",
            "T::\n a = 1 : 2\n::\n",
        ],
    )
    def test_malformed_input_raises(self, parser, source) -> None:
        """Test that each of these raises rather than parsing.

        Notable cases: a table must be opened with ``::`` and closed, a table entry holds
        one assignment per line, the comment character is ``#`` not ``!``, and a colon
        list admits no whitespace around its separators.
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
            ("crlf line endings", "A: 1\r\n"),
            ("no trailing newline", "A: 1"),
            ("tab after the colon", "A:\t1\n"),
            ("trailing blank line", "A: 1\n\n"),
            ("empty file", ""),
            ("table with no trailing newline", "T::\n a = 1\n::"),
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

    @pytest.mark.parametrize(
        "source",
        [
            "BLK::\n Verbosity = off\n::\n",
            "BLK::\n Verbosity= off\n::\n",
            "BLK::\n Verbosity =off\n::\n",
            "BLK::\n Verbosity=off\n::\n",
            "BLK::\n Verbosity  =  off\n::\n",
        ],
    )
    def test_whitespace_stays_on_its_own_side(self, parser, source) -> None:
        """Test that the position of a single whitespace run survives.

        The "=" is an anonymous literal, so Lark filters it out of the parse tree. With a
        bare "ws*" on each side, the tree records that a run exists but not which side of
        the "=" it fell on. Here that also let the reconstructor confuse this rule with the
        resource-file rule aliased to the same name, and write "Verbosity: off" instead.
        """
        assert str(parser.parse(source)) == source


class TestEditingAValue:
    """Writing a value back into the file, changing that and nothing else."""

    def test_a_top_level_scalar_rewrites_only_its_own_token(self, parser) -> None:
        """Test that the rest of the line, including a comment, is left alone."""
        config = parser.parse("A: 1  # note\nB: 2\n")

        config["A"] = 7

        assert str(config) == "A: 7  # note\nB: 2\n"

    def test_a_value_inside_a_table_reaches_its_own_line(self, parser) -> None:
        """Test that editing through a table writes to the entry, not to the table."""
        source = "T::\n X = 1\n Y = 2\n::\n"
        config = parser.parse(source)

        config["T"]["Y"] = 9

        assert str(config) == "T::\n X = 1\n Y = 9\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_each_level_keeps_its_own_list_separator(self, parser) -> None:
        """Test that rewriting a list does not move it to the other level's syntax."""
        config = parser.parse("TOP: a b\nT::\n a = 1:2\n::\n")

        config["TOP"] = ["c", "d"]
        config["T"]["a"] = [3, 4]

        assert str(config) == "TOP: c d\nT::\n a = 3:4\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)


class TestAddingAnEntry:
    """Writing a key the file does not have, in the syntax of the level it goes into."""

    def test_a_new_two_element_list_in_a_table_uses_the_table_separator(self, parser) -> None:
        """Test the length at which the two levels hold the same children.

        A list of two is the shape a top-level list can also take, so it is the case where
        the entry could be written with the top-level separator instead.
        """
        config = parser.parse("TEST::\n a = 1:2:3\n::\n")

        config["TEST"]["b"] = [4, 5]

        assert str(config) == "TEST::\n a = 1:2:3\n b = 4:5\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_lands_between_comments_rather_than_inside_one(self, parser) -> None:
        """Test that a comment run is stepped over, as far as the first blank line.

        A comment on the entry's own line is held by that entry, so the position one past
        it is the second line of the comment.
        """
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

    def test_a_new_table_body_is_indented(self, parser) -> None:
        """Test the ``entry_templates`` this parser declares.

        Nothing in the grammar requires a table entry to be indented, so the text derived
        from the grammar alone would put the first entry of a new table in column 0. Every
        table in the ESMF documentation and in a real file indents its body.
        """
        config = parser.parse("A: 1\n")

        config["NEW"] = {"X": "y"}

        assert str(config) == "A: 1\nNEW::\n X = y\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_table_copies_a_sibling_indent(self, parser) -> None:
        """Test that the declared template gives way once there is something to imitate."""
        config = parser.parse("OLD::\n  P = 1\n::\n")

        config["NEW"] = {"X": 2}

        assert str(config) == "OLD::\n  P = 1\n::\nNEW::\n  X = 2\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_an_added_key_matches_its_own_table(self, parser) -> None:
        """Test that a key added to one table copies that table, not another."""
        config = parser.parse("A::\n  P = 1\n::\nB::\n q = 2\n::\n")

        config["B"]["r"] = 3

        assert str(config) == "A::\n  P = 1\n::\nB::\n q = 2\n r = 3\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_multi_component_path_can_be_written(self, parser) -> None:
        """Test that a path with a directory in it is written as one value."""
        config = parser.parse("A: 1\n")

        config["P"] = Path("dir/file.nc")

        assert str(config) == "A: 1\nP: dir/file.nc\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_what_a_nuopc_config_cannot_express(self, parser) -> None:
        """Test the additions the format has no syntax for.

        This grammar has no quoted-string value type, so a string that is not a bare word
        cannot be written at all. A bare word that would read back as something else -- a
        number, or a path -- is refused for the same reason.
        """
        config = parser.parse("A: 1\n")

        for value in ("a string", "1", "a-b"):
            with pytest.raises(TypeError):
                config["Z"] = value

        assert str(config) == "A: 1\n"


class TestDeleting:
    """Removing an entry, and the line or lines it occupied."""

    def test_every_entry_that_wrote_a_key_is_removed(self, parser) -> None:
        """Test that a key assigned twice does not reappear the next time it is read.

        Only the last assignment survives in the configuration, so removing just the entry
        it came from left the earlier ones in the file.
        """
        config = parser.parse("A: 1\nA: 2\nB: 3\n")
        assert dict(config) == {"A": 2, "B": 3}

        del config["A"]

        assert str(config) == "B: 3\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_table_entry_can_go_on_its_own(self, parser) -> None:
        """Test that deleting from a table leaves the table and its other entries."""
        config = parser.parse("T::\n a = 1\n b = 2\n::\n")

        del config["T"]["a"]

        assert str(config) == "T::\n b = 2\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)


# =============================================================================
# End to end
# =============================================================================

SAMPLE = """DRIVER_attributes:: # Comment 1

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

SAMPLE_VALUES = {
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

SAMPLE_EDITED = """DRIVER_attributes:: # Comment 1

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
  NEWBLIST = 1:2:3

::
NEWTOPLIST: a b
NEWPATH: x.nc
NEWTABLE::
  X = y
::

"""


def test_a_whole_file_parses_edits_and_round_trips(parser):
    """Test that the rules above compose, over a file carrying all of them at once.

    Each focused test pins one construct in isolation. This one is the anchor: two tables,
    top-level entries between them, comments in five different positions and blank lines
    inside a table, all read, changed, added to and written back with one byte-exact
    expectation.

    What it covers that no focused test can: that an edit at one level leaves the other
    alone, that a new entry takes the syntax of the container it goes into rather than the
    file's first, and that a table created from nothing copies the indentation of the tables
    already there.
    """
    config = parser.parse(SAMPLE)
    assert dict(config) == SAMPLE_VALUES

    config["TEST"] = "Off"
    config["COMPONENTS"] = ["atm", "um"]
    config["ALLCOMP_attributes"]["ATM_model"] = "um"
    config["ALLCOMP_attributes"]["ocn2glc_levels"] = [1, 10, 19, 26, 30, 33, 36]

    config["ALLCOMP_attributes"]["NEWBLIST"] = [1, 2, 3]
    config["NEWTOPLIST"] = ["a", "b"]
    config["NEWPATH"] = Path("x.nc")
    config["NEWTABLE"] = {"X": "y"}

    assert str(config) == SAMPLE_EDITED
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("nuopc_config"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was generated for. These rows are the
    statement of what this grammar can express and how it spells it.
    """
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected)
