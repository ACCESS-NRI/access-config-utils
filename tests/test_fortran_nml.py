# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Fortran namelist dialect.

Two halves. The first has a test per rule of the grammar, or per assumption its comments
state -- why a terminal carries a lookahead, why a rule is named not aliased, why one
whitespace run belongs to the ``eq`` rule. Grouping them by rule is what makes an untested
construct visible as an empty spot rather than an absence nobody notices.

The second half covers the round trip: reading a file, changing it, and writing it back with
that change and nothing else. Almost every test asserts byte-exact output as well
as the value read, because for this package the text is half the result.

One end-to-end test closes the file. The focused tests prove each rule; that one proves they
compose, over a file carrying comments, blank lines, free text and several groups at once.
"""

from pathlib import Path

import pytest
from conftest import assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedInput

from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import compile_grammar
from access.config.grammar_values import UnsupportedEntryError


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser."""
    return FortranNMLParser()


# =============================================================================
# The grammar
# =============================================================================


class TestGroupDelimiters:
    """``nml_start`` and ``nml_end``: how a namelist group opens and closes."""

    @pytest.mark.parametrize(
        "source",
        [
            "&LIST TEST='a'/",
            "&LIST\nTEST='a'/",
            "&LIST\nTEST='a'\n/",
            "&LIST\nTEST='a'\n&end",
            "&LIST\nTEST='a'\n&End",
            "&LIST\nTEST='a'\n&END",
        ],
    )
    def test_a_group_closes_with_a_slash_or_an_end_token(self, parser, source) -> None:
        """Test both terminators, in either case, with or without a line of their own.

        ``nml_end`` is ``ws* ("/"|/&end/i) line_end``: the alternation is what admits the
        older ``&end`` spelling, and the ``i`` flag is why its case does not matter.
        """
        assert dict(parser.parse(source)) == {"LIST": {"TEST": "a"}}

    def test_the_group_name_may_be_followed_by_the_body_on_one_line(self, parser) -> None:
        """Test the optional ``line_end`` between the group name and its block."""
        source = "&LIST TEST='a' Z=1/"
        config = parser.parse(source)

        assert dict(config["LIST"]) == {"TEST": "a", "Z": 1}
        assert str(config) == source

    def test_the_ampersand_may_be_indented(self, parser) -> None:
        """Test the ``ws*`` of ``nml_start``: a group need not begin in the first column."""
        source = "  &LIST\nTEST='a'\n  /\n"
        config = parser.parse(source)

        assert dict(config) == {"LIST": {"TEST": "a"}}
        assert str(config) == source

    def test_several_groups_in_one_file(self, parser) -> None:
        """Test the repetition in ``start``, and that the groups keep the order written."""
        source = "&A\nX = 1\n/\n&B\nY = 2\n/\n"
        config = parser.parse(source)

        assert list(config) == ["A", "B"]
        assert str(config) == source


class TestAssignmentForms:
    """``key_value``, ``key_list``, ``key_null`` and the wrapped forms of the first two."""

    def test_a_single_value_is_a_scalar(self, parser) -> None:
        """Test ``key_value``: one value reads as the value, not as a list of one."""
        config = parser.parse("&L\nA = 1\n/\n")

        assert config["L"]["A"] == 1

    @pytest.mark.parametrize(
        "text",
        ["'a', 'b'", "'a','b'", "'a', \n'b'"],
    )
    def test_two_or_more_values_are_a_list(self, parser, text) -> None:
        """Test ``key_list``, whose second value may follow a separator or a ``line_break``.

        The rule requires two: one value is a ``key_value``, which is what makes a list and
        a scalar different kinds of entry rather than the same one of different lengths.
        """
        source = f"&L\nA = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["A"] == ["a", "b"]
        assert str(config) == source

    def test_a_key_with_nothing_after_the_equals_is_valueless(self, parser) -> None:
        """Test ``key_null``: the entry exists and its value is ``None``."""
        source = " &LIST\nTEST = \n /\n"
        config = parser.parse(source)

        assert dict(config) == {"LIST": {"TEST": None}}
        assert str(config) == source

    def test_a_value_may_start_on_the_line_after_its_equals(self, parser) -> None:
        """Test ``wrapped_value`` and ``wrapped_list``.

        MOM6's test inputs write their longer lists that way. A key with nothing after it at
        all is still a valueless one, so this does not swallow the assignment that follows.
        """
        source = "&L\n  files =\n      'a',\n      'b',\n  empty =\n  z = 1\n/\n"
        config = parser.parse(source)

        assert dict(config["L"]) == {"FILES": ["a", "b"], "EMPTY": None, "Z": 1}
        assert str(config) == source

    def test_a_list_continues_across_lines(self, parser) -> None:
        """Test ``line_break``: a separator and a newline carry a list to the next line."""
        source = "&LIST\nTEST='a', \n'b', \n'c'\n/\n"
        config = parser.parse(source)

        assert config["LIST"]["TEST"] == ["a", "b", "c"]
        assert str(config) == source


class TestSeveralPerLine:
    """``nml_line``: several assignments on a line, and the optional trailing separator."""

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("VAR1=1, VAR2=2", {"VAR1": 1, "VAR2": 2}),
            ("VAR1=1, VAR2=2, VAR3=3", {"VAR1": 1, "VAR2": 2, "VAR3": 3}),
            ("VAR1=1, 2, VAR2=3", {"VAR1": [1, 2], "VAR2": 3}),
            ("VAR1=1, VAR2=2, 3", {"VAR1": 1, "VAR2": [2, 3]}),
            ("VAR1=1, 2, VAR2=3, 4", {"VAR1": [1, 2], "VAR2": [3, 4]}),
        ],
    )
    def test_where_one_assignment_ends_and_the_next_begins(self, parser, body, expected) -> None:
        """Test that a comma continues a list until a key and an "=" say otherwise.

        The separator between two assignments and the one between two values of a list is
        the same character, so which it is depends entirely on what follows it.
        """
        source = f"&LIST\n{body}\n/\n"
        config = parser.parse(source)

        assert dict(config["LIST"]) == expected
        assert str(config) == source

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("TEST='a',", {"TEST": "a"}),
            ("TEST='a' , ", {"TEST": "a"}),
            ("TEST='a','b',", {"TEST": ["a", "b"]}),
            ("VAR1=1, VAR2=2,", {"VAR1": 1, "VAR2": 2}),
        ],
    )
    def test_a_line_may_end_with_a_separator(self, parser, body, expected) -> None:
        """Test the ``(ws* separator)?`` of ``nml_line``, which real files write freely."""
        source = f"&LIST\n{body}\n/\n"
        config = parser.parse(source)

        assert dict(config["LIST"]) == expected
        assert str(config) == source


class TestDerivedTypes:
    """``key_derived`` and ``dtype_body``: ``a%b = 1`` is block ``a`` holding ``b = 1``."""

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
    def test_a_component_assignment_reads_as_a_nested_block(self, parser, text, expected) -> None:
        """Test that the key_block machinery is reused, so nesting comes for free."""
        config = parser.parse(text)

        assert dict(config)["L"] == expected
        assert str(config) == text

    def test_the_components_of_one_type_merge_wherever_their_lines_are(self, parser) -> None:
        """Test that components merge into one key rather than the last one winning.

        A derived type spells one component per line, and the lines need not be adjacent.
        Each is a separate entry in the tree under the same key, so without merging only the
        last would survive -- losing the others with no error. This is why ``dtype_body`` is
        held apart from the ``block`` rule of a group, whose repetitions are read as
        separate records instead.
        """
        text = "&L\n  filename%met = 'x'\n  z = 1\n  filename%out = 'y'\n/\n"
        config = parser.parse(text)

        assert dict(config["L"]) == {"FILENAME": {"MET": "x", "OUT": "y"}, "Z": 1}
        assert str(config) == text

    def test_a_body_holds_exactly_one_assignment(self, parser) -> None:
        """Test that a derived type does not swallow the lines that follow it.

        ``dtype_body`` holds one assignment rather than reusing the namelist's own ``block``
        rule, which would be greedy: it takes the following lines and mis-nests them under
        the component, and costs an order of magnitude in parse time.
        """
        text = "&L\n  a%b = 1\n  c = 2\n  d = 3\n/\n"
        config = parser.parse(text)

        assert dict(config["L"]) == {"A": {"B": 1}, "C": 2, "D": 3}
        assert str(config) == text

    def test_a_key_used_as_both_value_and_block_is_refused(self, parser) -> None:
        """Test that a key assigned a value and also used as a block raises.

        Merging the two would build a block around the value node and drop the value with no
        error, and deleting the key afterwards left the configuration disagreeing with the
        file.
        """
        with pytest.raises(ValueError):
            parser.parse("&L\n  a = 1\n  a%b = 2\n/\n")


class TestArrayQualifiers:
    """``index`` and ``INDEX_SPEC``: the ``(3)`` of ``v(3) = 1``."""

    @pytest.mark.parametrize(
        ("lines", "expected"),
        [
            (["v(1) = 1", "v(2) = 2"], [1, 2]),
            (["v(1:4) = 1,2,3,4"], [1, 2, 3, 4]),
            (["v(1:7:2) = 1, 3, 5, 7"], [1, None, 3, None, 5, None, 7]),
            (["v(0:3) = 1, 2, 3, 4"], [1, 2, 3, 4]),
            (["v(-2:2) = 1, 2, 3, 4, 5"], [1, 2, 3, 4, 5]),
            (["v(2) = 2", "v(4) = 4", "v(3) = 3", "v(1) = 1"], [1, 2, 3, 4]),
            (["v(1:) = 1,2,3,4"], [1, 2, 3, 4]),
            (["v(:4) = 1,2,3,4"], [1, 2, 3, 4]),
            (["v(1) = 1", "v(4) = 4"], [1, None, None, 4]),
            (["v(1) = 1"], [1]),
            (["v(2) = 2", "v = 1"], [1, 2]),
            (["v = 1", "v(2) = 2"], [1, 2]),
        ],
    )
    def test_indexed_assignments_make_up_one_array(self, parser, lines, expected) -> None:
        """Test that the entries are gathered rather than overwriting each other.

        An index writes part of an array, wherever the lines are and in whatever order. A
        position no entry writes -- skipped by a stride, or never named -- reads as null.
        An unqualified assignment joins the same array, from the first position Fortran
        numbers.
        """
        source = "&L\n" + "".join(f"{line}\n" for line in lines) + "/\n"
        config = parser.parse(source)

        assert config["L"]["V"] == expected
        assert str(config) == source

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("n_aero(1) = 1, 2, 3", [1, 2, 3]),
            ("n_aero(3) = 1, 2, 3", [1, 2, 3]),
            ("n_aero(2) = 2*7", [7, 7]),
        ],
    )
    def test_a_single_position_says_where_the_values_start(self, parser, line, expected) -> None:
        """Test that ``v(3) = 1, 2, 3`` fills three positions from the third.

        A qualifier says where an entry's values begin, not how many it may have. Reading it
        as one position and three values raised before the configuration was even built.
        """
        source = f"&L\n  {line}\n/\n"
        config = parser.parse(source)

        assert config["L"]["N_AERO"] == expected
        assert str(config) == source

    @pytest.mark.parametrize(
        ("lines", "expected"),
        [
            (["x = 1, 2", "x = 3"], [3, 2]),
            (["x = 1, 2, 3", "x = 9"], [9, 2, 3]),
            (["x = 1, 2, 3", "x = 8, 9"], [8, 9, 3]),
            (["x = 3*7", "x = 1"], [1, 7, 7]),
            # A later entry at least as long covers every position, so it simply wins.
            (["x = 1", "x = 2"], 2),
            (["x = 1, 2", "x = 3, 4, 5"], [3, 4, 5]),
        ],
    )
    def test_an_unqualified_reassignment_overlays_what_is_there(self, parser, lines, expected) -> None:
        """Test that a second assignment writes from position one and leaves the rest.

        Fortran does not reset the variable: ``x = 1, 2`` then ``x = 3`` is ``[3, 2]``.
        Taking the later entry whole reported the scalar ``3`` and dropped the second
        element with no error -- the overlay the qualified ``x(1) = 3`` already got right.
        """
        source = "&L\n" + "".join(f"  {line}\n" for line in lines) + "/\n"
        config = parser.parse(source)

        assert config["L"]["X"] == expected
        assert str(config) == source

    def test_a_valueless_assignment_may_carry_one(self, parser) -> None:
        """Test that a qualified null sets one position of the array the others fill in."""
        source = "&L\n  v(2) = 3\n  v(1) =\n/\n"
        config = parser.parse(source)

        assert config["L"]["V"] == [None, 3]
        assert str(config) == source

    @pytest.mark.parametrize(
        ("lines", "keys"),
        [
            (["v(1,1) = 1", "v(3,3) = 2"], ["V(1,1)", "V(3,3)"]),
            (["a(1)%b = 1", "a(2)%b = 2"], ["A(1)", "A(2)"]),
        ],
    )
    def test_a_shape_this_does_not_model_stays_in_the_key(self, parser, lines, keys) -> None:
        """Test that a multidimensional index and an indexed derived type keep their text.

        Both address a shape this flat model cannot build. Leaving the qualifier in the key
        is what stops two of them reading as one key, which would drop all but the last
        with no error.
        """
        source = "&L\n" + "".join(f"{line}\n" for line in lines) + "/\n"
        config = parser.parse(source)

        assert list(config["L"]) == keys
        assert str(config) == source

    @pytest.mark.parametrize("qualifier", ["(1:2:3:4)", "(+)", "(1:3:0)"])
    def test_a_qualifier_naming_no_positions_stays_in_the_key(self, parser, qualifier) -> None:
        """Test that too many bounds, a non-numeric bound and a zero stride are not guessed.

        Each describes no set of positions, so there is nothing to gather the values into.
        """
        source = f"&L\n  v{qualifier} = 1\n/\n"
        config = parser.parse(source)

        assert list(config["L"]) == [f"V{qualifier}"]
        assert str(config) == source

    @pytest.mark.parametrize(("qualifier", "key"), [("( )", "V( )"), ("(1,2)", "V(1,2)")])
    def test_a_valueless_entry_keeps_a_qualifier_it_cannot_place(self, parser, qualifier, key) -> None:
        """Test that the rule holds whether or not the assignment has a value."""
        source = f"&L\n  v{qualifier} =\n/\n"
        config = parser.parse(source)

        assert dict(config["L"]) == {key: None}
        assert str(config) == source


class TestRepeatCounts:
    """``repeat`` and ``REPEAT_COUNT``: ``n*v`` is *v* repeated *n* times."""

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
    def test_a_repeat_expands_and_a_bare_count_is_null(self, parser, text, expected) -> None:
        """Test that "n*v" reads as v repeated n times, and a bare "n*" as n nulls.

        A repeat is written where one value goes but means several, so even "x = 3*1" is a
        list. CABLE's parameter files are written almost entirely this way.
        """
        source = f"&L\nX = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["X"] == expected
        assert str(config) == source


class TestValueTypes:
    """The eight rules of ``?scalar``, one case each."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1", 1),
            ("-2", -2),
            ("1.5", 1.5),
            ("1e-10", 1e-10),
            ("1d-10", 1e-10),
            ("(3.0, 4.0)", 3 + 4j),
            ("(1d0, 2d0)", 1 + 2j),
            ("'text'", "text"),
            (".true.", True),
            ("continue", "continue"),
        ],
    )
    def test_each_value_type_reads_and_writes_back_unchanged(self, parser, text, expected) -> None:
        """Test integer, float, double, complex, double_complex, string, logical, bare word.

        ``double`` and ``double_complex`` are separate rules from ``float`` and ``complex``
        because Fortran's ``d`` exponent is not something Python spells, so the notation has
        to survive as text even though the value is an ordinary float.
        """
        source = f"&L\nA = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["A"] == expected
        assert str(config) == source

    @pytest.mark.parametrize("text", ["(1,2)", "(1.0, 2)", "(3.0, 4.0)", "(-1, +2)"])
    def test_either_component_of_a_complex_may_be_an_integer(self, parser, text) -> None:
        """Test that a complex is not required to write both components as floats."""
        source = f"&L\nA = {text}\n/\n"
        config = parser.parse(source)

        assert isinstance(config["L"]["A"], complex)
        assert str(config) == source

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("inf", float("inf")), ("-inf", float("-inf")), ("Infinity", float("inf"))],
    )
    def test_an_infinity_is_a_float(self, parser, text, expected) -> None:
        """Test the ``IEEE`` terminal, read as a number rather than the word spelling it.

        Without it the bare-word rule claims them and ``x = inf`` is the string ``"inf"``.
        """
        source = f"&L\n  x = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["X"] == expected
        assert str(config) == source

    def test_a_not_a_number_is_a_float(self, parser) -> None:
        """Test NaN, which needs a test of its own: it does not compare equal to itself."""
        source = "&L\n  x = nan\n/\n"
        config = parser.parse(source)

        assert isinstance(config["L"]["X"], float)
        assert config["L"]["X"] != config["L"]["X"]
        assert str(config) == source

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
    def test_a_quote_inside_a_string_is_written_twice(self, parser, text, expected) -> None:
        """Test Fortran's quoting rule, which is not the backslash escaping ``string`` uses.

        A "!" or a "/" inside a string is data, not a comment or a group terminator.
        """
        source = f"&L\nA = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["A"] == expected
        assert str(config) == source


class TestLogicals:
    """``FTRUE`` and ``FFALSE``: an optional dot, T or F, and an optional rest of word."""

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
    def test_every_spelling_reads_as_the_same_value(self, parser, text, expected) -> None:
        """Test the spellings Fortran accepts, all of which appear in real namelists.

        The WW3 namelists use a bare "T".
        """
        source = f"&L\nA = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["A"] is expected
        assert str(config) == source

    @pytest.mark.parametrize("name", ["f_icedir", "t_ref", "true_north", "false_x", "tx"])
    def test_a_logical_does_not_claim_the_start_of_a_key(self, parser, name) -> None:
        """Test the ``(?![A-Za-z0-9_])`` lookahead on both logical terminals.

        A trailing comma leaves the parser expecting another value where the next line
        actually starts a new assignment. Without the lookahead the key ``f_icedir`` is a
        second value ``.false.`` followed by the leftover key ``_icedir`` -- so the file
        parses, with a value invented and a key renamed, and no error anywhere.

        The word has to be *the key of the assignment that follows*: written as a value it
        is the last thing on its line, nothing can follow it, and the parse is unambiguous
        with or without the lookahead.
        """
        source = f"&L\nA = 'x',\n{name} = 1\n/\n"
        config = parser.parse(source)

        assert dict(config["L"]) == {"A": "x", name.upper(): 1}
        assert str(config) == source


class TestBareWords:
    """``BAREWORD``: an unquoted word as a string, tried last of all the value types."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("continue", "continue"),
            ("none", "none"),
            ("ocean_only", "ocean_only"),
            ("'quoted'", "quoted"),
            (".true.", True),
            ("T", True),
            ("12", 12),
        ],
    )
    def test_a_word_is_a_string_unless_another_type_claims_it(self, parser, text, expected) -> None:
        """Test the ``-1`` priority: the logical spellings and numbers still win.

        Real files write bare words: CICE has "runtype=continue" and the CESM data models
        have "import_data_fields = none".
        """
        source = f"&L\n  X = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["X"] == expected
        assert str(config) == source

    def test_a_word_cannot_be_read_as_the_key_that_follows_it(self, parser) -> None:
        """Test the ``(?![ \\t]*=)`` lookahead.

        A continuation comma leaves the parser expecting another value where the next line
        starts with a key, so without it "a = 'x',\\n b = 1" reads "b" as a second value of
        "a".
        """
        source = "&L\n  a = 'x',\n  b = 1\n/\n"
        config = parser.parse(source)

        assert dict(config["L"]) == {"A": "x", "B": 1}
        assert str(config) == source


class TestCommentsAndBlankLines:
    """``fortran_comment``, ``empty_line`` and the ``line_end`` that carries them."""

    @pytest.mark.parametrize(
        "source",
        [
            "&L ! after the group name\n  A = 1\n/\n",
            "&L\n  A = 1  ! after an assignment\n/\n",
            "&L\n  ! on a line of its own\n  A = 1\n/\n",
            "&L\n\n  A = 1\n\n/\n",
            "&L\n  A = 1,  ! inside a list\n      2\n/\n",
        ],
    )
    def test_a_comment_or_a_blank_line_survives_untouched(self, parser, source) -> None:
        """Test every position a comment may take, and the blank line that carries none.

        A blank line and a comment-only line are the same rule, told apart by whether a
        comment node hangs below it.
        """
        assert str(parser.parse(source)) == source


class TestFreeTextBetweenGroups:
    """``random_text`` and ``ANYTHING``: text a namelist file may carry around groups."""

    @pytest.mark.parametrize(
        "source",
        [
            "header\n&L\nA=1\n/\n",
            "&L\nA=1\n/\ntrailer\n",
            "\n&A\n  x = 1\n/\n\n! a comment\n some free text \n\n&B\n  y = 2\n/\n",
        ],
    )
    def test_text_before_between_and_after_groups_is_kept(self, parser, source) -> None:
        """Test that free text round-trips and does not become part of any group."""
        config = parser.parse(source)

        assert all(isinstance(value, dict) for value in config.values())
        assert str(config) == source

    def test_a_line_starting_with_an_ampersand_is_never_free_text(self, parser) -> None:
        """Test the ``(?![ \\t]*&)`` lookahead in ``ANYTHING``.

        Without it a group whose body did not fit the grammar was quietly read as more free
        text, and the file round-tripped byte for byte with the whole group missing from the
        configuration -- no error, no sign anything was wrong.
        """
        with pytest.raises(UnexpectedInput):
            parser.parse("&A\n  x = 1\n&B\n  y = 2\n/\n")


class TestRejected:
    """Input this dialect refuses, rather than reading as something else."""

    @pytest.mark.parametrize(
        "source",
        [
            "&LIST\nTEST : 'a'\n/",
            "&LIST\nTEST = @maybe\n/",
            "&LIST\nTEST = 'unterminated\n/",
            "%TEST\na=1\n%TEST",
            "TEST%\na=1\nTEST%",
            "&TEST\na=1 2\n/",
            "&TEST\na=1 \n2\n/",
            "BLOCK\n TEST ='a'/",
            "&BLOCK\n VAR1=1\n&e",
            "&L\n  nonsense @@@\n/\n",
            "&L\n  x = 1,,3\n/\n",
        ],
    )
    def test_malformed_input_raises(self, parser, source) -> None:
        """Test that each of these raises rather than parsing.

        Which ``UnexpectedInput`` Lark raises depends on how far it got, and that is not a
        promise the parser makes. Notable cases: values cannot be separated by whitespace
        alone, a group has to be closed rather than ended by the next "&", and a list cannot
        yet hold an elided element written as ``1,,3``.
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
            ("crlf line endings", "&L\r\nA = 1\r\n/\r\n"),
            ("no trailing newline", "&L\nA = 1\n/"),
            ("tab indentation", "&L\n\tA\t=\t1\n/\n"),
            ("no leading newline", "&L\nA = 1\n/\n"),
            ("empty group", "&L\n/\n"),
        ],
    )
    def test_the_shape_of_the_file_survives(self, parser, name, source) -> None:
        """Test the line and whitespace forms the terminals admit.

        ``NEWLINE`` is ``CR? LF``, so a file written on Windows has to come back with its
        carriage returns. ``parse`` appends a newline to simplify the grammar, so a file
        without one is the case where that has to be undone on the way out. ``ws`` is
        ``WS_INLINE``, which is tabs as well as spaces.
        """
        assert str(parser.parse(source)) == source

    def test_a_key_may_hold_digits_and_underscores(self, parser) -> None:
        """Test that ``key`` is a ``CNAME``, and that its case is kept in the file."""
        source = "&L\nA_1b = 1\n/\n"
        config = parser.parse(source)

        assert dict(config["L"]) == {"A_1B": 1}
        assert str(config) == source


class TestAssignmentSpacing:
    """The ``eq`` rule: whitespace is written back on the side of "=" it came from."""

    @pytest.mark.parametrize(
        "source",
        [
            "&L\nA = 1\n/\n",
            "&L\nA= 1\n/\n",
            "&L\nA =1\n/\n",
            "&L\nA=1\n/\n",
            "&L\nA  =  1\n/\n",
            "&L\nA=  0 , 1 ,\n/\n",
        ],
    )
    def test_whitespace_stays_on_its_own_side(self, parser, source) -> None:
        """Test that the position of a single whitespace run survives.

        The "=" is an anonymous literal, so Lark filters it out of the parse tree. With a
        bare "ws*" on each side, the tree records that a run exists but not which side of
        the "=" it fell on, and the reconstructor guesses -- writing "A= 1" as "A =1". The
        "eq" rule binds the leading whitespace to the "=" so the position survives.
        """
        assert str(parser.parse(source)) == source


class TestEditingAValue:
    """Writing a value back into the file, changing that and nothing else."""

    def test_a_scalar_rewrites_only_its_own_token(self, parser) -> None:
        """Test that the rest of the line, including a comment, is left alone."""
        config = parser.parse("&L\n  X = 1  ! note\n  Y = 2\n/\n")

        config["L"]["X"] = 7

        assert str(config) == "&L\n  X = 7  ! note\n  Y = 2\n/\n"

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
    def test_a_logical_keeps_its_dots_case_and_length(self, parser, written, value, rewritten) -> None:
        """Test that a file written with bare logicals does not acquire dotted ones."""
        config = parser.parse(f"&L\n  b = {written}\n/\n")

        config["L"]["B"] = value

        assert str(config) == f"&L\n  b = {rewritten}\n/\n"
        assert parser.parse(str(config))["L"]["B"] is value

    def test_a_string_holding_both_quote_characters(self, parser) -> None:
        """Test that doubling the quote puts every string within reach.

        A format that wraps a value verbatim has to refuse this, since neither quote can
        delimit it.
        """
        config = parser.parse("&L\nA = 'x'\n/\n")

        config["L"]["A"] = """he said "no" and it's fine"""

        assert str(config) == "&L\nA = 'he said \"no\" and it''s fine'\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize("value", ["true", "t", "F", "false", "café"])
    def test_a_bare_word_refuses_a_value_that_would_change_type(self, parser, value) -> None:
        """Test that ``true`` written into an unquoted value is a logical, not a string.

        An unquoted value has nothing marking where it starts and ends. A value the grammar
        cannot write unquoted at all, such as a non-ASCII word, is refused for the same
        reason.
        """
        config = parser.parse("&L\n  w = continue\n/\n")

        with pytest.raises(TypeError):
            config["L"]["W"] = value

        assert str(config) == "&L\n  w = continue\n/\n"

    def test_one_element_of_a_repeat_splits_the_run(self, parser) -> None:
        """Test that the notation is kept rather than expanded, and the comment survives.

        The six values of "6*9.0" are backed by a single node, so writing one of them has to
        rewrite the run. Neighbouring equal values are regrouped.
        """
        config = parser.parse("&L\n  X = 6*9.0, 4.0   ! keep me\n/\n")

        config["L"]["X"][2] = 8.0

        assert config["L"]["X"] == [9.0, 9.0, 8.0, 9.0, 9.0, 9.0, 4.0]
        assert str(config) == "&L\n  X = 2*9.0, 8.0, 3*9.0, 4.0   ! keep me\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_splitting_a_repeat_leaves_the_rest_of_the_entry_alone(self, parser) -> None:
        """Test that only the node being split is replaced.

        A comment or a line break between the entry's other values survives.
        """
        config = parser.parse("&L\n  x = 2*9.0, ! keep me\n      1.0\n/\n")

        config["L"]["X"][0] = 8.0

        assert str(config) == "&L\n  x = 8.0, 9.0, ! keep me\n      1.0\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize(
        ("values", "written"),
        [([5, 5, 5], "3*5"), ([1, 2, 3], "1, 2, 3"), ([1, 1, 2], "2*1, 2")],
    )
    def test_a_whole_list_may_be_replaced(self, parser, values, written) -> None:
        """Test that the entry rule itself changes with the values.

        Holding a different number of values makes it a different kind of entry: one is
        a scalar assignment and several are a list one.
        """
        config = parser.parse("&L\n  X = 3*1\n/\n")

        config["L"]["X"] = values

        assert str(config) == f"&L\n  X = {written}\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_repeat_is_as_strict_about_types_as_any_value(self, parser) -> None:
        """Test that a repeat of nulls names no type, so there is no notation to write."""
        nulls = parser.parse("&L\n  X = 3*\n/\n")
        with pytest.raises(TypeError):
            nulls["L"]["X"][0] = 1

        integers = parser.parse("&L\n  X = 3*1\n/\n")
        with pytest.raises(TypeError):
            integers["L"]["X"][0] = "a"

        assert str(integers) == "&L\n  X = 3*1\n/\n"

    def test_a_null_repeat_does_not_block_the_real_values(self, parser) -> None:
        """Test that only the run of nulls is unwritable, not the whole list."""
        config = parser.parse("&L\n  x = 2*, 1\n/\n")
        assert config["L"]["X"] == [None, None, 1]

        config["L"]["X"][2] = 5
        assert str(config) == "&L\n  x = 2*, 5\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

        # Writing *into* a null is still refused: there is no value there to change.
        with pytest.raises(TypeError):
            config["L"]["X"][0] = 5

    def test_a_slice_may_be_assigned_from_a_one_shot_iterable(self, parser) -> None:
        """Test that the values are not taken twice.

        They were: once to write the file and once to store, and the second got nothing,
        leaving the list shorter than the nodes behind it.
        """
        config = parser.parse("&L\n  x = 1.0, 2.0, 3.0, 4.0\n/\n")
        values = config["L"]["X"]

        values[1:3] = (value for value in [8.0, 9.0])

        assert list(values) == [1.0, 8.0, 9.0, 4.0]
        assert str(config) == "&L\n  x = 1.0, 8.0, 9.0, 4.0\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_list_handed_out_earlier_still_writes_after_replacement(self, parser) -> None:
        """Test that replacing a list does not strand the object the caller was holding.

        Replacing a list written as a repeat replaces the nodes behind it, and the old list
        went on writing through ones no longer in the tree.
        """
        config = parser.parse("&L\n  x = 3*1\n/\n")
        values = config["L"]["X"]

        config["L"]["X"] = [1, 2, 3]
        values[0] = 99

        assert str(config) == "&L\n  x = 99, 2, 3\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_an_array_spread_over_lines_is_edited_as_one_key(self, parser) -> None:
        """Test that each element is still backed by the node that wrote it."""
        source = "&L\n  latpnt(1)=90.0,\n  latpnt(2)=-65.0,\n  z = 5\n/\n"
        config = parser.parse(source)

        assert config["L"]["LATPNT"] == [90.0, -65.0]

        config["L"]["LATPNT"][1] = -60.0
        assert str(config) == source.replace("-65.0", "-60.0")
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_position_the_file_never_wrote_cannot_be_assigned(self, parser) -> None:
        """Test that a gap stays a gap: there is no text behind it to change."""
        config = parser.parse("&L\n  v(1) = 1\n  v(4) = 4\n/\n")

        with pytest.raises(ValueError):
            config["L"]["V"][1] = 9

        assert str(config) == "&L\n  v(1) = 1\n  v(4) = 4\n/\n"

    def test_a_whole_array_may_be_replaced_over_a_gap(self, parser) -> None:
        """Test that the replacement has to leave an unwritten position null."""
        config = parser.parse("&L\n  v(1) = 1\n  v(4) = 4\n/\n")

        config["L"]["V"] = [7, None, None, 9]

        assert str(config) == "&L\n  v(1) = 7\n  v(4) = 9\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

        with pytest.raises(ValueError):
            config["L"]["V"] = [7, 8, None, 9]

    def test_entries_whose_positions_interleave(self, parser) -> None:
        """Test writing an array whose entries alternate with each other.

        ``v(1:3:2)`` holds positions 1 and 3 and ``v(2)`` position 2, so the elements of
        one entry are not next to each other in the list. Grouping them by where they sit
        rather than by which entry wrote them rewrote the first entry twice, losing a value.
        """
        config = parser.parse("&L\n  v(1:3:2) = 2*1\n  v(2) = 7\n/\n")
        assert config["L"]["V"] == [1, 7, 1]

        config["L"]["V"] = [9, 8, 7]

        assert str(config) == "&L\n  v(1:3:2) = 9, 7\n  v(2) = 8\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_entries_whose_positions_overlap_refuse_writes(self, parser) -> None:
        """Test that a list whose positions are written twice is refused, not mangled.

        Rewriting one entry means writing every value it holds, and an entry partly
        overridden by a later one no longer appears in the list in full. Writing it would
        silently shorten the line.
        """
        config = parser.parse("&L\n  a = 1, 3*7\n  a(1) = 0\n/\n")
        assert config["L"]["A"] == [0, 7, 7, 7]

        with pytest.raises(ValueError):
            config["L"]["A"] = [0, 7, 7, 7]

        assert str(config) == "&L\n  a = 1, 3*7\n  a(1) = 0\n/\n"

    def test_a_merged_derived_type_is_editable_in_place(self, parser) -> None:
        """Test that writing through a merged block reaches the line that holds the value.

        The merged block is a node the parse tree does not contain, but its children are the
        real ones, so only that line is rewritten -- including the comment on a sibling.
        """
        text = "&L\n  filename%met = 'x.nc'   ! met file\n  z = 1\n  filename%out = 'y.nc'\n/\n"
        config = parser.parse(text)

        config["L"]["FILENAME"]["OUT"] = "z.nc"

        assert str(config) == text.replace("'y.nc'", "'z.nc'")
        assert dict(parser.parse(str(config))) == dict(config)


class TestAddingAnEntry:
    """Writing a key the file does not have, in the format's own syntax."""

    def test_the_values_a_namelist_can_write(self, parser) -> None:
        """Test the notation chosen for each type of new value."""
        config = parser.parse("&L\n  X = 1\n/\n")

        config["L"]["B"] = True
        config["L"]["S"] = "text"
        config["L"]["N"] = None

        # Fortran logicals, and a quoted string since a new value is never written bare.
        assert str(config) == '&L\n  X = 1\n  B = .true.\n  S = "text"\n  N =\n/\n'
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize(
        ("donor", "added"),
        [("  x =1", "  N =5"), ("  x= 1", "  N= 5"), ("  x = 1", "  N = 5"), ("  x=1", "  N=5")],
    )
    def test_the_spacing_of_a_neighbour_is_copied_to_the_right_side(self, parser, donor, added) -> None:
        """Test that a new entry puts the whitespace where its neighbour put it.

        A single run is ambiguous -- ``x =1`` and ``x= 1`` both have one -- so the style has
        to record which side of the assignment it fell on.
        """
        config = parser.parse(f"&L\n{donor}\n/\n")

        config["L"]["N"] = 5

        assert str(config) == f"&L\n{donor}\n{added}\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_lands_after_a_comment_run(self, parser) -> None:
        """Test that a comment continuing below an assignment is not written into.

        The comment on the assignment's own line is held by the ``nml_line`` the index is
        measured past, so the position one past it is the second line of that comment.
        """
        config = parser.parse("&L\n  X = 1  ! what X is\n  ! ...continued\n/\n")

        config["L"]["NEW"] = 2

        assert str(config) == "&L\n  X = 1  ! what X is\n  ! ...continued\n  NEW = 2\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_is_written_as_given_and_stored_upper_cased(self, parser) -> None:
        """Test that a key matches case-insensitively however the file spelled it."""
        config = parser.parse("&L\n  Var = 4\n/\n")

        config["L"]["newvar"] = 1

        assert "NEWVAR" in dict(config["L"])
        assert config["L"]["NewVar"] == 1
        assert str(config) == "&L\n  Var = 4\n  newvar = 1\n/\n"

        # Assigning again finds the existing key rather than adding a second entry.
        config["L"]["NEWVAR"] = 2
        assert str(config) == "&L\n  Var = 4\n  newvar = 2\n/\n"

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("&L\n  X = 1\n/\n", "&L\n  X = 1\n/\n&NEW\n  A = 1\n/\n"),
            ("&L\nX = 1\n/\n", "&L\nX = 1\n/\n&NEW\nA = 1\n/\n"),
        ],
    )
    def test_a_new_group_indents_its_contents_like_the_groups_already_there(self, parser, source, expected) -> None:
        """Test that a group created by an assignment copies its neighbour's indentation.

        It is created empty, so it holds no entry of its own to copy a style from and its
        contents would otherwise be written flush left. Nothing is invented either: a file
        that indents nothing gets an unindented group.
        """
        config = parser.parse(source)

        config["NEW"] = {"A": 1}

        assert str(config) == expected
        assert dict(parser.parse(str(config))) == dict(config)

    def test_the_style_of_a_new_group_reaches_keys_added_later(self, parser) -> None:
        """Test that a block keeps the style until it has an entry of its own to copy."""
        config = parser.parse("&L\n  X = 1\n/\n")

        config["NEW"] = {}
        config["NEW"]["A"] = 1

        assert str(config) == "&L\n  X = 1\n/\n&NEW\n  A = 1\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_group_goes_above_a_trailing_comment(self, parser) -> None:
        """Test the known limitation of free text between groups.

        Between groups a comment is not a node of its own: the grammar lumps every line
        there into one ``random_text`` blob, so a new group still goes above it. That is a
        limitation of this format, not of the placement rule.
        """
        config = parser.parse("&L\n  X = 1\n/\n! a top-level comment\n")

        config["NEW"] = {"A": 1}

        assert str(config) == "&L\n  X = 1\n/\n&NEW\n  A = 1\n/\n! a top-level comment\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_what_a_namelist_cannot_express(self, parser) -> None:
        """Test the additions the format has no syntax for."""
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

    def test_a_derived_type_has_no_body_to_add_to(self, parser) -> None:
        """Test that a block with nowhere for an entry to go refuses, not drops it.

        A derived type is written one component per line, so there is nowhere inside it, and
        a merged block's root is not in the parse tree at all. The group around them
        is a real block, so it still takes new keys.
        """
        merged = parser.parse("&L\n A%B = 1\n A%C = 2\n/\n")
        with pytest.raises(UnsupportedEntryError):
            merged["L"]["A"]["D"] = 3

        single = parser.parse("&L\n A%B = 1\n/\n")
        with pytest.raises(UnsupportedEntryError):
            single["L"]["A"]["C"] = 2

        single["L"]["NEW"] = 2
        assert str(single) == "&L\n A%B = 1\n NEW = 2\n/\n"
        assert dict(parser.parse(str(single))) == dict(single)


class TestDeleting:
    """Removing an entry, and the line or lines it occupied."""

    def test_an_entry_alone_on_its_line_takes_the_line_with_it(self, parser) -> None:
        """Test that the trailing comment goes too: it described the line that has gone.

        A namelist line is an assignment followed by an optional comment, so the line rule
        cannot derive what is left once the assignment goes. A line holding two assignments
        keeps the one that remains.
        """
        config = parser.parse("&L\n  X = 1  ! note\n  Y = 2\n/\n")
        del config["L"]["X"]
        assert str(config) == "&L\n  Y = 2\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

        shared = parser.parse("&L\n X = 1, Y = 2\n/\n")
        del shared["L"]["X"]
        assert dict(shared["L"]) == {"Y": 2}
        assert dict(parser.parse(str(shared))) == dict(shared)

    def test_every_entry_that_wrote_a_key_is_removed(self, parser) -> None:
        """Test that a key written twice does not reappear the next time the file is read.

        Only the last assignment survives in the configuration, so removing only that one
        left the key behind in the file.
        """
        config = parser.parse("&L\n  x = 1\n  x = 2\n/\n")

        del config["L"]["X"]

        assert str(config) == "&L\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_an_array_spread_over_lines_loses_every_line(self, parser) -> None:
        """Test that removing an indexed key removes each entry making it up."""
        config = parser.parse("&L\n  latpnt(1)=90.0,\n  latpnt(2)=-65.0,\n  z = 5\n/\n")

        del config["L"]["LATPNT"]

        assert str(config) == "&L\n  z = 5\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_derived_type_goes_when_it_loses_its_last_component(self, parser) -> None:
        """Test that a key with nothing left to write goes from the configuration too.

        A derived type is written a component per line, so removing the last one leaves
        nothing. An empty namelist group is still a namelist group, so that one stays.
        """
        config = parser.parse("&L\n  grid%nx = 96\n  grid%ny = 72\n  z = 1\n/\n")

        del config["L"]["GRID"]["NX"]
        assert str(config) == "&L\n  grid%ny = 72\n  z = 1\n/\n"

        del config["L"]["GRID"]["NY"]
        assert str(config) == "&L\n  z = 1\n/\n"

        del config["L"]["Z"]
        assert str(config) == "&L\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_merged_derived_type_deletes_every_line_it_occupies(self, parser) -> None:
        """Test that deleting the key removes both of its lines and leaves the rest."""
        config = parser.parse("&L\n  filename%met = 'x'\n  z = 1\n  filename%out = 'y'\n/\n")

        del config["L"]["FILENAME"]

        assert str(config) == "&L\n  z = 1\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_whole_group_can_go_and_the_text_around_it_is_rejoined(self, parser) -> None:
        """Test that removing a group leaves the comments and free text that surrounded it.

        A namelist and the text around it alternate in the grammar, so removing one leaves
        two runs of text side by side. They are rejoined, keeping the tree in a shape the
        grammar derives. A namelist file needs at least one group, so once they are all gone
        there is no text left to write.
        """
        config = parser.parse(SAMPLE)

        del config["LIST_B"]

        output = str(config)
        assert "LIST_B" not in output
        assert "! Yet another comment" in output
        assert "This is some random text" in output
        assert dict(parser.parse(output)) == dict(config)

        assert list(config) == ["LIST_A", "LIST_C"]
        del config["LIST_A"]
        del config["LIST_C"]
        assert str(config) == ""


class TestRepeatedGroups:
    """A group the file writes more than once, read as one record per occurrence."""

    def test_each_occurrence_is_its_own_block(self, parser) -> None:
        """Test that a group named twice is not merged into one.

        Fortran's ``READ`` scans forward and stops at the first group of the name, so a
        program reading the group once sees the first occurrence and one in a loop
        sees each in turn. Neither of those is the merge, and only the occurrences produce
        both.
        """
        source = "&L\n V = 1\n/\n&L\n W = 2\n/\n"
        config = parser.parse(source)

        assert [dict(block) for block in config["L"]] == [{"V": 1}, {"W": 2}]
        assert str(config) == source

    def test_a_key_in_both_stays_with_its_own_group(self, parser) -> None:
        """Test that neither occurrence's value is lost when both assign the same key.

        Merging reported the last value for such a key while keeping the keys only the first
        one set, which is a combination no single ``READ`` returns.
        """
        source = "&L\n V = 1\n P = 9\n/\n&L\n V = 2\n/\n"
        config = parser.parse(source)

        assert [dict(block) for block in config["L"]] == [{"V": 1, "P": 9}, {"V": 2}]
        assert str(config) == source

    def test_a_group_written_once_is_not_a_list(self, parser) -> None:
        """Test the usual case is untouched: one occurrence is handed out as a block."""
        config = parser.parse("&L\n V = 1\n/\n")

        assert dict(config["L"]) == {"V": 1}

    def test_one_occurrence_is_edited_on_its_own(self, parser) -> None:
        """Test that writing through one occurrence leaves every other one as it was."""
        config = parser.parse("&L\n V = 1\n/\n&L\n V = 2\n/\n")

        config["L"][1]["V"] = 5

        assert str(config) == "&L\n V = 1\n/\n&L\n V = 5\n/\n"

    def test_an_occurrence_accepts_a_new_key(self, parser) -> None:
        """Test that each occurrence is a real block node, so it has a body to add to.

        A merged block is a node the parse tree does not contain, so nothing can be spliced
        into it.
        """
        config = parser.parse("&L\n V = 1\n/\n&L\n W = 2\n/\n")

        config["L"][0]["Z"] = 7

        assert str(config) == "&L\n V = 1\n Z = 7\n/\n&L\n W = 2\n/\n"

    def test_deleting_the_key_removes_every_occurrence(self, parser) -> None:
        """Test that the group does not come back the next time the file is read."""
        config = parser.parse("&L\n V = 1\n/\n&L\n W = 2\n/\n")

        del config["L"]

        assert str(config) == ""


# =============================================================================
# End to end
# =============================================================================

SAMPLE = """
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

SAMPLE_VALUES = {
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

SAMPLE_EDITED = """
&LIST_A ! This is a comment
  Var = 6 , ! This is a comment after an assignment
  LIST = 1, 2, 3

! This is another comment

  Float = 900.0  ,
  COMPLEX = (3.0, 4.0),

  NOVALUE =
  NEWNULL =
  NEWCOMPLEX = (3.0, 4.0)
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
  NEWLIST = 1, 2, 3
/

&LIST_C
NEWVAR = 9
/
&NEWNML
A = 1
/
"""


def test_a_whole_file_parses_edits_and_round_trips(parser):
    """Test that the rules above compose, over a file carrying all of them at once.

    Each focused test pins one construct in isolation. This one is the anchor: comments,
    blank lines, free text, a list continued over three lines and an empty group, all in one
    file that is read, changed, added to and written back with one byte-exact expectation.

    What it covers that no focused test can: where each new entry lands relative to the
    trailing blank lines and the group terminator, that editing one group leaves the others
    alone, and that a group created from nothing follows the layout of the file it joins --
    ``LIST_C`` is empty, so it has no style to copy and its new key is written canonically,
    and ``NEWNML`` then copies that.
    """
    config = parser.parse(SAMPLE)
    assert dict(config) == SAMPLE_VALUES

    config["LIST_A"]["VAR"] = 6
    config["LIST_A"]["FLOAT"] = 900.0
    config["LIST_B"]["BOOL"] = False
    config["LIST_B"]["STRING"] = "another string"

    config["LIST_A"]["NEWNULL"] = None
    config["LIST_A"]["NEWCOMPLEX"] = 3.0 + 4.0j
    config["LIST_B"]["NEWLIST"] = [1, 2, 3]
    config["LIST_C"]["NEWVAR"] = 9
    config["NEWNML"] = {"A": 1}

    assert str(config) == SAMPLE_EDITED
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("fortran_nml"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was generated for. These rows are the
    statement of what this grammar can express and how it spells it.
    """
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected)
