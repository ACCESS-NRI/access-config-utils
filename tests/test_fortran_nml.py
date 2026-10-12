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
from conftest import assert_canonical_entry, canonical_rows, chain_rows
from lark.exceptions import UnexpectedInput

from access.config.fortran_nml import FortranNMLParser
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

        ``&end`` is the older spelling, matched case-insensitively (``/&end/i``).
        """
        assert dict(parser.parse(source)) == {"LIST": {"TEST": "a"}}

    @pytest.mark.parametrize(
        "source",
        [
            "$g\n x = 1\n$end\n",
            "$g\n x = 1\n/\n",
            "&g\n x = 1\n$end\n",
            "$g\n x = 1\n&end\n",
        ],
    )
    def test_a_group_may_open_with_a_dollar(self, parser, source) -> None:
        """Test the older ``$`` delimiter, and that the one written is the one written back.

        Each delimiter is its own rule; an anonymous literal would come back as ``&``.
        """
        config = parser.parse(source)

        assert dict(config["G"]) == {"X": 1}
        assert str(config) == source

    def test_a_bare_dollar_does_not_close_a_group(self, parser) -> None:
        """Test that ``$name … $`` is refused, as gfortran refuses it.

        Some readers take it (f90nml's ``dollar.nml``), but gfortran gives iostat 5010.
        """
        with pytest.raises(UnexpectedInput):
            parser.parse("$g\n x = 1\n$\n")

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

        One value is a ``key_value``: a scalar and a list are different kinds of entry.
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
        """Test ``wrapped_value`` and ``wrapped_list``, which real files use for long lists.

        A key with nothing after it at all stays valueless: it swallows no later line.
        """
        source = "&L\n  files =\n      'a',\n      'b',\n  empty =\n  z = 1\n/\n"
        config = parser.parse(source)

        assert dict(config["L"]) == {"FILES": ["a", "b"], "EMPTY": None, "Z": 1}
        assert str(config) == source

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1 2 3", [1, 2, 3]),
            ("1  2   3", [1, 2, 3]),
            ("'ocn' 'atm' 'ice'", ["ocn", "atm", "ice"]),
            ("hello world", ["hello", "world"]),
            ("1 T", [1, True]),
            ("1 , 2 3", [1, 2, 3]),
            ("2*7 1", [7, 7, 1]),
        ],
    )
    def test_a_blank_may_part_two_values(self, parser, text, expected) -> None:
        """Test that whitespace alone separates values, as Fortran says it does.

        WW3's inputs are written this way throughout (``v = 1 2 3 4`` is four elements).
        """
        source = f"&L\nV = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["V"] == expected
        assert str(config) == source

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1, , 3", [1, None, 3]),
            ("1,,3", [1, None, 3]),
            ("1, , , 4", [1, None, None, 4]),
            ("1, , 3, 4", [1, None, 3, 4]),
        ],
    )
    def test_a_list_may_skip_a_position(self, parser, text, expected) -> None:
        """Test that nothing between two separators leaves that element unset.

        gfortran keeps the variable's old value there; here it reads as null, unwritable.
        """
        source = f"&L\nV = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["V"] == expected
        assert str(config) == source

    def test_a_skipped_position_cannot_be_written(self, parser) -> None:
        """Test that the null is not a value: there is no text behind it to change."""
        config = parser.parse("&L\nV = 1, , 3\n/\n")

        with pytest.raises((TypeError, ValueError)):
            config["L"]["V"][1] = 9

        assert str(config) == "&L\nV = 1, , 3\n/\n"

    def test_two_assignments_on_a_line_need_no_comma(self, parser) -> None:
        """Test that a blank between values does not swallow the key that follows.

        A lookahead on the bare word keeps ``b`` from being read as a value of ``a``.
        """
        source = "&L\nA = 1 B = 2\n/\n"
        config = parser.parse(source)

        assert dict(config["L"]) == {"A": 1, "B": 2}
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

        The separator between assignments and between list values is the same character.
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

        Each line is its own entry under one key; without merging only the last survives.
        """
        text = "&L\n  filename%met = 'x'\n  z = 1\n  filename%out = 'y'\n/\n"
        config = parser.parse(text)

        assert dict(config["L"]) == {"FILENAME": {"MET": "x", "OUT": "y"}, "Z": 1}
        assert str(config) == text

    def test_a_body_holds_exactly_one_assignment(self, parser) -> None:
        """Test that a derived type does not swallow the lines that follow it.

        A greedy ``block`` would mis-nest the next lines and cost an order of magnitude.
        """
        text = "&L\n  a%b = 1\n  c = 2\n  d = 3\n/\n"
        config = parser.parse(text)

        assert dict(config["L"]) == {"A": {"B": 1}, "C": 2, "D": 3}
        assert str(config) == text

    def test_a_key_used_as_both_value_and_block_is_refused(self, parser) -> None:
        """Test that a key assigned a value and also used as a block raises.

        Merging would build a block around the value node and drop the value with no error.
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

        A position no entry writes reads as null; an unqualified entry starts at one.
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

        A qualifier says where the values begin, not how many: reading it as one raised.
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

        ``x = 1, 2`` then ``x = 3`` is ``[3, 2]``; the scalar ``3`` dropped an element.
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

        Else two of them would read as one key, dropping all but the last with no error.
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

        It is written where one value goes, so even "x = 3*1" is a list.
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

        ``double`` keeps Fortran's ``d`` exponent, which Python does not spell, as text.
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

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("1+0", 1.0), ("5-1", 0.5), ("1.5+2", 150.0), ("-2-1", -0.2)],
    )
    def test_the_exponent_letter_may_be_left_out(self, parser, text, expected) -> None:
        """Test ``BARE_EXPONENT``: a signed exponent needs no ``e`` before it.

        ``1+0`` is 1.0 (gfortran); the handler adds the ``e`` to convert, the text is kept.
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

        Without it, after a trailing comma, ``f_icedir`` reads as the value ``.false.`` and
        a key ``_icedir``: the file parses with a value invented and no error.
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

        CICE has "runtype=continue"; the CESM data models "import_data_fields = none".
        """
        source = f"&L\n  X = {text}\n/\n"
        config = parser.parse(source)

        assert config["L"]["X"] == expected
        assert str(config) == source

    def test_a_word_cannot_be_read_as_the_key_that_follows_it(self, parser) -> None:
        """Test the lookahead that stops a bare word being the key of the next assignment.

        After a continuation comma, "b" in "b = 1" would otherwise read as a value of "a".
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

        A blank line and a comment-only line are one rule, told apart by a comment node.
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
        """Test that a line opening a group is never free text.

        Else a group the grammar cannot derive was read as text and silently dropped.
        """
        with pytest.raises(UnexpectedInput):
            parser.parse("&A\n  x = 1\n&B\n  y = 2\n/\n")

    def test_a_dollar_group_is_not_free_text_either(self, parser) -> None:
        """Test the second lookahead, which closes the same hole for the older delimiter.

        A file like this used to lose ``MYGRP`` and still round-trip byte for byte.
        """
        source = "$mygrp\n  x = 1\n$end\n&other\n  y = 2\n/\n"
        config = parser.parse(source)

        assert list(config) == ["MYGRP", "OTHER"]
        assert str(config) == source

    @pytest.mark.parametrize("comment", ["$ ---- a comment ----", "$", "$   spaced"])
    def test_a_bare_dollar_still_starts_free_text(self, parser, comment) -> None:
        """Test that the lookahead needs a name after the ``$`` before it stops being text.

        WW3 writes comments this way (884 lines in the namelists on hand).
        """
        source = f"{comment}\n&g\n  x = 1\n/\n"
        config = parser.parse(source)

        assert list(config) == ["G"]
        assert str(config) == source


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
            "&TEST\na=1 \n2\n/",
            "BLOCK\n TEST ='a'/",
            "&BLOCK\n VAR1=1\n&e",
            "&L\n  nonsense @@@\n/\n",
        ],
    )
    def test_malformed_input_raises(self, parser, source) -> None:
        """Test that each of these raises rather than parsing.

        Which ``UnexpectedInput`` Lark raises is not a promise the parser makes.
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

        CRLF files keep their carriage returns, and a missing final newline stays missing.
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

        The ``eq`` rule binds the leading whitespace to "=", else "A= 1" comes back "A =1".
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

        A format that wraps a value verbatim has to refuse such a string.
        """
        config = parser.parse("&L\nA = 'x'\n/\n")

        config["L"]["A"] = """he said "no" and it's fine"""

        assert str(config) == "&L\nA = 'he said \"no\" and it''s fine'\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize("value", ["true", "t", "F", "false", "café"])
    def test_a_bare_word_refuses_a_value_that_would_change_type(self, parser, value) -> None:
        """Test that ``true`` written into an unquoted value is a logical, not a string.

        Nothing delimits an unquoted value, so what would change its type is refused.
        """
        config = parser.parse("&L\n  w = continue\n/\n")

        with pytest.raises(TypeError):
            config["L"]["W"] = value

        assert str(config) == "&L\n  w = continue\n/\n"

    @pytest.mark.parametrize(
        ("text", "value", "error"),
        [
            ("x = 1, 2", [5, "text"], TypeError),
            ("x = 1.0, 2.0", [5.0, float("nan")], UnsupportedEntryError),
            ("x = 1, 3*2", [5, "t", "t", "t"], TypeError),
            ("x = 1, 3*2", [5, 2, 2, 7], ValueError),
        ],
    )
    def test_a_list_the_file_cannot_take_leaves_the_file_and_the_list_as_they_were(
        self, parser, text, value, error
    ) -> None:
        """Test that a refused list is refused as a whole.

        No element ahead of the refused one may be written, or file and dict disagree.
        """
        config = parser.parse(f"&L\n  {text}\n/\n")
        before = list(config["L"]["X"])

        with pytest.raises(error):
            config["L"]["X"] = value

        assert str(config) == f"&L\n  {text}\n/\n"
        assert config["L"]["X"] == before

    def test_one_list_element_leaves_the_others_as_written(self, parser) -> None:
        """Test that writing one element does not rewrite the notation of the rest."""
        config = parser.parse("&L\n  x = 1.0d0, 2.0D0, 1.5e3, 1e-5\n/\n")

        config["L"]["X"][1] = 3.0

        assert str(config) == "&L\n  x = 1.0d0, 3.0, 1.5e3, 1e-5\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_nan_in_a_list_does_not_block_writing_the_rest(self, parser) -> None:
        """Test that an element which cannot be written anew is left alone."""
        config = parser.parse("&L\n  x = 1.0, nan, 2.0\n/\n")

        config["L"]["X"][0] = 5.0

        assert str(config) == "&L\n  x = 5.0, nan, 2.0\n/\n"

    def test_one_element_of_a_repeat_is_not_written_on_its_own(self, parser) -> None:
        """Test that a run is not split: "6*9.0" is six values written once.

        The caller is told to delete the key and assign it again.
        """
        config = parser.parse("&L\n  X = 6*9.0, 4.0   ! keep me\n/\n")

        with pytest.raises(ValueError, match="delete it and assign it again"):
            config["L"]["X"][2] = 8.0

        assert str(config) == "&L\n  X = 6*9.0, 4.0   ! keep me\n/\n"
        assert config["L"]["X"] == [9.0] * 6 + [4.0]

    def test_a_repeat_given_one_new_value_is_rewritten_in_place(self, parser) -> None:
        """Test that setting a run alike keeps its count and the comment."""
        config = parser.parse("&L\n  X = 6*9.0, 4.0   ! keep me\n/\n")

        config["L"]["X"] = [9.5] * 6 + [4.0]

        assert str(config) == "&L\n  X = 6*9.5, 4.0   ! keep me\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_rewriting_a_repeat_leaves_the_rest_of_the_entry_alone(self, parser) -> None:
        """Test that a comment or a line break between the entry's other values survives."""
        config = parser.parse("&L\n  x = 2*9.0, ! keep me\n      1.0\n/\n")

        config["L"]["X"] = [8.0, 8.0, 1.0]

        assert str(config) == "&L\n  x = 2*8.0, ! keep me\n      1.0\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize(("values", "written"), [([5, 5, 5], "3*5"), ([1, 1, 1], "3*1")])
    def test_a_whole_list_of_one_value_may_replace_a_repeat(self, parser, values, written) -> None:
        """Test that a list all alike is written back as the run it replaces."""
        config = parser.parse("&L\n  X = 3*1\n/\n")

        config["L"]["X"] = values

        assert str(config) == f"&L\n  X = {written}\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize("values", [[1, 2, 3], [1, 1, 2]])
    def test_a_list_that_is_not_all_alike_cannot_replace_a_repeat(self, parser, values) -> None:
        """Test that a run cannot be split in place, however the whole list is given."""
        config = parser.parse("&L\n  X = 3*1\n/\n")

        with pytest.raises(ValueError, match="delete it and assign it again"):
            config["L"]["X"] = values

        assert str(config) == "&L\n  X = 3*1\n/\n"

    def test_a_repeat_is_as_strict_about_types_as_any_value(self, parser) -> None:
        """Test that a repeat of nulls names no type, so there is no notation to write."""
        nulls = parser.parse("&L\n  X = 3*\n/\n")
        with pytest.raises(TypeError):
            nulls["L"]["X"][0] = 1

        integers = parser.parse("&L\n  X = 3*1\n/\n")
        with pytest.raises(TypeError):
            integers["L"]["X"] = ["a", "a", "a"]

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

        A second read of the iterable got nothing, leaving the list shorter than its nodes.
        """
        config = parser.parse("&L\n  x = 1.0, 2.0, 3.0, 4.0\n/\n")
        values = config["L"]["X"]

        values[1:3] = (value for value in [8.0, 9.0])

        assert list(values) == [1.0, 8.0, 9.0, 4.0]
        assert str(config) == "&L\n  x = 1.0, 8.0, 9.0, 4.0\n/\n"
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

        ``v(1:3:2)`` and ``v(2)`` interleave, and a repeat is written as a whole.
        """
        config = parser.parse("&L\n  v(1:3:2) = 2*1\n  v(2) = 7\n/\n")
        assert config["L"]["V"] == [1, 7, 1]

        config["L"]["V"] = [9, 8, 9]

        assert str(config) == "&L\n  v(1:3:2) = 2*9\n  v(2) = 8\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

        with pytest.raises(ValueError, match="delete it and assign it again"):
            config["L"]["V"] = [9, 8, 7]

    def test_entries_whose_positions_overlap_are_written_through_what_survives(self, parser) -> None:
        """Test an entry partly overridden by a later one, whose rest the list holds.

        Only the nodes the list reads from are written; the overridden position stays.
        """
        config = parser.parse("&L\n  a = 1, 3*7\n  a(1) = 0\n/\n")
        assert config["L"]["A"] == [0, 7, 7, 7]

        config["L"]["A"] = [5, 7, 7, 7]

        assert str(config) == "&L\n  a = 1, 3*7\n  a(1) = 5\n/\n"

    def test_a_merged_derived_type_is_editable_in_place(self, parser) -> None:
        """Test that writing through a merged block reaches the line that holds the value.

        The merged node is not in the tree, but its children are the real ones.
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

        A single run is ambiguous: ``x =1`` and ``x= 1`` each have one.
        """
        config = parser.parse(f"&L\n{donor}\n/\n")

        config["L"]["N"] = 5

        assert str(config) == f"&L\n{donor}\n{added}\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize(
        ("source", "key", "value", "expected"),
        [
            pytest.param("&L\n  S = 'abc'\n/\n", "T", "x", "&L\n  S = 'abc'\n  T = 'x'\n/\n", id="quote"),
            pytest.param(
                "&L\n  S = 'a', 'b'\n/\n",
                "T",
                ["x", "y", "z"],
                "&L\n  S = 'a', 'b'\n  T = 'x', 'y', 'z'\n/\n",
                id="list-quote",
            ),
            pytest.param("&L\n  F = T\n/\n", "G", False, "&L\n  F = T\n  G = F\n/\n", id="bare-logical"),
            pytest.param(
                "&L\n  F = .TRUE.\n/\n", "G", False, "&L\n  F = .TRUE.\n  G = .FALSE.\n/\n", id="dotted-logical"
            ),
            pytest.param(
                "&L\r\n  X = 1 ! one\r\n/\r\n", "Y", 2, "&L\r\n  X = 1 ! one\r\n  Y = 2\r\n/\r\n", id="windows-line-end"
            ),
            pytest.param("&L X = 1 /\n", "Y", 2, "&L X = 1 Y = 2 /\n", id="one-line-group"),
        ],
    )
    def test_a_new_entry_is_written_like_its_neighbour(self, parser, source, key, value, expected) -> None:
        """Test that the nearest entry of the same kind is copied, notation and all.

        Quotes, logical spelling, line ending, and a one-line group staying on one line.
        """
        config = parser.parse(source)

        config["L"][key] = value

        assert str(config) == expected
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_group_copies_the_delimiters_of_the_last(self, parser) -> None:
        """Test that a file written with ``$`` and ``$end`` gets a new group written so."""
        config = parser.parse("$L\n  X = 1\n$end\n")

        config["NEW"] = {"A": 1}

        assert str(config) == "$L\n  X = 1\n$end\n$NEW\n  A = 1\n$end\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_lands_after_a_comment_run(self, parser) -> None:
        """Test that a comment continuing below an assignment is not written into.

        One past the entry is the second line of a comment begun on the entry's own line.
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

        Created empty, it has no entry to copy; an unindented file gets an unindented group.
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

        Between groups the grammar lumps every line into one ``random_text`` blob.
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

        # Namelists do not nest, and a derived type is written by its components, so one
        # with none has nothing to write.
        with pytest.raises(TypeError):
            config["L"]["NESTED"] = {}

        # The grammar has no value type for a path.
        with pytest.raises(TypeError):
            config["L"]["P"] = Path("/x/y")

        assert str(config) == "&L\n  X = 1\n/\n"


class TestAddingADerivedTypeComponent:
    """A key added to a derived type, which is written one component per line.

    No body of its own: the line goes in the group, after the last line of the type.
    """

    @pytest.mark.parametrize(
        ("source", "path", "key", "expected"),
        [
            pytest.param(
                "&L\n A%B = 1\n A%C = 2\n/\n", ["L", "A"], "d", "&L\n A%B = 1\n A%C = 2\n A%d = 3\n/\n", id="merged"
            ),
            pytest.param("&L\n A%B = 1\n/\n", ["L", "A"], "C", "&L\n A%B = 1\n A%C = 3\n/\n", id="single"),
            pytest.param(
                "&l\n  a%b = 1\n  x = 2\n/\n", ["l", "a"], "c", "&l\n  a%b = 1\n  a%c = 3\n  x = 2\n/\n", id="mid-group"
            ),
            pytest.param(
                "&l\n  a%b%c = 1\n  x = 2\n/\n",
                ["l", "a", "b"],
                "d",
                "&l\n  a%b%c = 1\n  a%b%d = 3\n  x = 2\n/\n",
                id="nested",
            ),
            pytest.param(
                "&l\n  a%b%c = 1\n  x = 2\n/\n",
                ["l", "a"],
                "e",
                "&l\n  a%b%c = 1\n  a%e = 3\n  x = 2\n/\n",
                id="outer-level",
            ),
        ],
    )
    def test_goes_after_the_last_line_of_the_type(self, parser, source, path, key, expected) -> None:
        """Test that the type keeps its lines together, spelled as the file spells it."""
        config = parser.parse(source)
        held = config
        for step in path:
            held = held[step]

        held[key] = 3

        assert str(config) == expected
        assert dict(parser.parse(str(config))) == dict(config)
        # A reference held to the type sees the new component.
        assert held[key] == 3

    def test_a_new_type_is_written_by_its_components(self, parser) -> None:
        """Test that a dict for a type the group does not have is a line per component."""
        config = parser.parse("&L\n  X = 1\n/\n")

        config["L"]["NESTED"] = {"A": 1, "B": {"C": 2, "D": 3}, "E": 4}

        assert str(config) == ("&L\n  X = 1\n  NESTED%A = 1\n  NESTED%B%C = 2\n  NESTED%B%D = 3\n  NESTED%E = 4\n/\n")
        assert dict(parser.parse(str(config))) == dict(config)
        assert config["L"]["NESTED"] == {"A": 1, "B": {"C": 2, "D": 3}, "E": 4}

    def test_the_group_around_a_type_still_takes_keys_of_its_own(self, parser) -> None:
        """Test that a key of the group goes after the last entry, as ever."""
        config = parser.parse("&L\n A%B = 1\n/\n")

        config["L"]["NEW"] = 2

        assert str(config) == "&L\n A%B = 1\n NEW = 2\n/\n"
        assert dict(parser.parse(str(config))) == dict(config)


class TestResizingAList:
    """A list that grows or shrinks, laid out around the elements that stay."""

    MIXED = "&L\n  x = 1, 2,  ! c\n     3, 4\n/\n"
    ONE_PER_LINE = "&L\n  x = 1,\n      2,\n      3\n/\n"

    @pytest.mark.parametrize(
        ("source", "change", "written"),
        [
            pytest.param(MIXED, lambda x: x.insert(2, 9), "&L\n  x = 1, 2, 9,  ! c\n     3, 4\n/\n", id="mixed-insert"),
            pytest.param(MIXED, lambda x: x.remove(2), "&L\n  x = 1,  ! c\n     3, 4\n/\n", id="mixed-delete-before"),
            pytest.param(MIXED, lambda x: x.remove(3), "&L\n  x = 1, 2,  ! c\n     4\n/\n", id="mixed-delete-after"),
            pytest.param(
                ONE_PER_LINE,
                lambda x: x.append(4),
                "&L\n  x = 1,\n      2,\n      3,\n      4\n/\n",
                id="one-per-line-append",
            ),
            pytest.param(ONE_PER_LINE, lambda x: x.pop(), "&L\n  x = 1,\n      2\n/\n", id="one-per-line-pop"),
            pytest.param(
                ONE_PER_LINE,
                lambda x: x.insert(0, 0),
                "&L\n  x = 0,\n      1,\n      2,\n      3\n/\n",
                id="one-per-line-prepend",
            ),
            pytest.param("&nml x = 1, 2 /\n", lambda x: x.append(3), "&nml x = 1, 2, 3 /\n", id="single-line-namelist"),
            pytest.param(
                "&L\n  s = 'a' 'b'\n/\n", lambda x: x.append("it's"), "&L\n  s = 'a' 'b' 'it''s'\n/\n", id="strings"
            ),
        ],
    )
    def test_layout(self, parser, source, change, written) -> None:
        """Test that line breaks, comments and notation stay, and new elements match."""
        config = parser.parse(source)
        group = next(iter(config))

        change(config[group][next(iter(config[group]))])

        assert str(config) == written
        assert dict(parser.parse(str(config))) == dict(config)

    def test_another_entry_on_the_same_line_is_left_alone(self, parser) -> None:
        """Test that only the list's own entry is rewritten, the line's other one kept."""
        config = parser.parse("&L\n  x = 1, 2, y = 3, 4,\n/\n")
        other = config["L"]["X"]

        config["L"]["Y"].append(5)
        other[0] = 7

        assert str(config) == "&L\n  x = 7, 2, y = 3, 4, 5,\n/\n"

    def test_a_derived_type_component(self, parser) -> None:
        """Test a list that is an entry inside the entry for its type."""
        config = parser.parse("&L\n  a%b = 1, 2\n/\n")

        config["L"]["A"]["B"].append(3)

        assert str(config) == "&L\n  a%b = 1, 2, 3\n/\n"

    def test_a_whole_list_of_another_length(self, parser) -> None:
        """Test that assigning a list matches the old values to keep their places."""
        config = parser.parse("&L\n  x = 1, 2, 3, 4  ! c\n/\n")

        config["L"]["X"] = [1, 4]

        assert str(config) == "&L\n  x = 1, 4  ! c\n/\n"

    @pytest.mark.parametrize(
        ("source", "reason"),
        [
            pytest.param("&L\n  x(2) = 1, 2\n/\n", "array qualifiers", id="qualifier"),
            pytest.param("&L\n  x = 1, , 3\n/\n", "unset", id="skipped"),
            pytest.param("&L\n  x = 1, 2\n  x = 3, 4\n/\n", "more than once", id="assigned-twice"),
            pytest.param("&L\n  x = 3*1\n/\n", "repeat count", id="repeat"),
        ],
    )
    def test_what_cannot_be_resized(self, parser, source, reason) -> None:
        """Test the lists whose values are not written one token apiece in one entry."""
        config = parser.parse(source)

        with pytest.raises(ValueError, match=reason):
            config["L"]["X"].append(9)
        assert str(config) == source


class TestDeleting:
    """Removing an entry, and the line or lines it occupied."""

    def test_a_derived_type_component_then_the_type(self, parser) -> None:
        """Test that the group no longer names the line a component deletion removed.

        Else deleting ``a`` next tries to unlink the line of ``a%b`` again.
        """
        config = parser.parse("&L\n  a%b = 1\n  a%c = 2\n  x = 3\n/\n")

        del config["L"]["A"]["B"]
        assert str(config) == "&L\n  a%c = 2\n  x = 3\n/\n"
        del config["L"]["A"]

        assert str(config) == "&L\n  x = 3\n/\n"
        assert dict(config["L"]) == {"X": 3}

    def test_the_last_component_of_a_nested_type_takes_the_type_with_it(self, parser) -> None:
        """Test that a type left with nothing to write leaves the dict at every level."""
        config = parser.parse("&L\n  a%b%c = 1\n  x = 3\n/\n")

        del config["L"]["A"]["B"]["C"]

        assert str(config) == "&L\n  x = 3\n/\n"
        assert dict(config["L"]) == {"X": 3}

    def test_a_nested_type_emptied_beside_another_component(self, parser) -> None:
        """Test that only the levels the file no longer has leave the dict."""
        config = parser.parse("&L\n  a%b%c = 1\n  a%d = 2\n/\n")

        del config["L"]["A"]["B"]["C"]

        assert str(config) == "&L\n  a%d = 2\n/\n"
        assert dict(config["L"]["A"]) == {"D": 2}

    def test_an_entry_alone_on_its_line_takes_the_line_with_it(self, parser) -> None:
        """Test that the trailing comment goes too: it described the line that has gone.

        A namelist line is an assignment plus an optional comment: it cannot derive less.
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

        Only the last assignment is in the dict; removing only it left the key in the file.
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

        An empty namelist group is still a group, so that one stays.
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

        Groups and text alternate in the grammar, so runs left side by side are rejoined.
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

        ``READ`` stops at the first group of a name, so no single read sees a merge.
        """
        source = "&L\n V = 1\n/\n&L\n W = 2\n/\n"
        config = parser.parse(source)

        assert [dict(block) for block in config["L"]] == [{"V": 1}, {"W": 2}]
        assert str(config) == source

    def test_a_key_in_both_stays_with_its_own_group(self, parser) -> None:
        """Test that neither occurrence's value is lost when both assign the same key.

        A merge mixes the last value with keys only the first set: no ``READ`` returns that.
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

        A merged block is not in the parse tree, so nothing can be spliced into it.
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

    It pins what no focused test can: where each new entry lands relative to blank lines
    and terminators, that editing one group leaves the others alone, and that a group
    created from nothing copies the layout of the file it joins.
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

    Byte-exact, and re-parsed in the container it was written for.
    """
    assert_canonical_entry(parser, container, category, expected)


@pytest.mark.parametrize(("container", "category", "depth", "expected"), chain_rows("fortran_nml"))
def test_canonical_chain_entry_text(parser, container, category, depth, expected) -> None:
    """Test the text of a new derived-type component, with no neighbour to copy from."""
    keys = ("A", "B", "C")[:depth]
    assert_canonical_entry(parser, container, category, expected, keys=keys)
