# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the value types a grammar can express.

``grammar_values`` imports nothing from the package, so nothing is faked and no grammar is
compiled: each handler is a bundle of four small functions, and they are called directly.
That matters because the four are reached from three different places in the parser -- one
on parse, one on assignment, two only when a key is added -- so exercising a type through
the parser leaves parts of its handler untouched.
"""

import math
from pathlib import Path

import pytest

from access.config.grammar_values import (
    VALUE_TYPE_HANDLER_REGISTRY,
    UnsupportedEntryError,
    identical_values,
    select_value_rule,
)

# One representative round trip per value type: the rule, the text a file holds, the Python
# value it reads back as, and the text that value is written as into a brand-new entry.
ROUND_TRIPS = [
    ("logical", ".true.", True, ".true."),
    ("logical", ".FALSE.", False, ".false."),
    ("bool", "True", True, "True"),
    ("bool", "False", False, "False"),
    ("integer", "-42", -42, "-42"),
    ("float", "1.5", 1.5, "1.5"),
    ("float", "1.0e-2", 1.0e-2, "0.01"),
    ("double", "1.0d-2", 1.0e-2, "0.01"),
    ("double", "-2.0D2", -200.0, "-200.0"),
    ("complex", "(1.0, -2.0)", 1.0 - 2.0j, "(1.0, -2.0)"),
    ("double_complex", "(1.0d0, -2.0d0)", 1.0 - 2.0j, "(1.0, -2.0)"),
    ("identifier", "a_word", "a_word", "a_word"),
    ("string", "'quoted'", "quoted", '"quoted"'),
    ("fortran_string", "'quoted'", "quoted", '"quoted"'),
    ("text_line", "MED -> OCN :remapMethod=redist", "MED -> OCN :remapMethod=redist", "MED -> OCN :remapMethod=redist"),
    ("path", "/dir/file.nc", Path("/dir/file.nc"), "/dir/file.nc"),
    ("inferred_string", '"kg/kg"', "kg/kg", '"kg/kg"'),
    ("inferred_string", '"3.e-6"', 3.0e-6, '"3e-06"'),
    ("inferred_text", "3", 3, "3"),
    ("inferred_text", ".true.", True, ".true."),
    (
        "named_assignments",
        '"fixed", "surface_value=1.e30"',
        {"value": "fixed", "surface_value": 1.0e30},
        '"fixed", "surface_value=1e+30"',
    ),
]


@pytest.mark.parametrize(("rule", "text", "value", "written"), ROUND_TRIPS)
def test_handler_round_trip(rule, text, value, written) -> None:
    """Test three of the four operations of one handler: check, parse, and seed.

    ``to_token`` depends on the token it replaces and is asserted in the classes below.
    """
    handler = VALUE_TYPE_HANDLER_REGISTRY[rule]

    assert handler.type_check(value)
    assert handler.from_token(text) == value
    assert handler.to_new_token(value) == written
    # Whatever it writes has to read back as the same value again.
    assert handler.from_token(handler.to_new_token(value)) == value


class TestTypeChecking:
    """Which Python values each handler claims, and which it refuses."""

    @pytest.mark.parametrize(("rule", "value"), [(rule, value) for rule, _, value, _ in ROUND_TRIPS])
    def test_accepts_its_own_type(self, rule, value) -> None:
        """Test that every handler recognises the value it parses."""
        assert VALUE_TYPE_HANDLER_REGISTRY[rule].type_check(value)

    @pytest.mark.parametrize("rule", ["integer", "float", "double"])
    def test_a_bool_is_not_a_number(self, rule) -> None:
        """Test that ``bool`` being a subclass of ``int`` does not leak into the numbers.

        The checks are ``type(value) is ...``: ``True`` written as ``1`` would change type.
        """
        assert not VALUE_TYPE_HANDLER_REGISTRY[rule].type_check(True)

    def test_an_identifier_must_be_a_bare_word(self) -> None:
        """Test that only a string usable unquoted as a name is an ``identifier``."""
        handler = VALUE_TYPE_HANDLER_REGISTRY["identifier"]

        assert handler.type_check("a_word")
        assert not handler.type_check("a string")
        assert not handler.type_check("1")
        assert not handler.type_check(Path("/x"))

    @pytest.mark.parametrize(
        ("value", "accepted"),
        [
            pytest.param("@3600", True, id="loop"),
            pytest.param("MED med_phases_post_ocn", True, id="phase"),
            pytest.param("a :: b", True, id="inner-colons"),
            pytest.param("", False, id="empty"),
            pytest.param(" padded", False, id="leading-space"),
            pytest.param("padded\t", False, id="trailing-tab"),
            pytest.param("a # b", False, id="comment"),
            pytest.param("two\nlines", False, id="line-break"),
            pytest.param("::", False, id="table-closer"),
            pytest.param(3600, False, id="int"),
        ],
    )
    def test_a_text_line_is_one_trimmed_line(self, value, accepted) -> None:
        """Test that only a string reading back unchanged as a line of text is one."""
        assert VALUE_TYPE_HANDLER_REGISTRY["text_line"].type_check(value) is accepted

    def test_a_path_is_the_only_type_taking_one(self) -> None:
        """Test that a ``Path`` is not mistaken for a string, or the reverse."""
        assert VALUE_TYPE_HANDLER_REGISTRY["path"].type_check(Path("/x"))
        assert not VALUE_TYPE_HANDLER_REGISTRY["string"].type_check(Path("/x"))
        assert not VALUE_TYPE_HANDLER_REGISTRY["path"].type_check("/x")


class TestStringQuoting:
    """Choosing a quote character, given that values are wrapped rather than escaped."""

    def test_keeps_the_quote_already_in_use(self) -> None:
        """Test that rewriting a value does not gratuitously requote the line."""
        handler = VALUE_TYPE_HANDLER_REGISTRY["string"]

        assert handler.to_token("plain", "'old'") == "'plain'"
        assert handler.to_token("plain", '"old"') == '"plain"'

    @pytest.mark.parametrize(
        ("value", "token", "expected"),
        [("has ' inside", "'old'", '"has \' inside"'), ('has " inside', '"old"', "'has \" inside'")],
    )
    def test_switches_when_the_value_contains_it(self, value, token, expected) -> None:
        """Test the switch: keeping the original quote would break the line."""
        assert VALUE_TYPE_HANDLER_REGISTRY["string"].to_token(value, token) == expected

    def test_refuses_a_value_holding_both(self) -> None:
        """Test that a string with no usable quote is refused, not written unescaped."""
        with pytest.raises(UnsupportedEntryError, match="both quote characters"):
            VALUE_TYPE_HANDLER_REGISTRY["string"].to_token("has ' and \"", "'old'")

    def test_a_new_string_prefers_double_quotes(self) -> None:
        """Test the seed, which decides the quoting of a string with no previous token."""
        assert VALUE_TYPE_HANDLER_REGISTRY["string"].to_new_token("fresh") == '"fresh"'
        assert VALUE_TYPE_HANDLER_REGISTRY["string"].to_new_token('has " inside') == "'has \" inside'"


class TestFortranStringQuoting:
    """Fortran escapes a quote by doubling it, so no value is out of reach."""

    def test_reads_a_doubled_quote_as_one(self) -> None:
        """Test the escape on the way in: ``'it''s'`` holds one apostrophe."""
        handler = VALUE_TYPE_HANDLER_REGISTRY["fortran_string"]

        assert handler.from_token("'it''s'") == "it's"
        assert handler.from_token('"say ""hi"""') == 'say "hi"'

    def test_doubles_a_quote_on_the_way_out(self) -> None:
        """Test that the quote in use is kept and the value escaped against it."""
        handler = VALUE_TYPE_HANDLER_REGISTRY["fortran_string"]

        assert handler.to_token("it's", "'old'") == "'it''s'"
        assert handler.to_token('say "hi"', '"old"') == '"say ""hi"""'

    def test_a_value_holding_both_quotes_can_still_be_written(self) -> None:
        """Test the difference from the general string type, which has to refuse this.

        Doubling keeps the quote in use, so a value holding both is representable.
        """
        handler = VALUE_TYPE_HANDLER_REGISTRY["fortran_string"]
        both = "has ' and \""

        written = handler.to_token(both, "'old'")

        assert handler.from_token(written) == both


class TestFloatNotation:
    """Preserving the exponent character a file already uses."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [("1.0e5", "1e-20"), ("1.0E5", "1E-20"), ("1.0d5", "1d-20"), ("1.0D5", "1D-20"), ("1.0", "1e-20")],
    )
    def test_uses_the_exponent_character_of_the_previous_token(self, token, expected) -> None:
        """Test that a Fortran ``D`` exponent survives an edit, ``e`` being the default."""
        assert VALUE_TYPE_HANDLER_REGISTRY["float"].to_token(1e-20, token) == expected

    def test_a_double_reads_either_exponent_character(self) -> None:
        """Test that the Fortran double form is understood on the way in as well as out."""
        handler = VALUE_TYPE_HANDLER_REGISTRY["double"]

        assert handler.from_token("1.0d3") == 1000.0
        assert handler.from_token("1.0D3") == 1000.0

    def test_a_new_double_carries_its_marker_only_with_an_exponent(self) -> None:
        """Test that the ``d`` seed shows up only where Python's own repr uses an exponent.

        A value whose repr has no ``e`` comes out plain, which ``DOUBLE`` also parses.
        """
        handler = VALUE_TYPE_HANDLER_REGISTRY["double"]

        assert handler.to_new_token(1e-20) == "1d-20"
        assert handler.to_new_token(0.01) == "0.01"
        # Either way it reads back as the value asked for, which is what insertion checks.
        assert handler.from_token(handler.to_new_token(0.01)) == 0.01

    @pytest.mark.parametrize("rule", ["float", "double"])
    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_refuses_a_non_finite_value(self, rule, value) -> None:
        """Test that an infinity or a NaN is refused rather than written as a bare word.

        No supported format reads ``inf`` or ``nan`` back as a number.
        """
        with pytest.raises(UnsupportedEntryError, match="cannot be represented"):
            VALUE_TYPE_HANDLER_REGISTRY[rule].to_token(value, "1.0")

    @pytest.mark.parametrize("rule", ["complex", "double_complex"])
    def test_a_complex_refuses_a_non_finite_part(self, rule) -> None:
        """Test that the check reaches both halves of a complex value."""
        with pytest.raises(UnsupportedEntryError):
            VALUE_TYPE_HANDLER_REGISTRY[rule].to_token(complex(math.inf, 0.0), "(1.0, 2.0)")


class TestBareExponent:
    """The exponent letter a Fortran real may leave out when the exponent is signed."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("1+0", 1.0), ("5-1", 0.5), ("1.5+2", 150.0), ("-2-1", -0.2), ("+3+1", 30.0)],
    )
    def test_the_letter_is_put_back_before_converting(self, text, expected) -> None:
        """Test the spellings ``float`` cannot take, confirmed against gfortran."""
        assert VALUE_TYPE_HANDLER_REGISTRY["bare_exponent"].from_token(text) == expected

    @pytest.mark.parametrize("text", ["1.5", "-2.5", "1e5", "1E5", "-1e-5", "+0.0"])
    def test_an_ordinary_spelling_is_left_alone(self, text) -> None:
        """Test that a number already carrying an exponent letter is not touched.

        A leading sign is not an exponent, and a text holding ``e`` or ``d`` is left alone.
        """
        assert VALUE_TYPE_HANDLER_REGISTRY["bare_exponent"].from_token(text) == float(text)

    def test_the_plain_float_handler_is_untouched(self) -> None:
        """Test that the notation stayed out of the handler every format shares.

        Teaching ``float`` this spelling would teach it to every grammar importing it.
        """
        assert VALUE_TYPE_HANDLER_REGISTRY["float"].from_token("1.5") == 1.5
        with pytest.raises(ValueError):
            VALUE_TYPE_HANDLER_REGISTRY["float"].from_token("1+0")


class TestSelectValueRule:
    """Choosing the rule to write a value with, among those the format has."""

    def test_takes_the_first_admitted_rule_that_accepts_the_value(self) -> None:
        """Test that the order of the registry decides, among the rules the grammar has."""
        admitted = {"logical", "bool", "integer"}

        assert select_value_rule(True, admitted) == "logical"
        assert select_value_rule(1, admitted) == "integer"

    def test_a_format_without_the_preferred_rule_gets_the_next(self) -> None:
        """Test the effect that makes ``True`` a namelist ``.true.`` but a MOM6 ``True``."""
        assert select_value_rule(True, {"bool", "integer"}) == "bool"

    def test_reports_nothing_when_no_admitted_rule_accepts_the_value(self) -> None:
        """Test the case that becomes ``UnsupportedEntryError`` further up."""
        assert select_value_rule("text", {"integer", "float"}) is None
        assert select_value_rule(object(), set(VALUE_TYPE_HANDLER_REGISTRY)) is None

    def test_an_admitted_name_that_is_not_a_value_rule_is_skipped(self) -> None:
        """Test that a name absent from the registry cannot be chosen."""
        assert select_value_rule(1, {"not_a_value_rule"}) is None


class TestRewrite:
    """Writing a new value over the token that held the old one."""

    def test_a_token_already_holding_the_value_keeps_its_text(self) -> None:
        """Test that an unchanged element is written exactly as it was.

        ``1.0d0`` would come back as ``1.0``, and a NaN could not be written over itself.
        """
        assert VALUE_TYPE_HANDLER_REGISTRY["double"].rewrite(1.0, "1.0d0") == "1.0d0"
        assert VALUE_TYPE_HANDLER_REGISTRY["float"].rewrite(math.nan, "nan") == "nan"

    def test_a_new_value_takes_the_notation_of_the_token_it_replaces(self) -> None:
        """Test that an edit does not restyle what it writes over."""
        assert VALUE_TYPE_HANDLER_REGISTRY["double"].rewrite(2.5, "1.0d0") == "2.5"
        assert VALUE_TYPE_HANDLER_REGISTRY["double"].rewrite(1e-20, "1.0d0") == "1d-20"

    def test_a_value_of_another_type_is_refused(self) -> None:
        """Test that the type is fixed by what the file already holds."""
        with pytest.raises(TypeError, match="change value type"):
            VALUE_TYPE_HANDLER_REGISTRY["integer"].rewrite("text", "1")

    def test_a_value_the_format_cannot_write_is_refused(self) -> None:
        """Test that a value of the right type is still refused if it cannot be written."""
        with pytest.raises(UnsupportedEntryError):
            VALUE_TYPE_HANDLER_REGISTRY["float"].rewrite(math.inf, "1.0")

    def test_a_handler_without_notation_writes_the_value_as_text(self) -> None:
        """Test the defaults a bare-word type relies on: read and written as the text."""
        handler = VALUE_TYPE_HANDLER_REGISTRY["identifier"]

        assert handler.from_token("word") == "word"
        assert handler.rewrite("other", "word") == "other"
        assert handler.to_new_token("fresh") == "fresh"


class TestIdenticalValues:
    """Whether two values are the same value, which equality does not say."""

    @pytest.mark.parametrize(
        ("first", "second", "identical"),
        [
            pytest.param(1, 1, True, id="same"),
            pytest.param(1, 1.0, False, id="int-float"),
            pytest.param(1, True, False, id="int-bool"),
            pytest.param(math.nan, math.nan, True, id="nan"),
            pytest.param(0.0, -0.0, False, id="signed-zero"),
            pytest.param("a", "a", True, id="str"),
            pytest.param(Path("a"), "a", False, id="path-str"),
        ],
    )
    def test_compares_type_and_representation(self, first, second, identical) -> None:
        """Test the cases where equality and identity of value disagree."""
        assert identical_values(first, second) is identical


class TestInferredTyping:
    """Reading a value's type from its spelling, as the FMS field manager does."""

    @pytest.mark.parametrize(
        ("text", "value"),
        [
            pytest.param("3", 3, id="integer"),
            pytest.param("-3", -3, id="negative"),
            pytest.param("+3", 3, id="signed"),
            pytest.param("3.0", 3.0, id="real"),
            pytest.param("1.e30", 1.0e30, id="exponent"),
            pytest.param("1.E-6", 1.0e-6, id="upper-exponent"),
            pytest.param(".5", 0.5, id="leading-point"),
            pytest.param("1e5", 1.0e5, id="no-point"),
            pytest.param("t", True, id="letter"),
            pytest.param(".F.", False, id="dotted-letter"),
            pytest.param("TRUE", True, id="word"),
            pytest.param(".false.", False, id="dotted-word"),
            pytest.param("kg/kg", "kg/kg", id="text"),
            pytest.param("specific humidity", "specific humidity", id="words"),
            pytest.param("'3'", "3", id="quoted"),
            # A letter other than an exponent's makes text of anything.
            pytest.param("1.0d0", "1.0d0", id="d-exponent"),
            pytest.param("inf", "inf", id="inf"),
            pytest.param(".T", ".T", id="half-dotted"),
            # Spellings the field manager would fail to read stay text.
            pytest.param("1.2.3", "1.2.3", id="two-points"),
            pytest.param(".", ".", id="point"),
        ],
    )
    def test_reads_the_type_from_the_spelling(self, text, value) -> None:
        """Test each spelling, bare and in double quotes with padding around it."""
        assert identical_values(VALUE_TYPE_HANDLER_REGISTRY["inferred_text"].from_token(text), value)
        assert identical_values(VALUE_TYPE_HANDLER_REGISTRY["inferred_string"].from_token(f'" {text} "'), value)

    @pytest.mark.parametrize(
        ("value", "accepted"),
        [
            pytest.param(True, True, id="bool"),
            pytest.param(3, True, id="int"),
            pytest.param(2.5, True, id="float"),
            pytest.param("a b", True, id="str"),
            pytest.param(float("nan"), False, id="nan"),
            pytest.param("", False, id="empty"),
            pytest.param(" a", False, id="padded"),
            pytest.param("a,b", False, id="comma"),
            pytest.param("a=b", False, id="equals"),
            pytest.param('a"b', False, id="double-quote"),
            pytest.param("a\nb", False, id="line-break"),
            pytest.param([1, 2], False, id="list"),
            pytest.param(None, False, id="none"),
        ],
    )
    def test_accepts_what_a_spelling_can_hold(self, value, accepted) -> None:
        """Test that only a value the field manager can read back as itself is taken."""
        assert VALUE_TYPE_HANDLER_REGISTRY["inferred_string"].type_check(value) is accepted
        assert VALUE_TYPE_HANDLER_REGISTRY["inferred_text"].type_check(value) is accepted

    @pytest.mark.parametrize(
        ("value", "token", "written"),
        [
            pytest.param(False, '"T"', '"F"', id="logical-letter"),
            pytest.param(False, '".TRUE."', '".FALSE."', id="logical-dotted"),
            pytest.param(2.0e-6, '"1.E-6"', '"2E-06"', id="exponent-letter"),
            pytest.param(4, '"3"', '"4"', id="integer"),
            pytest.param("g", '"  kg/kg "', '"  g "', id="padding"),
            pytest.param("x", "\"'y'\"", "\"'x'\"", id="quoted-stays-quoted"),
        ],
    )
    def test_writes_in_the_notation_it_replaces(self, value, token, written) -> None:
        """Test that a new value is spelled as the one it replaces was."""
        assert VALUE_TYPE_HANDLER_REGISTRY["inferred_string"].to_token(value, token) == written

    @pytest.mark.parametrize(
        ("value", "written"),
        [
            pytest.param("3", "'3'", id="number"),
            pytest.param("T", "'T'", id="logical"),
            pytest.param("'a", "''a'", id="leading-quote"),
            pytest.param("a/", "'a/'", id="closing-slash"),
        ],
    )
    def test_quotes_a_string_that_would_read_back_as_something_else(self, value, written) -> None:
        """Test the single quotes the field manager reads a string from.

        A string ending with ``/`` is quoted only where it would end its line (entry end).
        """
        assert VALUE_TYPE_HANDLER_REGISTRY["inferred_text"].to_new_token(value) == written
        expected = '"a/"' if value == "a/" else f'"{written}"'
        assert VALUE_TYPE_HANDLER_REGISTRY["inferred_string"].to_new_token(value) == expected

    @pytest.mark.parametrize(
        ("value", "token"), [pytest.param("x", "3", id="int-to-str"), pytest.param(1, "t", id="bool-to-int")]
    )
    def test_refuses_a_value_of_another_type(self, value, token) -> None:
        """Test that a value keeps the type its spelling gave it, as every format's does."""
        with pytest.raises(TypeError, match="change value type"):
            VALUE_TYPE_HANDLER_REGISTRY["inferred_text"].to_token(value, token)


class TestNamedAssignments:
    """A name and a control of comma-separated parameters, read as one record."""

    HANDLER = VALUE_TYPE_HANDLER_REGISTRY["named_assignments"]
    RAYLEIGH = '"KaraStr","itable=1333,jtable=953,rayleigh_damp_table=5400"'

    def test_reads_the_name_and_each_parameter(self) -> None:
        """Test a record as an OM2 ocean field table writes one."""
        assert self.HANDLER.from_token(self.RAYLEIGH) == {
            "value": "KaraStr",
            "itable": 1333,
            "jtable": 953,
            "rayleigh_damp_table": 5400,
        }

    def test_a_segment_with_no_equals_sign_continues_a_list(self) -> None:
        """Test ``a=1, 2,3``: the field manager splits the values of ``a`` on its commas."""
        record = self.HANDLER.from_token('"fixed", "a=1, 2,3, b = x"')

        assert record == {"value": "fixed", "a": [1, 2, 3], "b": "x"}

    @pytest.mark.parametrize(
        "control",
        [
            pytest.param("1,a=2", id="comma-before-equals"),
            pytest.param("a=b=c", id="two-equals"),
            pytest.param("a=1,a=2", id="twice"),
            pytest.param("value=1", id="named-value"),
            pytest.param("=1", id="no-name"),
        ],
    )
    def test_refuses_a_control_it_cannot_read(self, control) -> None:
        """Test the controls the field manager refuses, or that would not read as a dict."""
        with pytest.raises(ValueError):
            self.HANDLER.from_token(f'"fixed", "{control}"')

    def test_refuses_a_token_that_is_not_two_strings(self) -> None:
        """Test that only the shape the grammar matches is read."""
        with pytest.raises(ValueError, match="Not a name and a control"):
            self.HANDLER.from_token('"fixed"')

    @pytest.mark.parametrize(
        ("value", "accepted"),
        [
            pytest.param({"value": "fixed", "a": 1, "b": [1, 2.5]}, True, id="record"),
            pytest.param({"value": "fixed"}, False, id="no-parameter"),
            pytest.param({"a": 1}, False, id="no-name"),
            pytest.param({"value": 3, "a": 1}, False, id="numeric-name"),
            pytest.param({"value": "", "a": 1}, False, id="empty-name"),
            pytest.param({"value": 'a"b', "a": 1}, False, id="quote-in-name"),
            pytest.param({"value": "fixed", "a b,": 1}, False, id="comma-in-key"),
            pytest.param({"value": "fixed", " a": 1}, False, id="padded-key"),
            pytest.param({"value": "fixed", "a": [1]}, False, id="list-of-one"),
            pytest.param({"value": "fixed", "a": "x,y"}, False, id="comma-in-value"),
            pytest.param("fixed", False, id="not-a-dict"),
        ],
    )
    def test_accepts_a_record_a_control_can_hold(self, value, accepted) -> None:
        """Test that only what reads back as the same record is taken."""
        assert self.HANDLER.type_check(value) is accepted

    def test_writes_an_unchanged_record_as_it_was(self) -> None:
        """Test that writing a record over itself changes no character of it."""
        token = '"fixed", "a=1, 2,3, b = x"'

        assert self.HANDLER.to_token(self.HANDLER.from_token(token), token) == token

    def test_changes_only_what_changed(self) -> None:
        """Test a changed, a removed and a new parameter, and a renamed record.

        The changed one keeps its padding; the new one is parted like the last two.
        """
        token = '"fixed", "a=1, 2,3, b = x, c=1.E-6"'
        record = {"value": "other", "a": [1, 5], "c": 2.0e-6, "d": "new"}

        assert self.HANDLER.to_token(record, token) == '"other", "a=1, 5, c=2E-06, d=new"'

    def test_a_parameter_changing_type_is_written_afresh(self) -> None:
        """Test that a record, a dict, may hold a parameter of another type than before."""
        assert self.HANDLER.to_token({"value": "f", "a": "x"}, '"f", "a=1"') == '"f", "a=x"'

    def test_a_new_record(self) -> None:
        """Test the seed: a name and a control, parted as the field tables part them."""
        assert self.HANDLER.to_new_token({"value": "fixed", "a": 3.0e-6, "b": "3"}) == '"fixed", "a=3e-06,b=\'3\'"'

    def test_only_a_record_collects(self) -> None:
        """Test that a record is the one value an entry holds one of several of."""
        assert [rule for rule, handler in VALUE_TYPE_HANDLER_REGISTRY.items() if handler.collects] == [
            "named_assignments"
        ]


def test_the_registry_order_states_the_ambiguous_pairs() -> None:
    """Test the order of the six pairs whose members accept the same Python type.

    Without the order the choice would fall to whichever grammar rule matched first.
    """
    order = list(VALUE_TYPE_HANDLER_REGISTRY)

    assert order.index("logical") < order.index("bool")
    assert order.index("float") < order.index("double")
    assert order.index("complex") < order.index("double_complex")
    assert order.index("string") < order.index("identifier")
    assert order.index("identifier") < order.index("text_line")
    assert order.index("inferred_string") < order.index("inferred_text")
