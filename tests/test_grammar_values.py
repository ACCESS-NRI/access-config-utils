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
    VALUE_RULE_PRIORITY,
    VALUE_TYPE_HANDLER_REGISTRY,
    UnsupportedEntryError,
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
    ("path", "/dir/file.nc", Path("/dir/file.nc"), "/dir/file.nc"),
]


@pytest.mark.parametrize(("rule", "text", "value", "written"), ROUND_TRIPS)
def test_handler_round_trip(rule, text, value, written) -> None:
    """Test three of the four operations of one handler: check, parse, and seed.

    ``to_new_token`` is the path taken for a key that does not exist yet, where there is no
    previous token to preserve notation from, so it is what fixes how a brand-new value of
    each type is spelled. ``to_token`` is the fourth and is not asserted here, because what
    it writes depends on the token being replaced -- see the two classes below.
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

        The checks are ``type(value) is ...`` rather than ``isinstance`` for exactly this:
        ``True`` written as ``1`` would read back as an integer, silently changing the type.
        """
        assert not VALUE_TYPE_HANDLER_REGISTRY[rule].type_check(True)

    def test_an_identifier_must_be_a_bare_word(self) -> None:
        """Test that only a string usable unquoted as a name is an ``identifier``."""
        handler = VALUE_TYPE_HANDLER_REGISTRY["identifier"]

        assert handler.type_check("a_word")
        assert not handler.type_check("a string")
        assert not handler.type_check("1")
        assert not handler.type_check(Path("/x"))

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

        Doubling means the quote in use never has to change, so a value containing both is
        representable rather than an error.
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

        ``_float_to_str`` substitutes the marker into ``str(value)``, so a value whose repr
        has no ``e`` comes out plain. Both forms are valid: ``config.lark`` makes the
        exponent optional in ``DOUBLE``, so ``0.01`` parses as a double just as ``1d-20``
        does.
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

        Python writes them as ``inf`` and ``nan``, which no supported format reads back as a
        number, so writing one would produce a file that no longer means the same thing.
        """
        with pytest.raises(UnsupportedEntryError, match="cannot be represented"):
            VALUE_TYPE_HANDLER_REGISTRY[rule].to_token(value, "1.0")

    @pytest.mark.parametrize("rule", ["complex", "double_complex"])
    def test_a_complex_refuses_a_non_finite_part(self, rule) -> None:
        """Test that the check reaches both halves of a complex value."""
        with pytest.raises(UnsupportedEntryError):
            VALUE_TYPE_HANDLER_REGISTRY[rule].to_token(complex(math.inf, 0.0), "(1.0, 2.0)")


class TestSelectValueRule:
    """Choosing the rule to write a value with, among those the grammar admits."""

    def test_takes_the_first_admitted_rule_that_accepts_the_value(self) -> None:
        """Test that the priority order decides, filtered by what the grammar allows."""
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
        assert select_value_rule(1, {"not_a_value_rule"}, ("not_a_value_rule", "integer")) is None

    def test_a_caller_may_reverse_the_order(self) -> None:
        """Test that the priority is an argument, which is how a parser overrides it."""
        assert select_value_rule("word", {"identifier", "string"}) == "string"
        assert select_value_rule("word", {"identifier", "string"}, ("identifier", "string")) == "identifier"


def test_priority_covers_every_registered_rule() -> None:
    """Test that no value type is left unreachable when a new key is written.

    A rule in the registry but missing from the priority order would parse fine and never be
    chosen for a new entry, which is the kind of gap that shows up only as odd output.
    """
    assert set(VALUE_RULE_PRIORITY) == set(VALUE_TYPE_HANDLER_REGISTRY)


def test_priority_states_the_ambiguous_pairs() -> None:
    """Test the order of the four pairs whose members accept the same Python type.

    These are the whole reason the order exists: without it the choice would fall to
    whichever grammar rule happened to match first.
    """
    order = list(VALUE_RULE_PRIORITY)

    assert order.index("logical") < order.index("bool")
    assert order.index("float") < order.index("double")
    assert order.index("complex") < order.index("double_complex")
    assert order.index("string") < order.index("identifier")
