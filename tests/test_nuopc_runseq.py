# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NUOPC run sequence parser.

The first half takes the grammar a rule at a time; the second, the round trip and the edits
a run sequence gets: a coupling time step changed, rows inserted and removed, a whole
sequence replaced by another.

The CMEPS samples are written out line by line rather than as one literal, because CMEPS
ends some lines with spaces, which an editor stripping trailing whitespace would remove
from a literal without anyone noticing.
"""

import pytest
from conftest import TEXT_VALUES, assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedInput

from access.config.grammar_compiled import compile_grammar
from access.config.grammar_values import UnsupportedEntryError
from access.config.nuopc_runseq import NUOPCRunSeqParser


def lines(*rows: str) -> str:
    """Return *rows* as the lines of a file."""
    return "".join(f"{row}\n" for row in rows)


# The run sequence CMEPS generates for ACCESS-OM3's MOM6-CICE6 configuration.
MOM6_CICE6 = lines(
    "runSeq:: ",
    "@3600 ",
    "  MED med_phases_aofluxes_run",
    "  MED med_phases_prep_ocn_accum",
    "  MED med_phases_ocnalb_run",
    "  MED med_phases_diag_ocn",
    "  MED med_phases_prep_ice",
    "  MED -> ICE :remapMethod=redist",
    "  MED med_phases_prep_ocn_avg",
    "  MED -> OCN :remapMethod=redist",
    "  ICE",
    "  ROF",
    "  OCN",
    "  OCN -> MED :remapMethod=redist",
    "  MED med_phases_post_ocn",
    "  MED med_phases_diag_rof",
    "  MED med_phases_diag_ice_ice2med",
    "  ICE -> MED :remapMethod=redist",
    "  MED med_phases_post_ice",
    "  ATM",
    "  ATM -> MED :remapMethod=redist",
    "  MED med_phases_post_atm",
    "  MED med_phases_diag_atm",
    "  MED med_phases_diag_ice_med2ice",
    "  ROF -> MED :remapMethod=redist",
    "  MED med_phases_post_rof",
    "  MED med_phases_diag_accum",
    "  MED med_phases_diag_print",
    "  MED med_phases_history_write        ",
    "  MED med_phases_restart_write        ",
    "  MED med_phases_profile              ",
    "@ ",
    "::",
)

# The same configuration with WW3 coupled in, as CMEPS generates it.
MOM6_CICE6_WW3 = lines(
    "runSeq:: ",
    "@3600 ",
    "  MED med_phases_aofluxes_run",
    "  MED med_phases_prep_ocn_accum",
    "  MED med_phases_ocnalb_run",
    "  MED med_phases_diag_ocn",
    "  MED med_phases_prep_ice",
    "  MED -> ICE :remapMethod=redist",
    "  MED med_phases_prep_wav_accum",
    "  MED med_phases_prep_wav_avg",
    "  MED -> WAV :remapMethod=redist",
    "  MED med_phases_prep_ocn_avg",
    "  MED -> OCN :remapMethod=redist",
    "  ICE",
    "  WAV",
    "  ROF",
    "  OCN",
    "  OCN -> MED :remapMethod=redist",
    "  MED med_phases_post_ocn",
    "  MED med_phases_diag_rof",
    "  MED med_phases_diag_ice_ice2med",
    "  ICE -> MED :remapMethod=redist",
    "  MED med_phases_post_ice",
    "  ATM",
    "  ATM -> MED :remapMethod=redist",
    "  MED med_phases_post_atm",
    "  MED med_phases_diag_atm",
    "  MED med_phases_diag_ice_med2ice",
    "  ROF -> MED :remapMethod=redist",
    "  MED med_phases_post_rof",
    "  WAV -> MED :remapMethod=redist",
    "  MED med_phases_post_wav",
    "  MED med_phases_diag_accum",
    "  MED med_phases_diag_print",
    "  MED med_phases_history_write        ",
    "  MED med_phases_restart_write        ",
    "  MED med_phases_profile              ",
    "@ ",
    "::",
)

# Nested loops indented by depth, as the UFS coupled templates write them, with a comment
# and a blank line inside the table.
NESTED = lines(
    "runSeq::",
    "@3600   # slow coupling",
    "   MED med_phases_prep_ocn_avg",
    "   OCN",
    "   @1800",
    "",
    "     ATM",
    "   @",
    "   OCN -> MED :remapMethod=redist",
    "@",
    "::",
)


@pytest.fixture(scope="module")
def parser() -> NUOPCRunSeqParser:
    """Fixture instantiating the parser."""
    return NUOPCRunSeqParser()


def test_the_samples_keep_their_trailing_spaces() -> None:
    """Test that the samples hold the spaces CMEPS writes, so a round trip tests them."""
    assert MOM6_CICE6.count(" \n") == 6
    assert MOM6_CICE6_WW3.count(" \n") == 6


class TestTables:
    """``table_list``, ``table_value``, ``table_null``: the lines decide the category."""

    def test_several_lines_are_a_list(self, parser) -> None:
        """Test that each line of a table is one element, kept whole."""
        config = parser.parse(MOM6_CICE6)

        assert len(config["runSeq"]) == 31
        assert config["runSeq"][:3] == ["@3600", "MED med_phases_aofluxes_run", "MED med_phases_prep_ocn_accum"]
        assert config["runSeq"][-1] == "@"

    def test_one_line_is_a_string(self, parser) -> None:
        """Test that a single line reads as a scalar, as a single value does elsewhere."""
        assert parser.parse("runSeq::\n  OCN\n::\n")["runSeq"] == "OCN"

    def test_no_line_is_none(self, parser) -> None:
        """Test that an empty table is a valueless entry."""
        assert parser.parse("runSeq::\n::\n")["runSeq"] is None

    def test_several_tables(self, parser) -> None:
        """Test that a file may hold more than one table, each under its own label."""
        config = parser.parse("a::\n  x\n  y\n::\n\n  b::\n  z\n  ::\n")

        assert dict(config) == {"a": ["x", "y"], "b": "z"}

    def test_labels_are_case_sensitive(self, parser) -> None:
        """Test that ``runSeq`` and ``RUNSEQ`` are different labels, as they are to ESMF."""
        config = parser.parse("runSeq::\n  OCN\n::\n")

        assert "runSeq" in config
        assert "RUNSEQ" not in config


class TestLines:
    """``text_line``: what a line holds, and what around it is not part of it."""

    @pytest.mark.parametrize(
        "row",
        [
            "@3600",
            "@36000:72000",
            "@@600",
            "@",
            "OCN",
            "MED med_phases_post_ocn",
            "MED -> ICE :remapMethod=redist:srcTermProcessing=1:termOrder=srcseq",
        ],
    )
    def test_any_run_sequence_element(self, parser, row) -> None:
        """Test loops, alarms, components, phases and connectors, read as written."""
        source = f"runSeq::\n  {row}\n  OCN\n::\n"
        config = parser.parse(source)

        assert config["runSeq"] == [row, "OCN"]
        assert str(config) == source

    def test_indentation_trailing_space_and_comment_are_not_part_of_a_line(self, parser) -> None:
        """Test that only the text of the element is the value."""
        config = parser.parse("runSeq::\n\t  @3600  # every hour\n  OCN   \n::\n")

        assert config["runSeq"] == ["@3600", "OCN"]

    def test_comment_and_blank_lines_are_not_elements(self, parser) -> None:
        """Test that the lines between two elements are kept but not read."""
        config = parser.parse(NESTED)

        assert config["runSeq"] == [
            "@3600",
            "MED med_phases_prep_ocn_avg",
            "OCN",
            "@1800",
            "ATM",
            "@",
            "OCN -> MED :remapMethod=redist",
            "@",
        ]


class TestRejected:
    """Text the grammar refuses rather than misreads."""

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("junk\nrunSeq::\n  OCN\n::\n", id="text-outside-a-table"),
            pytest.param("runSeq::\n  @3600\n  OCN\n", id="unclosed-table"),
            pytest.param("runSeq: OCN\n", id="label-without-a-table"),
        ],
    )
    def test_malformed_input(self, parser, source) -> None:
        """Test that each is a parse error."""
        with pytest.raises(UnexpectedInput):
            parser.parse(source)


class TestRoundTripFidelity:
    """Text written back exactly as it was read."""

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(MOM6_CICE6, id="mom6-cice6"),
            pytest.param(MOM6_CICE6_WW3, id="mom6-cice6-ww3"),
            pytest.param(NESTED, id="nested"),
            pytest.param(MOM6_CICE6.replace("\n", "\r\n"), id="crlf"),
            pytest.param(MOM6_CICE6.rstrip("\n"), id="no-final-newline"),
            pytest.param("", id="empty"),
            pytest.param("# just a comment\n\n", id="no-table"),
        ],
    )
    def test_round_trip(self, parser, source) -> None:
        """Test that what was read is written back unchanged."""
        assert str(parser.parse(source)) == source


class TestEditingALine:
    """Writing one line back, changing that and nothing else."""

    def test_the_coupling_time_step(self, parser) -> None:
        """Test that the loop line keeps the trailing space CMEPS wrote after it."""
        config = parser.parse(MOM6_CICE6)

        config["runSeq"][0] = "@1800"

        assert str(config) == MOM6_CICE6.replace("@3600 \n", "@1800 \n")

    def test_a_connector(self, parser) -> None:
        """Test that a line in the middle changes alone, its indentation kept."""
        config = parser.parse(NESTED)

        config["runSeq"][6] = "OCN -> MED :remapMethod=bilinear"

        assert str(config) == NESTED.replace("remapMethod=redist", "remapMethod=bilinear")


class TestResizing:
    """Lines inserted and removed, and whole sequences replaced."""

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            pytest.param(MOM6_CICE6, MOM6_CICE6_WW3, id="add-ww3"),
            pytest.param(MOM6_CICE6_WW3, MOM6_CICE6, id="remove-ww3"),
        ],
    )
    def test_one_sequence_assigned_over_another(self, parser, source, target) -> None:
        """Test that the lines both have stay exactly as written, spaces included."""
        config = parser.parse(source)

        config["runSeq"] = list(parser.parse(target)["runSeq"])

        assert str(config) == target

    def test_a_line_inserted_after_the_loop_opener_is_indented(self, parser) -> None:
        """Test that a new line is indented like the loop's body, not like the loop line."""
        config = parser.parse(MOM6_CICE6)

        config["runSeq"].insert(1, "MED med_phases_new")

        assert str(config) == MOM6_CICE6.replace("@3600 \n", "@3600 \n  MED med_phases_new\n")

    def test_a_line_inserted_before_the_loop_closer_is_indented(self, parser) -> None:
        """Test the other end of the loop, where the closer is shallower than the body."""
        config = parser.parse(MOM6_CICE6)

        config["runSeq"].insert(-1, "MED med_phases_new")

        assert str(config) == MOM6_CICE6.replace("@ \n", "  MED med_phases_new\n@ \n")

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            pytest.param(4, NESTED.replace("   @1800\n", "   @1800\n     ICE\n"), id="after-inner-opener"),
            pytest.param(5, NESTED.replace("     ATM\n", "     ATM\n     ICE\n"), id="before-inner-closer"),
        ],
    )
    def test_a_line_inside_a_nested_loop(self, parser, index, expected) -> None:
        """Test that a line added inside the inner loop takes that loop's indentation."""
        config = parser.parse(NESTED)

        config["runSeq"].insert(index, "ICE")

        assert str(config) == expected

    def test_removing_a_line_keeps_the_lines_around_it(self, parser) -> None:
        """Test that the comment and blank lines in the table stay where they were."""
        config = parser.parse(NESTED)

        config["runSeq"].remove("OCN")

        assert str(config) == NESTED.replace("   OCN\n", "", 1)

    def test_a_new_line_ends_like_the_others(self, parser) -> None:
        """Test that a file with CRLF line endings gains a CRLF line."""
        source = MOM6_CICE6.replace("\n", "\r\n")
        config = parser.parse(source)

        config["runSeq"].insert(1, "OCN")

        assert str(config) == source.replace("@3600 \r\n", "@3600 \r\n  OCN\r\n")

    def test_a_table_keeps_two_lines(self, parser) -> None:
        """Test that a list cannot become a single line in place."""
        config = parser.parse("runSeq::\n  ATM\n  OCN\n::\n")

        with pytest.raises(ValueError, match="at least two elements"):
            config["runSeq"].pop()

    @pytest.mark.parametrize("value", [3600, "OCN # c", "  OCN", ":: OCN"])
    def test_a_value_that_is_not_a_line_of_text(self, parser, value) -> None:
        """Test that what would not read back as the same line is refused."""
        config = parser.parse("runSeq::\n  ATM\n  OCN\n::\n")

        with pytest.raises(UnsupportedEntryError):
            config["runSeq"].append(value)
        assert str(config) == "runSeq::\n  ATM\n  OCN\n::\n"


class TestAddingAndDeleting:
    """Whole tables added and removed."""

    @pytest.mark.parametrize(
        ("value", "added"),
        [
            pytest.param(["@1800", "ATM", "@"], "other::\n@1800\nATM\n@\n::\n", id="list"),
            pytest.param("ATM", "other::\nATM\n::\n", id="single-line"),
            pytest.param(None, "other::\n::\n", id="empty"),
        ],
    )
    def test_adding_a_table(self, parser, value, added) -> None:
        """Test that a new table is written after the last.

        Neither table above is copied: the only one is a list, whose rows are indented by
        how deep in the loops each is, which says nothing about the rows of another.
        """
        source = "runSeq::\n@3600\n  OCN\n@\n::\n"
        config = parser.parse(source)

        config["other"] = value

        assert str(config) == source + added
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_one_line_table_copies_the_one_above(self, parser) -> None:
        """Test that a table of one row is written like the last such table, indent too."""
        config = parser.parse("runSeq::\n  @3600\n::\n")

        config["other"] = "ATM"

        assert str(config) == "runSeq::\n  @3600\n::\nother::\n  ATM\n::\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_deleting_a_table(self, parser) -> None:
        """Test that the table goes, and the blank line before it stays."""
        config = parser.parse("runSeq::\n@3600\n  OCN\n@\n::\n\nother::\n  ATM\n::\n")

        del config["other"]

        assert str(config) == "runSeq::\n@3600\n  OCN\n@\n::\n\n"


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("nuopc_runseq"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text of a new table, with no neighbour to copy a style from."""
    assert_canonical_entry(compile_grammar(parser.grammar), container, category, expected, TEXT_VALUES)
