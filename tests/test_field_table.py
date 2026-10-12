# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FMS field table parser.

The first half takes the format a construct at a time: headers, methods, how a value is
typed, and what is refused. The second, the round trip and the edits an experiment makes to
a field table: a restart file renamed, a rayleigh damping record changed, added or removed,
a tracer added or deleted.

The samples are real field tables, trimmed: ACCESS-OM2's 1-degree ocean table, the
header-only table of MOM6 run on its own, written without a final line break, the table FMS
tests its field manager with, and the start of a UFS table, written with Windows line
endings.

Every assertion on the text written out is paired with one that the text reads back as the
configuration it was written from.
"""

import pytest
from conftest import assert_canonical_entry, canonical_rows, chain_rows
from lark.exceptions import UnexpectedInput

from access.config.config_dict import ConfigRecord
from access.config.field_table import FieldTableParser
from access.config.grammar_values import UnsupportedEntryError


def lines(*rows: str, end: str = "\n") -> str:
    """Return *rows* as the lines of a file."""
    return "".join(f"{row}{end}" for row in rows)


# ACCESS-OM2's 1-degree ocean field table, with the rayleigh damping table cut to three
# records: bare assignments, FRE's atmosphere tracers in pairs, and records.
OM2 = lines(
    '"prog_tracers","ocean_mod","temp"',
    "horizontal-advection-scheme = mdppm",
    "vertical-advection-scheme = mdppm",
    "restart_file  = ocean_temp_salt.res.nc",
    "ppm_hlimiter = 3",
    "ppm_vlimiter = 3",
    "/",
    '"prog_tracers","ocean_mod","salt"',
    "horizontal-advection-scheme = mdppm",
    "vertical-advection-scheme = mdppm",
    "restart_file  = ocean_temp_salt.res.nc",
    "ppm_hlimiter = 3",
    "ppm_vlimiter = 3",
    "/",
    '"tracer_packages","ocean_mod","ocean_age_tracer"',
    "names = global",
    "const_init_tracer = .true.",
    "restart_file = ocean_age.res.nc",
    "min_tracer_limit=0.0",
    "/",
    '"namelists","ocean_mod","ocean_age_tracer/global"',
    "slat = -90.0",
    "nlat =  90.0",
    "/",
    "# added by FRE: sphum must be present in atmos",
    "# specific humidity for moist runs",
    ' "TRACER", "atmos_mod", "sphum"',
    '           "longname",     "specific humidity"',
    '           "units",        "kg/kg"',
    '       "profile_type", "fixed", "surface_value=3.e-6" /',
    "# prognotic cloud scheme tracers",
    '  "TRACER", "atmos_mod", "liq_wat"',
    '            "longname",     "cloud liquid specific humidity"',
    '            "units",        "kg/kg" /',
    "# test tracer for radon",
    '# "TRACER", "atmos_mod", "radon"',
    '#           "longname",     "radon test tracer"',
    '#           "units",        "kg/kg" /',
    '       "TRACER", "land_mod", "sphum"',
    '       "longname",     "specific humidity"',
    '       "units",        "kg/kg" /',
    "",
    '"rayleigh_damp_table","ocean_mod","rayleigh_damp_table"',
    '"rayleigh","Lombok","itable=36,jtable=112,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"',
    '"rayleigh","Ombai","itable=44,jtable=111,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"',
    '"rayleigh","Torres","itable=62,jtable=106,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"/',
)

# The rayleigh damping records of OM2, as lines, to state edits to them.
LOMBOK = '"rayleigh","Lombok","itable=36,jtable=112,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"\n'
OMBAI = '"rayleigh","Ombai","itable=44,jtable=111,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"\n'
TORRES = '"rayleigh","Torres","itable=62,jtable=106,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"/\n'

# MOM6's field table when the ocean runs on its own: two entries with no methods, and no
# line break at the end of the file.
MOM6 = '"TRACER", "atmos_mod", "sphum" /\n"TRACER", "land_mod", "sphum" /'

# The table FMS tests its field manager with: headers with and without spaces after the
# commas, a record, an entry with no methods, and a closer written against the last value.
FMS_BASE = lines(
    "# Simplified field table to run the field table unit tests",
    ' "TRACER", "ocean_mod", "biotic1"',
    '           "diff_horiz", "linear", "slope=ok"',
    '           "longname", "biotic one" /',
    ' "TRACER", "ocean_mod", "age_ctl" /',
    ' "TRACER", "atmos_mod","radon"',
    '           "longname","radon-222"',
    '           "units","VMR*1E21"',
    '           "profile_type","fixed","surface_value=0.0E+00"',
    '           "convection","all"/',
    ' "TRACER", "land_mod", "sphum"',
    '           "longname",     "specific humidity"',
    '            "units",        "kg/kg" /',
    "",
)

# The start of a UFS regional atmosphere table, which is written with Windows line endings.
UFS = lines(
    "# added by FRE: sphum must be present in atmos",
    "# specific humidity for moist runs",
    ' "TRACER", "atmos_mod", "sphum"',
    '           "longname",     "specific humidity"',
    '           "units",        "kg/kg"',
    '       "profile_type", "fixed", "surface_value=3.e-6" /',
    "# prognostic cloud water mixing ratio",
    ' "TRACER", "atmos_mod", "liq_wat"',
    '           "longname",     "cloud water mixing ratio"',
    '           "units",        "kg/kg"',
    '       "profile_type", "fixed", "surface_value=1.e30" /',
    end="\r\n",
)


@pytest.fixture(scope="module")
def parser() -> FieldTableParser:
    """Return the parser."""
    return FieldTableParser()


def assert_written(parser: FieldTableParser, config, expected: str) -> None:
    """Assert the text *config* writes, and that it reads back as *config*."""
    assert str(config) == expected
    assert dict(parser.parse(expected)) == dict(config)


class TestHeaders:
    """The header line: a field type, a model and a field, as three levels of blocks."""

    def test_a_header_names_three_levels(self, parser) -> None:
        """Test that each header opens a block for its methods, two levels down."""
        config = parser.parse(OM2)

        assert list(config) == ["prog_tracers", "tracer_packages", "namelists", "tracer", "rayleigh_damp_table"]
        assert config["prog_tracers"]["ocean_mod"]["temp"]["ppm_hlimiter"] == 3

    def test_keys_read_in_lower_case_and_match_in_any(self, parser) -> None:
        """Test the case the field manager reads every name in, whatever the file writes."""
        config = parser.parse(OM2)

        assert "tracer" in dict(config)
        assert config["TRACER"]["ATMOS_MOD"]["SPHUM"] is config["tracer"]["atmos_mod"]["sphum"]

    def test_headers_sharing_a_type_or_a_model_merge(self, parser) -> None:
        """Test that the entries of one type and model read as one block, as in FMS."""
        config = parser.parse(OM2)

        assert list(config["tracer"]) == ["atmos_mod", "land_mod"]
        assert list(config["tracer"]["atmos_mod"]) == ["sphum", "liq_wat"]

    def test_a_field_name_may_hold_a_slash(self, parser) -> None:
        """Test the name of a tracer package's namelist, which holds the package's."""
        assert parser.parse(OM2)["namelists"]["ocean_mod"]["ocean_age_tracer/global"] == {"slat": -90.0, "nlat": 90.0}

    def test_an_entry_with_no_methods(self, parser) -> None:
        """Test a header closed on its own line, which reads as an empty block."""
        assert parser.parse(MOM6) == {"tracer": {"atmos_mod": {"sphum": {}}, "land_mod": {"sphum": {}}}}


class TestMethods:
    """The lines of an entry, in each of the ways the field manager reads one."""

    def test_two_strings_are_a_name_and_a_value(self, parser) -> None:
        """Test the method FRE's atmosphere tables write most."""
        sphum = parser.parse(OM2)["tracer"]["atmos_mod"]["sphum"]

        assert (sphum["longname"], sphum["units"]) == ("specific humidity", "kg/kg")

    def test_a_bare_assignment(self, parser) -> None:
        """Test the method MOM5's ocean tables write, its key holding hyphens."""
        temp = parser.parse(OM2)["prog_tracers"]["ocean_mod"]["temp"]

        assert temp["horizontal-advection-scheme"] == "mdppm"
        assert temp["restart_file"] == "ocean_temp_salt.res.nc"

    def test_three_strings_are_a_record(self, parser) -> None:
        """Test a method with a name and a control, which reads as a list of one record."""
        profile = parser.parse(OM2)["tracer"]["atmos_mod"]["sphum"]["profile_type"]

        assert profile == [{"value": "fixed", "surface_value": 3.0e-6}]
        assert isinstance(profile[0], ConfigRecord)

    def test_the_lines_of_one_method_are_a_list_of_records(self, parser) -> None:
        """Test the rayleigh damping table, a record per line."""
        rayleigh = parser.parse(OM2)["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]

        assert [record["value"] for record in rayleigh] == ["Lombok", "Ombai", "Torres"]
        assert rayleigh[1] == {
            "value": "Ombai",
            "itable": 44,
            "jtable": 111,
            "ktable_1": 1,
            "ktable_2": 50,
            "rayleigh_damp_table": 5400,
        }

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param('  "a = 1, 2" /', id="one-string"),
            pytest.param("  a = 1, 2\n/", id="bare"),
        ],
    )
    def test_an_assignment_of_several_values_is_a_list(self, parser, method) -> None:
        """Test an assignment in one string, or none, holding comma-separated values."""
        assert parser.parse(f'"T", "m", "n"\n{method}\n')["t"]["m"]["n"] == {"a": [1, 2]}

    def test_comments_and_blank_lines_inside_an_entry(self, parser) -> None:
        """Test that a comment starting its line, and a blank line, are not methods."""
        assert parser.parse('"T", "m", "n"\n# note\n\n  "a", "1" /\n')["t"]["m"]["n"] == {"a": 1}


class TestValueTyping:
    """A value takes the type its spelling gives it, as the field manager types one."""

    def test_the_types_a_spelling_gives(self, parser) -> None:
        """Test an integer, a logical, a real and a string, bare and in quotes."""
        config = parser.parse(OM2)
        package = config["tracer_packages"]["ocean_mod"]["ocean_age_tracer"]

        assert package["const_init_tracer"] is True
        assert package["min_tracer_limit"] == 0.0
        assert isinstance(package["min_tracer_limit"], float)
        assert config["prog_tracers"]["ocean_mod"]["temp"]["ppm_hlimiter"] == 3
        assert config["tracer"]["atmos_mod"]["sphum"]["units"] == "kg/kg"

    def test_a_string_in_single_quotes(self, parser) -> None:
        """Test that single quotes make a string of what would otherwise be a number."""
        assert parser.parse('"T", "m", "n"\n  "a", "\'3\'" /\n')["t"]["m"]["n"]["a"] == "3"


class TestRejected:
    """What the field manager would misread or refuse, refused rather than misread."""

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param('"T", "m", "n"\n  "a", "b", "c" /\n', id="three-strings-without-equals"),
            pytest.param('"T", "m", "n"\n  "a", "b,c" /\n', id="comma-in-value"),
            pytest.param('"T", "m", "n"\n  "a", "b=c" /\n', id="equals-in-value"),
            pytest.param('"T", "m", "n"\n  "a", "" /\n', id="empty-value"),
            pytest.param(' # note\n"T", "m", "n" /\n', id="indented-comment"),
            pytest.param('"T", "m", "n"\n  # note\n  "a", "1" /\n', id="indented-comment-in-entry"),
            pytest.param('"T", "m", "n"\n# note /\n', id="comment-ending-the-entry"),
            pytest.param('"T", "m", "n" / extra\n', id="text-after-the-closer"),
            pytest.param('"T", "m" /\n', id="two-string-header"),
            pytest.param('"T" "m" "n" /\n', id="header-without-commas"),
            pytest.param('"T", "m", "n"\n  "a", "1"\n', id="unclosed"),
        ],
    )
    def test_malformed_input(self, parser, source) -> None:
        """Test each shape that does not parse."""
        with pytest.raises(UnexpectedInput):
            parser.parse(source)

    @pytest.mark.parametrize(
        ("source", "match"),
        [
            pytest.param('"T", "m", "n"\n  "a", "b", "1,x=2" /\n', "comma before the first", id="control"),
            pytest.param(
                '"T", "m", "n"\n  "a", "b", "x=1"\n  "a", "2" /\n',
                "both as records and otherwise",
                id="records-and-pair",
            ),
        ],
    )
    def test_entries_that_do_not_read_as_one_configuration(self, parser, source, match) -> None:
        """Test a control the field manager refuses, and a method written two ways."""
        with pytest.raises(ValueError, match=match):
            parser.parse(source)


@pytest.mark.parametrize("source", [OM2, MOM6, FMS_BASE, UFS], ids=["om2", "mom6", "fms", "ufs"])
def test_round_trip(parser, source) -> None:
    """Test that each sample is written back out exactly as it was read."""
    assert str(parser.parse(source)) == source


class TestEditing:
    """Changing a value, which changes its token and nothing else."""

    def test_a_restart_file(self, parser) -> None:
        """Test a bare assignment, its spacing kept."""
        config = parser.parse(OM2)

        config["prog_tracers"]["ocean_mod"]["temp"]["restart_file"] = "new.res.nc"

        assert_written(parser, config, OM2.replace("ocean_temp_salt.res.nc", "new.res.nc", 1))

    def test_one_parameter_of_a_record(self, parser) -> None:
        """Test a rayleigh damping table index, the rest of the line left as it was."""
        config = parser.parse(OM2)
        rayleigh = config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]

        rayleigh[1] = {**rayleigh[1], "itable": 45}

        assert_written(parser, config, OM2.replace(OMBAI, OMBAI.replace("itable=44", "itable=45")))

    def test_the_name_of_a_record(self, parser) -> None:
        """Test the second string of the method, held under ``"value"``."""
        config = parser.parse(OM2)
        rayleigh = config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]

        rayleigh[1] = {**rayleigh[1], "value": "Bali"}

        assert_written(parser, config, OM2.replace(OMBAI, OMBAI.replace("Ombai", "Bali")))

    def test_a_parameter_added_to_and_removed_from_a_record(self, parser) -> None:
        """Test that the control gains and loses a parameter and keeps the others."""
        config = parser.parse(OM2)
        rayleigh = config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]
        profile = config["tracer"]["atmos_mod"]["sphum"]["profile_type"]

        rayleigh[0] = {name: value for name, value in rayleigh[0].items() if name != "ktable_2"}
        profile[0] = {**profile[0], "extra": 1}

        expected = OM2.replace(LOMBOK, LOMBOK.replace("ktable_2=50,", "")).replace(
            '"surface_value=3.e-6"', '"surface_value=3.e-6,extra=1"'
        )
        assert_written(parser, config, expected)

    def test_the_notation_of_a_value_is_kept(self, parser) -> None:
        """Test a logical keeping its dots, and a real its exponent letter."""
        config = parser.parse(OM2)
        profile = config["tracer"]["atmos_mod"]["sphum"]["profile_type"]

        config["tracer_packages"]["ocean_mod"]["ocean_age_tracer"]["const_init_tracer"] = False
        profile[0] = {**profile[0], "surface_value": 1.0e-5}

        expected = OM2.replace(".true.", ".false.").replace("surface_value=3.e-6", "surface_value=1e-05")
        assert_written(parser, config, expected)

    def test_a_string_that_would_read_as_a_number_is_quoted(self, parser) -> None:
        """Test that a string stays one, in the single quotes the field manager reads."""
        config = parser.parse(OM2)

        config["tracer_packages"]["ocean_mod"]["ocean_age_tracer"]["names"] = "3"

        assert_written(parser, config, OM2.replace("names = global", "names = '3'"))

    def test_a_value_of_another_type_is_refused(self, parser) -> None:
        """Test that a value keeps the type its spelling gave it, as in every format."""
        config = parser.parse(OM2)

        with pytest.raises(TypeError, match="change value type"):
            config["prog_tracers"]["ocean_mod"]["temp"]["ppm_hlimiter"] = "x"

        assert str(config) == OM2


class TestAddingAMethod:
    """A key added to an entry, written like the methods around it."""

    def test_in_an_entry_writing_pairs(self, parser) -> None:
        """Test a pair, spaced like the method above it, the closer moved after it."""
        config = parser.parse(OM2)

        config["tracer"]["atmos_mod"]["liq_wat"]["note"] = "x"

        old = '            "units",        "kg/kg" /\n# test'
        new = '            "units",        "kg/kg"\n            "note",        "x" /\n# test'
        assert_written(parser, config, OM2.replace(old, new))

    def test_in_an_entry_writing_assignments(self, parser) -> None:
        """Test a bare assignment, its key holding a hyphen as the others do."""
        config = parser.parse(OM2)

        config["prog_tracers"]["ocean_mod"]["temp"]["ppm-climiter"] = 2

        assert_written(
            parser, config, OM2.replace("ppm_vlimiter = 3\n/\n", "ppm_vlimiter = 3\nppm-climiter = 2\n/\n", 1)
        )

    def test_in_an_entry_with_none(self, parser) -> None:
        """Test the parser's template, used where the entry has no method to copy."""
        config = parser.parse(MOM6)

        config["tracer"]["atmos_mod"]["sphum"]["longname"] = "specific humidity"

        expected = (
            '"TRACER", "atmos_mod", "sphum"\n  "longname", "specific humidity" /\n"TRACER", "land_mod", "sphum" /'
        )
        assert_written(parser, config, expected)

    def test_a_new_list_of_records(self, parser) -> None:
        """Test a method of three strings, written as the first record of a new key."""
        config = parser.parse(FMS_BASE)

        config["tracer"]["ocean_mod"]["biotic1"]["diff_vert"] = [{"value": "linear", "slope": "ok"}]

        old = '           "longname", "biotic one" /\n'
        new = '           "longname", "biotic one"\n           "diff_vert", "linear", "slope=ok" /\n'
        assert_written(parser, config, FMS_BASE.replace(old, new))

    def test_records_added_and_removed(self, parser) -> None:
        """Test that a new record copies its neighbour, and that a removed one goes."""
        config = parser.parse(OM2)
        rayleigh = config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]

        rayleigh.append({**rayleigh[2], "value": "Timor", "itable": 70})
        rayleigh.insert(0, {**rayleigh[0], "value": "Bali", "itable": 30})
        assert rayleigh.pop(2)["value"] == "Ombai"

        bali = LOMBOK.replace("Lombok", "Bali").replace("itable=36", "itable=30")
        timor = TORRES.replace("Torres", "Timor").replace("itable=62", "itable=70")
        expected = OM2.replace(LOMBOK + OMBAI + TORRES, bali + LOMBOK + TORRES.replace("/\n", "\n") + timor)
        assert_written(parser, config, expected)

    def test_with_windows_line_endings(self, parser) -> None:
        """Test that a new line ends as the others do."""
        config = parser.parse(UFS)

        config["tracer"]["atmos_mod"]["liq_wat"]["note"] = "x"

        old = '"surface_value=1.e30" /\r\n'
        new = '"surface_value=1.e30"\r\n       "note", "x" /\r\n'
        assert_written(parser, config, UFS.replace(old, new))


class TestAddingAnEntry:
    """A field added: a whole header, written like the one above it, then its methods."""

    def test_a_new_field_of_a_model(self, parser) -> None:
        """Test the header after the last of its type and model, spelled the way it is.

        The entry above it is followed by comments, which stay with it. The methods are
        spaced like those of the entry above.
        """
        config = parser.parse(OM2)
        atmosphere = config["tracer"]["atmos_mod"]

        atmosphere["radon"] = {"longname": "radon test tracer", "units": "VMR*1E21"}

        old = '#           "units",        "kg/kg" /\n'
        new = old + (
            '  "TRACER", "atmos_mod", "radon"\n'
            '            "longname",        "radon test tracer"\n'
            '            "units",        "VMR*1E21" /\n'
        )
        assert_written(parser, config, OM2.replace(old, new))
        # A reference held to the model sees the new field.
        assert atmosphere["radon"] == {"longname": "radon test tracer", "units": "VMR*1E21"}

    def test_a_new_field_with_no_methods(self, parser) -> None:
        """Test a header closed on its own line, in a file of nothing else."""
        config = parser.parse(MOM6)

        config["tracer"]["atmos_mod"]["liq_wat"] = {}

        expected = (
            '"TRACER", "atmos_mod", "sphum" /\n"TRACER", "atmos_mod", "liq_wat" /\n"TRACER", "land_mod", "sphum" /'
        )
        assert_written(parser, config, expected)

    def test_a_new_type(self, parser) -> None:
        """Test a type the file does not have, written down to its methods."""
        config = parser.parse(OM2)

        config["diag_tracers"] = {"ocean_mod": {"x": {"units": "m"}}}

        assert_written(parser, config, OM2 + '"diag_tracers","ocean_mod","x"\n"units","m"/\n')

    def test_a_type_with_nothing_in_it(self, parser) -> None:
        """Test that a header needs a field, which an empty dict does not name."""
        config = parser.parse(OM2)

        with pytest.raises(UnsupportedEntryError):
            config["empty"] = {}

        assert str(config) == OM2


class TestDeleting:
    """Removing a method, a record, a field, a model or a type."""

    def test_the_methods_of_an_entry(self, parser) -> None:
        """Test that the closer moves back onto the header once the last method goes."""
        config = parser.parse(OM2)
        liquid = config["tracer"]["atmos_mod"]["liq_wat"]

        del liquid["units"]
        del liquid["longname"]

        old = (
            '  "TRACER", "atmos_mod", "liq_wat"\n'
            '            "longname",     "cloud liquid specific humidity"\n'
            '            "units",        "kg/kg" /\n'
        )
        assert_written(parser, config, OM2.replace(old, '  "TRACER", "atmos_mod", "liq_wat" /\n'))

    def test_a_list_of_records(self, parser) -> None:
        """Test that deleting the key removes every line writing it."""
        config = parser.parse(OM2)

        del config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]

        old = '"rayleigh_damp_table","ocean_mod","rayleigh_damp_table"\n' + LOMBOK + OMBAI + TORRES
        assert_written(parser, config, OM2.replace(old, '"rayleigh_damp_table","ocean_mod","rayleigh_damp_table"/\n'))

    def test_a_field(self, parser) -> None:
        """Test that the whole entry goes, the one beside it staying."""
        config = parser.parse(OM2)

        del config["prog_tracers"]["ocean_mod"]["salt"]

        salt = OM2[OM2.index('"prog_tracers","ocean_mod","salt"') : OM2.index('"tracer_packages"')]
        assert_written(parser, config, OM2.replace(salt, ""))

    def test_a_merged_model_then_its_type(self, parser) -> None:
        """Test deleting through blocks merged from several entries, comments staying."""
        config = parser.parse(OM2)

        del config["tracer"]["atmos_mod"]
        del config["tracer"]

        expected = OM2
        for entry in (
            ' "TRACER", "atmos_mod", "sphum"\n',
            '           "longname",     "specific humidity"\n',
            '           "units",        "kg/kg"\n',
            '       "profile_type", "fixed", "surface_value=3.e-6" /\n',
            '  "TRACER", "atmos_mod", "liq_wat"\n',
            '            "longname",     "cloud liquid specific humidity"\n',
            '            "units",        "kg/kg" /\n',
            '       "TRACER", "land_mod", "sphum"\n',
            '       "longname",     "specific humidity"\n',
            '       "units",        "kg/kg" /\n',
        ):
            expected = expected.replace(entry, "", 1)
        assert_written(parser, config, expected)
        assert "tracer" not in config


def test_a_whole_table_edited(parser) -> None:
    """Test the edits an experiment makes, together, on the OM2 table.

    A restart file renamed, a rayleigh record removed and another changed, and an age
    tracer added after the temperature and salinity ones.
    """
    config = parser.parse(OM2)
    ocean = config["prog_tracers"]["ocean_mod"]
    rayleigh = config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]

    ocean["salt"]["restart_file"] = "salt.res.nc"
    rayleigh.remove(rayleigh[0])
    rayleigh[-1] = {**rayleigh[-1], "jtable": 107}
    ocean["age"] = {"restart_file": "age.res.nc", "ppm_hlimiter": 3}

    expected = (
        OM2.replace("restart_file  = ocean_temp_salt.res.nc", "restart_file  = salt.res.nc")
        .replace("restart_file  = salt.res.nc", "restart_file  = ocean_temp_salt.res.nc", 1)
        .replace(LOMBOK, "")
        .replace("jtable=106", "jtable=107")
        .replace(
            '/\n"tracer_packages"',
            '/\n"prog_tracers","ocean_mod","age"\nrestart_file = age.res.nc\nppm_hlimiter = 3\n/\n"tracer_packages"',
        )
    )
    assert_written(parser, config, expected)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("field_table"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text of a new method with no neighbour to copy, as the grammar ranks it."""
    assert_canonical_entry(parser, container, category, expected)


@pytest.mark.parametrize(("container", "category", "depth", "expected"), chain_rows("field_table"))
def test_canonical_chain_entry_text(parser, container, category, depth, expected) -> None:
    """Test the text of a new header with no neighbour to copy."""
    assert_canonical_entry(parser, container, category, expected, keys=("A", "B", "C")[:depth])
