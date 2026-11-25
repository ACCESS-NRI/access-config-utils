# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rose configuration file format.

Two halves. The first has a test per rule of the grammar, or per assumption its comments
state -- what a value is, how it goes on over the lines below it, which section the comment
above a header belongs to. Grouping them by rule is what makes an untested construct visible
as an empty spot rather than an absence nobody notices. What a file reads as is checked
against what ``ConfigLoader.load`` of Rose 2019.01 reads, which the module docstring of
``rose_conf`` sums up.

The second half covers the round trip: reading a file, changing it, and writing it back with
that change and nothing else. Almost every test asserts byte-exact output as well as the
value read, because for this package the text is half the result.

What is worth naming about this format: every value is a string, a section runs until the
next header, and the marks of an ignored setting are part of its key. The samples are made
up, in the shape of the files of a Unified Model suite.
"""

import pytest
from conftest import TEXT_VALUES, assert_canonical_entry, canonical_rows
from lark.exceptions import UnexpectedInput

from access.config.grammar_values import UnsupportedEntryError
from access.config.rose_conf import RoseConfParser


@pytest.fixture(scope="module")
def parser() -> RoseConfParser:
    """Fixture instantiating the parser."""
    return RoseConfParser()


# =============================================================================
# The grammar
# =============================================================================


class TestSettings:
    """``key_value`` and ``key_null``, a key and the text after its equals sign."""

    def test_an_equals_assigns_the_rest_of_the_line(self, parser) -> None:
        """Test that a value is text, whatever it looks like: a number stays a string."""
        source = "MAIN_ATM_PROCX=24\nMAIN_CLOCK='PT1H10M'\nltimer=.true.\n"
        config = parser.parse(source)

        assert dict(config) == {"MAIN_ATM_PROCX": "24", "MAIN_CLOCK": "'PT1H10M'", "ltimer": ".true."}
        assert str(config) == source

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("*=/scratch/tm70/x", "*=/scratch/tm70/x"),
            ("1 # not a comment", "1 # not a comment"),
            ("#not a comment either", "#not a comment either"),
            ("a=b", "a=b"),
            ("18,16", "18,16"),
            ("$HOME/cylc-run", "$HOME/cylc-run"),
        ],
    )
    def test_a_value_keeps_every_character(self, parser, text, expected) -> None:
        """Test that a value runs to the end of its line, ``=`` and ``#`` included.

        Only a ``#`` that starts a line is a comment.
        """
        source = f"root-dir={text}\n"
        config = parser.parse(source)

        assert config["root-dir"] == expected
        assert str(config) == source

    def test_a_value_is_trimmed(self, parser) -> None:
        """Test that the blanks around a value are not part of it, as Rose strips them."""
        source = "A =  1 2  \n"
        config = parser.parse(source)

        assert config["A"] == "1 2"
        assert str(config) == source

    def test_a_setting_before_the_first_section_is_top_level(self, parser) -> None:
        """Test that the settings above every header belong to no section."""
        source = "root-dir=*=/x\n[jinja2:suite.rc]\nA=1\n"
        config = parser.parse(source)

        assert dict(config) == {"root-dir": "*=/x", "jinja2:suite.rc": {"A": "1"}}
        assert str(config) == source

    @pytest.mark.parametrize("key", ["root-dir{share}", "prune{log}", "MAIN_ATM_PROCX", "rg01_rs01_m01_nproc", "meta"])
    def test_a_key_is_not_an_identifier(self, parser, key) -> None:
        """Test that a key holds anything but whitespace and an equals sign."""
        source = f"{key}=x\n"
        config = parser.parse(source)

        assert config[key] == "x"
        assert str(config) == source

    def test_a_setting_with_nothing_after_the_equals_is_null(self, parser) -> None:
        """Test ``key_null``: ``KEY=`` reads as ``None``, the empty string Rose reads."""
        source = "A=\nB =  \n"
        config = parser.parse(source)

        assert dict(config) == {"A": None, "B": None}
        assert str(config) == source

    def test_a_key_written_twice_keeps_the_last_value(self, parser) -> None:
        """Test that the later of two settings of a key is the one read, as in Rose."""
        source = "A=1\nA=2\n"
        config = parser.parse(source)

        assert config["A"] == "2"
        assert str(config) == source


class TestSections:
    """``key_block``: a header starts a section, which runs until the next one."""

    def test_a_section_holds_the_settings_up_to_the_next_header(self, parser) -> None:
        """Test that nothing closes a section: the next header ends it."""
        source = "[a]\nx=1\ny=2\n[b]\nz=3\n"
        config = parser.parse(source)

        assert dict(config) == {"a": {"x": "1", "y": "2"}, "b": {"z": "3"}}
        assert str(config) == source

    @pytest.mark.parametrize(
        "name",
        [
            "jinja2:suite.rc",
            "namelist:nlsizes",
            "namelist:items(abc123)",
            "file:app/um/input.nml",
            "env",
            "ana:grepper.FilePattern(A test of one)",
            "namelist:x=y",
        ],
    )
    def test_a_section_name_is_not_an_identifier(self, parser, name) -> None:
        """Test names holding colons, brackets, slashes, blanks and an equals sign."""
        source = f"[{name}]\nk=v\n"
        config = parser.parse(source)

        assert dict(config) == {name: {"k": "v"}}
        assert str(config) == source

    def test_a_section_may_be_empty(self, parser) -> None:
        """Test that a header followed by nothing is still a section."""
        source = "[a]\n\n[b]\n"
        config = parser.parse(source)

        assert dict(config) == {"a": {}, "b": {}}
        assert str(config) == source

    def test_a_section_written_twice_reads_as_one(self, parser) -> None:
        """Test that the settings of both are read, the later value of a key winning."""
        source = "[a]\nx=1\ny=2\n[b]\n[a]\nx=3\n"
        config = parser.parse(source)

        assert dict(config["a"]) == {"x": "3", "y": "2"}
        assert str(config) == source

    def test_no_key_can_be_added_to_a_section_written_twice(self, parser) -> None:
        """Test that a key has nowhere to go in a section written in two places."""
        source = "[a]\nx=1\n[a]\ny=2\n"
        config = parser.parse(source)

        with pytest.raises(UnsupportedEntryError):
            config["a"]["z"] = "3"

        assert str(config) == source


class TestContinuationLines:
    """``rose_text``: a value goes on over the indented lines below it."""

    def test_an_indented_line_continues_the_value(self, parser) -> None:
        """Test that each continuation line is trimmed and joined after a line break."""
        source = "[jinja2:suite.rc]\nSITES='nci',\n      'other'  \nNEXT=1\n"
        config = parser.parse(source)

        assert dict(config["jinja2:suite.rc"]) == {"SITES": "'nci',\n'other'", "NEXT": "1"}
        assert str(config) == source

    def test_one_leading_equals_is_dropped_and_the_blanks_after_it_kept(self, parser) -> None:
        """Test the equals sign that keeps the blanks a continuation line starts with."""
        source = "meta=line 1\n    =    line 2\n    ==line 3\n"
        config = parser.parse(source)

        assert config["meta"] == "line 1\n    line 2\n=line 3"
        assert str(config) == source

    def test_a_bare_equals_is_an_empty_line(self, parser) -> None:
        """Test that a continuation line of a lone equals sign adds an empty line."""
        source = "meta=one\n   =\n   =three\n"
        config = parser.parse(source)

        assert config["meta"] == "one\n\nthree"
        assert str(config) == source

    def test_blank_and_comment_lines_do_not_end_a_value(self, parser) -> None:
        """Test that the lines between two continuation lines add nothing to the value."""
        source = "root-dir=hpc*=$WORKDIR\n\n  # elsewhere\n# not indented\n        =*=$DATADIR\n"
        config = parser.parse(source)

        assert config["root-dir"] == "hpc*=$WORKDIR\n*=$DATADIR"
        assert str(config) == source

    def test_a_value_may_start_on_a_continuation_line(self, parser) -> None:
        """Test a setting whose own line ends at the equals sign, its value below it."""
        source = "root-dir=\n    =*=/scratch\n"
        config = parser.parse(source)

        assert config["root-dir"] == "\n*=/scratch"
        assert str(config) == source

    def test_an_indented_header_continues_a_value(self, parser) -> None:
        """Test that any indented line after a setting continues it, as Rose reads one."""
        source = "x=1\n  [s]\n"
        config = parser.parse(source)

        assert dict(config) == {"x": "1\n[s]"}
        assert str(config) == source


class TestIgnoredState:
    """``!`` and ``!!``, which mark a setting or a section ignored."""

    def test_the_marks_are_part_of_the_key(self, parser) -> None:
        """Test that a setting and a section keep their marks in their keys."""
        source = "[jinja2:suite.rc]\n!!MAIN_IOS_NPROC=0\n!USER_SET=1\n[!namelist:x]\na=1\n[!!namelist:y]\n"
        config = parser.parse(source)

        assert dict(config) == {
            "jinja2:suite.rc": {"!!MAIN_IOS_NPROC": "0", "!USER_SET": "1"},
            "!namelist:x": {"a": "1"},
            "!!namelist:y": {},
        }
        assert str(config) == source

    def test_an_ignored_setting_is_not_the_setting(self, parser) -> None:
        """Test that a setting Rose ignores is not found under the key Rose would use."""
        config = parser.parse("[s]\n!!A=0\n")

        assert "A" not in config["s"]
        assert "!!A" in config["s"]

    def test_a_new_key_does_not_copy_the_marks_of_its_neighbour(self, parser) -> None:
        """Test that a setting added beside an ignored one is a setting in use.

        The neighbour is copied with its key replaced, and the marks are part of the key.
        """
        config = parser.parse("[s]\n!!A=0\n")

        config["s"]["A"] = "4"

        assert str(config) == "[s]\n!!A=0\nA=4\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_section_does_not_copy_the_marks_of_its_neighbour(self, parser) -> None:
        """Test that a section added after an ignored one is a section in use."""
        config = parser.parse("[!s]\nk=0\n")

        config["t"] = {"a": "1"}

        assert str(config) == "[!s]\nk=0\n[t]\na=1\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_an_ignored_setting_can_be_added_and_edited(self, parser) -> None:
        """Test that an ignored setting is written as any other, under its marked key."""
        config = parser.parse("[s]\n!!A=0\n")

        config["s"]["!!A"] = "2"
        config["s"]["!B"] = "3"

        assert str(config) == "[s]\n!!A=2\n!B=3\n"
        assert dict(parser.parse(str(config))) == dict(config)


class TestCommentsAndBlankLines:
    """``comment`` and ``blank_line``, and the comment and blank lines above a header."""

    @pytest.mark.parametrize(
        "source",
        [
            "# at the top\nA=1\n",
            "  # indented\nA=1\n",
            "[s]\nA=1\n    # indented, below a setting\n",
            "[s]\n# about A\nA=1\n",
            "A=1\n\n# about s\n# and more\n[s]\nB=2\n",
            "\n\nA=1\n\n",
            "# nothing but a comment\n",
            "",
        ],
    )
    def test_a_comment_or_a_blank_line_survives_untouched(self, parser, source) -> None:
        """Test every position a comment may take, and a file of nothing but a comment."""
        assert str(parser.parse(source)) == source


class TestRejected:
    """Input this format refuses, rather than reading as something else."""

    @pytest.mark.parametrize(
        "source",
        [
            "  A=1",
            "A",
            "=1",
            "a b=1",
            "[s",
            "[[s]]",
            "[s] x",
            "[s]\n  x=1",
            # Rose reads these four: the first takes it back to the settings of no section,
            # and the last is the setting "[a]".
            "[]",
            " [s]",
            "[ s ]",
            "[a]=1",
        ],
    )
    def test_malformed_input_raises(self, parser, source) -> None:
        """Test that each of these raises rather than parsing.

        A setting cannot be indented except as part of a value, nor follow a header on its
        line, and a header holds no bracket and no blank at either end.
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
            ("crlf line endings", "A=1\r\n[s]\r\nB=2\r\n  =3\r\n"),
            ("no trailing newline", "[s]\nA=1"),
            ("tab whitespace", "A\t=\t1\t\n"),
            ("trailing blank lines", "[s]\nA=1\n\n\n"),
            ("whitespace-only line", "A=1\n  \n[s]\n"),
            ("empty file", ""),
        ],
    )
    def test_the_shape_of_the_file_survives(self, parser, name, source) -> None:
        """Test the line and whitespace forms the terminals admit.

        CRLF files keep their carriage returns, and a missing final newline stays missing.
        """
        assert str(parser.parse(source)) == source


class TestAssignmentSpacing:
    """The ``eq`` rule: whitespace is written back on the side of "=" it came from."""

    @pytest.mark.parametrize("source", ["A = 1\n", "A= 1\n", "A =1\n", "A=1\n", "A  =  1\n", "A =\n"])
    def test_whitespace_stays_on_its_own_side(self, parser, source) -> None:
        """Test that the position of a single whitespace run survives."""
        assert str(parser.parse(source)) == source


class TestEditingAValue:
    """Writing a value back into the file, changing that and nothing else."""

    @pytest.mark.parametrize("value", ["16", "'P1M'", ".true.", "18,16", "*=/scratch/tm70/model_scaling", "true"])
    def test_a_value_is_written_exactly_as_given(self, parser, value) -> None:
        """Test that the string is the text after the equals sign, quotes and all."""
        config = parser.parse("[s]\nA=1\nB=2\n")

        config["s"]["A"] = value

        assert str(config) == f"[s]\nA={value}\nB=2\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_one_line_replaces_a_value_of_several(self, parser) -> None:
        """Test that the continuation lines go with the value they continued."""
        config = parser.parse("[s]\nA=one\n  =two\nB=2\n")

        config["s"]["A"] = "single"

        assert str(config) == "[s]\nA=single\nB=2\n"

    def test_a_value_of_several_lines_can_be_written_over_itself(self, parser) -> None:
        """Test that a token already holding a value keeps its text."""
        source = "[s]\nA=one\n  =two\n"
        config = parser.parse(source)

        config["s"]["A"] = "one\ntwo"

        assert str(config) == source

    @pytest.mark.parametrize("value", ["", " padded", "padded ", "two\nlines"])
    def test_what_is_not_one_line_of_text_is_refused(self, parser, value) -> None:
        """Test the strings the line after an equals sign cannot hold as they are.

        Rose trims a value, and new continuation lines would need an indent to copy.
        """
        config = parser.parse("[s]\nA=1\n")

        with pytest.raises(UnsupportedEntryError):
            config["s"]["A"] = value
        with pytest.raises(UnsupportedEntryError):
            config["s"]["B"] = value

        assert str(config) == "[s]\nA=1\n"

    @pytest.mark.parametrize("value", [16, True, 1.5, ["a", "b"]])
    def test_a_value_that_is_not_a_string_is_refused(self, parser, value) -> None:
        """Test that a number, a boolean or a list is refused, for Rose has no types."""
        config = parser.parse("[s]\nA=1\n")

        with pytest.raises(TypeError):
            config["s"]["A"] = value
        with pytest.raises(UnsupportedEntryError):
            config["s"]["B"] = value

        assert str(config) == "[s]\nA=1\n"

    def test_a_value_and_a_null_do_not_replace_each_other(self, parser) -> None:
        """Test that ``KEY=`` and ``KEY=1`` are different kinds of entry.

        Deleting the key and assigning it again is how one becomes the other.
        """
        config = parser.parse("[s]\nA=1\nB=\n")

        with pytest.raises(TypeError):
            config["s"]["A"] = None
        with pytest.raises(TypeError):
            config["s"]["B"] = "2"

        del config["s"]["B"]
        config["s"]["B"] = "2"

        assert str(config) == "[s]\nA=1\nB=2\n"


class TestAddingAnEntry:
    """Writing a key the file does not have, in the format's own syntax."""

    def test_a_new_setting_follows_the_last_of_its_section(self, parser) -> None:
        """Test that a setting joins its section, before the blank line parting the next."""
        config = parser.parse("[a]\nx=1\n\n[b]\ny=2\n")

        config["a"]["z"] = "3"

        assert str(config) == "[a]\nx=1\nz=3\n\n[b]\ny=2\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_setting_follows_a_value_of_several_lines(self, parser) -> None:
        """Test that the continuation lines of the setting above are not split."""
        config = parser.parse("[s]\nA=one\n  =two\n")

        config["s"]["B"] = "3"

        assert str(config) == "[s]\nA=one\n  =two\nB=3\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_top_level_setting_goes_above_the_first_section(self, parser) -> None:
        """Test that a setting of no section is not written into the last section.

        Nothing closes a section, so a line after the last one would be read as part of it.
        """
        config = parser.parse("root-dir=*=/x\n\n[a]\nx=1\n")

        config["root-dir{share}"] = "*=/y"

        assert str(config) == "root-dir=*=/x\nroot-dir{share}=*=/y\n\n[a]\nx=1\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize(
        ("source", "written"),
        [
            ("[a]\nx=1\n", "A=1\n[a]\nx=1\n"),
            ("# about the file\n\n[a]\nx=1\n", "# about the file\nA=1\n\n[a]\nx=1\n"),
            ("# about the file\n[a]\nx=1\n", "# about the file\nA=1\n[a]\nx=1\n"),
            ("\n# about a\n[a]\nx=1\n", "A=1\n\n# about a\n[a]\nx=1\n"),
        ],
        ids=["no comment", "a file comment", "a file comment above a header", "a comment heading a"],
    )
    def test_a_new_top_level_setting_where_the_file_has_none(self, parser, source, written) -> None:
        """Test that the first setting of no section goes first, below the file's comments.

        The comments opening the file are the file's, up to its first blank line, as Rose
        reads them, while a comment right above a header after that heads its section.
        """
        config = parser.parse(source)

        config["A"] = "1"

        assert str(config) == written
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_section_goes_last_parted_like_the_one_above(self, parser) -> None:
        """Test that a new section copies the blank line above the last one."""
        config = parser.parse("[a]\nx=1\n\n# about b\n[b]\ny=2\n")

        config["c"] = {"z": "3"}

        assert str(config) == "[a]\nx=1\n\n# about b\n[b]\ny=2\n\n[c]\nz=3\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_section_keeps_the_final_newline(self, parser) -> None:
        """Test a file with no final line break, whose new section ends it the same way."""
        config = parser.parse("[a]\nx=1\n\n[b]\ny=2")

        config["c"] = {"z": "3"}

        assert str(config) == "[a]\nx=1\n\n[b]\ny=2\n\n[c]\nz=3"

    def test_the_first_section_of_a_file(self, parser) -> None:
        """Test the declared template, a blank line and a header, with nothing to copy."""
        config = parser.parse("A=1\n")

        config["s"] = {"B": "2"}

        assert str(config) == "A=1\n\n[s]\nB=2\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_the_first_section_of_an_empty_file(self, parser) -> None:
        """Test that the line ``parse`` adds stays below a new section, as the file's end.

        The template's blank line comes first, so the file starts with one.
        """
        config = parser.parse("")

        config["s"] = {"B": "2"}

        assert str(config) == "\n[s]\nB=2\n"
        assert dict(parser.parse(str(config))) == dict(config)

    @pytest.mark.parametrize(
        ("source", "written"),
        [
            ("meta=x\n\n", "meta=x\n\n\n[s]\nB=2\n"),
            ("\n", "\n\n[s]\nB=2\n"),
            ("A=1\n\n# about the file\n", "A=1\n\n# about the file\n\n[s]\nB=2\n"),
            ("# about the file\n", "# about the file\n\n[s]\nB=2\n"),
        ],
        ids=["a blank line", "a single line break", "a comment", "only a comment"],
    )
    def test_a_new_section_goes_below_every_line_of_no_section(self, parser, source, written) -> None:
        """Test that a new section goes past the blank lines and comments of no section.

        Below a section they would be read as part of it, so its rule takes none there, and
        a section spliced in above them could not be written out.
        """
        config = parser.parse(source)

        config["s"] = {"B": "2"}

        assert str(config) == written
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_never_lands_between_a_comment_and_the_section_it_heads(self, parser) -> None:
        """Test that the comment above a header stays with it as the section above grows."""
        config = parser.parse("[a]\nx=1\n# about b\n[b]\ny=2\n")

        config["a"]["z"] = "3"

        assert str(config) == "[a]\nx=1\nz=3\n# about b\n[b]\ny=2\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_a_new_key_ends_its_line_like_its_neighbour(self, parser) -> None:
        """Test that a file with Windows line endings gets one on the new line too."""
        config = parser.parse("[s]\r\nA=1\r\n")

        config["s"]["B"] = "2"

        assert str(config) == "[s]\r\nA=1\r\nB=2\r\n"

    def test_a_null_setting(self, parser) -> None:
        """Test that ``None`` is written as a key and an equals sign."""
        config = parser.parse("[s]\nA=1\n")

        config["s"]["B"] = None

        assert str(config) == "[s]\nA=1\nB=\n"
        assert dict(parser.parse(str(config))) == dict(config)

    def test_what_a_rose_configuration_cannot_express(self, parser) -> None:
        """Test the additions the format has no syntax for.

        A section holds no section, and a new section is named as a key is: a name with a
        blank in it can be read and edited, but not added.
        """
        config = parser.parse("[s]\nA=1\n")

        with pytest.raises(UnsupportedEntryError):
            config["s"]["N"] = {"Y": "1"}
        with pytest.raises(ValueError, match="Not a valid configuration key"):
            config["a section"] = {"Y": "1"}
        with pytest.raises(ValueError, match="Not a valid configuration key"):
            config["s"]["a b"] = "1"

        assert str(config) == "[s]\nA=1\n"


class TestDeleting:
    """Removing an entry, and the line or lines it occupied."""

    def test_deleting_the_last_setting_leaves_the_section(self, parser) -> None:
        """Test that an empty section is still a section.

        ``block`` is a plain repetition, which pruning stops at.
        """
        config = parser.parse("[a]\nx=1\n\n[b]\ny=2\n")

        del config["a"]["x"]

        assert str(config) == "[a]\n\n[b]\ny=2\n"
        assert dict(parser.parse(str(config))) == {"a": {}, "b": {"y": "2"}}

    def test_a_section_goes_with_its_heading_comment(self, parser) -> None:
        """Test that the comment and the blank line above a header go with its section."""
        config = parser.parse("[a]\nx=1\n\n# about b\n[b]\ny=2\n\n[c]\nz=3\n")

        del config["b"]

        assert str(config) == "[a]\nx=1\n\n[c]\nz=3\n"

    def test_the_comments_opening_the_file_stay_when_the_first_section_goes(self, parser) -> None:
        """Test that the comments above the first header are the file's, as for Rose."""
        config = parser.parse("# Suite for X\n[jinja2:suite.rc]\nA=1\n[env]\nB=2\n")

        del config["jinja2:suite.rc"]

        assert str(config) == "# Suite for X\n[env]\nB=2\n"

    def test_deleting_the_last_section_keeps_the_final_newline(self, parser) -> None:
        """Test that the line break ending the file is not the last section's to take."""
        config = parser.parse("A=1\n\n[a]\nx=1\n")

        del config["a"]

        assert str(config) == "A=1\n"

    def test_every_entry_that_wrote_a_key_is_removed(self, parser) -> None:
        """Test that a key assigned twice does not reappear the next time it is read."""
        config = parser.parse("A=1\nA=2\nB=3\n")

        del config["A"]

        assert str(config) == "B=3\n"
        assert dict(parser.parse(str(config))) == dict(config)


# =============================================================================
# End to end
# =============================================================================

SAMPLE = """\
# A made-up suite configuration, in the shape of a Unified Model suite's.
root-dir=*=/scratch/tm70/x

[jinja2:suite.rc]
MAIN_ATM_PROCX=24
MAIN_ATM_PROCY=24
!!MAIN_IOS_NPROC=0
MAIN_CLOCK='PT1H10M'
SITES='nci',
     ='other'

# The model's own sizes.
[namelist:nlsizes]
global_row_length=192
global_rows=144

[!namelist:ignored]
x=1
"""

SAMPLE_VALUES = {
    "root-dir": "*=/scratch/tm70/x",
    "jinja2:suite.rc": {
        "MAIN_ATM_PROCX": "24",
        "MAIN_ATM_PROCY": "24",
        "!!MAIN_IOS_NPROC": "0",
        "MAIN_CLOCK": "'PT1H10M'",
        "SITES": "'nci',\n'other'",
    },
    "namelist:nlsizes": {"global_row_length": "192", "global_rows": "144"},
    "!namelist:ignored": {"x": "1"},
}

SAMPLE_EDITED = """\
# A made-up suite configuration, in the shape of a Unified Model suite's.
root-dir=*=/scratch/tm70/model_scaling
root-dir{share}=*=/scratch/tm70/share

[jinja2:suite.rc]
MAIN_ATM_PROCX=16
MAIN_ATM_PROCY=24
!!MAIN_IOS_NPROC=0
MAIN_CLOCK='P1M'
SITES='nci',
     ='other'
rg01_rs01_m01_nproc=18,16

# The model's own sizes.
[namelist:nlsizes]
global_row_length=192
global_rows=144
l_new=.true.

[!namelist:ignored]
x=1

[namelist:new]
a=true
"""


def test_a_whole_file_parses_edits_and_round_trips(parser) -> None:
    """Test that the rules above compose, over a file carrying all of them at once.

    It pins what no focused test can: where each new entry lands, and that editing one
    entry leaves every other line alone.
    """
    config = parser.parse(SAMPLE)
    assert dict(config) == SAMPLE_VALUES
    assert str(config) == SAMPLE

    config["root-dir"] = "*=/scratch/tm70/model_scaling"
    config["jinja2:suite.rc"]["MAIN_ATM_PROCX"] = "16"
    config["jinja2:suite.rc"]["MAIN_CLOCK"] = "'P1M'"

    config["root-dir{share}"] = "*=/scratch/tm70/share"
    config["jinja2:suite.rc"]["rg01_rs01_m01_nproc"] = "18,16"
    config["namelist:nlsizes"]["l_new"] = ".true."
    config["namelist:new"] = {"a": "true"}

    assert str(config) == SAMPLE_EDITED
    assert dict(parser.parse(str(config))) == dict(config)


@pytest.mark.parametrize(("container", "category", "expected"), canonical_rows("rose_conf"))
def test_canonical_entry_text(parser, container, category, expected) -> None:
    """Test the text this format writes for a new entry with no neighbour to copy from.

    Byte-exact, and re-parsed in the container it was written for.
    """
    assert_canonical_entry(parser, container, category, expected, TEXT_VALUES)
