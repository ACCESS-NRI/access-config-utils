# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for laying out the text of a list whose length changes.

Everything here is text, so no grammar is involved: a list's text is built directly as the
parts ``ListText`` holds, and what is checked is the candidate texts offered for a new list.
Which candidate a real format accepts is for ``config_resize``, and the format tests.
"""

import pytest

from access.config.list_layout import VALUE_MARKER, ListText, align_values, layout_candidates

# A Fortran list written a value per line, the first on the key's line.
ONE_PER_LINE = ListText("  x = ", (",\n      ", ",\n      "), "\n", (",\n      ", ",\n      "))

# A run sequence: rows on lines of their own, a comment heading one of them, and CIME's
# trailing spaces after the loop rows.
RUN_SEQUENCE = ListText(
    "runSeq:: \n",
    (" \n  ", "\n  # about OCN\n  ", "\n"),
    " \n::\n",
    (" \n  ", "\n\n  ", "\n"),
)
ROWS = ["@3600", "MED a", "OCN", "@"]

# A Fortran list partly on one line and partly on the next, a comment ending the first.
MIXED = ListText("  x = ", (", ", ",  ! c\n     ", ", "), "\n", (", ", ",\n     ", ", "))


class TestListText:
    """Taking apart the text of a list whose values were replaced by markers."""

    def test_splits_around_the_markers(self) -> None:
        """Test the prefix, the separators, the suffix and the clean separators."""
        raw = "x = " + VALUE_MARKER.format(0) + ", ! c\n  " + VALUE_MARKER.format(1) + "\n"
        clean = "x = " + VALUE_MARKER.format(0) + ", \n  " + VALUE_MARKER.format(1) + "\n"

        text = ListText.from_marked(raw, clean, 2)

        assert text == ListText("x = ", (", ! c\n  ",), "\n", (", \n  ",))

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("x = " + VALUE_MARKER.format(0) + "\n", id="missing"),
            pytest.param("x = " + VALUE_MARKER.format(1) + ", " + VALUE_MARKER.format(0), id="out-of-order"),
        ],
    )
    def test_refuses_markers_that_are_not_all_there_in_order(self, raw: str) -> None:
        """Test that a text which lost a value, or reordered them, cannot be laid out."""
        with pytest.raises(ValueError, match="value markers 0 to 1"):
            ListText.from_marked(raw, raw, 2)


class TestAlignValues:
    """Matching each new value to the old one it stands in for."""

    @pytest.mark.parametrize(
        ("old", "new", "origins"),
        [
            pytest.param([1, 2, 3], [1, 2, 3], [0, 1, 2], id="unchanged"),
            pytest.param([1, 2, 3], [1, 2, 3, 4], [0, 1, 2, None], id="appended"),
            pytest.param([1, 2, 3], [0, 1, 2, 3], [None, 0, 1, 2], id="prepended"),
            pytest.param([1, 2, 3], [1, 3], [0, 2], id="deleted"),
            pytest.param([1, 2, 3], [1, 9, 3], [0, 1, 2], id="changed-keeps-its-place"),
            pytest.param([1, 2], [7, 8, 9], [0, 1, None], id="replaced-by-more"),
            pytest.param([1, 2, 3], [7, 2], [0, 1], id="replaced-by-fewer"),
            pytest.param([1, 2], [1.0, 2], [0, 1], id="another-type-is-another-value"),
        ],
    )
    def test_origins(self, old: list, new: list, origins: list) -> None:
        """Test the alignment for the edits a whole-list assignment makes."""
        assert align_values(old, new) == origins


class TestLineMode:
    """A list each of whose elements owns a line."""

    def test_appending_offers_both_owners_of_the_closing_line_end(self) -> None:
        """Test the two readings of who owns the line ending that closes the list.

        Fortran puts its separator after an element, so only the second reads back; a run
        sequence has no separator, and the first keeps a trailing space where it was.
        """
        candidates = layout_candidates(ONE_PER_LINE, ["1", "2", "3", "4"], [0, 1, 2, None])

        assert candidates == [
            "  x = 1,\n      2,\n      3\n      4,\n",
            "  x = 1,\n      2,\n      3,\n      4\n",
        ]

    def test_popping_the_last_element(self) -> None:
        """Test that the new last element takes the line ending the list closed on."""
        candidates = layout_candidates(ONE_PER_LINE, ["1", "2"], [0, 1])

        assert candidates[-1] == "  x = 1,\n      2\n"

    def test_prepending_moves_the_first_element_off_the_key_line(self) -> None:
        """Test that the old first element takes the indent of the element after it."""
        candidates = layout_candidates(ONE_PER_LINE, ["0", "1", "2", "3"], [None, 0, 1, 2])

        assert candidates == ["  x = 0,\n      1,\n      2,\n      3\n"]

    def test_deleting_the_first_element_brings_the_next_onto_the_key_line(self) -> None:
        """Test that whatever is first sits on the key's line, without an indent."""
        candidates = layout_candidates(ONE_PER_LINE, ["2", "3"], [1, 2])

        assert candidates[-1] == "  x = 2,\n      3\n"

    def test_a_row_inserted_after_a_loop_opener_takes_the_body_indent(self) -> None:
        """Test that the deeper neighbour sets the indent, and trailing spaces stay put."""
        candidates = layout_candidates(RUN_SEQUENCE, ["@3600", "MED b", *ROWS[1:]], [0, None, 1, 2, 3])

        assert candidates == ["runSeq:: \n@3600 \n  MED b\n  MED a\n  # about OCN\n  OCN\n@ \n::\n"]

    def test_a_row_inserted_before_a_loop_closer_takes_the_body_indent(self) -> None:
        """Test the other side of the loop: the closer is shallower than the row above."""
        candidates = layout_candidates(RUN_SEQUENCE, [*ROWS[:3], "ICE", "@"], [0, 1, 2, None, 3])

        assert candidates == ["runSeq:: \n@3600 \n  MED a\n  # about OCN\n  OCN\n  ICE\n@ \n::\n"]

    def test_a_deleted_row_leaves_the_comment_heading_the_next(self) -> None:
        """Test that the comment lines above a row stay with it when the row above goes."""
        candidates = layout_candidates(RUN_SEQUENCE, ["@3600", "OCN", "@"], [0, 2, 3])

        assert candidates == ["runSeq:: \n@3600 \n  # about OCN\n  OCN\n@ \n::\n"]

    def test_the_comment_above_a_deleted_row_moves_down(self) -> None:
        """Test that deleting a row keeps the comment lines that headed it."""
        candidates = layout_candidates(RUN_SEQUENCE, ["@3600", "MED a", "@"], [0, 1, 3])

        assert candidates == ["runSeq:: \n@3600 \n  MED a\n  # about OCN\n@ \n::\n"]

    def test_a_row_appended_after_the_last_offers_both_readings(self) -> None:
        """Test that the closer's trailing space stays with it in the first candidate."""
        candidates = layout_candidates(RUN_SEQUENCE, [*ROWS, "WAV"], [0, 1, 2, 3, None])

        assert candidates == [
            "runSeq:: \n@3600 \n  MED a\n  # about OCN\n  OCN\n@ \nWAV\n::\n",
            "runSeq:: \n@3600 \n  MED a\n  # about OCN\n  OCN\n@\nWAV \n::\n",
        ]

    def test_a_list_closing_without_a_line_end_has_one_reading(self) -> None:
        """Test that the element-owned reading needs a line ending to give the old last."""
        text = ListText("x = ", ("\n    ",), "", ("\n    ",))

        assert layout_candidates(text, ["1", "2", "3"], [0, 1, None]) == ["x = 1\n    2\n    3"]


class TestInlineMode:
    """A list some of whose elements share a line."""

    @pytest.mark.parametrize(
        ("values", "origins", "expected"),
        [
            pytest.param(
                ["1", "2", "9", "3", "4"], [0, 1, None, 2, 3], "  x = 1, 2, 9,  ! c\n     3, 4\n", id="insert"
            ),
            pytest.param(["1", "3", "4"], [0, 2, 3], "  x = 1,  ! c\n     3, 4\n", id="delete-before-a-break"),
            pytest.param(["1", "2", "4"], [0, 1, 3], "  x = 1, 2,  ! c\n     4\n", id="delete-after-a-break"),
            pytest.param(["0", *"1234"], [None, 0, 1, 2, 3], "  x = 0, 1, 2,  ! c\n     3, 4\n", id="prepend"),
            pytest.param([*"1234", "5"], [0, 1, 2, 3, None], "  x = 1, 2,  ! c\n     3, 4, 5\n", id="append"),
            pytest.param(["a", "b"], [None, None], "  x = a, b\n", id="all-new"),
        ],
    )
    def test_layout(self, values: list, origins: list, expected: str) -> None:
        """Test that line breaks stay where they were, and new gaps copy an inline one."""
        assert layout_candidates(MIXED, values, origins) == [expected]

    def test_searches_forwards_when_nothing_inline_precedes(self) -> None:
        """Test a gap whose nearest separators before it all break the line."""
        text = ListText("x = ", (",\n  ", ", "), "\n", (",\n  ", ", "))

        candidates = layout_candidates(text, ["1", "9", "2", "3"], [0, None, 1, 2])

        assert candidates == ["x = 1, 9,\n  2, 3\n"]
