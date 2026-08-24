# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for choosing a template and writing the text of a new entry.

These exercise ranking and rendering directly, against the three shipped grammars, so that a
change in which candidate wins is caught here rather than only through the end-to-end tests
in ``test_config.py`` and the per-format modules.
"""

from access.config import entry_render
from access.config.entry_fragments import Fragment
from access.config.entry_render import iter_entry_snippets, render
from access.config.entry_style import EntryStyle


def test_value_type_rendering(infos) -> None:
    """Test the notation each grammar uses for a value written into a new entry.

    These pin ``VALUE_RULE_PRIORITY`` as it applies to the real formats: a namelist writes
    Fortran logicals and has no bare-word type, while MOM6 writes Python-style booleans.
    """
    fortran = infos["fortran_nml"]
    for value, expected in [
        (True, "K = .true.\n"),
        (7, "K = 7\n"),
        (1.5, "K = 1.5\n"),
        (3.0 + 4.0j, "K = (3.0, 4.0)\n"),
        ("text", 'K = "text"\n'),
    ]:
        assert next(iter(iter_entry_snippets(fortran, "block", "key_value", "K", value, EntryStyle()))) == expected

    mom6 = infos["mom6_input"]
    assert next(iter(iter_entry_snippets(mom6, "start", "key_value", "K", True, EntryStyle()))) == "K = True\n"

    nuopc = infos["nuopc_config"]
    assert next(iter(iter_entry_snippets(nuopc, "start", "key_value", "K", True, EntryStyle()))) == "K: .true.\n"
    assert next(iter(iter_entry_snippets(nuopc, "start", "key_value", "K", "datm", EntryStyle()))) == "K: datm\n"


def test_no_candidate_for_unrepresentable_value(infos) -> None:
    """Test that a value no admitted rule accepts produces no candidate at all."""
    # NUOPC has neither a quoted string nor any other type accepting this value.
    assert list(iter_entry_snippets(infos["nuopc_config"], "start", "key_value", "K", "a string", EntryStyle())) == []
    # MOM6 has no complex type.
    assert list(iter_entry_snippets(infos["mom6_input"], "start", "key_value", "K", 1 + 2j, EntryStyle())) == []


def test_value_rule_fallback(infos) -> None:
    """Test that every accepting value type is offered, not only the most preferred one.

    A grammar admitting both a quoted string and a bare word offers both renderings, so a
    caller whose preferred one reads back as the wrong type can fall through to the next.
    """
    snippets = list(iter_entry_snippets(infos["nuopc_config"], "start", "key_value", "K", "datm", EntryStyle()))
    assert "K: datm\n" in snippets


def test_snippet_cap_is_enforced(infos, monkeypatch) -> None:
    """Test that the number of candidates offered for one insertion is bounded."""
    monkeypatch.setattr(entry_render, "_MAX_SNIPPETS", 2)

    snippets = list(iter_entry_snippets(infos["mom6_input"], "start", "key_value", "K", 1, EntryStyle()))
    assert len(snippets) == 2


def test_ranking_prefers_a_complete_line(infos) -> None:
    """Test that the chosen entry ends the line it is written on.

    A Fortran assignment can be written with no line terminator, since the terminator sits
    on the enclosing line rule. An entry rendered that way would run into what follows it.
    """
    info = infos["fortran_nml"]
    first = next(iter(iter_entry_snippets(info, "block", "key_value", "K", 1, EntryStyle())))
    assert first.endswith("\n")


def test_ranking_avoids_a_trailing_separator(infos) -> None:
    """Test that an optional trailing separator is not added.

    Fortran allows a comma after the last value on a line, and that form has exactly the
    whitespace runs a spaced neighbour wants. Style must not be allowed to buy it.
    """
    style = EntryStyle(indent="  ", key_pads=(" ", " "), element_pads=(" ",), has_donor=True)
    first = next(iter(iter_entry_snippets(infos["fortran_nml"], "block", "key_list", "K", [1, 2], style)))

    assert first == "  K = 1, 2\n"
    assert not first.rstrip("\n").endswith(",")


def test_ranking_avoids_a_line_continuation(infos) -> None:
    """Test that a list is written on one line.

    Fortran can separate two values with a comma and a newline, which is a legal list but
    reads as a continuation. The entry should stay on the line it started on.
    """
    first = next(iter(iter_entry_snippets(infos["fortran_nml"], "block", "key_list", "K", [1, 2, 3], EntryStyle())))
    assert first.count("\n") == 1


def test_ranking_puts_a_new_block_body_on_its_own_line(infos) -> None:
    """Test that a new namelist group leaves its body room.

    ``&NAME/`` is a legal empty namelist, but a group written that way has nowhere to put
    the keys that are about to be added to it.
    """
    first = next(iter(iter_entry_snippets(infos["fortran_nml"], "start", "key_block", "K", {}, EntryStyle())))
    assert first == "&K\n/\n"


def test_render_ignores_a_block_placeholder() -> None:
    """Test that the contents of a new block render as nothing."""
    template = (Fragment("key"), Fragment("literal", "<"), Fragment("block"), Fragment("literal", ">"))
    assert render(template, "K", [], EntryStyle()) == "K<>"
