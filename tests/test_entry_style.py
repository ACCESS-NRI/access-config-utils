# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reading whitespace off a file and applying it to a new entry.

Whitespace is copied, never invented, so these cover both halves of that: what
``probe_entry_style`` reads from a real container, and what a style then produces once a
grammar has had its say about where whitespace may go.
"""

from access.config.entry_render import iter_entry_snippets
from access.config.entry_style import EntryStyle, probe_entry_style, probe_sibling_block_style


def test_style_is_copied_from_neighbours(infos) -> None:
    """Test that a neighbour's indentation and spacing are reproduced."""
    info = infos["fortran_nml"]
    style = EntryStyle(indent="  ", key_pads=(" ", " "), element_pads=(" ",), has_donor=True)

    assert next(iter(iter_entry_snippets(info, "block", "key_value", "NEW", 42, style))) == "  NEW = 42\n"
    assert next(iter(iter_entry_snippets(info, "block", "key_list", "NEW", [1, 2, 3], style))) == "  NEW = 1, 2, 3\n"
    assert next(iter(iter_entry_snippets(info, "block", "key_null", "NEW", None, style))) == "  NEW =\n"


def test_style_without_padding(infos) -> None:
    """Test that a neighbour written without spaces is matched without spaces."""
    info = infos["mom6_input"]
    tight = EntryStyle(indent="", key_pads=(), element_pads=(), has_donor=True)

    assert next(iter(iter_entry_snippets(info, "start", "key_value", "NEW", 1, tight))) == "NEW=1\n"
    assert next(iter(iter_entry_snippets(info, "start", "key_list", "NEW", [1, 2], tight))) == "NEW=1,2\n"


def test_style_indent_ignored_when_unsupported(infos) -> None:
    """Test that indentation is dropped where the grammar does not allow it.

    A MOM6 assignment cannot be indented, so an indented neighbour must not produce an entry
    that fails to parse.
    """
    info = infos["mom6_input"]
    style = EntryStyle(indent="    ", key_pads=(" ", " "), element_pads=None, has_donor=True)

    assert next(iter(iter_entry_snippets(info, "start", "key_value", "NEW", 1, style))) == "NEW = 1\n"


def test_probe_entry_style(larks) -> None:
    """Test the whitespace read back from the entries of a real container."""
    tree = larks["fortran_nml"].parse("&L\n  Var = 4\n  LIST = 1, 2, 3\n/\n", start="start")
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")

    style = probe_entry_style(block)
    assert style.has_donor
    assert style.indent == "  "
    assert style.key_pads == (" ", " ")
    assert style.element_pads == (" ",)


def test_probe_entry_style_separates_the_two_kinds(larks) -> None:
    """Test that a scalar neighbour does not dictate how a list is spaced.

    The last entry here is a scalar written without spaces around its separator, and the
    only list is spaced. Element spacing has to come from the list, and assignment spacing
    from the scalar, or a new entry of either kind comes out looking like the other.
    """
    tree = larks["mom6_input"].parse("BLIST = 1, 2, 3\nVAR=4\n", start="start")

    style = probe_entry_style(tree)
    assert style.key_pads == ()
    assert style.element_pads == (" ",)


def test_probe_entry_style_without_entries(larks) -> None:
    """Test that a container holding no entries reports no style to copy."""
    tree = larks["mom6_input"].parse("B%\n%B\n", start="start")
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")

    assert probe_entry_style(block) == EntryStyle()


def test_probe_entry_style_without_values(larks) -> None:
    """Test the style read from an entry that has a key but no value."""
    tree = larks["fortran_nml"].parse("&L\n  NOVALUE =\n/\n", start="start")
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")

    style = probe_entry_style(block)
    assert style.has_donor
    assert style.indent == "  "
    # With no value, everything after the key spaces out the key, and nothing says how
    # elements are separated.
    assert style.element_pads is None


def test_probe_entry_style_stops_at_a_nested_block(larks) -> None:
    """Test that the style of a container is read from its own entries only."""
    tree = larks["mom6_input"].parse("A = 1\nB%\nX = 1\n%B\n", start="start")

    style = probe_entry_style(tree)
    assert style.has_donor
    # The block's own entry must not be visible to the outer container.
    assert style.indent == ""


def test_probe_sibling_block_style(larks) -> None:
    """Test the style a new block takes from the blocks already in its container."""
    tree = larks["fortran_nml"].parse("&L\n  Var = 4\n/\n&EMPTY\n/\n", start="start")

    style = probe_sibling_block_style(tree)
    assert style.has_donor
    assert style.indent == "  "
    assert style.key_pads == (" ", " ")


def test_probe_sibling_block_style_takes_the_nearest(larks) -> None:
    """Test that the last block able to donate wins, as the last entry does for a style.

    A new block is written at the end of its container, so the block just above it is its
    closest neighbour. An empty one in between says nothing and is passed over.
    """
    tree = larks["fortran_nml"].parse("&A\n    Far = 1\n/\n&B\n  Near = 2\n/\n&C\n/\n", start="start")

    assert probe_sibling_block_style(tree).indent == "  "


def test_probe_sibling_block_style_without_blocks(larks) -> None:
    """Test that a container with no block to copy from reports no style.

    Both cases have to report it: a container holding no block at all, and one whose only
    block is empty -- which is what a block just created looks like.
    """
    entries_only = larks["mom6_input"].parse("A = 1\n", start="start")
    empty_block = larks["mom6_input"].parse("B%\n%B\n", start="start")

    assert probe_sibling_block_style(entries_only) == EntryStyle()
    assert probe_sibling_block_style(empty_block) == EntryStyle()
