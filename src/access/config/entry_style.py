# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The whitespace a new entry should be written with.

Whitespace is copied, never invented. Everything here exists to answer one question -- how
should the entry about to be added be spaced -- and it answers it by looking at the entries
already in the file. ``probe_entry_style`` reads them; ``EntryStyle`` carries what it found;
``canonical_ws`` and the slot classification decide where each recorded run goes in a
template, and what to write where nothing was recorded.

Where a grammar admits no whitespace at the position a donated indent would occupy, the
indent is dropped rather than written into a line that would not parse. That is what
``style_matches`` is for: it lets ranking prefer a template that *can* reproduce the
neighbouring style over one that cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

from lark import Tree

from access.config.entry_fragments import EntryTemplate, neighbour_fragment
from access.config.grammar_contract import BLOCK_RULE, KEY_BLOCK, KEY_RULE, WS_RULE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY
from access.config.tree_navigation import entries_of


@dataclass(frozen=True)
class EntryStyle:
    """Whitespace copied from existing entries, so a new one matches its neighbours.

    Args:
        indent (str): Whitespace before the key.
        key_pads (tuple[str, ...] | None): Whitespace runs between the key and the first
            value, in order, or ``None`` when no entry demonstrates it.
        element_pads (tuple[str, ...] | None): Whitespace runs between the first and second
            values, in order, or ``None`` when no entry demonstrates it. That is the case
            whenever no entry holds two values, so a scalar neighbour does not dictate how a
            list is spaced.
        has_donor (bool): Whether any entry was found to copy from. When false, rendering
            falls back to a canonical layout.
    """

    indent: str = ""
    key_pads: tuple[str, ...] | None = None
    element_pads: tuple[str, ...] | None = None
    has_donor: bool = False


# --- Whitespace policy ---


def _is_punctuation(text: str) -> bool:
    """Report whether *text* is non-empty and made only of punctuation."""
    return text != "" and not any(character.isalnum() or character.isspace() for character in text)


def canonical_ws(template: EntryTemplate, index: int) -> str:
    """Return the whitespace to emit at *index* when there is no entry to copy from.

    Empty at the start and the end of a line, and before a separator that follows a value,
    so a canonically rendered entry carries no indentation, no trailing whitespace, and
    reads as ``a, b`` rather than ``a , b``. A single space anywhere else, which is where a
    grammar allows optional whitespace for readability, as in ``KEY = value``.

    Args:
        template (EntryTemplate): The template being rendered.
        index (int): Index of the whitespace placeholder.

    Returns:
        str: The whitespace text.
    """
    before = neighbour_fragment(template, index, -1)
    after = neighbour_fragment(template, index, 1)
    if before is None or before.kind == "block" or (before.kind == "literal" and before.text.endswith("\n")):
        return ""
    if after is None or after.kind == "block" or (after.kind == "literal" and after.text.startswith("\n")):
        return ""
    if before.kind == "value" and after.kind == "literal" and _is_punctuation(after.text):
        return ""
    return " "


@dataclass(frozen=True)
class _Slots:
    """Indices of a template's whitespace placeholders, grouped by what they space out.

    Args:
        leading (tuple[int, ...]): Placeholders before the key, which carry the indentation.
        key (tuple[int, ...]): Stylable placeholders between the key and the first value.
        element (tuple[int, ...]): Stylable placeholders between the first two values.
    """

    leading: tuple[int, ...]
    key: tuple[int, ...]
    element: tuple[int, ...]


def classify_slots(template: EntryTemplate) -> _Slots:
    """Group the whitespace placeholders of *template* into regions.

    Only placeholders that render as a space canonically are collected for the key and
    element regions. The others are fixed by the line structure -- a line never starts or
    ends with copied whitespace -- and must not consume the copied whitespace runs.

    Args:
        template (EntryTemplate): The template to classify.

    Returns:
        _Slots: The grouped indices.
    """
    keys = [index for index, fragment in enumerate(template) if fragment.kind == "key"]
    values = [index for index, fragment in enumerate(template) if fragment.kind == "value"]
    first_key = keys[0] if keys else len(template)
    # With no value to stop at, everything after the key spaces out the key, as in "KEY =".
    key_end = values[0] if values else len(template)
    element_start, element_end = (values[0], values[1]) if len(values) > 1 else (len(template), len(template))

    def stylable(start: int, stop: int) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(start, stop)
            if template[index].kind == "ws" and canonical_ws(template, index) == " "
        )

    leading = tuple(index for index in range(first_key) if template[index].kind == "ws")
    return _Slots(leading, stylable(first_key, key_end), stylable(element_start, element_end))


def style_matches(template: EntryTemplate, style: EntryStyle) -> bool:
    """Report whether *template* can reproduce the whitespace *style* was taken from."""
    if not style.has_donor:
        return True
    slots = classify_slots(template)
    if style.indent and not slots.leading:
        return False
    for indices, pads in ((slots.key, style.key_pads), (slots.element, style.element_pads)):
        if pads is not None and len(indices) != len(pads):
            return False
    return True


# --- Style probing ---


def probe_entry_style(container: Tree) -> EntryStyle:
    """Read the whitespace of the entries in *container*, to match when adding a new one.

    Each kind of whitespace is taken from the last entry that actually demonstrates it: the
    indentation and the spacing around the assignment from the last entry with both a key
    and a value, and the spacing around the element separator from the last entry with two
    values. So a scalar neighbour does not dictate how a list is spaced, and a list
    neighbour does not stop a scalar being spaced like the scalars around it.

    Falls back to a canonical layout when the container holds no entries at all, which is
    the case for a block that was empty in the file. A block that was *just created* has
    nothing to copy from either, but is not left canonical: see
    ``probe_sibling_block_style``.

    Args:
        container (Tree): The container rule node the new entry will be added to.

    Returns:
        EntryStyle: The whitespace to copy, or a style with ``has_donor`` false.

    Examples:
        The effect, through the public interface: an unusually spaced neighbour is matched
        rather than normalised, because a minimal diff matters more than tidiness.

        >>> from access.config import MOM6InputParser
        >>> config = MOM6InputParser().parse("A   =   1\\n")
        >>> config["B"] = 2
        >>> print(str(config), end="")
        A   =   1
        B   =   2
    """
    entries = entries_of(container)
    if not entries:
        return EntryStyle()

    shapes = [_shape_of(entry) for entry in entries]
    with_value = [shape for shape in shapes if shape[2]]
    indent, key_pads, _ = with_value[-1] if with_value else shapes[-1]
    with_elements = [shape for shape in shapes if len(shape[2]) > 1]
    element_pads = with_elements[-1][2][1] if with_elements else None
    return EntryStyle(indent, key_pads, element_pads, has_donor=True)


def probe_sibling_block_style(container: Tree) -> EntryStyle:
    """Read the whitespace used inside the blocks of *container*, for a block just created.

    A block created by an assignment starts empty, so ``probe_entry_style`` finds nothing in
    it to copy and its contents would be written with no indentation at all -- foreign to a
    file whose other blocks are indented. The style is therefore taken from a sibling block
    instead, searching from the nearest one upwards, since that is where the new block was
    added. This mirrors ``probe_entry_style`` taking its style from the *last* entry: in
    both cases the closest neighbour wins.

    Only the rule named ``BLOCK_RULE`` is looked at, and only as a direct child of the
    entry, which is the same contract ``ConfigToDict.key_block`` relies on to find the body
    of a block.

    Args:
        container (Tree): The container rule node the new block was added to.

    Returns:
        EntryStyle: The whitespace to copy, or a style with ``has_donor`` false when no
            sibling block holds an entry -- including the new block itself, which is empty.

    Examples:
        A new block is laid out like the block next to it, not like a fresh file.

        >>> from access.config import NUOPCParser
        >>> config = NUOPCParser().parse("OLD::\\n  P = 1\\n::\\n")
        >>> config["NEW"] = {"X": 2}
        >>> print(str(config), end="")
        OLD::
          P = 1
        ::
        NEW::
          X = 2
        ::
    """
    for entry in reversed(entries_of(container)):
        if entry.data != KEY_BLOCK:
            continue
        for child in entry.children:
            if isinstance(child, Tree) and child.data == BLOCK_RULE:
                style = probe_entry_style(child)
                if style.has_donor:
                    return style
    return EntryStyle()


def _shape_of(entry: Tree) -> tuple[str, tuple[str, ...], list[tuple[str, ...]]]:
    """Split the whitespace runs of an entry rule node into the parts a style is built from.

    Args:
        entry (Tree): An entry rule node.

    Returns:
        tuple[str, tuple[str, ...], list[tuple[str, ...]]]: The indentation, the runs
            between the key and the first value, and the runs before each value in turn --
            so the second element, when present, is the element separator.
    """
    kinds: list[tuple[str, str]] = []
    for child in (child for child in entry.children if isinstance(child, Tree)):
        if child.data == WS_RULE:
            kinds.append(("ws", "".join(str(token) for token in child.children)))
        elif child.data == KEY_RULE:
            kinds.append((KEY_RULE, ""))
        elif child.data in VALUE_TYPE_HANDLER_REGISTRY:
            kinds.append(("value", ""))

    def runs(start: int, stop: int) -> tuple[str, ...]:
        return tuple(text for kind, text in kinds[start:stop] if kind == "ws")

    keys = [index for index, (kind, _) in enumerate(kinds) if kind == KEY_RULE]
    key_at = keys[0] if keys else len(kinds)
    values = [index for index, (kind, _) in enumerate(kinds) if kind == "value"]

    # The run before each value, measured from the key for the first and from the previous
    # value thereafter. With no value at all, everything after the key spaces out the key.
    if values:
        boundaries = [key_at, *values[:-1]]
        gaps = [runs(previous, index) for previous, index in zip(boundaries, values, strict=True)]
        key_pads = gaps[0]
    else:
        gaps = []
        key_pads = runs(key_at, len(kinds))
    return "".join(runs(0, key_at)), key_pads, gaps
