# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Laying out the text of a list whose length changes.

A list in a file is more than its values. What parts two of them may be a comma, a line
break and an indent, or a comment ending one line and blank lines before the next. When a
list grows or shrinks, the text around the elements that stay has to stay, and the text
around a new element has to be made up from what its neighbours do. This module does only
that, and only on text: finding the text of a list in the parse tree, and checking that a
candidate reads back as the values asked for, is ``config_resize``.

The text of a list is taken apart into a prefix (everything before the first value, the key
included), the separators between consecutive values, and a suffix (everything after the
last value). Two layouts are told apart:

- **Line mode**, when every separator holds a line break. Each element then owns a line: the
  comment and blank lines above it, its indent, and the rest of its own line -- a comment
  there included. An element that stays keeps all three, and a new element takes the indent
  of the deeper of its neighbours and the line ending most of the list uses.
- **Inline mode**, otherwise. Two elements that stay next to each other keep the separator
  between them. A gap that closes over deleted elements keeps the separator with the most
  line breaks among those it replaces, so a line structure is not lost, and every other gap
  copies the nearest separator that holds no line break.

Which element owns the line ending that closes a list is a genuine ambiguity -- it depends
on whether the grammar puts the separator after the element or before the next one -- so
line mode offers both readings, and the caller keeps the first that reads back correctly.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

VALUE_MARKER = "\x00{}\x00"
"""Stand-in token text for the value at a position, so that its place in a text is known.

A NUL cannot occur in the text of a configuration file, and is not a character the
reconstructor would put a space next to.
"""

_MARKER_PATTERN = re.compile("\x00([0-9]+)\x00")


@dataclass(frozen=True)
class ListText:
    """The text of a list, taken apart around its values.

    Args:
        prefix (str): Everything before the first value, the key included.
        separators (tuple[str, ...]): The text between each pair of consecutive values.
        suffix (str): Everything after the last value.
        clean_separators (tuple[str, ...]): The separators again, with the text of every
            comment removed, so that a new element can copy one without copying a comment
            that described another.
    """

    prefix: str
    separators: tuple[str, ...]
    suffix: str
    clean_separators: tuple[str, ...]

    @classmethod
    def from_marked(cls, raw: str, clean: str, count: int) -> ListText:
        """Take apart the text of a list whose values were replaced by ``VALUE_MARKER``.

        Args:
            raw (str): The text with markers in place of the values.
            clean (str): The same text with every comment removed as well.
            count (int): How many values the list holds.

        Returns:
            ListText: The parts.

        Raises:
            ValueError: If either text does not hold the markers 0 to *count* - 1, in order.
        """
        prefix, separators, suffix = _split(raw, count)
        _, clean_separators, _ = _split(clean, count)
        return cls(prefix, separators, suffix, clean_separators)


def _split(text: str, count: int) -> tuple[str, tuple[str, ...], str]:
    """Split a marked text into the text before, between and after its markers.

    Args:
        text (str): The marked text.
        count (int): How many markers it should hold.

    Returns:
        tuple[str, tuple[str, ...], str]: The prefix, the separators and the suffix.

    Raises:
        ValueError: If the markers are not 0 to *count* - 1, in order.
    """
    pieces = _MARKER_PATTERN.split(text)
    indices = [int(piece) for piece in pieces[1::2]]
    if indices != list(range(count)):
        raise ValueError(f"Expected the value markers 0 to {count - 1} in order, found {indices}")
    return pieces[0], tuple(pieces[2:-1:2]), pieces[-1]


def align_values(old: Sequence[Any], new: Sequence[Any]) -> list[int | None]:
    """Return, for each new value, the position of the old one it stands in for.

    Values that are the same, in type and in value, are matched up in order. Where the old
    and the new lists differ, the values on either side are paired by position as far as
    both go, so a changed value keeps the place -- and the line -- the old one had; the
    rest are new or gone.

    Args:
        old (Sequence[Any]): The values the list holds.
        new (Sequence[Any]): The values it is to hold.

    Returns:
        list[int | None]: One entry per new value: an old position, or ``None`` for a value
            with nothing to stand in for.
    """
    matcher = SequenceMatcher(None, [_identity(value) for value in old], [_identity(value) for value in new], False)
    origins: list[int | None] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            origins.extend(range(old_start, old_end))
        elif tag in ("replace", "insert"):
            paired = min(old_end - old_start, new_end - new_start) if tag == "replace" else 0
            origins.extend(range(old_start, old_start + paired))
            origins.extend([None] * (new_end - new_start - paired))
    return origins


def _identity(value: Any) -> tuple[type, str]:
    """Return what tells two values apart, hashable: their type and representation."""
    return type(value), repr(value)


def layout_candidates(text: ListText, value_texts: Sequence[str], origins: Sequence[int | None]) -> list[str]:
    """Return candidate texts for a list holding *value_texts*, most preferred first.

    Args:
        text (ListText): The text of the list as it is.
        value_texts (Sequence[str]): The token text of each new value.
        origins (Sequence[int | None]): For each new value, the old position it stands in
            for, or ``None``. The old positions have to be in increasing order.

    Returns:
        list[str]: The candidates, without repetitions.
    """
    if all("\n" in separator for separator in text.separators):
        candidates = _line_mode(text, value_texts, origins)
    else:
        candidates = [_inline_mode(text, value_texts, origins)]
    return list(dict.fromkeys(candidates))


# --- Inline mode ---


def _inline_mode(text: ListText, value_texts: Sequence[str], origins: Sequence[int | None]) -> str:
    """Return the text of the list laid out inline.

    Args:
        text (ListText): The text of the list as it is.
        value_texts (Sequence[str]): The token text of each new value.
        origins (Sequence[int | None]): The old position each new value stands in for.

    Returns:
        str: The text.
    """
    gaps = _inline_gaps(text.separators, origins)
    pieces = [text.prefix, value_texts[0]]
    for gap, value in zip(gaps, value_texts[1:], strict=True):
        pieces += [gap, value]
    return "".join([*pieces, text.suffix])


def _inline_gaps(separators: Sequence[str], origins: Sequence[int | None]) -> list[str]:
    """Return the separator for each gap between two consecutive new values.

    A run of new values between two that stay is spaced with copies of the nearest
    separator holding no line break, except for its last gap, which takes the separator the
    two that stay had between them -- so a line break that ended the first one's line still
    does, rather than moving to before the new values.

    Args:
        separators (Sequence[str]): The separators of the list as it is.
        origins (Sequence[int | None]): The old position each new value stands in for.

    Returns:
        list[str]: One separator per gap.
    """
    gaps: list[str] = []
    previous: int | None = None  # The old position of the last value that stayed, if any.
    for index in range(1, len(origins)):
        left, right = origins[index - 1], origins[index]
        if left is not None:
            previous = left
        if right is not None and previous is not None:
            # The gap just before a value that stays, after one that stayed earlier: the
            # text the two had between them, keeping its line breaks.
            gaps.append(max(separators[previous:right], key=lambda separator: separator.count("\n")))
        else:
            gaps.append(_nearest_inline(separators, previous if previous is not None else (right or 1) - 1))
    return gaps


def _nearest_inline(separators: Sequence[str], at: int) -> str:
    """Return the separator holding no line break that is nearest to position *at*.

    Searched backwards from *at* first, then forwards, since a new value is usually written
    after the ones it follows. There is always one: the list is laid out inline only when
    some separator holds no line break.

    Args:
        separators (Sequence[str]): The separators of the list as it is.
        at (int): The position to search from.

    Returns:
        str: The separator.
    """
    at = min(max(at, 0), len(separators) - 1)
    order = [*range(at, -1, -1), *range(at + 1, len(separators))]
    return next(separators[index] for index in order if "\n" not in separators[index])


# --- Line mode ---


@dataclass(frozen=True)
class _Line:
    """The text one element of a list laid out a line each owns.

    Args:
        above (str): The comment and blank lines above it, whole lines.
        indent (str): The text from the start of its line to the value.
        end (str): The rest of its line, up to and including the line break.
    """

    above: str
    indent: str
    end: str


def _line_mode(text: ListText, value_texts: Sequence[str], origins: Sequence[int | None]) -> list[str]:
    """Return the candidate texts of the list laid out a line per element.

    Args:
        text (ListText): The text of the list as it is.
        value_texts (Sequence[str]): The token text of each new value.
        origins (Sequence[int | None]): The old position each new value stands in for.

    Returns:
        list[str]: The candidates.
    """
    head, lines, rest, on_key_line = _owned_lines(text)
    common_end = Counter(_line_break_head(separator) for separator in text.clean_separators).most_common(1)[0][0]
    last = len(lines) - 1
    candidates = []
    # Element-owned: the old last element keeps the line ending the list closed on. Only a
    # reading at all when that ending is a line of its own.
    if "\n" in lines[last].end:
        candidates.append(_render_lines(head, lines, rest, value_texts, origins, common_end, on_key_line, None))
    # Position-owned: whatever is last now closes the list, and the old last takes the
    # ending the rest of the list uses.
    candidates.append(_render_lines(head, lines, rest, value_texts, origins, common_end, on_key_line, last))
    return candidates


def _line_break_head(separator: str) -> str:
    """Return a separator up to and including its first line break."""
    return separator[: separator.index("\n") + 1]


def _owned_lines(text: ListText) -> tuple[str, list[_Line], str, bool]:
    """Take a list laid out a line per element apart into the lines its elements own.

    Args:
        text (ListText): The text of the list.

    Returns:
        tuple[str, list[_Line], str, bool]: The text before the first element's own line,
            each element's line, the text after the last element's line, and whether the
            first element shares its line with the key -- and so owns no indent of its own.
    """
    on_key_line = "\n" not in text.prefix
    cut = 0 if on_key_line else text.prefix.rindex("\n") + 1
    head, indent = (text.prefix, "") if on_key_line else (text.prefix[:cut], text.prefix[cut:])
    aboves, indents, ends = [""], [indent], []
    for separator in text.separators:
        first, last = separator.index("\n") + 1, separator.rindex("\n") + 1
        ends.append(separator[:first])
        aboves.append(separator[first:last])
        indents.append(separator[last:])
    suffix_cut = text.suffix.index("\n") + 1 if "\n" in text.suffix else 0
    ends.append(text.suffix[:suffix_cut])
    lines = [_Line(above, indent, end) for above, indent, end in zip(aboves, indents, ends, strict=True)]
    return head, lines, text.suffix[suffix_cut:], on_key_line


def _render_lines(
    head: str,
    lines: Sequence[_Line],
    rest: str,
    value_texts: Sequence[str],
    origins: Sequence[int | None],
    common_end: str,
    on_key_line: bool,
    position_owned: int | None,
) -> str:
    """Write the list a line per element.

    Args:
        head (str): Text before the first element's own line.
        lines (Sequence[_Line]): The lines the old elements own.
        rest (str): Text after the last element's line.
        value_texts (Sequence[str]): The token text of each new value.
        origins (Sequence[int | None]): The old position each new value stands in for.
        common_end (str): The line ending most elements use, for a new element's line.
        on_key_line (bool): Whether the first element shares its line with the key.
        position_owned (int | None): The old position whose line ending passes to whatever
            is last now, or ``None`` to leave every ending with its element.

    Returns:
        str: The text.
    """
    carried = _carried_above(lines, origins)
    indents = _indents(lines, origins, on_key_line)
    pieces = [head]
    final = len(origins) - 1
    for index, (value, origin) in enumerate(zip(value_texts, origins, strict=True)):
        above = "" if index == 0 and on_key_line else carried.get(index, "")
        # A new element, and the one whose ending passes to the last, end like the rest.
        takes_common = origin is None or (origin == position_owned and index != final)
        end = common_end if takes_common else lines[origin].end
        if index == final and position_owned is not None:
            end = lines[position_owned].end
        pieces += [above, indents[index], value, end]
    return "".join([*pieces, rest])


def _carried_above(lines: Sequence[_Line], origins: Sequence[int | None]) -> dict[int, str]:
    """Return the comment lines each new position carries above it.

    An element that stays keeps the lines above it, and gains those above any elements
    deleted just before it, so that removing an element removes its line but not the
    comments heading the lines that follow. Lines above a deleted last element have no
    element left to head, and go with it.

    Args:
        lines (Sequence[_Line]): The lines the old elements own.
        origins (Sequence[int | None]): The old position each new value stands in for.

    Returns:
        dict[int, str]: The lines to write above each new position that has any.
    """
    carried: dict[int, str] = {}
    previous = -1
    for index, origin in enumerate(origins):
        if origin is None:
            continue
        carried[index] = "".join(line.above for line in lines[previous + 1 : origin + 1])
        previous = origin
    return carried


def _indents(lines: Sequence[_Line], origins: Sequence[int | None], on_key_line: bool) -> list[str]:
    """Return the indent of each new position.

    An element that stays keeps its own indent, except for one moving onto the key's line,
    which needs none, and the old first element moving off it, which takes the indent of
    the element after it. A new element takes the deeper indent of the nearest elements
    that stay on either side, so one added inside a nested loop is indented like the loop's
    body rather than like the line opening it.

    Args:
        lines (Sequence[_Line]): The lines the old elements own.
        origins (Sequence[int | None]): The old position each new value stands in for.
        on_key_line (bool): Whether the first element shares its line with the key.

    Returns:
        list[str]: One indent per new position.
    """
    own = [_own_indent(lines, origin, on_key_line) for origin in origins]
    indents = []
    for index, indent in enumerate(own):
        if index == 0 and on_key_line:
            indents.append("")
        elif indent is not None:
            indents.append(indent)
        else:
            before = next((own[at] for at in range(index - 1, -1, -1) if own[at] is not None), "")
            after = next((own[at] for at in range(index + 1, len(own)) if own[at] is not None), "")
            indents.append(max(before, after, key=len))
    return indents


def _own_indent(lines: Sequence[_Line], origin: int | None, on_key_line: bool) -> str | None:
    """Return the indent an old element brings with it, or ``None`` for a new element.

    Args:
        lines (Sequence[_Line]): The lines the old elements own.
        origin (int | None): The old position, or ``None``.
        on_key_line (bool): Whether the first element shares its line with the key.

    Returns:
        str | None: The indent.
    """
    if origin is None:
        return None
    if origin == 0 and on_key_line:
        # It had no line of its own; off the key's line it is indented like the next one.
        return lines[1].indent
    return lines[origin].indent
