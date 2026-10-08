# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""How a new entry should be written so that it looks like its neighbours.

Layout is copied, never invented. Everything here exists to answer one question -- how
should the entry about to be added be written -- and it answers it by looking at the entries
already in the file. ``probe_entry_style`` reads them; ``EntryStyle`` carries what it found.

It finds two things. The first is a whole entry to copy: ``donor_template`` turns the
nearest entry of the kind wanted into a template of its own, so that a format with several
ways of writing one entry -- quoted or bare, one delimiter or another -- gets the one the
file uses rather than the one the grammar's ranking prefers. The second is whitespace, for a
template the grammar derives when there is no such entry: ``canonical_ws`` and the slot
classification decide where each recorded run goes, and what to write where nothing was
recorded.

Where a grammar admits no whitespace at the position a donated indent would occupy, the
indent is dropped rather than written into a line that would not parse. That is what
``style_matches`` is for: it lets ranking prefer a template that *can* reproduce the
neighbouring style over one that cannot.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lark import Token, Tree

from access.config.entry_fragments import EntryTemplate, Fragment, neighbour_fragment
from access.config.grammar_contract import (
    BLOCK_RULE,
    ELIDED_RULE,
    ENTRY_CATEGORIES,
    INDEX_RULE,
    KEY_BLOCK,
    KEY_LIST,
    KEY_RULE,
    REPEAT_RULE,
    TRANSPARENT_RULES,
    WS_RULE,
)
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY
from access.config.tree_edits import blank_comments
from access.config.tree_navigation import entry_chain, entry_nodes

if TYPE_CHECKING:
    from lark.reconstruct import Reconstructor

DonorSignature = tuple[int, str]
"""What a template copied from an entry can write: how many keys deep, and what category.

The depth counts the keys an entry chains together, so it is 1 for ``a = 1`` and 2 for the
Fortran ``a%b = 1``. The category is the innermost entry's.
"""


@dataclass(frozen=True)
class EntryStyle:
    """What a new entry copies from the existing ones, so that it matches its neighbours.

    Args:
        indent (str): Whitespace before the key.
        key_pads (tuple[str, ...] | None): Whitespace runs between the key and the first
            value, in order, or ``None`` when no entry demonstrates it.
        key_split (int): How many of *key_pads* fall before the assignment itself. A single
            run is otherwise ambiguous -- ``x =1`` and ``x= 1`` both record one -- and the
            new entry would take whichever side the chosen template's slot sits on.
        element_pads (tuple[str, ...] | None): Whitespace runs between the first and second
            values, in order, or ``None`` when no entry demonstrates it. That is the case
            whenever no entry holds two values, so a scalar neighbour does not dictate how a
            list is spaced.
        has_donor (bool): Whether any entry was found to copy from. When false, rendering
            falls back to a canonical layout.
        donors (tuple[tuple[DonorSignature, EntryTemplate], ...]): A template copied whole
            from the nearest entry of each kind, keyed by what it can write. A new entry of
            that kind is written as a copy of it before any template the grammar derives
            is tried.
    """

    indent: str = ""
    key_pads: tuple[str, ...] | None = None
    key_split: int = 0
    element_pads: tuple[str, ...] | None = None
    has_donor: bool = False
    donors: tuple[tuple[DonorSignature, EntryTemplate], ...] = ()

    def donor(self, depth: int, category: str) -> EntryTemplate | None:
        """Return the template copied from an existing entry for a new one of this kind.

        Args:
            depth (int): How many keys the new entry chains together; 1 for a plain entry.
            category (str): The category of the innermost entry.

        Returns:
            EntryTemplate | None: The template, or ``None`` if no entry of the kind was
                found to copy.
        """
        return next((template for signature, template in self.donors if signature == (depth, category)), None)


# --- Whitespace policy ---


def _is_punctuation(text: str) -> bool:
    """Report whether *text* is non-empty and made only of punctuation."""
    return text != "" and not any(character.isalnum() or character.isspace() for character in text)


def canonical_ws(template: EntryTemplate, index: int) -> str:
    """Return the whitespace to emit at *index* when there is no entry to copy from.

    Empty at the start and the end of a line, before a separator that follows a value or
    any comma, and between punctuation and a key, so a canonically rendered entry carries no
    indentation, no trailing whitespace, and reads as ``a, b`` rather than ``a , b`` and
    ``a%b`` rather than ``a% b``. A single space anywhere else, which is where a grammar
    allows optional whitespace for readability, as in ``KEY = value``.

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
    if after.kind == "literal" and after.text.startswith(","):
        return ""
    if before.kind == "literal" and _is_punctuation(before.text) and after.kind == "key":
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
        key_split (int): How many of *key* precede the assignment literal.
        element (tuple[int, ...]): Stylable placeholders between the first two values.
    """

    leading: tuple[int, ...]
    key: tuple[int, ...]
    key_split: int
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
    key_slots = stylable(first_key, key_end)
    # The assignment is the first fixed text after the key, so the slots before it space out
    # the key and the ones after it space out the value.
    literals = [index for index in range(first_key, key_end) if template[index].kind == "literal"]
    boundary = literals[0] if literals else key_end
    key_split = sum(1 for index in key_slots if index < boundary)
    return _Slots(leading, key_slots, key_split, stylable(element_start, element_end))


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
    # The counts can agree while the runs sit on the wrong side of the assignment.
    return style.key_pads is None or slots.key_split == style.key_split


# --- Style probing ---


def probe_entry_style(
    container: Tree, reconstructor: Reconstructor | None = None, upto: int | None = None
) -> EntryStyle:
    """Read how the entries in *container* are written, to match when adding a new one.

    Each kind of whitespace is taken from the last entry that actually demonstrates it: the
    indentation and the spacing around the assignment from the last entry with both a key
    and a value, and the spacing around the element separator from the last entry with two
    values. So a scalar neighbour does not dictate how a list is spaced, and a list
    neighbour does not stop a scalar being spaced like the scalars around it.

    Given a reconstructor, the last entry of each kind is also copied whole, as a template a
    new entry of the same kind is written with first; see ``donor_template``. Only an entry
    on a line of its own is copied, since a line holding several is laid out for all of
    them.

    Falls back to a canonical layout when the container holds no entries at all, which is
    the case for a block that was empty in the file. A block that was *just created* has
    nothing to copy from either, but is not left canonical: see
    ``probe_sibling_block_style``.

    Args:
        container (Tree): The container rule node the new entry will be added to.
        reconstructor (Reconstructor | None): Writes a parse tree out as text, to copy
            whole entries with; ``None`` to read the whitespace alone.
        upto (int | None): Where the new entry is to go, as an index into the children of
            *container*: only the entries above it are read, so the nearest of them is the
            one written just before it. ``None`` reads them all.

    Returns:
        EntryStyle: The layout to copy, or a style with ``has_donor`` false.

    An unusually spaced neighbour is matched rather than normalised: a key added beside
    ``A   =   1`` is written ``B   =   2``, since a minimal diff matters more than tidiness.
    """
    lines = container.children[:upto]
    entries = [entry for line in lines if isinstance(line, Tree) for entry in entry_nodes(line)]
    if not entries:
        return EntryStyle()

    shapes = [_shape_of(entry) for entry in entries]
    # A block entry shows the indentation but carries no assignment, so it says nothing
    # about the spacing around one -- and taking its empty run would write "KEY=value". A
    # valueless entry does show that spacing, just not how elements are set out.
    with_value = [shape for shape in shapes if shape[3]]
    with_assignment = [shape for entry, shape in zip(entries, shapes, strict=True) if entry.data != KEY_BLOCK]
    assigning = with_value or with_assignment
    with_elements = [shape for shape in shapes if len(shape[3]) > 1]
    element_pads = with_elements[-1][3][1] if with_elements else None
    key_pads, split = (assigning[-1][1], assigning[-1][2]) if assigning else (None, 0)
    donors = _donors(lines, reconstructor) if reconstructor is not None else ()
    return EntryStyle(shapes[-1][0], key_pads, split, element_pads, has_donor=True, donors=donors)


def probe_sibling_block_style(
    container: Tree, reconstructor: Reconstructor | None = None, upto: int | None = None
) -> EntryStyle:
    """Read the whitespace used inside the blocks of *container*, for a block just created.

    A block created by an assignment starts empty, so ``probe_entry_style`` finds nothing in
    it to copy and its contents would be written with no indentation at all -- foreign to a
    file whose other blocks are indented. The style is therefore taken from a sibling block
    instead, searching from the nearest one upwards, since that is where the new block was
    added. This mirrors ``probe_entry_style`` taking its style from the *last* entry: in
    both cases the closest neighbour wins.

    Only the rule named ``BLOCK_RULE`` is looked at, and only as a direct child of the
    entry, which is the same contract ``ConfigToDict.key_block`` relies on to find the body
    of a block. For an entry that chains others, such as an FMS field table's
    ``"tracer", "ocean_mod", "age"``, that entry is the innermost, ``age``.

    Args:
        container (Tree): The container rule node the new block was added to.
        reconstructor (Reconstructor | None): Writes a parse tree out as text, to copy
            whole entries with; ``None`` to read the whitespace alone.
        upto (int | None): The index of the child of *container* the new block is written
            in, so that the search starts at the block just above it. ``None`` searches
            them all, from the last.

    Returns:
        EntryStyle: The layout to copy, or a style with ``has_donor`` false when no sibling
            block holds an entry -- including the new block itself, which is empty.

    A new block is laid out like the block next to it, not like a fresh file: if the block
    above indents its entries, so does this one.
    """
    lines = container.children[:upto]
    for entry in reversed([entry for line in lines if isinstance(line, Tree) for entry in entry_nodes(line)]):
        innermost = entry_chain(entry)[-1]
        if innermost.data != KEY_BLOCK:
            continue
        for child in innermost.children:
            if isinstance(child, Tree) and child.data == BLOCK_RULE:
                style = probe_entry_style(child, reconstructor)
                if style.has_donor:
                    return style
    return EntryStyle()


def _shape_of(entry: Tree) -> tuple[str, tuple[str, ...], int, list[tuple[str, ...]]]:
    """Split the whitespace runs of an entry rule node into the parts a style is built from.

    The runs between the key and the first value come with a count of how many of them fall
    *before* the assignment itself. Without it a single run is ambiguous: ``x =1`` and
    ``x= 1`` both record one run, and a template with one whitespace slot would reproduce
    whichever side its slot happens to be on.

    Args:
        entry (Tree): An entry rule node.

    Returns:
        tuple[str, tuple[str, ...], int, list[tuple[str, ...]]]: The indentation, the runs
            between the key and the first value, how many of those precede the assignment,
            and the runs before each value in turn -- so the second element, when present,
            is the element separator.
    """
    kinds: list[tuple[str, str]] = []
    for child in (child for child in entry.children if isinstance(child, Tree)):
        if child.data in TRANSPARENT_RULES:
            # A transparent rule groups whitespace with the assignment literal. Its runs
            # count as though written beside the key, and the marker records where the
            # literal fell so the two sides stay apart.
            for grandchild in child.children:
                if isinstance(grandchild, Tree) and grandchild.data == WS_RULE:
                    kinds.append(("ws", "".join(str(token) for token in grandchild.children)))
            kinds.append(("assign", ""))
        elif child.data == WS_RULE:
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
    assigns = [index for index, (kind, _) in enumerate(kinds) if kind == "assign"]

    # The run before each value, measured from the key for the first and from the previous
    # value thereafter. With no value at all, everything after the key spaces out the key.
    if values:
        boundaries = [key_at, *values[:-1]]
        gaps = [runs(previous, index) for previous, index in zip(boundaries, values, strict=True)]
        key_pads = gaps[0]
    else:
        gaps = []
        key_pads = runs(key_at, len(kinds))
    # With no marker at all the grammar does not group whitespace with its assignment, so
    # every run counts as preceding it and the split constrains nothing.
    split = len(runs(key_at, assigns[0])) if assigns else len(key_pads)
    return "".join(runs(0, key_at)), key_pads, split, gaps


# --- Donor templates ---
#
# Copying an entry's whitespace into a template the grammar derives reproduces its spacing,
# but not its punctuation: a format with several ways of writing one entry gets whichever
# the ranking prefers. So the nearest entry of the kind wanted is also copied whole, as a
# template of its own, with markers standing in for what a new entry writes differently.

_KEY_MARKER = "\x00K{}\x00"
_VALUE_MARKER = "\x00V\x00"
_MARKERS = re.compile("\x00(K[0-9]+|V)\x00")


def _donors(
    lines: Sequence[Tree | Token], reconstructor: Reconstructor
) -> tuple[tuple[DonorSignature, EntryTemplate], ...]:
    """Return a template copied from the last entry of each kind on *lines*.

    The search runs from the last line upwards, so only one entry of each kind is copied
    unless an entry cannot be, when the one above it is tried.

    Args:
        lines (Sequence[Tree | Token]): Children of a container rule node.
        reconstructor (Reconstructor): Writes a parse tree out as text.

    Returns:
        tuple[tuple[DonorSignature, EntryTemplate], ...]: Each kind found, with its
            template.
    """
    found: dict[DonorSignature, EntryTemplate] = {}
    for line in reversed(lines):
        entries = entry_nodes(line) if isinstance(line, Tree) else []
        if len(entries) != 1:
            continue
        chain = entry_chain(entries[0])
        signature = (len(chain), str(chain[-1].data))
        if signature not in found:
            template = donor_template(line, entries[0], reconstructor)
            if template is not None:
                found[signature] = template
    return tuple(found.items())


def donor_template(line: Tree, entry: Tree, reconstructor: Reconstructor) -> EntryTemplate | None:
    """Return a template that writes a new entry the way *entry* is written.

    The entry is written out with a marker in place of each key and each value, and the
    text between the markers becomes fixed text, whitespace and punctuation alike. A key
    marker records which key of a chain it stands for, and a value marker the rule and the
    token text it replaced, so that a new value written with the same rule can keep the
    notation. What belongs to this entry alone is left out: its comments, an array
    qualifier, and the contents of a block. A list keeps its first separator and the text
    after its last value, which is what ``expand_value_slots`` needs to grow it to any
    length.

    A list is copied only if its first two values are on one line. Where a list breaks, the
    indentation of the next line is that value's own -- aligned under its key, or nested in
    a run sequence loop -- and a new list repeating it after every value would not look like
    anything in the file.

    The line break comes too. In most formats it is part of the entry; where a line wraps
    the entry instead, it is taken from the line, and only it: a separator after the entry
    is part of how that line is written, not of how an entry is.

    Args:
        line (Tree): The container child *entry* is written in, which may be *entry*.
        entry (Tree): The entry to copy, with any entries chained inside it.
        reconstructor (Reconstructor): Writes a parse tree out as text.

    Returns:
        EntryTemplate | None: The template, or ``None`` for an entry that cannot be copied:
            one writing a repeat count or skipping a list position, which a new entry has
            no reason to copy, a list broken after its first value, or an entry holding
            several entries.
    """
    copied = copy.deepcopy(entry)
    notations: list[tuple[str, str]] = []
    if not _mark(copied, len(entry_chain(copied)), 0, notations):
        return None
    blank_comments(copied)
    text = reconstructor.reconstruct(copied) + _line_break_after(line, entry)
    fragments = _fragments(text, notations)
    if entry_chain(entry)[-1].data != KEY_LIST:
        return fragments
    values = [index for index, fragment in enumerate(fragments) if fragment.kind == "value"]
    if any("\n" in fragment.text for fragment in fragments[values[0] : values[1]]):
        return None
    return fragments[: values[1] + 1] + fragments[values[-1] + 1 :]


def _mark(node: Tree, depth: int, level: int, notations: list[tuple[str, str]]) -> bool:
    """Put markers in place of the keys and the values below *node*, in a copied entry.

    Args:
        node (Tree): The rule node to mark, changed in place.
        depth (int): How many entries the chain *node* is in holds.
        level (int): Which of them *node* is in.
        notations (list[tuple[str, str]]): Collects the rule and the token text of each
            value replaced, in the order written.

    Returns:
        bool: False if the entry is not one to copy.
    """
    node.children = [child for child in node.children if not (isinstance(child, Tree) and child.data == INDEX_RULE)]
    return all(_mark_child(child, depth, level, notations) for child in node.children if isinstance(child, Tree))


def _mark_child(child: Tree, depth: int, level: int, notations: list[tuple[str, str]]) -> bool:
    """Mark one rule node of a copied entry, and what is below it; see ``_mark``."""
    if child.data in (REPEAT_RULE, ELIDED_RULE):
        return False
    if child.data == KEY_RULE:
        _write_token(child, _KEY_MARKER.format(level))
    elif child.data in VALUE_TYPE_HANDLER_REGISTRY:
        notations.append((str(child.data), _write_token(child, _VALUE_MARKER)))
    elif child.data == BLOCK_RULE:
        child.children = []
    elif child.data in ENTRY_CATEGORIES:
        # Only the next link of a chain: an entry holding several is no chain.
        return level + 1 < depth and _mark(child, depth, level + 1, notations)
    else:
        return _mark(child, depth, level, notations)
    return True


def _write_token(node: Tree, text: str) -> str:
    """Write *text* as the token of a key or a value node, and return what it replaced."""
    token = node.children[0]
    assert isinstance(token, Token)
    node.children[0] = token.update(value=text)
    return str(token)


def _line_break_after(line: Tree, entry: Tree) -> str:
    """Return the line break ending the line *entry* is in, if the entry leaves it out.

    Args:
        line (Tree): The container child *entry* is written in.
        entry (Tree): The entry.

    Returns:
        str: The line break, or nothing if the entry holds its own or the line has none.
    """
    written = ""
    for text in _written_after(line, entry):
        written += text
        found = re.search(r"\r?\n", written)
        if found is not None:
            return found.group()
    return ""


def _written_after(node: Tree, entry: Tree) -> Iterator[str]:
    """Yield the text of each token below *node* written after *entry*, in order."""
    stack: list[Tree | str] = list(reversed(node.children))
    passed = False
    while stack:
        child = stack.pop()
        if child is entry:
            passed = True
        elif isinstance(child, Tree):
            stack.extend(reversed(child.children))
        elif passed:
            yield str(child)


def _fragments(text: str, notations: Sequence[tuple[str, str]]) -> EntryTemplate:
    """Split the text of a marked entry into fixed text and placeholders.

    Args:
        text (str): The text, with markers in place of the keys and the values.
        notations (Sequence[tuple[str, str]]): The rule and token text of each value
            marker, in order.

    Returns:
        EntryTemplate: The fragments.
    """
    remaining = iter(notations)
    fragments = []
    # Splitting on a pattern with a group keeps what the group matched at the odd indices.
    for index, piece in enumerate(_MARKERS.split(text)):
        if index % 2 == 0:
            if piece:
                fragments.append(Fragment("literal", piece))
        elif piece == "V":
            fragments.append(Fragment("value", notation=next(remaining)))
        else:
            fragments.append(Fragment("key", level=int(piece[1:])))
    return tuple(fragments)
