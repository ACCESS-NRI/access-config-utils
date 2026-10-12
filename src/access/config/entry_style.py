# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""How a new entry should be written so that it looks like its neighbours.

Layout is copied, never invented. Everything here exists to answer one question -- how
should the entry about to be added be written -- and it answers it by looking at the entries
already in the file. ``probe_entry_style`` reads them; ``EntryStyle`` carries what it found.

It finds two things. The first is a whole entry to copy: ``donor_template`` turns the
nearest entry of the kind wanted into a template of its own, so that a format with several
ways of writing one entry -- quoted or bare, one delimiter or another -- gets the one the
file uses. The second is the indentation of the entry above, for the first entry of a kind,
which has none to copy and is written from the template its parser declares: the file still
shows how its entries are indented.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lark import Token, Tree

from access.config.entry_fragments import EntryTemplate, Fragment
from access.config.grammar_contract import (
    BLOCK_RULE,
    ELIDED_RULE,
    ENTRY_CATEGORIES,
    INDEX_RULE,
    KEY_BLOCK,
    KEY_LIST,
    KEY_RULE,
    REPEAT_RULE,
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
        indent (str | None): The whitespace the line of the last entry above starts with,
            or ``None`` when there is no entry above to show one, as in a block that was
            empty in the file.
        donors (tuple[tuple[DonorSignature, EntryTemplate], ...]): A template copied whole
            from the nearest entry of each kind, keyed by what it can write. A new entry of
            that kind is written as a copy of it before the template its parser declares is
            tried.
    """

    indent: str | None = None
    donors: tuple[tuple[DonorSignature, EntryTemplate], ...] = ()

    def donor(self, depth: int, category: str) -> EntryTemplate | None:
        """Return the template copied from an existing entry for a new one of this kind.

        A kind is a *depth* and a *category*, as in ``DonorSignature``. The result is
        ``None`` if no entry of the kind was found to copy.
        """
        return next((template for signature, template in self.donors if signature == (depth, category)), None)


# --- Style probing ---


def probe_entry_style(container: Tree, reconstructor: Reconstructor, upto: int | None = None) -> EntryStyle:
    """Read how the entries in *container* are written, to match when adding a new one.

    The indentation is that of the last entry above the new one. The last entry of each
    kind is also copied whole, as a template a new entry of the same kind is written with
    first; see ``donor_template``. Only an entry on a line of its own is copied, since a
    line holding several is laid out for all of them.

    An unusually spaced neighbour is matched rather than normalised: a key added beside
    ``A   =   1`` is written ``B   =   2``, since a minimal diff matters more than tidiness.

    *upto* is where the new entry is to go, as an index into the children of *container*:
    only the entries above it are read, so the nearest of them is the one written just
    before it. ``None`` reads them all.

    Falls back to a style with no indentation when the container holds no entries at all,
    which is the case for a block that was empty in the file. A block that was *just
    created* has nothing to copy either, but is not left bare: see
    ``probe_sibling_block_style``.
    """
    lines = container.children[:upto]
    entries = [entry for line in lines if isinstance(line, Tree) for entry in entry_nodes(line)]
    if not entries:
        return EntryStyle()
    return EntryStyle(indent_of(entries[-1]), _donors(lines, reconstructor))


def probe_sibling_block_style(container: Tree, reconstructor: Reconstructor, upto: int | None = None) -> EntryStyle:
    """Read the style used inside the blocks of *container*, for a block just created.

    A block created by an assignment starts empty, so ``probe_entry_style`` finds nothing in
    it to copy and its contents would be written with no indentation at all -- foreign to a
    file whose other blocks are indented. The style is therefore taken from a sibling block
    instead, searching from the nearest one upwards, since that is where the new block was
    added. This mirrors ``probe_entry_style`` taking its style from the *last* entry: in
    both cases the closest neighbour wins.

    Only the rule named ``BLOCK_RULE`` is looked at, and only as a direct child of the
    entry, which is the same contract ``EntryReader.key_block`` relies on to find the body
    of a block. For an entry that chains others, such as an FMS field table's
    ``"tracer", "ocean_mod", "age"``, that entry is the innermost, ``age``.

    *upto* is the index of the child of *container* the new block is written in, so that
    the search starts at the block just above it. ``None`` searches them all, from the
    last. The result is a style with no indentation when no sibling block holds an entry,
    which includes the new block itself, since it is empty.
    """
    lines = container.children[:upto]
    for entry in reversed([entry for line in lines if isinstance(line, Tree) for entry in entry_nodes(line)]):
        innermost = entry_chain(entry)[-1]
        if innermost.data != KEY_BLOCK:
            continue
        for child in innermost.children:
            if isinstance(child, Tree) and child.data == BLOCK_RULE:
                style = probe_entry_style(child, reconstructor)
                if style.indent is not None:
                    return style
    return EntryStyle()


def indent_of(entry: Tree) -> str:
    """Return the whitespace the line of an entry starts with: its ``ws`` before the key."""
    indent = ""
    for child in entry.children:
        if isinstance(child, Tree):
            if child.data == KEY_RULE:
                break
            if child.data == WS_RULE:
                indent += "".join(str(token) for token in child.children)
    return indent


# --- Donor templates ---
#
# A new entry of a kind the file already has is written as a copy of the nearest entry of
# that kind, whole: its spacing and its punctuation both, as a template with markers
# standing in for what a new entry writes differently.

_KEY_MARKER = "\x00K{}\x00"
_VALUE_MARKER = "\x00V\x00"
_MARKERS = re.compile("\x00(K[0-9]+|V)\x00")


def _donors(
    lines: Sequence[Tree | Token], reconstructor: Reconstructor
) -> tuple[tuple[DonorSignature, EntryTemplate], ...]:
    """Return a template copied from the last entry of each kind on *lines*.

    The search runs from the last line upwards, so only one entry of each kind is copied
    unless an entry cannot be, when the one above it is tried.
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
    the entry instead, it is taken from *line*, the container child *entry* is written in
    (which may be *entry* itself), and only it: a separator after the entry is part of how
    that line is written, not of how an entry is.

    The result is ``None`` for an entry that cannot be copied: one writing a repeat count
    or skipping a list position, which a new entry has no reason to copy, a list broken
    after its first value, or an entry holding several entries.
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

    *node* is link *level* of a chain of *depth* entries; the rule and token text of each
    value replaced go to *notations*. The result is False if the entry is not to be copied.
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

    *line* is the container child *entry* is written in. The result is empty if the entry
    holds its own line break or the line has none.
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

    *notations* is the rule and token text of each value marker, in order.
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
