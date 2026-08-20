# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The grammar rule names every format grammar must use, and what each one promises.

These names are an API, not cosmetics. The reader is a Lark ``Interpreter``, which
dispatches on a node's rule name to a *method of that name*, and adding a key parses
synthesised text with a container rule as the start symbol, which needs that rule to be
nameable. So a grammar that spells one of these differently does not merely read oddly --
it silently stops working.

This module is the single statement of that contract. Collecting the names here means a
rename is one edit rather than a hunt through five modules. One duplication is irreducible:
the reader has to define a method literally named after each entry category, because that is
how Lark dispatches. ``ENTRY_CATEGORIES`` and those method names are held together by a test
rather than by construction.

The contract
------------
A grammar written against this package must:

- Express every key-value assignment with a rule named, or aliased with ``->``, to one of
  the four entry categories: ``key_value`` (one scalar), ``key_list`` (two or more values),
  ``key_block`` (a nested block of entries) and ``key_null`` (a key with no value, as in
  ``a=``). Aliasing is the normal case, so that one category can have several syntaxes:
  ``rfile_key_value: ws* key ":" ws* value line_end -> key_value``.
- Name the rule holding a key ``key``, with a ``Token`` as its first child whose text is the
  key name. ``config.lark`` provides one that suits most formats.
- Write scalar values using value-type rules imported from ``config.lark``, whose names are
  registered in ``VALUE_TYPE_HANDLER_REGISTRY``. Such a node has exactly one ``Token``
  child.
- Capture whitespace that must survive a round trip in the explicit ``ws`` rule, never
  discard it with ``%ignore``, and bind the run *before* an assignment literal into an
  ``eq`` rule with it -- see ``TRANSPARENT_RULES``.
- Name the top-level rule ``start``, and leave it non-transparent.
- Hold the contents of a ``key_block`` in a rule *named* ``block``, not one aliased to it.

Two rules aliased to the same category must produce **distinguishable children**. The
reconstructor picks the rule to write a node with by matching that node's children, and Lark
filters anonymous string tokens out of the tree before it looks, so punctuation that tells
two syntaxes apart has to sit in a rule of its own -- ``assign: "="`` in ``nuopc_config``,
``separator: ","`` in ``fortran_nml``.

Where the consequences are spelled out: the parse tree these rules produce in
``parse_tree_ops``, the value types in ``grammar_values``, how an entry is written back out
in ``entry_templates``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

START_RULE: Final = "start"
"""Name of the top-level rule, fixed by the grammar contract.

Must not be transparent (``start:``, never ``?start:``): a transparent rule collapses when
it has a single child, so a file holding one entry would have that entry as the root of the
tree, leaving no container to add a second entry to.
"""

KEY_RULE: Final = "key"
"""Name of the rule holding a key name, fixed by the grammar contract."""

BLOCK_RULE: Final = "block"
"""Name of the rule holding the contents of a block, fixed by the grammar contract.

Must be the rule's own name rather than an alias applied with ``->``: Lark resolves start
symbols against rule names, while a node carries the alias, and the two have to agree.
"""

WS_RULE: Final = "ws"
"""Name of the rule capturing whitespace, fixed by the grammar contract.

Whitespace that must survive a round trip is captured in this rule rather than discarded
with ``%ignore``, so that it can be reproduced verbatim -- and, when a key is added, copied
from a neighbouring entry.
"""

COMMENT_RULES: Final = ("comment", "fortran_comment")
"""Names of the rules holding the text of a comment, fixed by the grammar contract.

The shared grammar defines one rule per comment character in use, and both are rules rather
than terminals, so the name is what identifies a comment. The terminal each wraps is an
anonymous regular expression, which Lark names ``__ANON_0``.
"""

REPEAT_RULE: Final = "repeat"
"""Name of the rule holding a repeat count, ``n*v`` for *v* repeated *n* times.

Unlike a value-type rule it is not in ``VALUE_TYPE_HANDLER_REGISTRY``: it holds a count
token and, unless the repeat is of null values, the value-type rule node being repeated. One
such node therefore backs several list elements, which is what ``write_values`` has to undo
when one of them is written.
"""

TRANSPARENT_RULES: Final = frozenset({"eq"})
"""Rules that group whitespace with a punctuation literal, and hold nothing else.

A grammar needs one of these because an anonymous literal is filtered out of the parse tree,
so ``ws* "=" ws*`` records which whitespace runs exist but not which side of the ``=`` each
one fell on -- and the reconstructor then guesses, turning ``A= 1`` into ``A =1``. Binding
the leading whitespace to the literal in a rule of its own fixes the position.

Such a rule is a grouping device rather than structure, so reading an entry's whitespace
looks through it: the runs inside it space out the same things as the runs beside it do.
"""

KEY_VALUE: Final = "key_value"
"""Name of the entry category holding a key and one scalar value, as in ``a = 1``."""

KEY_LIST: Final = "key_list"
"""Name of the entry category holding a key and several values, as in ``a = 1, 2, 3``."""

KEY_BLOCK: Final = "key_block"
"""Name of the entry category holding a key and a nested block of entries."""

KEY_NULL: Final = "key_null"
"""Name of the entry category holding a key and no value at all, as in ``a =``."""

VALUE_SLOT_COUNTS: Mapping[str, int] = {
    KEY_VALUE: 1,
    KEY_LIST: 2,
    KEY_BLOCK: 0,
    KEY_NULL: 0,
}
"""Number of value placeholders a template for each entry category must have.

``KEY_LIST`` needs exactly two. The second is what makes the element separator visible, so
that it can be replicated for a list of any length -- see ``expand_value_slots``.
"""

ENTRY_CATEGORIES: Final = tuple(VALUE_SLOT_COUNTS)
"""Rule names (or aliases) that ``ConfigToDict`` recognises as a configuration entry.

Derived from ``VALUE_SLOT_COUNTS`` rather than written out again, so that adding a category
cannot leave the two disagreeing.
"""
