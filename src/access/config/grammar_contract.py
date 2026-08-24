# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The grammar rule names every format grammar must use.

These names are an API, not cosmetics. ``ConfigToDict`` is a Lark ``Interpreter``, which
dispatches on a node's rule name to a *method of that name*, and adding a key parses
synthesised text with a container rule as the start symbol, which needs that rule to be
nameable. So a grammar that spells one of these differently does not merely read oddly --
it silently stops working.

Collecting the names here means a rename is one edit rather than a hunt through five
modules. One duplication is irreducible: ``ConfigToDict`` has to define a method literally
named after each entry category, because that is how Lark dispatches. ``ENTRY_CATEGORIES``
and those method names are held together by a test rather than by construction.

The rules a grammar must provide, and where each is documented in full: the entry
categories and the ``key``, ``start`` and ``block`` rules in ``ConfigParser``; the parse
tree those rules produce in ``parse_tree_ops``; the value-type rules in ``parser_types``.
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
