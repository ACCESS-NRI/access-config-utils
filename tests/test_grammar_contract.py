# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the rule names every grammar must use.

The module is nothing but constants, so what is worth testing is the places two of them have
to agree. Both duplications below are held together here rather than by construction: one
because Lark dispatches on method names, the other because a category with no value count
declared would be silently unwritable.
"""

from access.config.grammar_contract import (
    BLOCK_RULE,
    COMMENT_RULES,
    ENTRY_CATEGORIES,
    KEY_BLOCK,
    KEY_LIST,
    KEY_NULL,
    KEY_RULE,
    KEY_VALUE,
    START_RULE,
    VALUE_SLOT_COUNTS,
    WS_RULE,
)
from access.config.tree_reader import EntryReader


def test_entry_categories_are_the_four_declared() -> None:
    """Test that the categories are exactly the four entry kinds, derived not repeated.

    ``ENTRY_CATEGORIES`` comes from the keys of ``VALUE_SLOT_COUNTS`` so that adding a
    category cannot leave the two disagreeing.
    """
    assert ENTRY_CATEGORIES == (KEY_VALUE, KEY_LIST, KEY_BLOCK, KEY_NULL)
    assert tuple(VALUE_SLOT_COUNTS) == ENTRY_CATEGORIES


def test_value_slot_counts_match_what_each_category_holds() -> None:
    """Test the number of values a template for each category must have.

    A list needs exactly two, not one per element: the second is what makes the element
    separator visible, so a list of any length can be grown from the pair.
    """
    assert VALUE_SLOT_COUNTS[KEY_VALUE] == 1
    assert VALUE_SLOT_COUNTS[KEY_LIST] == 2
    assert VALUE_SLOT_COUNTS[KEY_BLOCK] == 0
    assert VALUE_SLOT_COUNTS[KEY_NULL] == 0


def test_the_reader_has_a_callback_per_category() -> None:
    """Test that ``EntryReader`` implements every entry category the contract declares.

    This is the one duplication of the category names that cannot be removed: Lark's
    ``Interpreter`` dispatches on a node's rule name to a method of that name, so the
    callbacks have to be spelled out. Adding a category without its callback would leave
    entries of that category silently unvisited, so the two lists are pinned together here
    instead of by construction.
    """
    for category in ENTRY_CATEGORIES:
        assert callable(getattr(EntryReader, category, None)), category


def test_the_rule_names_are_the_ones_config_lark_defines() -> None:
    """Test the names themselves, which every format grammar has to spell out.

    They are an API rather than cosmetics: a grammar spelling one differently does not read
    oddly, it silently stops working. Writing them out once here is what makes a rename
    visible as a test change.
    """
    assert (START_RULE, KEY_RULE, BLOCK_RULE, WS_RULE) == ("start", "key", "block", "ws")
    assert COMMENT_RULES == ("comment", "fortran_comment")
    assert ENTRY_CATEGORIES == ("key_value", "key_list", "key_block", "key_null")
