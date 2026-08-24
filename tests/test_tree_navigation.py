# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for finding one's way around a parse tree.

The predicates here decide where a new entry goes, so they are exercised directly rather
than only through the end-to-end insertion tests.
"""

import pytest
from lark import Lark

from access.config.grammar_compiled import compile_grammar
from access.config.mom6_input import MOM6InputParser
from access.config.tree_navigation import contains_entry, entries_of, is_comment_line


@pytest.fixture(scope="module")
def lark() -> Lark:
    """Fixture returning a compiled parser for the MOM6 grammar."""
    return compile_grammar(MOM6InputParser().grammar).lark


def test_contains_entry(lark) -> None:
    """Test the test for a container child holding an entry."""
    tree = lark.parse("A = 1\n\n", start="start")

    kinds = [contains_entry(child) for child in tree.children]
    assert kinds.count(True) == 1
    # A blank line is a child, but not an entry.
    assert False in kinds


def test_is_comment_line(lark) -> None:
    """Test the test for a container child holding nothing but a comment.

    A comment-only line and a blank line are the same rule, and an entry's own trailing
    comment must not make its line count as one.
    """
    tree = lark.parse("A = 1 ! trailing\n! standalone\n\n", start="start")

    assert [is_comment_line(child) for child in tree.children] == [False, True, False]


def test_entries_of_stops_at_a_nested_block(lark) -> None:
    """Test that the entries of a container exclude those of a block inside it."""
    tree = lark.parse("A = 1\nB%\nX = 1\n%B\n", start="start")

    # The outer container sees its own scalar and the block entry, but not the block's X.
    assert [str(entry.data) for entry in entries_of(tree)] == ["key_value", "key_block"]
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")
    assert [str(entry.data) for entry in entries_of(block)] == ["key_value"]
