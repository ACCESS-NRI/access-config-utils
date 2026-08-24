# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the checks an entry has to pass before it is written.

Insertion is covered end to end by the ``test_config_add_*`` tests and by the per-format
modules. What is tested here are the two decisions those rest on: which category a value
needs, and whether a candidate read back from its own text is really what was asked for.
"""

import pytest

from access.config.config_insert import entry_category, entry_matches
from access.config.grammar_contract import ENTRY_CATEGORIES, VALUE_SLOT_COUNTS


def test_entry_matches_rejections(parser):
    """Test the checks that reject a candidate entry read back from its text.

    The candidate is spelled as the text a synthesised entry would be, and then read back
    through the ordinary path, so the references compared here are the ones insertion sees.
    """

    def read(text):
        return parser.parse(text)._refs

    # A candidate that produced the wrong key, or more than one entry.
    assert not entry_matches(read("other=1"), "z", 1)
    assert not entry_matches(read("z=1 other=2"), "z", 1)

    # The value read back has to match in type as well as in value.
    assert not entry_matches(read("z=1"), "z", True)
    assert not entry_matches(read("z='1'"), "z", 1)
    assert not entry_matches(read("z=2"), "z", 1)
    assert entry_matches(read("z=1"), "z", 1)

    # A valueless entry, and a list of the wrong length or element type.
    assert not entry_matches(read("z=1"), "z", None)
    assert entry_matches(read("z="), "z", None)
    assert not entry_matches(read("z=1,2"), "z", [1, 2, 3])
    assert not entry_matches(read("z=1"), "z", [1, 2])
    assert not entry_matches(read("z=1,'2'"), "z", [1, 2])
    assert entry_matches(read("z=1,2"), "z", [1, 2])

    # A new block has to come back empty; its contents are added afterwards.
    assert not entry_matches(read("z=1"), "z", {"a": 1})
    assert not entry_matches(read("z<a:1>"), "z", {"a": 1})
    assert entry_matches(read("z< >"), "z", {"a": 1})


@pytest.mark.parametrize(
    ("value", "expected"),
    [({}, "key_block"), (None, "key_null"), ([1, 2], "key_list"), (1, "key_value"), ("s", "key_value")],
)
def test_entry_category_is_a_declared_category(value, expected) -> None:
    """Test that the category chosen for a value is one the contract declares."""
    assert entry_category(value) == expected
    assert entry_category(value) in ENTRY_CATEGORIES
    assert entry_category(value) in VALUE_SLOT_COUNTS
