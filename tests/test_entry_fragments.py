# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the template type and the operations on one template alone."""

import pytest

from access.config.entry_fragments import Fragment, compile_template, expand_value_slots
from access.config.entry_render import render
from access.config.entry_style import EntryStyle


def test_expand_value_slots() -> None:
    """Test that a two-value template grows by repeating its element separator."""
    template = (
        Fragment("key"),
        Fragment("literal", "="),
        Fragment("value"),
        Fragment("literal", ","),
        Fragment("ws"),
        Fragment("value"),
    )

    grown = expand_value_slots(template, 4)
    assert sum(1 for fragment in grown if fragment.kind == "value") == 4
    assert render(grown, "K", ["1", "2", "3", "4"], EntryStyle()) == "K=1, 2, 3, 4"


def test_compile_template() -> None:
    """Test the compilation of a parser-declared template."""
    template = compile_template("{key} = {value}\n", "key_value", frozenset({"integer"}))
    assert render(template, "K", ["1"], EntryStyle()) == "K = 1\n"

    block = compile_template("{key}%\n{block}%{key}\n", "key_block", frozenset())
    assert render(block, "K", [], EntryStyle()) == "K%\n%K\n"


def test_compile_template_validation() -> None:
    """Test that a template whose markers do not fit its category is rejected."""
    # Wrong number of values for the category.
    with pytest.raises(ValueError):
        compile_template("{key}={value}", "key_null", frozenset())
    with pytest.raises(ValueError):
        compile_template("{key}={value}", "key_list", frozenset())
    # A block template needs exactly one block marker.
    with pytest.raises(ValueError):
        compile_template("{key}=", "key_block", frozenset())
    # Every template needs a key.
    with pytest.raises(ValueError):
        compile_template("={value}", "key_value", frozenset())
