# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adding an entry a configuration does not have yet.

Everything ``insert_entry`` calls is faked: the candidates it is offered, the parse of each,
and the read-back it validates against. What is left is the module's own decision-making --
which candidate to accept, when to give up, and what to splice where -- and it can be driven
one branch at a time. A candidate that parses but reads back wrong is a single line here; it
would take a purpose-built grammar to arrange through a real parser.

That the accepted text is one a real grammar actually produces is proved elsewhere, by
``test_parser.py`` and the four format modules.
"""

from typing import Any

import pytest
from conftest import FakeContext, entry_node, value_node
from lark import Token, Tree
from lark.exceptions import UnexpectedInput

from access.config import config_insert
from access.config.config_insert import entry_category, entry_matches, insert_entry
from access.config.entry_style import EntryStyle
from access.config.grammar_values import UnsupportedEntryError
from access.config.tree_reader import EntryRef


@pytest.fixture
def container() -> Tree:
    """Return a container node holding one entry, so an insertion has a neighbour."""
    return Tree("start", [entry_node("key_value", "a", value_node())])


@pytest.fixture
def offer(monkeypatch):
    """Return a function scripting one insertion attempt end to end.

    Each candidate is declared as ``(text, outcome)``: a node list to splice, an exception
    the parse should raise, or the refs a read-back should report. The fakes patch the
    module under test, since every import here binds its own reference.
    """

    def _offer(*candidates: tuple[str, Any], admitted: set[str] | None = None):
        scripted = dict(candidates)
        # Which candidate the reader is being asked about. Tracked rather than deduced from
        # the tree, because two candidates can parse into structurally equal nodes.
        current: list[str] = []

        def fake_snippets(*args, **kwargs):
            return iter([text for text, _ in candidates])

        def fake_parse(lark, container_rule, snippet):
            outcome = scripted[snippet]
            if isinstance(outcome, Exception):
                raise outcome
            current.append(snippet)
            return [Tree("line", [outcome[1]])]

        def fake_read(tree, ctx):
            refs, _ = scripted[current[-1]]
            if isinstance(refs, Exception):
                raise refs
            return refs

        monkeypatch.setattr(config_insert, "iter_entry_snippets", fake_snippets)
        monkeypatch.setattr(config_insert, "parse_entry_nodes", fake_parse)
        monkeypatch.setattr(config_insert, "read_entries", fake_read)
        monkeypatch.setattr(config_insert, "entry_insertion_index", lambda c: 1)
        monkeypatch.setattr(config_insert, "admitted_value_rules", lambda i, c, cat: frozenset(admitted or ()))

    return _offer


def scalar_ref(key: str = "z") -> tuple[dict[str, EntryRef], Tree]:
    """Return a read-back reporting one scalar entry, and the node it hangs from."""
    node = entry_node("key_value", key, value_node())
    return {key: EntryRef("key_value", node, (node.children[1],))}, node


class TestEntryCategory:
    """Which kind of entry a Python value needs."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ({}, "key_block"),
            ({"a": 1}, "key_block"),
            (None, "key_null"),
            ([1, 2], "key_list"),
            (1, "key_value"),
            ("s", "key_value"),
            (True, "key_value"),
        ],
    )
    def test_category_for_a_value(self, value, expected) -> None:
        """Test the mapping from a Python value to the entry that has to hold it."""
        assert entry_category(value) == expected


class TestEntryMatches:
    """The check a candidate has to pass before anything is written."""

    def test_accepts_an_exact_match(self) -> None:
        """Test the case that lets an insertion proceed."""
        refs, _ = scalar_ref()
        assert entry_matches(refs, "z", 1)

    def test_rejects_the_wrong_key_or_too_many_entries(self) -> None:
        """Test that a candidate producing anything but the one requested key is refused."""
        refs, _ = scalar_ref("other")
        assert not entry_matches(refs, "z", 1)

        two, _ = scalar_ref("z")
        two.update(scalar_ref("other")[0])
        assert not entry_matches(two, "z", 1)

    def test_rejects_a_value_read_back_as_another_type(self) -> None:
        """Test that the type is compared as well as the value.

        Several value types share a Python type, so a candidate can be written with exactly
        the right text and still be read back meaning something else.
        """
        refs, _ = scalar_ref()
        assert not entry_matches(refs, "z", True)
        assert not entry_matches(refs, "z", 2)

    def test_rejects_a_candidate_of_the_wrong_category(self) -> None:
        """Test that a scalar offered for a list, or the reverse, is refused."""
        refs, _ = scalar_ref()
        assert not entry_matches(refs, "z", [1, 2])
        assert not entry_matches(refs, "z", None)
        assert not entry_matches(refs, "z", {"a": 1})

    def test_a_valueless_entry_needs_no_value_check(self) -> None:
        """Test that ``key_null`` matches on its category alone."""
        node = entry_node("key_null", "z")
        assert entry_matches({"z": EntryRef("key_null", node)}, "z", None)

    def test_checks_a_list_element_by_element(self) -> None:
        """Test that length and every element's type have to agree."""
        values = (value_node(text="1"), value_node(text="2"))
        node = entry_node("key_list", "z", *values)
        refs = {"z": EntryRef("key_list", node, values)}

        assert entry_matches(refs, "z", [1, 2])
        assert not entry_matches(refs, "z", [1, 2, 3])
        assert not entry_matches(refs, "z", [1, "2"])

    def test_a_new_block_has_to_come_back_empty(self) -> None:
        """Test that a block is accepted only when it holds nothing yet.

        Its contents are added afterwards, one key at a time; a candidate arriving with
        entries already in it means the grammar wrote something other than a fresh block.
        """
        empty = Tree("block", [])
        filled = Tree("block", [entry_node("key_value", "p", value_node())])

        for body, expected in ((empty, True), (filled, False)):
            node = entry_node("key_block", "z", body)
            refs = {"z": EntryRef("key_block", node, block_node=body)}
            assert entry_matches(refs, "z", {"a": 1}) is expected


class TestInsertEntryRejections:
    """What is refused before any candidate is even asked for."""

    @pytest.mark.parametrize("key", ["", "a b", "2fast", "a-b", "a.b"])
    def test_a_key_that_is_not_a_usable_name(self, container, key) -> None:
        """Test that the key has to be spellable in the file: it is written verbatim."""
        with pytest.raises(ValueError, match="Not a valid configuration key"):
            insert_entry(container, FakeContext(), key, key, 1, EntryStyle())

    def test_a_key_that_is_not_a_string(self, container) -> None:
        """Test that a non-string key is refused rather than formatted into the text."""
        with pytest.raises(ValueError, match="Not a valid configuration key"):
            insert_entry(container, FakeContext(), "1", 1, 1, EntryStyle())

    @pytest.mark.parametrize("value", [[], [1]])
    def test_a_list_too_short_to_be_one(self, container, value) -> None:
        """Test the two-element floor.

        Every supported format needs at least two values to be a list rather than a scalar,
        and writing a scalar would put the wrong type in the dict.
        """
        with pytest.raises(ValueError, match="at least two elements"):
            insert_entry(container, FakeContext(), "z", "z", value, EntryStyle())


class TestInsertEntryCandidates:
    """Working through the candidates until one is both parseable and right."""

    def test_splices_the_first_acceptable_candidate(self, container, offer) -> None:
        """Test the accepted path: nodes land at the index, and the ref comes back."""
        refs, node = scalar_ref()
        offer(("z = 1\n", (refs, node)))

        ref = insert_entry(container, FakeContext(), "z", "z", 1, EntryStyle())

        assert ref is refs["z"]
        # Spliced at the index reported, not appended.
        assert len(container.children) == 2
        assert container.children[1].children == [node]

    def test_skips_a_candidate_that_does_not_parse(self, container, offer) -> None:
        """Test that a candidate the grammar rejects is passed over silently."""
        refs, node = scalar_ref()
        offer(("bad\n", UnexpectedInput()), ("z = 1\n", (refs, node)))

        assert insert_entry(container, FakeContext(), "z", "z", 1, EntryStyle()) is refs["z"]

    @pytest.mark.parametrize("error", [ValueError("no values"), TypeError("no key")])
    def test_skips_a_candidate_the_reader_rejects(self, container, offer, error) -> None:
        """Test that a candidate parsing into a shape the reader refuses is passed over.

        Both are shapes a grammar can produce and the reader cannot use, so neither may
        escape as an exception from an insertion that has a good candidate behind it.
        """
        refs, node = scalar_ref()
        offer(("odd\n", (error, Tree("line", []))), ("z = 1\n", (refs, node)))

        assert insert_entry(container, FakeContext(), "z", "z", 1, EntryStyle()) is refs["z"]

    def test_skips_a_candidate_that_reads_back_wrong(self, container, offer) -> None:
        """Test the check that makes ranking a matter of style rather than of correctness.

        The first candidate parses cleanly and means something else -- a bare word a format
        reads as a path, say. It has to be discarded rather than written out.
        """
        wrong, wrong_node = scalar_ref()
        right, right_node = scalar_ref()
        offer(("z = foo\n", (wrong, wrong_node)), ("z = 1\n", (right, right_node)))

        # The first read-back reports the string "1", not the integer asked for.
        wrong["z"] = EntryRef("key_value", wrong_node, (value_node("string", "'1'"),))

        assert insert_entry(container, FakeContext(), "z", "z", 1, EntryStyle()) is right["z"]

    def test_nothing_is_spliced_until_a_candidate_passes(self, container, offer) -> None:
        """Test that a rejected candidate leaves the tree untouched."""
        before = list(container.children)
        offer(("bad\n", UnexpectedInput()))

        with pytest.raises(UnsupportedEntryError):
            insert_entry(container, FakeContext(), "z", "z", 1, EntryStyle())

        assert container.children == before


class TestInsertEntryExhausted:
    """What is reported when no candidate can express the value."""

    def test_names_the_value_the_key_and_the_container(self, container, offer) -> None:
        """Test that the error says what could not be written and where."""
        offer()

        with pytest.raises(UnsupportedEntryError, match="cannot store .* as 'z'"):
            insert_entry(container, FakeContext(), "z", "z", object(), EntryStyle())

    def test_lists_the_value_types_the_container_admits(self, container, offer) -> None:
        """Test the hint, which is what turns "no" into something actionable."""
        offer(admitted={"integer", "float"})

        with pytest.raises(UnsupportedEntryError, match=r"value types allowed here: \['float', 'integer'\]"):
            insert_entry(container, FakeContext(), "z", "z", "text", EntryStyle())

    def test_omits_the_hint_where_nothing_is_admitted(self, container, offer) -> None:
        """Test that a container admitting nothing does not offer an empty list."""
        offer()

        with pytest.raises(UnsupportedEntryError) as raised:
            insert_entry(container, FakeContext(), "z", "z", "text", EntryStyle())

        assert "value types allowed here" not in str(raised.value)


def test_insert_entry_passes_the_style_and_container_through(container, monkeypatch) -> None:
    """Test that what the caller decided about layout reaches synthesis unchanged.

    ``config_store`` works out which style applies -- the container's own entries, or what
    a created block inherited -- and this module's only job with it is to hand it on.
    """
    seen: dict[str, object] = {}
    style = EntryStyle(indent="    ", has_donor=True)
    ctx = FakeContext()
    ctx.entry_templates = {("start", "key_value"): "{key}={value}"}
    ctx.value_rule_priority = ("integer",)

    def record(info, container_rule, category, key, value, given_style, overrides, priority):
        seen.update(
            container=container_rule,
            category=category,
            key=key,
            style=given_style,
            overrides=overrides,
            priority=priority,
        )
        return iter(())

    monkeypatch.setattr(config_insert, "iter_entry_snippets", record)
    monkeypatch.setattr(config_insert, "entry_insertion_index", lambda c: 0)
    monkeypatch.setattr(config_insert, "admitted_value_rules", lambda i, c, cat: frozenset())

    with pytest.raises(UnsupportedEntryError):
        insert_entry(container, ctx, "Z", "Z", 1, style)

    assert seen == {
        "container": "start",
        "category": "key_value",
        "key": "Z",
        "style": style,
        "overrides": ctx.entry_templates,
        "priority": ctx.value_rule_priority,
    }


def test_insert_entry_reads_the_candidate_in_a_throwaway_container(container, monkeypatch) -> None:
    """Test that a candidate is interpreted apart from the tree it may never join.

    The read-back happens on a fresh node of the container's own rule, so a candidate that
    parses but means the wrong thing leaves no trace behind it.
    """
    read_from: list[Tree] = []
    node = Tree("line", [Token("T", "z=1")])

    monkeypatch.setattr(config_insert, "iter_entry_snippets", lambda *a: iter(["z=1"]))
    monkeypatch.setattr(config_insert, "parse_entry_nodes", lambda *a: [node])
    monkeypatch.setattr(config_insert, "entry_insertion_index", lambda c: 0)
    monkeypatch.setattr(config_insert, "admitted_value_rules", lambda i, c, cat: frozenset())

    def fake_read(tree, ctx):
        read_from.append(tree)
        return {}

    monkeypatch.setattr(config_insert, "read_entries", fake_read)

    with pytest.raises(UnsupportedEntryError):
        insert_entry(container, FakeContext(), "z", "z", 1, EntryStyle())

    [throwaway] = read_from
    assert throwaway is not container
    assert throwaway.data == container.data
    assert throwaway.children == [node]
