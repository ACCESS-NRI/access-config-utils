# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the dict and list a parsed configuration is handed out as.

No grammar, no parse tree walk, no text. ``Config`` holds a ``ConfigStore`` and calls it;
here it holds a ``FakeStore`` that records what it was asked to do. What is left to test is
the dict semantics -- normalising a key, telling an update from an addition, wrapping what
the store reports -- and whether each mapping method reaches the store at all.

That last one is the point. Every method ``dict`` implements in C has to be overridden or it
would skip both the key normalisation and the tree update; a fake store is the only way to
see that a call arrived, rather than inferring it from text that happened to come out right.
"""

import pytest
from conftest import FakeContext, FakeStore, entry_node, value_node
from lark import Tree

from access.config import config_dict
from access.config.config_dict import Config, ConfigList
from access.config.tree_reader import EntryRef


def ref(category: str = "key_value", key: str = "a", values: int = 1) -> EntryRef:
    """Return a reference of *category*, with *values* value nodes below it."""
    nodes = tuple(value_node(text=str(index + 1)) for index in range(values))
    node = entry_node(category, key, *nodes)
    if category == "key_block":
        body = Tree("block", [])
        return EntryRef(category, entry_node(category, key, body), block_node=body)
    return EntryRef(category, node, nodes)


@pytest.fixture
def store() -> FakeStore:
    """Return a store reporting one scalar, one list, one block and one valueless entry."""
    return FakeStore(
        {
            "a": ref("key_value", "a"),
            "b": ref("key_list", "b", values=2),
            "c": ref("key_block", "c"),
            "d": ref("key_null", "d"),
        }
    )


class TestReadingAConfig:
    """What the dict holds once the store has reported its entries."""

    def test_wraps_each_category_as_the_right_object(self, store) -> None:
        """Test the three wrappings: a value, a ``ConfigList``, a nested ``Config``."""
        config = Config(store)

        assert config["a"] == 1
        assert isinstance(config["b"], ConfigList)
        assert config["b"] == [1, 2]
        assert isinstance(config["c"], Config)
        assert config["d"] is None

    def test_a_block_gets_the_store_the_parent_hands_it(self, store) -> None:
        """Test that a nested config is built over the child store, not the parent's."""
        config = Config(store)

        assert config["c"]._store is not store
        assert store.asked("child") == [store.refs["c"]]

    def test_an_empty_configuration(self) -> None:
        """Test the file with no entries in it, which is still a usable dict."""
        assert dict(Config(FakeStore())) == {}

    def test_reads_the_store_once(self, store) -> None:
        """Test that the dict is filled at construction rather than on each access."""
        config = Config(store)
        store.refs["e"] = ref("key_value", "e")

        assert "e" not in config


class TestUpdatingAnExistingKey:
    """Assignment where the key is already there."""

    def test_a_scalar_reaches_the_store(self, store) -> None:
        """Test that the value is both stored in the dict and passed on to be written."""
        config = Config(store)

        config["a"] = 42

        assert config["a"] == 42
        assert store.asked("replace") == [("a", 42)]

    def test_a_list_is_rewrapped_over_the_nodes_the_store_reports(self, store) -> None:
        """Test that element updates still reach the tree after a whole-list assignment."""
        config = Config(store)

        config["b"] = [7, 8]

        assert isinstance(config["b"], ConfigList)
        assert config["b"] == [7, 8]
        assert store.asked("replace") == [("b", [7, 8])]

    def test_assigning_none_to_a_valueless_entry_changes_nothing(self, store) -> None:
        """Test the one no-op: a key with no value assigned no value."""
        config = Config(store)

        config["d"] = None

        assert config["d"] is None
        assert store.asked("replace") == []

    def test_a_valueless_entry_cannot_gain_a_value(self, store) -> None:
        """Test that the category is fixed by the parse: ``a =`` cannot become ``a = 1``."""
        config = Config(store)

        with pytest.raises(TypeError, match="change the type"):
            config["d"] = 1

    def test_a_block_cannot_be_replaced(self, store) -> None:
        """Test that a block may be created but not assigned over once it exists."""
        config = Config(store)

        with pytest.raises(SyntaxError, match="entire block"):
            config["c"] = {"x": 1}


class TestAddingAKey:
    """Assignment where the key is new."""

    @pytest.fixture
    def adding(self) -> FakeStore:
        """Return a store that will report a scalar reference for a newly added key."""
        return FakeStore({}, added={"z": ref("key_value", "z")})

    def test_passes_both_spellings_and_stores_the_result(self, adding) -> None:
        """Test that the normalised key indexes the dict, the written one the file."""
        config = Config(adding)

        config["z"] = 5

        assert config["z"] == 1
        assert adding.asked("add") == [("z", "z", 5)]

    def test_a_case_insensitive_format_normalises_the_dict_key_only(self) -> None:
        """Test the namelist: the entry is keyed uppercase, the file keeps the spelling."""
        store = FakeStore({}, ctx=FakeContext(case_sensitive=False), added={"Z": ref("key_value", "Z")})
        config = Config(store)

        config["z"] = 5

        assert "Z" in config
        assert store.asked("add") == [("Z", "z", 5)]

    def test_a_new_block_is_given_a_style_and_then_filled(self) -> None:
        """Test the two things that only happen for a block.

        It arrives empty, because nothing in it can say how its contents should be spaced
        until it has some; it is handed the style of its nearest sibling and then filled
        through this same path, one key at a time, so nesting to any depth works.
        """
        inner = FakeStore({}, added={"x": ref("key_value", "x")})
        store = FakeStore({}, added={"blk": ref("key_block", "blk")})
        store.child = lambda reference: inner  # type: ignore[method-assign]
        config = Config(store)

        config["blk"] = {"x": 1}

        assert store.asked("created_block_style") == [None]
        assert inner.fallback_style.indent == "  "
        assert inner.asked("add") == [("x", "x", 1)]


class TestDeleting:
    """Removing a key from both the dict and the tree."""

    def test_reaches_the_store(self, store) -> None:
        """Test that the entry is unlinked, not merely dropped from the dict."""
        config = Config(store)

        del config["a"]

        assert "a" not in config
        assert store.asked("remove") == ["a"]

    def test_a_key_that_is_not_there(self, store) -> None:
        """Test that deleting an absent key raises before the store is troubled."""
        config = Config(store)

        with pytest.raises(KeyError):
            del config["nope"]
        assert store.asked("remove") == []


class TestKeyCase:
    """Matching keys the way the format matches them."""

    def test_case_sensitive_by_default(self, store) -> None:
        """Test that most formats distinguish two spellings of a name."""
        config = Config(store)

        assert config["a"] == 1
        with pytest.raises(KeyError):
            config["A"]

    def test_case_insensitive_reaches_the_same_entry_either_way(self) -> None:
        """Test the namelist, where the spelling used to look a key up does not matter."""
        store = FakeStore({"A": ref("key_value", "A")}, ctx=FakeContext(case_sensitive=False))
        config = Config(store)

        assert config["a"] == config["A"] == 1
        assert "a" in config and "A" in config

        config["a"] = 9
        assert config["A"] == 9

        del config["a"]
        assert "A" not in config

    def test_a_non_string_key_is_left_alone_by_the_membership_test(self, store) -> None:
        """Test that ``in`` does not try to normalise something that is not a name."""
        assert 1 not in Config(store)


class TestRemainingMappingProtocol:
    """The methods ``dict`` implements in C, which have to be overridden.

    Inheriting any of them would skip both the key normalisation and the tree update. A
    ``pop`` followed by an assignment is the damaging case: the removed entry would stay in
    the tree and a second one would be added, writing a file with the key twice.
    """

    @pytest.fixture
    def config(self) -> tuple[Config, FakeStore]:
        """Return a config over one scalar, with the store ready to accept two more."""
        store = FakeStore(
            {"a": ref("key_value", "a")},
            added={"y": ref("key_value", "y"), "z": ref("key_value", "z")},
        )
        return Config(store), store

    def test_get(self, config) -> None:
        """Test the defaulted read, which has to normalise the key like ``__getitem__``."""
        cfg, _ = config

        assert cfg.get("a") == 1
        assert cfg.get("nope") is None
        assert cfg.get("nope", "fallback") == "fallback"

    def test_pop(self, config) -> None:
        """Test that a removal through ``pop`` reaches the tree."""
        cfg, store = config

        assert cfg.pop("a") == 1
        assert store.asked("remove") == ["a"]
        assert cfg.pop("nope", "fallback") == "fallback"
        with pytest.raises(KeyError):
            cfg.pop("nope")

    def test_popitem(self, config) -> None:
        """Test that the last entry is the one removed, and that it reaches the tree."""
        cfg, store = config
        cfg["z"] = 3

        assert cfg.popitem() == ("z", 1)
        assert store.asked("remove") == ["z"]

    def test_update_from_a_mapping_and_from_keywords(self, config) -> None:
        """Test both call shapes, each landing as an ordinary assignment."""
        cfg, store = config

        cfg.update({"y": 2}, z=3)

        assert store.asked("add") == [("y", "y", 2), ("z", "z", 3)]

    def test_update_from_pairs(self, config) -> None:
        """Test the iterable-of-pairs form, which takes the other branch."""
        cfg, store = config

        cfg.update([("y", 2)])

        assert store.asked("add") == [("y", "y", 2)]

    def test_setdefault(self, config) -> None:
        """Test that an absent key is added and a present one is left as it is."""
        cfg, store = config

        assert cfg.setdefault("a", 99) == 1
        assert cfg.setdefault("z", 3) == 1
        assert store.asked("add") == [("z", "z", 3)]

    def test_in_place_merge(self, config) -> None:
        """Test that ``|=`` goes through assignment rather than dict's own merge."""
        cfg, store = config

        cfg |= {"z": 3}

        assert store.asked("add") == [("z", "z", 3)]

    def test_clear(self, config) -> None:
        """Test that emptying the dict empties the tree with it."""
        cfg, store = config

        cfg.clear()

        assert dict(cfg) == {}
        assert store.asked("remove") == ["a"]


def test_str_asks_the_store_to_render(store) -> None:
    """Test that writing a configuration out is the store's job, not the dict's."""
    config = Config(store)

    assert str(config) == "<rendered>"
    assert store.asked("render") == [None]


class TestConfigList:
    """The list handed out for a key holding several values."""

    @pytest.fixture
    def nodes(self) -> list[Tree]:
        """Return three value nodes, one per element."""
        return [value_node(text=str(index)) for index in range(3)]

    @pytest.fixture
    def values(self, nodes, monkeypatch) -> tuple[ConfigList, list]:
        """Return a list over those nodes, the tree update recorded rather than made."""
        written: list[tuple[Tree, object]] = []
        monkeypatch.setattr(config_dict, "update_node_value", lambda node, value: written.append((node, value)))
        return ConfigList([0, 1, 2], nodes), written

    def test_an_element_update_reaches_its_node(self, values, nodes) -> None:
        """Test the pairing: element *i* writes to the node holding element *i*."""
        listed, written = values

        listed[1] = 20

        assert listed == [0, 20, 2]
        assert written == [(nodes[1], 20)]

    def test_a_negative_index(self, values, nodes) -> None:
        """Test that indexing from the end reaches the same node the list does."""
        listed, written = values

        listed[-1] = 30

        assert written == [(nodes[-1], 30)]

    def test_a_slice(self, values, nodes) -> None:
        """Test that a slice writes each of its elements, in order."""
        listed, written = values

        listed[0:2] = [10, 11]

        assert listed == [10, 11, 2]
        assert written == [(nodes[0], 10), (nodes[1], 11)]

    def test_a_slice_with_a_step(self, values, nodes) -> None:
        """Test that the nodes are selected by the same slice as the elements."""
        listed, written = values

        listed[::2] = [10, 30]

        assert listed == [10, 1, 30]
        assert written == [(nodes[0], 10), (nodes[2], 30)]

    def test_a_slice_of_the_wrong_length(self, values) -> None:
        """Test that a slice assignment cannot change how many elements there are."""
        listed, written = values

        with pytest.raises(ValueError, match="change list length"):
            listed[0:2] = [1]
        assert written == []

    @pytest.mark.parametrize(
        "operation",
        [
            lambda values: values.append(4),
            lambda values: values.extend([4]),
            lambda values: values.insert(0, 4),
            lambda values: values.remove(1),
            lambda values: values.pop(),
            lambda values: values.clear(),
            lambda values: values.sort(),
            lambda values: values.reverse(),
            lambda values: values.__delitem__(0),
            lambda values: values.__iadd__([4]),
            lambda values: values.__imul__(2),
        ],
    )
    def test_refuses_what_would_break_the_pairing(self, values, operation) -> None:
        """Test every operation that would add, remove or reorder the elements.

        Each element is paired with the node holding it, so any of these would leave the
        list and the file disagreeing. Assigning a whole new list is the supported way.
        """
        listed, _ = values

        with pytest.raises(NotImplementedError, match="assign a whole new list"):
            operation(listed)
