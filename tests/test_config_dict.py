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

import copy
import pickle

import pytest
from conftest import FakeContext, FakeStore, entry_node, value_node
from lark import Tree

from access.config.config_dict import Config, ConfigBlockList, ConfigList
from access.config.tree_reader import EntryRef


def ref(category: str = "key_value", key: str = "a", values: int = 1) -> EntryRef:
    """Return a reference of *category*, with *values* value nodes below it."""
    nodes = tuple(value_node(text=str(index + 1)) for index in range(values))
    node = entry_node(category, key, *nodes)
    if category == "key_block":
        body = Tree("block", [])
        return EntryRef(category, (entry_node(category, key, body),), block_nodes=(body,))
    return EntryRef(category, (node,), nodes)


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
        store.child = lambda reference, index=0: inner  # type: ignore[method-assign]
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


class TestResyncAfterDeleting:
    """Reading the configurations above one again, after a deletion pruned the tree."""

    @staticmethod
    def _nested(block_rule: str, repetition: bool, reread: dict[str, EntryRef] | None = None):
        """Return a parent config whose one key is a block held by *block_rule*.

        Args:
            block_rule: The rule holding the block's contents.
            repetition: Whether the grammar reports that rule a plain repetition.
            reread: What the parent reports when read again; unchanged when not given.

        Returns:
            The parent configuration, its store and the block's store.
        """

        class Info:
            def is_repetition_rule(self, name: str) -> bool:
                return repetition

        ctx = FakeContext(info=Info())
        inner = FakeStore({"x": ref("key_value", "x")}, ctx=ctx)
        inner.tree = Tree(block_rule, [])
        outer = FakeStore({"blk": ref("key_block", "blk")}, ctx=ctx, reread=reread)
        outer.child = lambda reference, index=0: inner  # type: ignore[method-assign]
        return Config(outer), outer, inner

    def test_a_block_the_tree_no_longer_has_leaves_the_dict(self) -> None:
        """Test the derived type that goes when its last component does.

        It is spelled a component per line; once the last one goes there is nothing left
        to write, the lines are already out of the tree, and the parent no longer reads it.
        """
        config, outer, _ = self._nested("dtype_body", repetition=False, reread={})
        block = config["blk"]

        del config["blk"]["x"]

        assert "blk" not in config
        assert outer.asked("reread") == [None]
        assert block._parent is None

    def test_a_block_that_survives_is_read_again(self) -> None:
        """Test that the block's store follows the node the parent now reads for it."""
        fresh = ref("key_block", "blk")
        config, outer, inner = self._nested("dtype_body", repetition=False, reread={"blk": fresh})

        del config["blk"]["x"]

        assert "blk" in config
        assert inner.asked("retarget") == [fresh]
        assert inner.asked("reread") == [None]

    def test_an_empty_namelist_group_stays(self) -> None:
        """Test the block whose rule *is* a repetition: an empty group is still a group."""
        config, outer, _ = self._nested("block", repetition=True)

        del config["blk"]["x"]

        assert "blk" in config
        assert dict(config["blk"]) == {}
        assert outer.asked("reread") == []

    def test_a_merged_block_is_read_again_from_its_parent(self) -> None:
        """Test that a block not held in the tree itself is no anchor, whatever its rule."""
        config, outer, inner = self._nested("block", repetition=True, reread={"blk": ref("key_block", "blk")})
        inner.addable = False

        del config["blk"]["x"]

        assert outer.asked("reread") == [None]

    def test_the_configuration_of_a_whole_file_is_not_read_again(self) -> None:
        """Test that a top-level configuration is its own anchor."""
        store = FakeStore({"a": ref("key_value", "a")})
        config = Config(store)

        del config["a"]

        assert dict(config) == {}
        assert store.asked("reread") == []


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
    """The list handed out for a key holding several values, writing through its store."""

    @pytest.fixture
    def list_store(self) -> FakeStore:
        """Return a store holding one three-element list under ``a``."""
        return FakeStore({"a": ref("key_list", "a", values=3)})

    @pytest.fixture
    def listed(self, list_store) -> ConfigList:
        """Return the list over that store's entry."""
        return ConfigList([0, 1, 2], list_store, "a")

    @pytest.mark.parametrize(
        ("write", "expected"),
        [
            pytest.param(lambda values: values.__setitem__(1, 20), [0, 20, 2], id="index"),
            pytest.param(lambda values: values.__setitem__(-1, 30), [0, 1, 30], id="negative-index"),
            pytest.param(lambda values: values.__setitem__(slice(0, 2), [10, 11]), [10, 11, 2], id="slice"),
            pytest.param(lambda values: values.__setitem__(slice(None, None, 2), [10, 30]), [10, 1, 30], id="step"),
            pytest.param(lambda values: values.sort(reverse=True), [2, 1, 0], id="sort"),
            pytest.param(lambda values: values.reverse(), [2, 1, 0], id="reverse"),
        ],
    )
    def test_a_write_of_the_same_length_replaces_the_list(self, listed, list_store, write, expected) -> None:
        """Test that the whole list is written, position by position, then held."""
        write(listed)

        assert listed == expected
        assert list_store.asked("replace") == [("a", expected)]
        assert list_store.asked("resize") == []

    @pytest.mark.parametrize(
        ("change", "expected", "origins"),
        [
            pytest.param(lambda values: values.append(3), [0, 1, 2, 3], [0, 1, 2, None], id="append"),
            pytest.param(
                lambda values: values.extend(iter([3, 4])), [0, 1, 2, 3, 4], [0, 1, 2, None, None], id="extend"
            ),
            pytest.param(lambda values: values.insert(1, 9), [0, 9, 1, 2], [0, None, 1, 2], id="insert"),
            pytest.param(lambda values: values.remove(1), [0, 2], [0, 2], id="remove"),
            pytest.param(lambda values: values.pop(), [0, 1], [0, 1], id="pop"),
            pytest.param(lambda values: values.__delitem__(slice(0, 2)), [2], [2], id="del-slice"),
            pytest.param(
                lambda values: values.__setitem__(slice(1, 2), iter([7, 8])),
                [0, 7, 8, 2],
                [0, None, None, 2],
                id="slice",
            ),
            pytest.param(lambda values: values.__iadd__([3]), [0, 1, 2, 3], [0, 1, 2, None], id="iadd"),
            pytest.param(lambda values: values.__imul__(2), [0, 1, 2, 0, 1, 2], [0, 1, 2, None, None, None], id="imul"),
            pytest.param(lambda values: values.__imul__(0), [], [], id="imul-zero"),
            pytest.param(lambda values: values.clear(), [], [], id="clear"),
        ],
    )
    def test_a_change_of_length_is_a_resize(self, listed, list_store, change, expected, origins) -> None:
        """Test that the store is told which old element each new one stands in for."""
        change(listed)

        assert listed == expected
        assert list_store.asked("resize") == [("a", expected, origins)]

    def test_pop_hands_back_the_element(self, listed) -> None:
        """Test that ``pop`` still returns what it removed."""
        assert listed.pop(1) == 1
        assert listed == [0, 2]

    def test_an_extended_slice_of_the_wrong_size_is_refused_first(self, listed, list_store) -> None:
        """Test that the list's own rules apply before anything reaches the store."""
        with pytest.raises(ValueError):
            listed[::2] = [1]

        assert listed == [0, 1, 2]
        assert list_store.calls == []

    def test_the_list_is_unchanged_when_the_store_refuses(self, listed, list_store) -> None:
        """Test that a write the file cannot take leaves the list as it was too."""

        def refuse(key, values, origins):
            raise ValueError("refused")

        list_store.resize = refuse  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="refused"):
            listed.append(3)

        assert listed == [0, 1, 2]

    def test_a_copy_is_a_plain_list(self, listed) -> None:
        """Test that copying or pickling gives the data, not a second view of the file."""
        for duplicate in (copy.copy(listed), copy.deepcopy(listed), pickle.loads(pickle.dumps(listed))):
            assert type(duplicate) is list
            assert duplicate == [0, 1, 2]


class TestConfigBlockList:
    """The list handed out for a block the file writes more than once."""

    @pytest.fixture
    def blocks(self) -> ConfigBlockList:
        """Return a list standing for two occurrences of one block."""
        return ConfigBlockList([Config(FakeStore()), Config(FakeStore())])

    def test_holds_one_configuration_per_occurrence(self, blocks) -> None:
        """Test that the occurrences are reachable by index, in the order written."""
        assert len(blocks) == 2
        assert all(isinstance(block, Config) for block in blocks)

    @pytest.mark.parametrize(
        "operation",
        [
            lambda blocks: blocks.append({}),
            lambda blocks: blocks.extend([{}]),
            lambda blocks: blocks.insert(0, {}),
            lambda blocks: blocks.remove(blocks[0]),
            lambda blocks: blocks.pop(),
            lambda blocks: blocks.clear(),
            lambda blocks: blocks.sort(),
            lambda blocks: blocks.reverse(),
            lambda blocks: blocks.__setitem__(0, {}),
            lambda blocks: blocks.__delitem__(0),
            lambda blocks: blocks.__iadd__([{}]),
            lambda blocks: blocks.__imul__(2),
        ],
    )
    def test_refuses_to_add_remove_or_replace_a_whole_block(self, blocks, operation) -> None:
        """Test that only the entries of an occurrence are editable, not the list itself.

        Where a new occurrence would go, and how it should be spaced, is not something the
        list can answer, so these would leave it and the file disagreeing.
        """
        with pytest.raises(NotImplementedError, match="occurrence of a repeated block"):
            operation(blocks)
