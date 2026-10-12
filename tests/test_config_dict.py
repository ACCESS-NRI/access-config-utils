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

from access.config.config_dict import Config, ConfigBlockList, ConfigList, ConfigRecord
from access.config.grammar_values import UnsupportedEntryError
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

    def test_passes_the_key_as_written_and_stores_the_result(self, adding) -> None:
        """Test that the store gets the key as written, and the result is stored."""
        config = Config(adding)

        config["z"] = 5

        assert config["z"] == 1
        assert adding.asked("add_path") == [(("z",), 5)]

    def test_a_case_insensitive_format_normalises_the_dict_key_only(self) -> None:
        """Test the namelist: the entry is keyed uppercase, the file keeps the spelling."""
        store = FakeStore({}, ctx=FakeContext(case_sensitive=False), added={"Z": ref("key_value", "Z")})
        config = Config(store)

        config["z"] = 5

        assert "Z" in config
        assert store.asked("add_path") == [(("z",), 5)]

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

        # Asked about the entry just written, so siblings are searched from above it.
        assert store.asked("created_block_style") == [store.refs["blk"].entry_nodes[0]]
        assert inner.fallback_style.indent == "  "
        assert inner.asked("add_path") == [(("x",), 1)]


class TestAddingBelowABlockWithNoBody:
    """A key added to a block the file writes one component per line, or merged.

    Such a block has no body of its own in the file, so the line is written by the nearest
    configuration holding it, carrying the whole key path, and the configurations on the
    path are read again in place.
    """

    @staticmethod
    def _nested(outer_refs: dict, inner_refs: dict, **outer_kwargs):
        """Return a parent config over *outer_refs* whose block reads as *inner_refs*.

        The block's store has no body for an entry; it reads every key it is asked about.

        Returns:
            The parent configuration, its store and the block's store.
        """
        inner = FakeStore({key: inner_refs[key] for key in list(inner_refs)[:1]}, reread=inner_refs)
        inner.addable = False
        outer = FakeStore(outer_refs, **outer_kwargs)
        outer.child = lambda reference, index=0: inner  # type: ignore[method-assign]
        return Config(outer), outer, inner

    def test_the_holding_configuration_writes_the_whole_path(self) -> None:
        """Test that a held reference to the block sees the key afterwards."""
        fresh = ref("key_block", "blk")
        inner_refs = {"x": ref("key_value", "x"), "z": ref("key_value", "z")}
        config, outer, inner = self._nested(
            {"blk": ref("key_block", "blk")}, inner_refs, added={("blk", "z"): fresh}, reread={"blk": fresh}
        )
        block = config["blk"]

        block["z"] = 5

        assert outer.asked("add_path") == [(("blk", "z"), 5)]
        assert inner.asked("reread") == [fresh]
        assert config["blk"] is block
        assert block["z"] == 1

    def test_a_dict_the_format_cannot_hold_goes_one_level_down(self) -> None:
        """Test the Fortran ``t%a = 1`` written for ``t = {"a": 1, "b": 2}``.

        The format has no block of its own for ``t``, so its first item is written as one
        line carrying both keys, and the rest is added to the block that line created.
        """
        inner_refs = {"a": ref("key_value", "a"), "b": ref("key_value", "b")}
        config, outer, _ = self._nested(
            {},
            inner_refs,
            added={("t", "a"): ref("key_block", "t"), ("t", "b"): ref("key_block", "t")},
            refused=[("t",)],
            reread={"t": ref("key_block", "t")},
        )

        config["t"] = {"a": 1, "b": 2}

        assert outer.asked("add_path") == [
            (("t",), {"a": 1, "b": 2}),
            (("t", "a"), 1),
            (("t", "b"), 2),
        ]
        assert dict(config["t"]) == {"a": 1, "b": 1}

    def test_a_dict_one_level_down_is_filled_before_the_level_above(self) -> None:
        """Test that what is left of each dict is added deepest first, as the file reads."""
        inner_refs = {"a": ref("key_block", "a"), "b": ref("key_value", "b")}
        config, outer, inner = self._nested(
            {},
            inner_refs,
            added={("t", "a"): ref("key_block", "t"), ("t", "b"): ref("key_block", "t")},
            refused=[("t",)],
            reread={"t": ref("key_block", "t")},
        )
        innermost = FakeStore({}, added={"y": ref("key_value", "y")})
        inner.child = lambda reference, index=0: innermost  # type: ignore[method-assign]

        config["t"] = {"a": {"y": 1}, "b": 2}

        assert outer.asked("add_path")[-1] == (("t", "b"), 2)
        assert innermost.asked("add_path") == [(("y",), 1)]
        # The block one level down is spaced like a sibling of the line just written.
        assert outer.asked("created_block_style") == [outer.refs["t"].entry_nodes[0]]

    @pytest.mark.parametrize(
        ("value", "refused"),
        [
            pytest.param({}, [("t",)], id="nothing-to-go-down-into"),
            pytest.param({"a": 1}, [("t",), ("t", "a")], id="refused-one-level-down-too"),
        ],
    )
    def test_the_refusal_of_the_path_asked_for_is_reported(self, value, refused) -> None:
        """Test that the error names what the caller asked for, not a level below it."""
        store = FakeStore({}, refused=refused)

        with pytest.raises(UnsupportedEntryError, match=r"Refused \['t'\]"):
            Config(store)["t"] = value

        assert "t" not in Config(store)

    def test_a_block_left_without_a_parent_refuses_itself(self) -> None:
        """Test a block whose parent dropped it, which has nowhere to go for a line."""
        store = FakeStore({}, refused=[("z",)])
        store.addable = False

        with pytest.raises(UnsupportedEntryError):
            Config(store)["z"] = 1

        assert store.asked("add_path") == [(("z",), 1)]


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
        assert inner.asked("reread") == [fresh]

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

        assert store.asked("add_path") == [(("y",), 2), (("z",), 3)]

    def test_update_from_pairs(self, config) -> None:
        """Test the iterable-of-pairs form, which takes the other branch."""
        cfg, store = config

        cfg.update([("y", 2)])

        assert store.asked("add_path") == [(("y",), 2)]

    def test_setdefault(self, config) -> None:
        """Test that an absent key is added and a present one is left as it is."""
        cfg, store = config

        assert cfg.setdefault("a", 99) == 1
        assert cfg.setdefault("z", 3) == 1
        assert store.asked("add_path") == [(("z",), 3)]

    def test_in_place_merge(self, config) -> None:
        """Test that ``|=`` goes through assignment rather than dict's own merge."""
        cfg, store = config

        cfg |= {"z": 3}

        assert store.asked("add_path") == [(("z",), 3)]

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
    def test_a_change_of_length_says_which_old_element_each_new_one_stands_in_for(
        self, listed, list_store, change, expected, origins
    ) -> None:
        """Test that the store is told the origins, so that what stays keeps its place."""
        change(listed)

        assert listed == expected
        assert list_store.asked("replace") == [("a", expected, origins)]

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

        list_store.replace = refuse  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="refused"):
            listed.append(3)

        assert listed == [0, 1, 2]

    def test_a_copy_is_a_plain_list(self, listed) -> None:
        """Test that copying or pickling gives the data, not a second view of the file."""
        for duplicate in (copy.copy(listed), copy.deepcopy(listed), pickle.loads(pickle.dumps(listed))):
            assert type(duplicate) is list
            assert duplicate == [0, 1, 2]


def records_ref(key: str = "r", *controls: str) -> EntryRef:
    """Return a list of records under *key*, one entry per control, each named ``a``."""
    nodes = tuple(value_node("named_assignments", f'"a", "{control}"') for control in controls)
    return EntryRef("key_list", tuple(entry_node("key_value", key, node) for node in nodes), nodes, collected=True)


class TestConfigRecord:
    """A record of a list written one entry per record, as a dict writing through it."""

    FIRST, SECOND = {"value": "a", "x": 1}, {"value": "b", "x": 2}

    @pytest.fixture
    def records(self) -> tuple[ConfigList, FakeStore]:
        """Return a list of two records over a store."""
        store = FakeStore({"r": records_ref("r", "x=1", "x=2")})
        return ConfigList([dict(self.FIRST), dict(self.SECOND)], store, "r"), store

    def test_each_dict_of_a_list_is_a_record(self, records) -> None:
        """Test that the list hands out records, which read as plain dicts."""
        listed, _ = records

        assert all(isinstance(element, ConfigRecord) for element in listed)
        assert listed == [self.FIRST, self.SECOND]

    @pytest.mark.parametrize(
        ("write", "expected"),
        [
            pytest.param(lambda r: r.__setitem__("x", 5), {"value": "a", "x": 5}, id="set"),
            pytest.param(lambda r: r.__delitem__("x"), {"value": "a"}, id="del"),
            pytest.param(lambda r: r.update({"y": 1}, z=2), {"value": "a", "x": 1, "y": 1, "z": 2}, id="update"),
            pytest.param(lambda r: r.setdefault("y", 3), {"value": "a", "x": 1, "y": 3}, id="setdefault"),
            pytest.param(lambda r: r.pop("x"), {"value": "a"}, id="pop"),
            pytest.param(lambda r: r.popitem(), {"value": "a"}, id="popitem"),
            pytest.param(lambda r: r.__ior__({"y": 1}), {"value": "a", "x": 1, "y": 1}, id="ior"),
            pytest.param(lambda r: r.clear(), {}, id="clear"),
        ],
    )
    def test_a_write_is_a_write_of_the_whole_record(self, records, write, expected) -> None:
        """Test that every way of changing the dict reaches the file, record and all."""
        listed, store = records
        record = listed[0]

        write(record)

        assert record == expected
        assert listed[0] is record
        assert store.asked("replace") == [("r", [expected, self.SECOND])]

    def test_reading_a_parameter_writes_nothing(self, records) -> None:
        """Test the calls that leave the record as it is."""
        listed, store = records
        record = listed[0]

        assert record.setdefault("x", 9) == 1
        assert record.pop("nope", "fallback") == "fallback"
        with pytest.raises(KeyError):
            record.pop("nope")
        assert store.asked("replace") == []

    def test_a_write_the_file_refuses_leaves_the_record_as_it_was(self, records) -> None:
        """Test that the dict changes only once the file has."""
        listed, store = records

        def refuse(key, values, origins):
            raise TypeError("refused")

        store.replace = refuse  # type: ignore[method-assign]
        with pytest.raises(TypeError, match="refused"):
            listed[0]["x"] = 5

        assert listed[0] == self.FIRST

    def test_a_record_stays_one_through_a_change_of_length(self, records) -> None:
        """Test that a record held by a caller follows its place in the list."""
        listed, _ = records
        second = listed[1]

        listed.insert(0, {"value": "c"})

        assert listed[2] is second
        assert isinstance(listed[0], ConfigRecord)

    def test_a_record_the_list_no_longer_holds_refuses_a_write(self, records) -> None:
        """Test that a removed record says so rather than writing somewhere else."""
        listed, _ = records
        first = listed[0]
        listed.pop(0)

        with pytest.raises(ValueError, match="no longer in the list"):
            first["x"] = 5

    def test_a_copy_is_plain_data(self, records) -> None:
        """Test that copying a record, or a list of them, gives dicts, not views."""
        listed, _ = records
        for duplicate in (copy.copy(listed[0]), copy.deepcopy(listed[0]), pickle.loads(pickle.dumps(listed[0]))):
            assert type(duplicate) is dict
            assert duplicate == self.FIRST
        assert [type(element) for element in copy.copy(listed)] == [dict, dict]


class TestAddingRecords:
    """A list of dicts assigned to a new key: one record per entry."""

    def test_one_record_is_written_as_an_entry(self) -> None:
        """Test that the dict is asked for as a scalar entry, not as a block."""
        store = FakeStore({}, added={"r": records_ref("r", "x=1")})
        config = Config(store)

        config["r"] = [{"value": "a", "x": 1}]

        assert store.asked("add_path") == [(("r",), {"value": "a", "x": 1}, "key_value")]
        assert config["r"] == [{"value": "a", "x": 1}]
        assert isinstance(config["r"][0], ConfigRecord)

    def test_the_others_are_added_beside_the_first(self) -> None:
        """Test that the rest arrive as new records after the one written."""
        store = FakeStore({}, added={"r": records_ref("r", "x=1")})
        config = Config(store)

        config["r"] = [{"value": "a", "x": 1}, {"value": "b", "x": 2}]

        assert store.asked("replace") == [("r", [{"value": "a", "x": 1}, {"value": "b", "x": 2}], [0, None])]
        assert config["r"] == [{"value": "a", "x": 1}, {"value": "b", "x": 2}]

    def test_the_key_goes_again_if_a_record_is_refused(self) -> None:
        """Test that a refused record leaves no part of the list behind."""
        store = FakeStore({}, added={"r": records_ref("r", "x=1")})

        def refuse(key, values, origins):
            raise TypeError("refused")

        store.replace = refuse  # type: ignore[method-assign]
        config = Config(store)

        with pytest.raises(TypeError, match="refused"):
            config["r"] = [{"value": "a", "x": 1}, {"value": "b", "x": 2}]

        assert "r" not in config
        assert store.asked("remove") == ["r"]

    def test_a_whole_list_of_records_assigned_over_one(self) -> None:
        """Test that the records stay records, matched to the ones held by position."""
        store = FakeStore({"r": records_ref("r", "x=1")})
        config = Config(store)
        held = config["r"][0]

        config["r"] = [{"value": "a", "x": 5}]

        assert store.asked("replace") == [("r", [{"value": "a", "x": 5}])]
        assert config["r"][0] is held
        assert held == {"value": "a", "x": 5}


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
