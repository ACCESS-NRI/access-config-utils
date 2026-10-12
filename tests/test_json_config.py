# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import math
import re

import pytest

from access.config.json_config import JSONConfig, JSONParser, JSONStyle, _LayoutScanner, detect_style

# A trimmed ACCESS-OM2 forcing.json, written as libaccessom2's own test inputs are: by
# json.dumps(data, indent=2) followed by a newline.
FORCING = """{
  "description": "JRA55-do V1.3 RYF 1990-91 forcing",
  "inputs": [
    {
      "filename": "INPUT/RYF.rsds.1990_1991.nc",
      "fieldname": "rsds",
      "cname": "swfld_ai",
      "perturbations": [
        {
          "type": "scaling",
          "dimension": "spatiotemporal",
          "value": "../test_data/scaling.RYF.rsds.1990_1991.nc",
          "calendar": "forcing"
        },
        {
          "type": "offset",
          "dimension": "constant",
          "value": 5,
          "calendar": "forcing"
        }
      ]
    },
    {
      "filename": "INPUT/RYF.rlds.1990_1991.nc",
      "fieldname": "rlds",
      "cname": "lwfld_ai"
    }
  ]
}
"""

# A hand-formatted file, which no set of json.dumps options reproduces.
HAND_WRITTEN = """{
    "a": [1, 2],
    "b": {"c": 3}
}
"""


@pytest.fixture(scope="module")
def parser() -> JSONParser:
    """Fixture instantiating the parser."""
    return JSONParser()


@pytest.fixture()
def forcing(parser: JSONParser) -> JSONConfig:
    """Fixture returning the forcing configuration, freshly parsed."""
    return parser.parse(FORCING)


def edited(text: str, old: str, new: str) -> str:
    """Return *text* with the single occurrence of *old* replaced by *new*."""
    assert text.count(old) == 1
    return text.replace(old, new)


class TestReading:
    def test_values(self, forcing: JSONConfig) -> None:
        """The data reads exactly as json.loads reads it."""
        assert forcing == json.loads(FORCING)
        assert forcing["inputs"][0]["perturbations"][1]["value"] == 5

    def test_nested_values_are_plain(self, forcing: JSONConfig) -> None:
        """Nested objects and arrays are plain dicts and lists, edited directly."""
        assert type(forcing["inputs"]) is list
        assert type(forcing["inputs"][0]) is dict

    def test_round_trip(self, forcing: JSONConfig) -> None:
        """An unchanged configuration is written back exactly."""
        assert str(forcing) == FORCING

    def test_top_level_must_be_an_object(self, parser: JSONParser) -> None:
        """A JSON text whose top level is not an object is not a configuration."""
        with pytest.raises(ValueError, match="must be an object, not list"):
            parser.parse("[1, 2]\n")

    def test_invalid_json_is_refused(self, parser: JSONParser) -> None:
        """Text the standard library refuses is refused, a trailing comma included."""
        with pytest.raises(json.JSONDecodeError):
            parser.parse('{\n  "inputs": [\n    {"a": 1},\n  ]\n}\n')


class TestEditing:
    def test_a_nested_value(self, forcing: JSONConfig) -> None:
        """Changing a value changes only its line."""
        forcing["inputs"][1]["filename"] = "INPUT/RYF.rlds.1984_1985.nc"
        assert str(forcing) == edited(FORCING, "RYF.rlds.1990_1991", "RYF.rlds.1984_1985")

    def test_adding_a_key(self, forcing: JSONConfig) -> None:
        """A new key is written in the file's own style."""
        forcing["inputs"][1]["domain"] = "land"
        expected = edited(FORCING, '"cname": "lwfld_ai"\n', '"cname": "lwfld_ai",\n      "domain": "land"\n')
        assert str(forcing) == expected
        assert JSONParser().parse(str(forcing)) == forcing

    def test_adding_a_perturbation(self, forcing: JSONConfig) -> None:
        """A new list element, holding an object, is written in the file's own style."""
        forcing["inputs"][1]["perturbations"] = [{"type": "offset", "value": 1.5}]
        expected = edited(
            FORCING,
            '"cname": "lwfld_ai"\n',
            '"cname": "lwfld_ai",\n'
            '      "perturbations": [\n'
            "        {\n"
            '          "type": "offset",\n'
            '          "value": 1.5\n'
            "        }\n"
            "      ]\n",
        )
        assert str(forcing) == expected

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(lambda c: c.__delitem__("description"), id="del"),
            pytest.param(lambda c: c.pop("description"), id="pop"),
            pytest.param(lambda c: c.popitem(), id="popitem"),
            pytest.param(lambda c: c.update(extra=1), id="update"),
            pytest.param(lambda c: c.setdefault("extra", [1, 2]), id="setdefault"),
            pytest.param(lambda c: c.__ior__({"extra": None}), id="ior"),
            pytest.param(lambda c: c.clear(), id="clear"),
        ],
    )
    def test_mapping_methods(self, forcing: JSONConfig, change) -> None:
        """Every dict method changes the text, which is regenerated from the data."""
        change(forcing)
        expected = json.dumps(dict(forcing), indent=2) + "\n"
        assert str(forcing) == expected
        assert JSONParser().parse(str(forcing)) == forcing

    def test_a_reverted_change_gives_the_original_text(self, parser: JSONParser) -> None:
        """Text is regenerated only when the data differs from what was read."""
        config = parser.parse(HAND_WRITTEN)
        config["a"].append(3)
        assert str(config) != HAND_WRITTEN
        config["a"].pop()
        assert str(config) == HAND_WRITTEN

    def test_a_change_of_type_is_a_change(self, forcing: JSONConfig) -> None:
        """5 and 5.0 are equal, but written differently."""
        forcing["inputs"][0]["perturbations"][1]["value"] = 5.0
        assert str(forcing) == edited(FORCING, '"value": 5,', '"value": 5.0,')

    def test_a_change_of_order_is_a_change(self, forcing: JSONConfig) -> None:
        """Moving a key to the end reorders the text, though the dict compares equal."""
        forcing["description"] = forcing.pop("description")
        assert str(forcing) != FORCING
        assert str(forcing).index('"inputs"') < str(forcing).index('"description"')

    def test_nan(self, parser: JSONParser) -> None:
        """NaN, which the standard library reads and writes, survives a round trip."""
        text = '{\n  "a": NaN,\n  "b": 1\n}\n'
        config = parser.parse(text)
        assert str(config) == text
        config["b"] = 2
        assert str(config) == '{\n  "a": NaN,\n  "b": 2\n}\n'
        assert math.isnan(parser.parse(str(config))["a"])

    def test_deep_copy(self, forcing: JSONConfig) -> None:
        """A deep copy is written exactly as the original is."""
        clone = copy.deepcopy(forcing)
        assert str(clone) == FORCING
        clone["description"] = "x"
        assert str(forcing) == FORCING


class TestHandWritten:
    def test_written_back_unchanged_until_edited(self, parser: JSONParser) -> None:
        """A layout json.dumps cannot reproduce is kept for as long as the data is."""
        assert str(parser.parse(HAND_WRITTEN)) == HAND_WRITTEN

    def test_edited_in_the_guessed_style(self, parser: JSONParser) -> None:
        """Once edited, it is written with the indentation and separators it showed."""
        config = parser.parse(HAND_WRITTEN)
        config["b"]["c"] = 4
        assert str(config) == '{\n    "a": [1, 2],\n    "b": {\n        "c": 4\n    }\n}\n'


MIXED = """{
    "a": [1, 2],
    "m": [
        1,
        2
    ],
    "n": [[1, 2], [3, "x]"]],
    "o": [{"k": [1,2]}],
    "e": [],
    "s": "[\\"not, a list]"
}
"""


class TestListLayout:
    """A list is written as the file wrote it, wherever the edit is."""

    @pytest.fixture()
    def mixed(self, parser: JSONParser) -> JSONConfig:
        """Fixture returning a file with lists in several layouts."""
        return parser.parse(MIXED)

    def test_unchanged_until_edited(self, mixed: JSONConfig) -> None:
        assert str(mixed) == MIXED

    @pytest.mark.parametrize(
        "edit",
        [
            pytest.param(lambda c: c.update(first=1), id="added-last"),
            pytest.param(lambda c: c.__setitem__("s", "text"), id="changed"),
            pytest.param(lambda c: c.pop("e"), id="removed"),
            pytest.param(lambda c: c["o"][0].__setitem__("j", 1), id="in-a-nested-object"),
            pytest.param(lambda c: c.__setitem__("a", c["a"] + [3]), id="list-grows"),
        ],
    )
    def test_each_list_keeps_its_layout(self, mixed: JSONConfig, edit) -> None:
        edit(mixed)
        text = str(mixed)
        assert json.loads(text) == dict(mixed)
        assert '"m": [\n        1,\n        2\n    ]' in text
        assert '"n": [[1, 2], [3, "x]"]]' in text
        assert re.search(r'"a": \[1, 2(, 3)?\]', text)

    def test_a_list_holding_an_object_is_never_on_one_line(self, mixed: JSONConfig) -> None:
        mixed["z"] = 0
        assert '"o": [\n        {\n            "k": [1, 2]\n        }\n    ]' in str(mixed)

    def test_empty_values(self, mixed: JSONConfig) -> None:
        mixed["z"] = {}
        text = str(mixed)
        assert '"e": []' in text
        assert '"z": {}' in text

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param(
                '{\n  "a": [1, 2],\n  "b": [3, 4],\n  "c": [\n    5,\n    6\n  ]\n}\n', "[7, 8]", id="mostly-inline"
            ),
            pytest.param(
                '{\n  "a": [1,\n 2],\n  "b": [3,\n 4],\n  "c": [5, 6]\n}\n',
                "[\n    7,\n    8\n  ]",
                id="mostly-wrapped",
            ),
        ],
    )
    def test_a_new_list_follows_the_majority(self, parser: JSONParser, text: str, expected: str) -> None:
        config = parser.parse(text)
        config["new"] = [7, 8]
        assert str(config).endswith(f'"new": {expected}\n}}\n')

    def test_a_file_with_no_lists_keeps_the_default_for_a_new_one(self, parser: JSONParser) -> None:
        config = parser.parse('{\n "a": 1,\n    "b": {"c": 2}\n}\n')
        config["n"] = [1, 2]
        assert '"n": [\n' in str(config)

    def test_separators_without_a_space(self, parser: JSONParser) -> None:
        config = parser.parse('{\n  "a": [1,2],\n  "b": {"c": 1}\n}\n')
        config["d"] = 1
        assert '"a": [1,2]' in str(config)

    @pytest.mark.parametrize(
        ("indent", "newline", "ensure_ascii", "text"),
        [
            pytest.param("\t", "\n", False, '{\n\t"a": [1, 2],\n\t"b": {"c": "é"}\n}\n', id="tab"),
            pytest.param(3, "\r\n", True, '{\r\n   "a": [1, 2],\r\n   "b": {"c": 1}\r\n}\r\n', id="crlf"),
            pytest.param(2, "\n", False, '{\n  "a": [1, 2],\n  "é": {"c": "é"}\n}\n', id="non-ascii"),
        ],
    )
    def test_other_styles(self, parser: JSONParser, indent, newline, ensure_ascii, text: str) -> None:
        config = parser.parse(text)
        assert config.style.indent == indent
        config["z"] = [1.5, float("nan"), None, True]
        written = str(config)
        assert newline in written
        assert '"a": [1, 2]' in written
        assert '"z": [1.5, NaN, null, true]' in written
        assert (config.style.ensure_ascii, config.style.newline) == (ensure_ascii, newline)

    def test_the_written_text_reads_back_as_the_same_style(self, mixed: JSONConfig, parser: JSONParser) -> None:
        mixed["z"] = [1, 2]
        text = str(mixed)
        again = parser.parse(text)
        assert again.style.inline_lists == mixed.style.inline_lists | {("z",)}
        again["y"] = 0
        assert str(again).replace('  "y": 0,\n', "") != ""
        assert json.loads(str(again)) == dict(again)

    def test_layouts_of_the_text(self) -> None:
        scanner = _LayoutScanner('{"a": [[1, 2], [3,\n4]], "b": {"c": [1,2]}, "d": [{}], "e": [], "f": "[,"}')
        assert scanner.inline == {("a", 0), ("b", "c")}
        assert scanner.wrapped == {("a",), ("a", 1)}
        assert scanner.spaced


class TestDetectStyle:
    @pytest.mark.parametrize(
        ("style", "data"),
        [
            pytest.param(JSONStyle(), {"a": [1, {"b": "c"}]}, id="indent-2"),
            pytest.param(JSONStyle(indent=4), {"a": [1, 2]}, id="indent-4"),
            pytest.param(JSONStyle(indent="\t"), {"a": {"b": 1}}, id="tab"),
            pytest.param(JSONStyle(indent=None, separators=(", ", ": ")), {"a": [1, 2]}, id="compact"),
            pytest.param(JSONStyle(indent=None, separators=(",", ":")), {"a": [1, 2]}, id="minified"),
            pytest.param(JSONStyle(indent=2, separators=(", ", ": ")), {"a": [1, 2]}, id="old-separators"),
            pytest.param(JSONStyle(newline="\r\n", trailing="\r\n"), {"a": [1, 2]}, id="crlf"),
            pytest.param(JSONStyle(ensure_ascii=False), {"a": "café"}, id="non-ascii"),
            pytest.param(JSONStyle(), {"a": "café"}, id="ascii-escapes"),
            pytest.param(JSONStyle(trailing=""), {"a": 1}, id="no-final-newline"),
            pytest.param(JSONStyle(leading="\n", trailing="\n\n"), {"a": 1}, id="surrounding-blank-lines"),
            pytest.param(JSONStyle(indent=0), {"a": 1}, id="indent-0"),
        ],
    )
    def test_reproducing_style(self, style: JSONStyle, data: dict) -> None:
        """A text written by json.dumps is recognised by the options that wrote it."""
        text = style.dump(data)
        assert detect_style(text, data) == style
        assert str(JSONParser().parse(text)) == text

    @pytest.mark.parametrize(
        ("text", "indent", "separators", "ensure_ascii"),
        [
            pytest.param('{"a": [1, 2],  "b": 3}\n', None, (", ", ": "), True, id="single-line-spaced"),
            pytest.param('{"a":[1,2] , "b":3}\n', None, (", ", ":"), True, id="single-line-tight"),
            pytest.param('{\n\t"a": [1, 2]}\n', "\t", (",", ": "), True, id="tab-indented"),
            pytest.param('{\n   "a": [1,\n 2]}\n', 3, (",", ": "), True, id="space-indented"),
            pytest.param('{\n"a": [1,\n2], "é": 1}\n', 0, (",", ": "), False, id="unindented"),
        ],
    )
    def test_guessed_style(self, text: str, indent, separators: tuple[str, str], ensure_ascii: bool) -> None:
        """A text json.dumps cannot reproduce gets the style it shows."""
        style = detect_style(text, json.loads(text))
        assert (style.indent, style.separators, style.ensure_ascii) == (indent, separators, ensure_ascii)


class TestWritingErrors:
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            pytest.param({1: "a"}, "keys must be strings, not 1 in \\['x'\\]", id="int-key"),
            pytest.param((1, 2), "tuple at \\['x'\\] cannot be written", id="tuple"),
            pytest.param({"y": {1, 2}}, "set at \\['x'\\]\\['y'\\] cannot be written", id="set"),
            pytest.param([object()], "object at \\['x'\\]\\[0\\] cannot be written", id="object"),
        ],
    )
    def test_values_that_would_not_read_back(self, forcing: JSONConfig, value, message: str) -> None:
        """A value json.dumps would write as something else, or not at all, is refused."""
        forcing["x"] = value
        with pytest.raises(TypeError, match=message):
            str(forcing)

    def test_a_non_string_top_level_key(self) -> None:
        """The location is left out of the message for a key of the top-level object."""
        with pytest.raises(TypeError, match="keys must be strings, not 2$"):
            str(JSONConfig({2: "a"}))

    def test_a_top_level_value_that_is_not_json(self) -> None:
        """The check names the top level when that is where the problem is."""
        from access.config.json_config import _check_types

        with pytest.raises(TypeError, match="set at the top level"):
            _check_types({1})


def test_made_in_code_uses_the_default_style() -> None:
    """A configuration with no source text is written as json.dumps(indent=2) writes it."""
    assert str(JSONConfig({"a": [1, 2]})) == '{\n  "a": [\n    1,\n    2\n  ]\n}\n'
