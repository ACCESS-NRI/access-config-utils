# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle YAML-based configuration files.

The ruamel.yaml parser provides round-trip parsing of YAML files and has all capabilities
we require. Here we simply provide wrappers around the ruamel.yaml classes so that the API
is the same/similar to the other parsers.

Round-trip loading keeps the comments, the key order and, with ``preserve_quotes``, the
quoting of every scalar. What it does not keep is the indentation: ruamel.yaml writes
everything back with one set of indents, whatever the file used. So the parser reads the
indentation off the text first and dumps with that, which is what lets an untouched file
come back out unchanged.
"""

from __future__ import annotations

import re
import sys
from io import StringIO
from typing import Any

from ruamel.yaml import YAML, CommentedMap

# Indentation used where the text does not show it, e.g. a file with no nested mapping or
# no block sequence. A sequence's dash is indented by two, as in the ACCESS configurations.
DEFAULT_INDENT = {"mapping": 2, "sequence": 4, "offset": 2}

# A trailing comment: a "#" preceded by whitespace, as YAML defines one.
_TRAILING_COMMENT = re.compile(r"\s+#.*$")


def _round_trip_yaml(indent: dict[str, int] | None = None) -> YAML:
    """Return a round-trip ``YAML`` instance that changes nothing it does not have to.

    Args:
        indent (dict[str, int] | None): The ``mapping``, ``sequence`` and ``offset``
            indentation to dump with. ruamel.yaml's own defaults when ``None``.

    Returns:
        YAML: The configured instance.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    # Never fold a long line, such as a path, onto the next one.
    yaml.width = sys.maxsize
    if indent is not None:
        yaml.indent(**indent)
    return yaml


def _line_content(line: str) -> tuple[int, str]:
    """Split a line into the column its content starts at and the content itself.

    The dash of a block sequence entry is not content: in ``  - name: ocean`` the content is
    ``name: ocean``, starting at column 4. That is the column a mapping nested in the entry
    is indented from.

    Args:
        line (str): A line of YAML.

    Returns:
        tuple[int, str]: The column and the content, without any trailing comment.
    """
    stripped = line.lstrip(" ")
    column = len(line) - len(stripped)
    if stripped.startswith("- "):
        item = stripped[1:].lstrip(" ")
        column += len(stripped) - len(item)
        stripped = item
    return column, _TRAILING_COMMENT.sub("", stripped).rstrip()


def _is_bare_key(content: str) -> bool:
    """Whether a line's content is a key with nothing after the colon.

    Only such a key can have a nested mapping or block sequence below it.

    Args:
        content (str): The content of a line, as returned by ``_line_content``.

    Returns:
        bool: True for ``key:``, False for ``key: value`` and anything else.
    """
    return content.endswith(":") and ": " not in content


def guess_indent(text: str) -> dict[str, int]:
    """Read the indentation a YAML text uses for its mappings and block sequences.

    The mapping indent is taken from the first mapping nested under a bare key, and the
    sequence indent and dash offset from the first block sequence nested under one, all
    relative to the column of that key -- which is how ``YAML.indent`` takes them. A block
    sequence may put its dashes in the key's own column, so an offset of zero counts too.
    Whatever the text does not show falls back to ``DEFAULT_INDENT``.

    Args:
        text (str): The YAML text.

    Returns:
        dict[str, int]: The ``mapping``, ``sequence`` and ``offset`` indentation.
    """
    found: dict[str, int] = {}
    key_column = None  # Column of the previous line's key, if it was a bare key.
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        column, content = _line_content(line)
        dash_column = len(line) - len(line.lstrip(" "))
        if key_column is not None:
            if stripped.startswith("- ") and dash_column >= key_column and "sequence" not in found:
                found["offset"] = dash_column - key_column
                found["sequence"] = column - key_column
            elif not stripped.startswith("-") and dash_column > key_column and "mapping" not in found:
                found["mapping"] = column - key_column
        if len(found) == len(DEFAULT_INDENT):
            break
        key_column = column if _is_bare_key(content) else None

    return {**DEFAULT_INDENT, **found}


class YAMLConfig(dict):
    """Class to store a YAML configuration as a dict.

    The YAML parsers generates an instance of CommentedMap, which in turn also behaves
    like a dictionary. Unfortunately we cannot simply subclass CommentedMap. This is
    because the dump method of YAML calls the __str__ method of CommentedMap, which leads
    to a infinite recursion when calling the __str__ method of this class. This means
    that, instead, we need to keep a copy of the CommentedMap and sync it with the dict.

    The mapping methods ``dict`` implements in C are all overridden, because they do not go
    through ``__setitem__`` / ``__delitem__`` and would therefore leave the CommentedMap,
    and so the text written out, out of step with the dict.

    Args:
        map (CommentedMap): The round-trip loaded configuration.
        yaml (YAML | None): The instance to dump with. One with ruamel.yaml's default
            indentation when ``None``.
    """

    map: CommentedMap  # The round-trip loaded configuration, kept in step with the dict.
    yaml: YAML  # The instance the configuration is written back out with.

    def __init__(self, map: CommentedMap, yaml: YAML | None = None) -> None:
        self.map = map
        self.yaml = yaml if yaml is not None else _round_trip_yaml()
        super().__init__(map)

    def __str__(self) -> str:
        output = StringIO("")
        self.yaml.dump(self.map, output)
        return output.getvalue()

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self.map[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.map[key]

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self.map[key]

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        """Override method to assign several items, keeping the CommentedMap in sync."""
        items = other.items() if hasattr(other, "keys") else other
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Override method to add an item if absent, keeping the CommentedMap in sync."""
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key: str, /, *default: Any) -> Any:
        """Override method to remove an item, keeping the CommentedMap in sync."""
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default:
            return default[0]
        raise KeyError(key)

    def popitem(self) -> tuple[str, Any]:
        """Override method to remove the last item, keeping the CommentedMap in sync."""
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(list(self)))
        return key, self.pop(key)

    def clear(self) -> None:
        """Override method to remove every item, keeping the CommentedMap in sync."""
        for key in list(self):
            del self[key]

    def __ior__(self, other: Any) -> YAMLConfig:  # type: ignore[override, misc]
        """Override in-place merge, keeping the CommentedMap in sync."""
        self.update(other)
        return self


class YAMLParser:
    """Wrapper class to the ruamel.yaml parser."""

    def __init__(self) -> None:
        self.parser = _round_trip_yaml()

    def parse(self, stream: str) -> YAMLConfig:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            YAMLConfig: The configuration, written back out with the text's own indentation.
        """
        return YAMLConfig(self.parser.load(stream), _round_trip_yaml(guess_indent(stream)))
