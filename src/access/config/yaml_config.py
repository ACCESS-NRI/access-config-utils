# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle YAML-based configuration files.

The ruamel.yaml parser provides round-trip parsing of YAML files and has all capabilities
we require. Here we simply provide wrappers around the ruamel.yaml classes so that the API
is the same/similar to the other parsers.

Round-trip loading keeps the comments, the key order and, with ``preserve_quotes``, the
quoting of every scalar. What it does not keep is the layout: ruamel.yaml writes everything
back with one set of indents, whatever the file used, and with its own spelling of some
values (``True`` as ``true``, ``[ 1, 2 ]`` as ``[1, 2]``). So the parser reads the
indentation off the text first and dumps with that, and the configuration keeps the text it
was read from: an untouched configuration is written back as that text, and an edited one
as that text with only the lines that changed replaced.
"""

from __future__ import annotations

import difflib
import re
import sys
from io import StringIO
from typing import Any

from ruamel.yaml import YAML, CommentedMap
from ruamel.yaml.representer import RoundTripRepresenter

# Indentation used where the text does not show it, e.g. a file with no nested mapping or
# no block sequence. A sequence's dash is indented by two, as in the ACCESS configurations.
DEFAULT_INDENT = {"mapping": 2, "sequence": 4, "offset": 2}

# A trailing comment: a "#" preceded by whitespace, as YAML defines one.
_TRAILING_COMMENT = re.compile(r"\s+#.*$")

# What ruamel.yaml writes differently from the text it read, besides the spacing: the spaces
# inside the brackets of a flow collection, the case of a boolean, and a null value.
_FLOW_BRACKET_SPACES = re.compile(r"(?<=[\[{]) +| +(?=[\]}])")
_BOOLEAN = re.compile(r"(?<![\w.])(?:True|TRUE|False|FALSE)(?![\w.])")
_NULL_VALUE = re.compile(r"([:-]) (?:null|Null|NULL|~)(?= #|$)")


def _round_trip_yaml(indent: dict[str, int] | None = None) -> YAML:
    """Return a round-trip ``YAML`` instance that changes nothing it does not have to.

    *indent* holds the ``mapping``, ``sequence`` and ``offset`` to dump with; ruamel.yaml's
    own defaults apply when it is ``None``.
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

    The content has no trailing comment and no sequence dash: in ``  - name: ocean`` it is
    ``name: ocean``, at column 4, which a nested mapping is indented from.
    """
    stripped = line.lstrip(" ")
    column = len(line) - len(stripped)
    if stripped.startswith("- "):
        item = stripped[1:].lstrip(" ")
        column += len(stripped) - len(item)
        stripped = item
    return column, _TRAILING_COMMENT.sub("", stripped).rstrip()


def guess_indent(text: str) -> dict[str, int]:
    """Read the indentation a YAML text uses for its mappings and block sequences.

    The mapping indent is taken from the first mapping nested under a bare key, and the
    sequence indent and dash offset from the first block sequence nested under one, all
    relative to the column of that key -- which is how ``YAML.indent`` takes them. A block
    sequence may put its dashes in the key's own column, so an offset of zero counts too.
    Whatever the text does not show falls back to ``DEFAULT_INDENT``, so the result always
    has its ``mapping``, ``sequence`` and ``offset`` keys.
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
        # Only a key with nothing after its colon can have a mapping or sequence below it.
        key_column = column if content.endswith(":") and ": " not in content else None

    return {**DEFAULT_INDENT, **found}


def _line_key(line: str) -> str:
    """Return a line as ruamel.yaml would write it, as far as the line alone shows that.

    Two lines with the same key are the same line of the configuration, written by different
    hands: the one in the text, and the one ruamel.yaml writes for it.
    """
    key = " ".join(line.split())
    key = _FLOW_BRACKET_SPACES.sub("", key)
    key = _BOOLEAN.sub(lambda match: match.group().lower(), key)
    return _NULL_VALUE.sub(r"\1", key)


def _indent(line: str) -> int:
    """Return the column a line's content, or its sequence dash, starts at."""
    return len(line) - len(line.lstrip(" "))


def _text_column(column: int, row: int, dumped: list[str], source: dict[int, int], text: list[str]) -> int:
    """Return the column in the text for *column* of the dump, at the dump's line *row*.

    The column is that of the nearest line above at the same column of the dump, as the text
    writes it. A line with no such line above it in its block is indented from its parent as
    in the dump.
    """
    for k in range(min(row, len(dumped)) - 1, -1, -1):
        line = dumped[k]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        above = _indent(line)
        if above <= column and k in source:
            return _indent(text[source[k]]) + column - above
        if above < column:
            break
    return column


def _patch(text: str, before: str, after: str) -> str | None:
    """Return *text* with the change from *before* to *after* made to it, or None.

    *before* and *after* are what ruamel.yaml writes for the configuration as it was
    read and as it is now, so they differ in the lines that changed and nowhere else. Each
    line of *before* is matched to the line of *text* it was written from, and the lines
    that changed are replaced there, at the indentation the text uses around them. The
    lines that did not change stay as the text writes them. None means a changed line could
    not be matched.
    """
    lines, dumped, new = (part.splitlines(keepends=True) for part in (text, before, after))

    # the line of the text each line of the dump was written from
    source: dict[int, int] = {}
    keys = difflib.SequenceMatcher(
        None, [_line_key(line) for line in dumped], [_line_key(line) for line in lines], False
    )
    for block in keys.get_matching_blocks():
        source.update({block.a + k: block.b + k for k in range(block.size)})

    changes = difflib.SequenceMatcher(None, dumped, new, False).get_opcodes()
    # from the last change to the first, so that the rows of the text yet to change stay put
    for tag, i1, i2, j1, j2 in reversed(changes):
        if tag == "equal":
            continue
        if i1 < i2:
            rows = [source.get(i) for i in range(i1, i2)]
            if None in rows or rows != list(range(rows[0], rows[0] + len(rows))):
                return None
            start, end = rows[0], rows[-1] + 1
        elif i1 == 0:
            start = end = 0
        elif i1 - 1 in source:
            start = end = source[i1 - 1] + 1
        else:
            return None
        if tag == "replace" and i2 - i1 == j2 - j1:
            # each line takes the place of the line it changed from, indented as that one is
            columns = [
                _indent(lines[source[i1 + k]]) + _indent(line) - _indent(dumped[i1 + k])
                for k, line in enumerate(new[j1:j2])
            ]
        else:
            columns = [_text_column(_indent(line), i1, dumped, source, lines) for line in new[j1:j2]]
        lines[start:end] = [
            " " * max(column, 0) + line.lstrip(" ") for column, line in zip(columns, new[j1:j2], strict=True)
        ]
    return "".join(lines)


def _reads_as(text: str, config: CommentedMap) -> bool:
    """Return whether *text* reads as the same data as *config*."""
    try:
        return _round_trip_yaml().load(text) == config
    except Exception:  # a text that does not even read is not the configuration
        return False


class YAMLConfig(CommentedMap):
    """Class to store a YAML configuration.

    The configuration is the round-trip loaded CommentedMap itself, so every change made to
    it, through any of the mapping methods, is in what ``__str__`` writes out.

    ``YAMLParser`` turns the CommentedMap it loads into a YAMLConfig by reassigning its
    class, which keeps everything ruamel.yaml recorded about it. That only works while the
    two classes have the same layout, so this class must not define ``__slots__``.

    A configuration read from a text is written back out as that text, with only the lines
    that changed replaced, so that the lines that did not change keep the indentation, the
    spacing and the spelling of the text, which ruamel.yaml would write its own way. That
    text is checked to read as the configuration does, and the whole configuration is
    written out by ruamel.yaml instead when it does not, or when a changed line cannot be
    placed in it.

    Attributes:
        yaml (YAML | None): The instance the configuration is written back out with. One
            with ruamel.yaml's default indentation when ``None``.
        text (str | None): The text the configuration was read from, if any.
        dumped (str | None): What ``yaml`` wrote for the configuration as it was read.
    """

    yaml: YAML | None = None
    text: str | None = None
    dumped: str | None = None

    def __str__(self) -> str:
        dumped = self._dump()
        if self.text is None or self.dumped is None:
            return dumped
        if dumped == self.dumped:
            return self.text
        patched = _patch(self.text, self.dumped, dumped)
        return patched if patched is not None and _reads_as(patched, self) else dumped

    def _dump(self) -> str:
        """Return the configuration as ruamel.yaml writes it."""
        output = StringIO("")
        (self.yaml if self.yaml is not None else _round_trip_yaml()).dump(self, output)
        return output.getvalue()

    def copy_attributes(self, t: Any, memo: Any = None) -> Any:
        """Override method so that a copy is written back out like the original."""
        t.yaml = self.yaml
        t.text = self.text
        t.dumped = self.dumped
        return super().copy_attributes(t, memo)


# ruamel.yaml finds the representer of an object by its exact type, so a subclass of
# CommentedMap needs registering too. Otherwise dumping it falls through to the representer
# for unknown objects, whose error message calls __str__, which dumps it again, and so on.
RoundTripRepresenter.add_representer(YAMLConfig, RoundTripRepresenter.represent_dict)


class YAMLParser:
    """Wrapper class to the ruamel.yaml parser."""

    def __init__(self) -> None:
        self.parser = _round_trip_yaml()

    def parse(self, stream: str) -> YAMLConfig:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            YAMLConfig: The configuration, written back out as the text with only the lines
                that change replaced.

        Raises:
            TypeError: If the top level of the text is not a mapping.
        """
        config = self.parser.load(stream)
        if not isinstance(config, CommentedMap):
            raise TypeError(f"The top level of a YAML configuration must be a mapping, not {type(config).__name__}")
        config.__class__ = YAMLConfig
        config.yaml = _round_trip_yaml(guess_indent(stream))
        config.text = stream
        config.dumped = config._dump()
        return config
