# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle JSON configuration files.

JSON has no comments, and none of its whitespace means anything, so a file is completely
described by its data and by how that data was laid out. The standard library reads the
data; what it cannot do is write it back the way the file had it. A file written by a
program is laid out by one set of ``json.dumps`` options throughout -- every
``forcing.json`` used by ACCESS-OM2 is byte-identical to ``json.dumps(data, indent=2)``
followed by a newline -- so the parser looks for the options that reproduce the text
exactly, and writes the data back with those. An edit then changes exactly the lines that
``json.dumps`` writes differently.

A hand-formatted file has no such options. It is returned exactly as read for as long as its
data is unchanged, but once edited it is rewritten in the closest style that can be guessed
from it. That includes which lists it wrote on one line: ``json.dumps`` puts every element
of a list on a line of its own, so the writer here keeps each list in the layout the file
gave it, found by the keys and indexes leading to it, and writes a list the file did not
have the way most of its lists are written.

Unlike the other configurations this one holds plain data: nested objects are ``dict``s and
arrays are ``list``s, edited directly with nothing to keep in step, because the text is
regenerated from the data rather than edited in place.

Note: files the standard library refuses are refused here too, among them a trailing comma
after the last element of an object or an array, which some readers tolerate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from json.decoder import scanstring
from typing import Any

# The candidate layouts, most likely first: the indents json.dumps is usually given, then
# the rest, each with the separators it is usually paired with.
_INDENTS: tuple[int | str | None, ...] = (2, 4, None, 1, 3, 5, 6, 7, 8, "\t", 0)
_SEPARATORS: tuple[tuple[str, str], ...] = ((",", ": "), (", ", ": "), (",", ":"))

# Whitespace as JSON defines it. str.strip() would also remove other Unicode spaces.
_JSON_WHITESPACE = " \t\r\n"

# Where a list sits: the object keys and list indexes leading to it.
_Path = tuple[str | int, ...]


@dataclass(frozen=True)
class JSONStyle:
    """How a JSON text is laid out, as the ``json.dumps`` options that produce it.

    Args:
        indent (int | str | None): The ``indent`` given to ``json.dumps``.
        separators (tuple[str, str]): The item and key separators given to ``json.dumps``.
        ensure_ascii (bool): Whether non-ASCII characters are written as escapes.
        newline (str): The line terminator, ``"\\n"`` or ``"\\r\\n"``.
        leading (str): Whitespace before the top-level object.
        trailing (str): Whitespace after it, usually a single line terminator.
        inline_lists (frozenset[tuple[str | int, ...]]): Where the text had a list on one
            line, as the keys and indexes leading to it. Empty for a style ``json.dumps``
            reproduces.
        wrapped_lists (frozenset[tuple[str | int, ...]]): The same for lists with a line of
            their own for every element.
        inline_new_lists (bool): Whether a list in neither set is written on one line.
        inline_separator (str): The separator between the elements of a list on one line.
    """

    indent: int | str | None = 2
    separators: tuple[str, str] = (",", ": ")
    ensure_ascii: bool = True
    newline: str = "\n"
    leading: str = ""
    trailing: str = "\n"
    inline_lists: frozenset[_Path] = frozenset()
    wrapped_lists: frozenset[_Path] = frozenset()
    inline_new_lists: bool = False
    inline_separator: str = ", "

    def dump(self, data: Any) -> str:
        """Write *data* in this style.

        Args:
            data (Any): The data to write.

        Returns:
            str: The JSON text.
        """
        if self.indent is not None and (self.inline_lists or self.inline_new_lists):
            text = self._write(data, (), 0)
        else:
            text = json.dumps(data, indent=self.indent, separators=self.separators, ensure_ascii=self.ensure_ascii)
        # json.dumps escapes a line break inside a string, so every one in its output is a
        # line terminator.
        if self.newline != "\n":
            text = text.replace("\n", self.newline)
        return self.leading + text + self.trailing

    def _write(self, value: Any, path: _Path, level: int) -> str:
        """Write *value* as ``json.dumps`` would, except that lists keep their layout.

        Args:
            value (Any): The value to write.
            path (tuple[str | int, ...]): The keys and indexes leading to it.
            level (int): How deep it is nested.

        Returns:
            str: The text, with ``"\\n"`` as its line terminator.
        """
        item_separator, key_separator = self.separators
        unit = " " * self.indent if isinstance(self.indent, int) else (self.indent or "")
        if isinstance(value, dict) and value:
            lines = [
                unit * (level + 1)
                + json.dumps(key, ensure_ascii=self.ensure_ascii)
                + key_separator
                + self._write(item, (*path, key), level + 1)
                for key, item in value.items()
            ]
            return "{\n" + (item_separator + "\n").join(lines) + "\n" + unit * level + "}"
        if isinstance(value, list) and value:
            if _holds_no_object(value) and (
                path in self.inline_lists or (path not in self.wrapped_lists and self.inline_new_lists)
            ):
                return "[" + self.inline_separator.join(self._write_inline(item) for item in value) + "]"
            lines = [
                unit * (level + 1) + self._write(item, (*path, index), level + 1) for index, item in enumerate(value)
            ]
            return "[\n" + (item_separator + "\n").join(lines) + "\n" + unit * level + "]"
        return json.dumps(value, ensure_ascii=self.ensure_ascii)

    def _write_inline(self, value: Any) -> str:
        """Write a value on one line, the lists inside it too."""
        if isinstance(value, list):
            return "[" + self.inline_separator.join(self._write_inline(item) for item in value) + "]"
        return json.dumps(value, ensure_ascii=self.ensure_ascii)


def _holds_no_object(value: list[Any]) -> bool:
    """Report whether a list holds no object, however deep, as a list on one line must."""
    return all(_holds_no_object(item) if isinstance(item, list) else not isinstance(item, dict) for item in value)


def detect_style(text: str, data: Any) -> JSONStyle:
    """Return the style *text* is written in.

    The ``json.dumps`` options that reproduce *text* exactly when given *data*, if there are
    any. Otherwise the style is guessed from the text: an edited file is then rewritten in
    that style, which is as close to the original as ``json.dumps`` can come.

    Args:
        text (str): The JSON text.
        data (Any): The data *text* holds, as ``json.loads`` read it.

    Returns:
        JSONStyle: The style.
    """
    body = text.strip(_JSON_WHITESPACE)
    leading = text[: len(text) - len(text.lstrip(_JSON_WHITESPACE))]
    trailing = text[len(text.rstrip(_JSON_WHITESPACE)) :]
    newline = "\r\n" if "\r\n" in body else "\n"
    for indent in _INDENTS:
        for separators in _SEPARATORS:
            for ensure_ascii in (True, False):
                style = JSONStyle(indent, separators, ensure_ascii, newline, leading, trailing)
                if style.dump(data) == text:
                    return style
    return _guess_style(body, newline, leading, trailing)


class _LayoutScanner:
    """Finds where a JSON text writes a list on one line, and where one element a line.

    A list holding an object, or nothing, says nothing about layout and is left out.

    Args:
        text (str): Valid JSON.

    Attributes are set by ``scan``: ``inline`` and ``wrapped`` hold the paths of the lists
    written on one line and over several, and ``spaced`` says whether the former put a
    space after a comma.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.inline: set[_Path] = set()
        self.wrapped: set[_Path] = set()
        self.spaced = False
        self.scan(0, ())

    def _skip(self, i: int) -> int:
        """Return the position of the first character from *i* that is not whitespace."""
        while self.text[i] in _JSON_WHITESPACE:
            i += 1
        return i

    def scan(self, i: int, path: _Path) -> tuple[int, bool]:
        """Return where the value at *i* ends, and whether it is or holds an object."""
        i = self._skip(i)
        opener = self.text[i]
        if opener == '"':
            return scanstring(self.text, i + 1)[1], False
        if opener == "[":
            return self._scan_array(i, path)
        if opener == "{":
            return self._scan_object(i, path)
        while self.text[i] not in ",]} \t\r\n":
            i += 1
        return i, False

    def _scan_object(self, start: int, path: _Path) -> tuple[int, bool]:
        """Return where the object starting at *start* ends, scanning it for lists."""
        i = self._skip(start + 1)
        while self.text[i] != "}":
            key, i = scanstring(self.text, i + 1)
            i, _ = self.scan(self._skip(self._skip(i) + 1), (*path, key))
            i = self._skip(i)
            i = self._skip(i + 1) if self.text[i] == "," else i
        return i + 1, True

    def _scan_array(self, start: int, path: _Path) -> tuple[int, bool]:
        """Return where the array at *start* ends, and whether it holds an object."""
        i = self._skip(start + 1)
        count, has_object = 0, False
        while self.text[i] != "]":
            i, nested = self.scan(i, (*path, count))
            has_object |= nested
            count += 1
            i = self._skip(i)
            i = self._skip(i + 1) if self.text[i] == "," else i
        written = self.text[start:i]
        if count and not has_object:
            if "\n" in written or "\r" in written:
                self.wrapped.add(path)
            else:
                self.inline.add(path)
                self.spaced = self.spaced or re.search(r",[ \t]", written) is not None
        return i + 1, has_object


def _guess_style(body: str, newline: str, leading: str, trailing: str) -> JSONStyle:
    """Guess the style of a text no set of ``json.dumps`` options reproduces.

    Args:
        body (str): The text, without its leading and trailing whitespace.
        newline (str): The line terminator in use.
        leading (str): Whitespace before the top-level object.
        trailing (str): Whitespace after it.

    Returns:
        JSONStyle: The guessed style.
    """
    lines = body.splitlines()
    layout = _LayoutScanner(body)
    inline, wrapped = frozenset(layout.inline), frozenset(layout.wrapped)
    indent: int | str | None = None
    if len(lines) > 1:
        # The first indented line shows one level of indentation.
        padding = next((line[: len(line) - len(line.lstrip(" \t"))] for line in lines[1:] if line[:1] in " \t"), "")
        indent = "\t" if padding.startswith("\t") else len(padding)
    key_separator = ": " if re.search(r'"\s*:[ \t]', body) else ":"
    item_separator = ", " if indent is None and re.search(r",[ \t]", body) else ","
    ensure_ascii = body.isascii()
    return JSONStyle(
        indent,
        (item_separator, key_separator),
        ensure_ascii,
        newline,
        leading,
        trailing,
        inline,
        wrapped,
        len(inline) > 0 and len(inline) >= len(wrapped),
        ", " if layout.spaced else ",",
    )


def _fingerprint(data: Any) -> str:
    """Return a text that changes whenever *data* does, including its order and types.

    Comparing data with ``==`` is not enough: it ignores the order of keys, and treats
    ``1``, ``1.0`` and ``True`` as equal, though each is written differently.

    Args:
        data (Any): The data.

    Returns:
        str: The fingerprint.
    """
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _check_types(value: Any, where: str = "") -> None:
    """Check that *value* reads back as itself once written as JSON.

    ``json.dumps`` quietly writes a tuple as an array and a number as an object key, both of
    which read back as something else. Refusing them keeps the text and the data in step.

    Args:
        value (Any): The value to check.
        where (str): Where *value* sits in the configuration, for the error message.

    Raises:
        TypeError: If *value*, or anything inside it, cannot be written as JSON.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, not {key!r}{' in ' + where if where else ''}")
            _check_types(item, f"{where}[{key!r}]")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_types(item, f"{where}[{index}]")
    elif not isinstance(value, str | int | float | type(None)):
        raise TypeError(f"{type(value).__name__} at {where or 'the top level'} cannot be written as JSON")


class JSONConfig(dict):
    """The top-level object of a JSON configuration, as a dict.

    Args:
        data (dict[str, Any]): The parsed object.
        source (str | None): The text it was parsed from. Returned verbatim for as long as
            the data is unchanged, and the source of the style an edit is written in.
            ``None`` for a configuration made in code, written in the default style.
    """

    style: JSONStyle  # How the configuration is written out.
    _source: str | None  # The text the configuration was parsed from.
    _fingerprint: str | None  # The fingerprint of the data as parsed.

    def __init__(self, data: dict[str, Any], source: str | None = None) -> None:
        super().__init__(data)
        self._source = source
        if source is None:
            self.style = JSONStyle()
            self._fingerprint = None
        else:
            self.style = detect_style(source, data)
            self._fingerprint = _fingerprint(data)

    def __str__(self) -> str:
        """Write the configuration back out as text.

        Raises:
            TypeError: If the configuration holds anything that does not read back as itself
                once written as JSON.
        """
        _check_types(self)
        if self._source is not None and _fingerprint(self) == self._fingerprint:
            return self._source
        return self.style.dump(self)


class JSONParser:
    """Parser for JSON configuration files, whose top level has to be an object."""

    def parse(self, stream: str) -> JSONConfig:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            JSONConfig: The configuration.

        Raises:
            json.JSONDecodeError: If *stream* is not valid JSON.
            ValueError: If the top level of *stream* is not an object.
        """
        data = json.loads(stream)
        if not isinstance(data, dict):
            raise ValueError(f"The top level of a JSON configuration must be an object, not {type(data).__name__}")
        return JSONConfig(data, stream)
