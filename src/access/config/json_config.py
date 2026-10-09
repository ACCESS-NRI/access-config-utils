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
from it.

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
from typing import Any

# The candidate layouts, most likely first: the indents json.dumps is usually given, then
# the rest, each with the separators it is usually paired with.
_INDENTS: tuple[int | str | None, ...] = (2, 4, None, 1, 3, 5, 6, 7, 8, "\t", 0)
_SEPARATORS: tuple[tuple[str, str], ...] = ((",", ": "), (", ", ": "), (",", ":"))

# Whitespace as JSON defines it. str.strip() would also remove other Unicode spaces.
_JSON_WHITESPACE = " \t\r\n"


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
    """

    indent: int | str | None = 2
    separators: tuple[str, str] = (",", ": ")
    ensure_ascii: bool = True
    newline: str = "\n"
    leading: str = ""
    trailing: str = "\n"

    def dump(self, data: Any) -> str:
        """Write *data* in this style.

        Args:
            data (Any): The data to write.

        Returns:
            str: The JSON text.
        """
        text = json.dumps(data, indent=self.indent, separators=self.separators, ensure_ascii=self.ensure_ascii)
        # json.dumps escapes a line break inside a string, so every one in its output is a
        # line terminator.
        if self.newline != "\n":
            text = text.replace("\n", self.newline)
        return self.leading + text + self.trailing


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
    indent: int | str | None = None
    if len(lines) > 1:
        # The first indented line shows one level of indentation.
        padding = next((line[: len(line) - len(line.lstrip(" \t"))] for line in lines[1:] if line[:1] in " \t"), "")
        indent = "\t" if padding.startswith("\t") else len(padding)
    key_separator = ": " if re.search(r'"\s*:[ \t]', body) else ":"
    item_separator = ", " if indent is None and re.search(r",[ \t]", body) else ","
    ensure_ascii = body.isascii()
    return JSONStyle(indent, (item_separator, key_separator), ensure_ascii, newline, leading, trailing)


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
