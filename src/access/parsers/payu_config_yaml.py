# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle YAML-based configuration files.

The ruamel.yaml parser provides round-trip parsing of YAML files and has all capabilities we require. Here we simply
provide wrappers around the ruamel.yaml classes so that the API is the same/similar to the other parsers.
"""

from io import StringIO
from typing import Any

from ruamel.yaml import YAML, CommentedMap


class YAMLConfig(dict):
    """Class to store a YAML configuration as a dict.

    The YAML parsers generates an instance of CommentedMap, which in turn also behaves like a dictionary. Unfortunately
    we cannot simply subclass CommentedMap. This is because the dump method of YAML calls the __str__ method of
    CommentedMap, which leads to a infinite recursion when calling the __str__ method of this class. This means that,
    instead, we need to keep a copy of the CommentedMap and sync it with the dict.
    """

    def __init__(self, map: CommentedMap) -> None:
        self.map = map
        super().__init__(map)

    def __str__(self) -> str:
        output = StringIO("")
        YAML().dump(self.map, output)
        return output.getvalue()

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self.map[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.map[key]

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self.map[key]


class YAMLParser:
    """Wrapper class to the ruamel.yaml parser."""

    def __init__(self) -> None:
        self.parser = YAML()

    def parse(self, stream: str) -> YAMLConfig:
        return YAMLConfig(self.parser.load(stream))
