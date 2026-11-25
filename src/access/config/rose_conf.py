# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for Rose configuration files.

Rose configures the suites that Cylc runs, such as those of the UK Met Office Unified Model,
in files like ``rose-suite.conf`` and ``rose-app.conf``. The format is described here:

<https://metomi.github.io/rose/2019.01.8/html/api/configuration/rose-configuration-format.html>

It is a modified INI format. This grammar follows what ``ConfigLoader.load`` in
``rose/config.py`` of Rose 2019.01 reads, which says more than the page:

- A setting is ``KEY=VALUE``, its key holding no whitespace and no equals sign. The value is
  the rest of the line, trimmed, and may hold anything, an equals sign or a ``#`` included.
  ``KEY=`` gives the empty string.
- A value goes on over the indented lines that follow it. Rose trims each of them, drops one
  equals sign it starts with, and joins it to the value after a line break. A blank or
  comment line between them adds nothing and does not end the value.
- A line whose first character other than a blank is ``#`` is a comment. A ``#`` anywhere
  else is part of a value. The comments that open the file, up to its first blank line or
  setting, are the file's, and the comments right above a header are its section's.
- ``[NAME]`` starts a section, which runs until the next one: nothing closes it. The
  settings above the first section belong to no section.
- ``!`` or ``!!`` before a key or a section name marks it ignored, by the user or by a
  program.
- A section written twice holds the settings of both, and a key written twice keeps its last
  value.

Every value is text. Rose gives a setting no type, and the same text means different
things in different sections: ``[jinja2:suite.rc]`` holds Jinja2 expressions, a
``[namelist:...]`` section Fortran values, and ``[env]`` shell text. So a value reads as the
string Rose reads, and a string is written as given: ``'P1M'`` keeps its quotes, which are
part of a Jinja2 string. ``KEY=`` reads as ``None``.

The marks of an ignored setting or section are part of its key: ``!!MAIN_IOS_NPROC`` and
``MAIN_IOS_NPROC`` are two keys. This means that a setting Rose ignores is never mistaken
for one it uses, and a new key copied from an ignored neighbour does not take on its marks.
The cost is that a file holding both ``KEY=`` and ``!KEY=`` reads as two keys, where Rose
reads one, in the state written last.

Four things Rose reads are refused: the empty header ``[]``, which takes Rose back to the
settings of no section, a header with blanks before it, blanks inside the brackets, and a
setting whose key starts with ``[``, such as ``[a]=1``.
"""

from __future__ import annotations

from collections.abc import Mapping

from access.config.grammar_contract import BLOCK_RULE, KEY_BLOCK, KEY_NULL, KEY_VALUE, START_RULE
from access.config.parser import ConfigParser


class RoseConfParser(ConfigParser):
    """Rose configuration file parser.

    Reads and edits ``rose-suite.conf``, ``rose-app.conf`` and the other files Rose writes
    in the same format. A section is a block, and the settings of no section are at the top
    level. Every value is a string.

    A new setting at the top level is written above the first section, since a section runs
    until the next one. A new section is written after the last one.
    """

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        """The text of a new setting, at the top level and in a section, and of a section.

        A new section starts with a blank line, as Rose writes one, to part it from what is
        above it. Once a section is there to copy, it is parted as that one is.

        Returns:
            Mapping[tuple[str, str], str]: Templates for an entry.
        """
        return {
            (START_RULE, KEY_VALUE): "{key}={value}\n",
            (START_RULE, KEY_NULL): "{key}=\n",
            (START_RULE, KEY_BLOCK): "\n[{key}]\n",
            (BLOCK_RULE, KEY_VALUE): "{key}={value}\n",
            (BLOCK_RULE, KEY_NULL): "{key}=\n",
        }

    @property
    def grammar(self) -> str:
        return r"""
// "start" must be a non-transparent rule: the root node is the container that new entries
// are added to, and a transparent rule collapses away for a single-entry file. The settings
// of no section come first, as nothing closes a section.
start: opening* line* section* last_line?
?line: key_value | key_null | comment | blank_line

// The comments that open the file, up to its first blank line or setting, are the file's,
// as Rose gives them to the root of the configuration: they stay where they are when the
// first section is deleted, and a new setting of no section goes below them.
opening.3: COMMENT -> comment

// The blank lines and the comments right above a header belong to its section, as Rose
// associates them, so a new section is parted from the one above as that one is, and a
// section deleted takes its comments with it. The comments that open the file are the
// exception, being the file's. The line "parse" adds to the end of the file
// stays at the top level, so a new section goes above it and the file keeps its last line
// break. The priorities say which rule takes such a line where both could.
section: gap* heading* "[" name "]" line_end block -> key_block
// Spelled out rather than reusing "line*": Lark would share one helper rule between
// "start" and "block", which is then no plain repetition, and deleting the last setting of
// a section would delete the section.
block: (key_value | key_null | comment | blank_line)*

key_value: key eq ws* rose_text line_end
key_null: key eq line_end
// The whitespace before "=" belongs to a rule of its own, so that the reconstructor writes
// it back on the side of "=" it came from.
eq: ws* "="

gap.2: ws* NEWLINE
heading.2: COMMENT -> comment
last_line.3: ws* NEWLINE

// A key holds no whitespace and no "=", and does not start with "[" or "#", which open a
// header and a comment. A section name holds no bracket and no line break, and has no
// blanks at either end.
key: KEY
KEY: /[^\s=\[#][^\s=]*/
name: NAME -> key
NAME: /[^\s\[\]](?:[^\[\]\r\n]*[^\s\[\]])?/

// A value: the rest of the line, then any indented line that continues it, with the blank
// and comment lines between them. The value may also start on an indented line.
rose_text: ROSE_TEXT
ROSE_TEXT: _TEXT _CONTINUATION* | _CONTINUATION+
_TEXT: /\S(?:[^\r\n]*\S)?/
_CONTINUATION: /(?:\r?\n[ \t]*(?:#[^\r\n]*)?)*\r?\n[ \t]+[^\s#](?:[^\r\n]*\S)?/

// A comment takes its line break, so that a new section copied from one headed by a
// comment does not keep an empty line where the comment was.
comment: COMMENT
COMMENT: /[ \t]*#[^\n]*\n/
blank_line: ws* NEWLINE
line_end: ws* NEWLINE

%import config.ws
%import config.NEWLINE
"""
