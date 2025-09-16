# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for Fortran namelists.

Fortran namelists allow format-free I/O of variables by key-value assignements. Initially they were an extension to the
languages, but became part of the standard in Fortran 90.
"""

from access.parsers.config import ConfigParser


class FortranNMLParser(ConfigParser):
    """Fortran Namelist parser.

    Note: Currently array qualifiers, substrings and derived types are not implemented in the grammar.
    """

    @property
    def case_sensitive_keys(self) -> bool:
        return False

    @property
    def grammar(self) -> str:
        return """
?start: namelists

?namelists: random_text? namelist (random_text? namelist)* random_text?

namelist.2: nml_start key line_end? nml_lines nml_end -> key_block
nml_start: ws* "&"
nml_end:  ws* ("/"|/&end/i) line_end
nml_lines: (nml_line | empty_line)* -> block

?nml_line: assignment (ws* "," assignment)* (ws* separator)? line_end?

?assignment: key_value | key_list | key_null

key_value: ws* key ws* "=" ws* value
key_list: ws* key ws* "=" ws* value ((line_break|ws* separator) ws* value)+
key_null: ws* key ws* "=" ws*
line_break: ws* separator line_end

?value: logical
      | integer
      | float
      | double
      | complex
      | double_complex
      | string

empty_line: line_end
line_end: (fortran_comment|ws*) NEWLINE
separator: ","

random_text: (/.+/|NEWLINE)*
ANYTHING: /.+/

%import config.key
%import config.logical
%import config.integer
%import config.float
%import config.double
%import config.complex
%import config.double_complex
%import config.string
%import config.fortran_comment
%import config.ws
%import config.NEWLINE
"""
