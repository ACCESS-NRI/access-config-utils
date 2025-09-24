# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for MOM6 parameter files.

The MOM6 parameter file format is described here:

https://mom6.readthedocs.io/en/main/api/generated/pages/Runtime_Parameter_System.html#mom6-parameter-file-syntax

It has similarities with a Fortran namelist, but with some notable differences:
 - no opening nor closing clauses ('&NAME' and '\')
 - usage of an override directive ('#override')
 - some character, like '*', are allowed in the MOM6 parameter files, but not in namelists
 - keys are case-sensitive
We have also found MOM6 parameter files with C-style comments in files used by CESM. These are ignored by MOM6, but
are actually not part of the specifications.
"""

from access.config.parser import ConfigParser


class MOM6InputParser(ConfigParser):
    """MOM6 input file parser.

    Note: The "override" directive is currently not implemented.
    """

    @property
    def case_sensitive_keys(self) -> bool:
        return True

    @property
    def grammar(self) -> str:
        return """
?start: lines*

?lines: key_value
      | key_list
      | key_block
      | empty_line

key_value: key ws* "=" ws* value line_end
key_list: key ws* "=" ws* value (ws* "," ws* value)+ line_end
key_block: key "%" line_end block "%" key line_end

block: (key_value | key_list | empty_line)*

?value: bool
      | integer
      | float
      | string

empty_line: line_end
line_end: (fortran_comment|ws*) NEWLINE

%import config.key
%import config.bool
%import config.integer
%import config.float
%import config.string
%import config.fortran_comment
%import config.ws
%import config.NEWLINE
"""
