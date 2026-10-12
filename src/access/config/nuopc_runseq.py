# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for NUOPC run sequence files.

The ``nuopc.runseq`` file of a CMEPS-driven model, such as ACCESS-OM3, says in which order
the components and the mediator run, and how often. CMEPS reads it in
``cesm/driver/esm.F90``::

    call ESMF_ConfigLoadFile(runSeq, "nuopc.runseq", rc=rc)
    runSeqFF = NUOPC_FreeFormatCreate(runSeq, label="runSeq::", rc=rc)
    call NUOPC_DriverIngestRunSequence(driver, runSeqFF, autoAddConnectors=.true., rc=rc)

So the file is an ESMF Resource File holding one table, labelled ``runSeq``, each line of
which is one element of the run sequence::

    runSeq::
    @3600
      MED med_phases_prep_ocn_avg
      MED -> OCN :remapMethod=redist
      OCN
      @1800
        ATM
        ATM -> MED :remapMethod=redist
      @
    @
    ::

``@<seconds>`` opens a time loop and ``@`` closes it, and loops may nest. A line naming a
component runs it, a component and a phase label runs that phase, and ``SRC -> DST`` with
optional ``:option=value`` settings runs a connector. ``#`` starts a comment, a table ends
at a line starting with ``::``, and the whitespace around a line means nothing to ESMF.
See the ESMF reference manual for Resource Files,
https://earthsystemmodeling.org/docs/release/ESMF_8_6_0/ESMF_refdoc/node6.html#SECTION06090000000000000000
and the NUOPC reference manual, under ``NUOPC_DriverIngestRunSequence``, for the run
sequence syntax: https://earthsystemmodeling.org/docs/release/latest/NUOPC_refdoc/

Each table reads as a list of its lines, kept whole as strings, without their indentation,
trailing whitespace or comments: ``config["runSeq"]`` is ``["@3600", "MED
med_phases_prep_ocn_avg", ..., "@"]``. Editing a line rewrites only that line, and since the
list can grow and shrink, a whole run sequence can be assigned, or lines inserted and
removed, with the lines that stay written exactly as they were and a new line indented like
its neighbours. The files CMEPS generates end some lines with spaces, which are kept.

Note: text outside a table is refused, though ESMF would ignore it. A table with a single
line reads as a string and an empty one as ``None``, as a single value does in every format,
so neither can become a list in place. And a label written twice reads as the last table,
where ESMF finds the first.
"""

from __future__ import annotations

from collections.abc import Mapping

from access.config.grammar_contract import KEY_LIST, KEY_NULL, KEY_VALUE, START_RULE
from access.config.parser import ConfigParser


class NUOPCRunSeqParser(ConfigParser):
    """NUOPC run sequence parser."""

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        """Mapping[tuple[str, str], str]: The text of a new table, a line per element."""
        return {
            (START_RULE, KEY_VALUE): "{key}::\n{value}\n::\n",
            (START_RULE, KEY_LIST): "{key}::\n{value}\n{value}\n::\n",
            (START_RULE, KEY_NULL): "{key}::\n::\n",
        }

    @property
    def grammar(self) -> str:
        return """
// "start" must be a non-transparent rule: the root node is the container that new entries
// are added to, and a transparent rule collapses away for a single-table file.
start: (empty_line | table_null | table_value | table_list)*

// The number of lines in a table decides its category, as the number of values does in
// every other format: none is a valueless entry, one a scalar, more a list.
table_null: ws* key open close -> key_null
table_value: ws* key open ws* text_line line_end close -> key_value
table_list: ws* key open ws* text_line (row_sep ws* text_line)+ line_end close -> key_list

// What parts two lines of a table: the end of the first, and any blank or comment lines
// between them. The indentation of the next line is the "ws*" in front of its text.
open: "::" line_end empty_line*
row_sep: line_end empty_line*
close: ws* "::" line_end

empty_line: line_end
line_end: (comment | ws*) NEWLINE

%import config.key
%import config.text_line
%import config.comment
%import config.ws
%import config.NEWLINE
"""
