# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for FMS field tables, in the legacy format.

A field table lists the tracers of a model built on FMS, such as MOM5, MOM6 and the UFS
atmosphere, and how each is to be treated. The field manager reads it in
``read_field_table_legacy`` of ``field_manager/field_manager.F90``, and what follows is
what that reads, https://github.com/NOAA-GFDL/FMS/blob/main/field_manager/field_manager.F90.

An entry is a header naming a field type, a model and a field, then the field's methods
one per line, closed by a ``/`` as the last character of a line -- of the header itself,
for an entry with no methods::

    "TRACER", "atmos_mod", "sphum"
        "longname", "specific humidity"
        "profile_type", "fixed", "surface_value=3.e-6" /
    "prog_tracers", "ocean_mod", "temp"
    restart_file = ocean_temp_salt.res.nc
    ppm_hlimiter = 3
    /

A method is written as two strings, a name and a value; as three, a name, a method and a
control of comma-separated parameters, ``k=v,k2=v2``; or as an assignment, ``k = v``, bare
or in one string. A value is typed by its spelling, as the field manager types one: see
``grammar_values``. A ``#`` in the first column of a line makes it a comment.

The configuration reads as the FMS YAML field table lays it out, its lists keyed by name::

    config["tracer"]["atmos_mod"]["sphum"] == {
        "longname": "specific humidity",
        "profile_type": [{"value": "fixed", "surface_value": 3e-06}],
    }

The header names its three levels, and a method its key. A method of three strings is a
record, its name under ``"value"`` and each parameter beside it, and the lines writing one
method make a list of records, one per line: the rayleigh damping tables of ACCESS-OM2
write dozens of lines of ``"rayleigh", "Lombok", "itable=36,jtable=112"``. A record is
read only, and is changed by assigning a dict to its place in the list, as in
``rayleigh[1] = {**rayleigh[1], "itable": 45}``. Keys are matched in any case, and read
in lower case as the field manager reads them.

Editing changes one token, and a new method or entry is written like the one above it.
A new entry goes after the last of its type and model, and spells those the way it does.

Note: what the field manager would misread or refuse is refused here: a three-string
method with no ``=`` in its control, a value holding a ``,`` or an ``=``, an empty value, a
comment that does not start its line or that ends with ``/`` inside an entry, text after
the closing ``/``, a header of fewer than three strings or of three not parted by commas,
and an entry never closed. A file holding one does not parse.

Note: the names of a record's parameters keep their case, where the field manager reads
them in lower case. A bare key holding slashes, ``a/b = 1``, is a key of its own rather
than a path. And an entry whose header the file writes twice reads as one, as the field
manager adds both to one list, but a key cannot then be added to it.

Note: parsing takes about half a second for a field table under 10 KB, and three seconds
for the 40 KB tables of the UFS air quality model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from access.config.grammar_contract import BLOCK_RULE, KEY_BLOCK, KEY_LIST, KEY_VALUE, START_RULE
from access.config.parser import ConfigParser


class FieldTableParser(ConfigParser):
    """FMS legacy field table parser."""

    @property
    def key_case(self) -> str:
        """str: ``"lower"``, the case the field manager reads every name in."""
        return "lower"

    @property
    def block_rules(self) -> Sequence[str]:
        """Sequence[str]: The methods of an entry, then its model and its field.

        A header writes the three levels in one line, so the model and the field are blocks
        with no body of their own: a new one is written as a whole header.
        """
        return (BLOCK_RULE, "model_body", "name_body")

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        """Templates for a new entry, and for the first method of an entry that has none.

        An entry is a header naming a field type, a model and a field, closed on its own
        line as an entry with no methods is. A method is indented, as it is in most field
        tables, and otherwise written like the one above it, or like those of the entry
        above.
        """
        return {
            (START_RULE, KEY_BLOCK): '"{keys:", "}"/\n',
            (BLOCK_RULE, KEY_VALUE): '\n  "{key}", {value}',
            (BLOCK_RULE, KEY_LIST): "\n  {key} = {value}, {value}",
        }

    @property
    def grammar(self) -> str:
        return r"""
// "start" must be a non-transparent rule: the root node is the container that new entries
// are added to. Between entries only blank lines and comments starting their line are
// allowed, as the field manager skips nothing else.
start: (blank_line | comment_line | type_entry)*
blank_line: ws* NEWLINE
comment_line: comment NEWLINE
comment: /#[^\r\n]*/

// The header: the field type, the model and the field, each a block holding the next, so a
// header is a chain of three keys and the entries sharing a type or a model merge.
type_entry: ws* "\"" key "\"" comma ws* model_body -> key_block
model_body: model_entry
model_entry: "\"" key "\"" comma ws* name_body -> key_block
name_body: name_entry
name_entry: "\"" key "\"" block closer -> key_block

// Each method line starts with its line break, so that the "/" closing the entry stays on
// the last line, after the methods, whatever is added.
block: (pair | record | assign | assign_list | quoted_assign | quoted_assign_list | method_comment | method_blank)*
pair: nl ws* "\"" key "\"" comma ws* inferred_string -> key_value
record: nl ws* "\"" key "\"" comma ws* named_assignments -> key_value
assign: nl ws* key eq ws* inferred_text -> key_value
assign_list: nl ws* key eq ws* inferred_text (comma ws* inferred_text)+ -> key_list
// One string holding an assignment. The quotes are a rule, as "comma" and "eq" are, so the
// reconstructor can tell this from a bare assignment holding the same children.
quoted_assign: nl ws* dquote ws* key eq ws* inferred_text ws* dquote -> key_value
quoted_assign_list: nl ws* dquote ws* key eq ws* inferred_text (comma ws* inferred_text)+ ws* dquote -> key_list
// A comment ending with "/" would end the entry, so it is not a comment here.
method_comment: nl entry_comment
entry_comment: /#(?:[^\r\n]*[^\/ \t\r\n])?[ \t]*(?=\r?\n)/ -> comment
method_blank: nl ws*

// The "/" ends the last method's line, or stands on a line of its own. The priority stops
// a "/" on its own line being read as a blank line then a "/" ending it.
closer: inline_close | own_line_close
inline_close: ws* "/" ws* NEWLINE
own_line_close.2: nl ws* "/" ws* NEWLINE

dquote: "\""
comma: ws* ","
eq: ws* "="
nl: NEWLINE
key: KEY
KEY: /[A-Za-z0-9_.\-\/\[\]]+/

%import config.inferred_string
%import config.inferred_text
%import config.named_assignments
%import config.ws
%import config.NEWLINE
"""
