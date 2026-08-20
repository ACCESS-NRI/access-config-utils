# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for Fortran namelists.

Fortran namelists allow format-free I/O of variables by key-value assignements. Initially
they were an extension to the languages, but became part of the standard in Fortran 90.
"""

from collections.abc import Sequence

from access.config.parser import BLOCK_RULE, ConfigParser


class FortranNMLParser(ConfigParser):
    """Fortran Namelist parser.

    Note: Currently array qualifiers and substrings are not implemented in the grammar.
    """

    @property
    def case_sensitive_keys(self) -> bool:
        return False

    @property
    def block_rules(self) -> Sequence[str]:
        """Sequence[str]: Rules holding block contents: a namelist body and a derived type.

        A derived type is a block written one component per line rather than as a delimited
        body, so its contents rule is named separately. Keeping the two distinguishable is
        what lets a new key be added to a namelist but refused inside a derived type, where
        there is no body for it to go in.
        """
        return (BLOCK_RULE, "dtype_body")

    @property
    def grammar(self) -> str:
        return """
// "start" must be a non-transparent rule: the root node is the container that new entries
// are added to, and a transparent rule collapses away for a single-namelist file.
start: random_text? namelist (random_text? namelist)* random_text?

namelist.2: nml_start key line_end? block nml_end -> key_block
nml_start: ws* "&"
nml_end:  ws* ("/"|/&end/i) line_end

// The block contents rule must be *named* "block", not aliased to it, so that the name can
// be used as a Lark start symbol when parsing the text of a new entry.
block: (nml_line | empty_line)*

?nml_line: assignment (ws* "," assignment)* (ws* separator)? line_end?

?assignment: key_value | key_list | key_null | key_derived

// A derived-type component assignment: "a%b = 1" is the block "a" holding "b = 1", so it
// reuses the key_block machinery and nests for free ("a%b%c" is a block in a block).
// "dtype_body" holds exactly one assignment. Reusing the namelist's own "block" rule here
// instead would be greedy -- it swallows the lines that follow, mis-nests them, and costs an
// order of magnitude in parse time.
key_derived: ws* key "%" dtype_body -> key_block
dtype_body: key_value | key_list | key_null | key_derived

key_value: ws* key eq ws* value
key_list: ws* key eq ws* value ((line_break|ws* separator) ws* value)+
key_null: ws* key eq ws*
line_break: ws* separator line_end

// The whitespace before "=" belongs to a rule of its own. Left as a bare "ws*" beside the
// "ws*" that follows "=", it is ambiguous which slot a single space falls in, and the
// reconstructor moves it: "A= 1" is written back as "A =1".
eq: ws* "="

// "n*v" is v repeated n times, and a bare "n*" is n null values. The count and the value it
// repeats are separate children, so the value is an ordinary value-type node that the usual
// handler reads -- the repetition is the only thing this rule adds.
?value: repeat | scalar
repeat: REPEAT_COUNT scalar?
REPEAT_COUNT: /[0-9]+\\*/

?scalar: logical
      | integer
      | float
      | double
      | complex
      | double_complex
      | fortran_string

empty_line: line_end
line_end: (fortran_comment|ws*) NEWLINE
separator: ","

random_text: (/.+/|NEWLINE)*
ANYTHING: /.+/

// Fortran writes a logical as an optional dot, T or F, and an optional rest of the word: all
// of ".true.", "true", ".t.", ".t" and a bare "T" mean the same thing, in either case. This
// is defined here rather than imported, because "config.lark"'s narrower version is shared
// with nuopc.runconfig, where a bare "T" is a plain identifier.
// The lookahead is what keeps a bare logical from claiming the start of a name: after a
// continuation comma, "f_icedir" would otherwise read as .false. followed by "_icedir".
logical: FTRUE | FFALSE
FTRUE: /\\.?t(rue)?\\.?(?![A-Za-z0-9_])/i
FFALSE: /\\.?f(alse)?\\.?(?![A-Za-z0-9_])/i

%import config.key
%import config.integer
%import config.float
%import config.double
%import config.complex
%import config.double_complex
%import config.fortran_string
%import config.fortran_comment
%import config.ws
%import config.NEWLINE
"""
