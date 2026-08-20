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

    Derived types, array qualifiers and repeat counts are all read. An array written with
    one-dimensional qualifiers is gathered into a single list, with a null wherever no entry
    wrote a position.

    Note: three qualifiers are not gathered, and keep their text in the key instead, so that
    two of them cannot be read as one: a multidimensional index (``v(1,2)``), an index on a
    derived type (``a(1)%b``), and one naming no positions at all (a stride of zero). Nor
    are character substrings implemented, a list cannot yet hold a null element written as
    ``x = 1,,3``, values cannot be separated by whitespace alone, and a group has to be
    closed -- the next ``&`` does not end it.

    Note: ``inf`` and ``nan`` are read as the floats they are, but cannot be written. Python
    spells them as bare words that no format reads back as a number, so assigning one raises
    ``UnsupportedEntryError``. Reading and round-tripping a file that contains one is fine,
    since its text is left alone.
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

?assignment: key_value | key_list | key_null | key_derived | wrapped_value | wrapped_list

// A derived-type component assignment: "a%b = 1" is the block "a" holding "b = 1", so it
// reuses the key_block machinery and nests for free ("a%b%c" is a block in a block).
// "dtype_body" holds exactly one assignment. Reusing the namelist's own "block" rule here
// instead would be greedy -- it swallows the lines that follow, mis-nests them, and costs an
// order of magnitude in parse time.
key_derived: ws* key index? "%" dtype_body -> key_block
dtype_body: key_value | key_list | key_null | key_derived

key_value: ws* key index? eq ws* value
key_list: ws* key index? eq ws* value ((line_break|ws* separator) ws* value)+
key_null: ws* key index? eq ws*

// The value may also start on the line after the "=", which MOM6's test inputs do for their
// longer lists. Separate rules with a *required* line_end rather than an optional one in the
// rules above: optional, it doubles the derivations Earley explores for every assignment in
// the file, and costs five times the parse time on a large namelist.
wrapped_value: ws* key index? eq line_end ws* value -> key_value
wrapped_list: ws* key index? eq line_end ws* value ((line_break|ws* separator) ws* value)+ -> key_list
line_break: ws* separator line_end

// An array qualifier: "v(3)", "v(2:5)", "v(1:7:2)" or "v(1,2)". Deliberately one opaque
// token rather than a structured rule -- built out of "integer" nodes its numbers would be
// collected as part of the entry's values, since values are found by rule name.
index: "(" INDEX_SPEC ")"
INDEX_SPEC: /[-+0-9:,\\t ]+/

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
      | fortran_identifier

empty_line: line_end
line_end: (fortran_comment|ws*) NEWLINE
separator: ","

random_text: (/.+/|NEWLINE)*
ANYTHING: /.+/

// Fortran writes an infinity or a not-a-number as a bare word. Read as one, rather than as
// the string "inf", which is what the bare-word rule below would otherwise make of it.
float: SIGNED_FLOAT | IEEE
IEEE.2: /[-+]?(inf(inity)?|nan)(?![A-Za-z0-9_])/i

// Fortran writes a logical as an optional dot, T or F, and an optional rest of the word: all
// of ".true.", "true", ".t.", ".t" and a bare "T" mean the same thing, in either case. This
// is defined here rather than imported, because "config.lark"'s narrower version is shared
// with nuopc.runconfig, where a bare "T" is a plain identifier.
// The lookahead is what keeps a bare logical from claiming the start of a name: after a
// continuation comma, "f_icedir" would otherwise read as .false. followed by "_icedir".
logical: FTRUE | FFALSE
FTRUE: /\\.?t(rue)?\\.?(?![A-Za-z0-9_])/i
FFALSE: /\\.?f(alse)?\\.?(?![A-Za-z0-9_])/i

// An unquoted string value: CICE writes "runtype=continue", and the CESM data models write
// "import_data_fields = none". The lower priority leaves the spellings above to "logical",
// and the lookaheads stop it matching part of a word, or a key that is about to be assigned
// to. Defined here rather than imported because "config.lark"'s identifier is a bare CNAME,
// which would do both.
fortran_identifier: BAREWORD
BAREWORD.-1: /[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])(?![ \\t]*=)/

%import config.key
%import config.integer
%import config.SIGNED_FLOAT
%import config.double
%import config.complex
%import config.double_complex
%import config.fortran_string
%import config.fortran_comment
%import config.ws
%import config.NEWLINE
"""
