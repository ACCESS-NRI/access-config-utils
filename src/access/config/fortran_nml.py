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
    wrote a position. A group may open with ``&`` or the older ``$`` and close with ``/``,
    ``&end`` or ``$end``. Values may be parted by commas or by blanks, a list may skip a
    position by writing nothing between two separators (``x = 1, , 3``), and a real may
    leave out the exponent letter before a signed exponent (``1+0`` is 1.0).

    A group written more than once is read as one block per occurrence rather than merged;
    see ``repeated_blocks``. An unqualified assignment over a longer entry overlays it from
    the first position, as Fortran does, so ``x = 1, 2`` then ``x = 3`` is ``[3, 2]``.

    A group whose body this cannot derive raises, rather than being read as the free text
    that may surround a group: a file that round-trips with a whole group missing from the
    configuration is worse than one that is refused.

    Note: three qualifiers are not gathered, and keep their text in the key instead, so that
    two of them cannot be read as one: a multidimensional index (``v(1,2)``), an index on a
    derived type (``a(1)%b``), and one naming no positions at all (a stride of zero). Nor
    are character substrings implemented, nor a character value continued across lines, and
    a group has to be closed -- neither the next ``&`` nor a bare ``$`` ends it.

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
    def repeated_blocks(self) -> str:
        """str: ``"separate"`` -- a group written twice is two records, not one merge.

        Fortran's ``READ`` scans forward and stops at the first group of the name, so a
        program reading a group once sees the first occurrence and one reading it in a loop
        sees each in turn. Merging them would report a set of values no single ``READ``
        returns, so each occurrence is kept and handed out as its own block.
        """
        return "separate"

    @property
    def grammar(self) -> str:
        return """
// "start" must be a non-transparent rule: the root node is the container that new entries
// are added to, and a transparent rule collapses away for a single-namelist file.
start: random_text? namelist (random_text? namelist)* random_text?

// A group opens with "&" or the older "$", and closes with "/", "&end" or "$end" -- all four
// accepted by the compiler. A bare "$" as terminator is not: gfortran rejects it, so it is
// left out even though some readers take it. Each delimiter is a rule of its own rather than
// an alternation of literals, because Lark filters an anonymous literal out of the tree and
// the reconstructor would then have nothing to tell "&" from "$" -- rewriting one as the
// other. This is the same device as "separator" and "eq"; see grammar_contract.
namelist.2: nml_start key line_end? block nml_end -> key_block
nml_start: ws* (amp_open | dollar_open)
amp_open: "&"
dollar_open: "$"
nml_end:  ws* (slash_close | amp_close | dollar_close) line_end
slash_close: "/"
amp_close: /&end/i
dollar_close: /\\$end/i

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

// Two values may be parted by a comma, by a line break, or by whitespace alone: Fortran
// treats a blank as a value separator, and WW3's inputs use it throughout ("v = 1 2 3", and
// "MODEL(4)%FORCING = 'ocn' 'ocn' 'atm'"). A bare "ws" here and not a newline: a list that ran
// on across a line break with nothing between the values would swallow the line below it.
key_value: ws* key index? eq ws* value
key_list: ws* key index? eq ws* value ((line_break|ws* separator elided*|ws) ws* value)+
key_null: ws* key index? eq ws*

// The value may also start on the line after the "=", which MOM6's test inputs do for their
// longer lists. Separate rules with a *required* line_end rather than an optional one in the
// rules above: optional, it doubles the derivations Earley explores for every assignment in
// the file, and costs five times the parse time on a large namelist.
wrapped_value: ws* key index? eq line_end ws* value -> key_value
wrapped_list: ws* key index? eq line_end ws* value ((line_break|ws* separator elided*|ws) ws* value)+ -> key_list
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
      | bare_exponent
      | double
      | complex
      | double_complex
      | fortran_string
      | fortran_identifier

// A list may skip a position by writing nothing between two separators: "v = 1, , 3" leaves
// the second element unset, which gfortran reports as the variable's previous value and this
// reads as a null, exactly like a position no indexed entry wrote. One "elided" node per
// position skipped, so the count survives into the tree.
elided: ws* separator

empty_line: line_end
line_end: (fortran_comment|ws*) NEWLINE
separator: ","

// Free text between namelist groups. A line starting with a group delimiter is never part of
// it: without that, a group whose body the grammar cannot derive is quietly read as text
// instead, and the file round-trips with the whole group missing from the configuration. Both
// delimiters have to be excluded, or a "$" group is swallowed exactly as an "&" one once was.
//
// "$" is excluded only when a name follows it, which is what makes it a group. WW3's inputs
// use a bare "$" as their own comment marker -- 884 lines of it in the namelists on hand --
// and those are ordinary free text.
random_text: (ANYTHING|NEWLINE)*
ANYTHING: /(?![ \\t]*&)(?![ \\t]*\\$[A-Za-z])[^\\n]+/

// Fortran writes an infinity or a not-a-number as a bare word. Read as one, rather than as
// the string "inf", which is what the bare-word rule below would otherwise make of it.
float: SIGNED_FLOAT | IEEE
IEEE.2: /[-+]?(inf(inity)?|nan)(?![A-Za-z0-9_])/i

// The exponent letter may be left out when the exponent carries a sign, so "1+0" is 1.0 and
// "5-1" is 0.5 -- confirmed against gfortran. A value-type rule of its own rather than a
// wider "float": the registry is keyed by rule name, so folding it in would teach every
// format's "float" a notation only this one can write.
bare_exponent: BARE_EXPONENT
BARE_EXPONENT.2: /[-+]?[0-9]+(\\.[0-9]*)?[-+][0-9]+(?![A-Za-z0-9_.])/

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
