# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The base class a configuration parser is built by extending.

One class, and the three properties a format has to declare: its grammar, whether its keys
are case-sensitive, and -- rarely -- how a brand-new entry should be laid out. Everything
that happens to the parsed result afterwards lives elsewhere; ``parse`` compiles the
grammar, hands the text to Lark, and wraps the tree in a ``Config``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from access.config.config_dict import Config
from access.config.config_store import ConfigStore
from access.config.grammar_compiled import ParseContext, compile_grammar
from access.config.grammar_contract import BLOCK_RULE, START_RULE
from access.config.grammar_values import VALUE_RULE_PRIORITY
from access.config.tree_navigation import AddParent


class ConfigParser(ABC):
    """Lark-based configuration parser base class.

    A parser built by extending this class is for a file that can be mapped onto a Python
    dict: a series of key-value assignments, where a value is a scalar (``a=1``), a list
    (``a=1,2,3``) or a block of further assignments (``blk: b=1, c=2``).

    Subclassing means supplying a Lark grammar and saying whether keys are case-sensitive.
    The grammar has to be written against the naming contract stated in full in
    ``grammar_contract`` -- the rule names there are what lets one set of machinery read,
    edit and extend every format.

    A new entry is written as a copy of the nearest entry of its kind. For the first entry
    of a kind, which has none to copy, a parser declares the text in ``entry_templates``;
    ``value_rule_priority`` says how a new value is spelled where several rules would do.

    This class is abstract to prevent instantiation, as it requires a grammar in order to
    work at all.
    """

    @property
    @abstractmethod
    def grammar(self) -> str:
        """The grammar is a property of the parser.

        Returns:
            str: The parser grammar.
        """

    @property
    @abstractmethod
    def case_sensitive_keys(self) -> bool:
        """Property indicating if the configuration uses case-sensitive keys or not.

        Returns:
            bool: Are the keys case-sensitive?
        """

    @property
    def key_case(self) -> str:
        """The case a key is stored in, for a format whose keys are not case-sensitive.

        ``"upper"``, the default, is what a Fortran namelist uses. ``"lower"`` is for a
        format whose own reader lowercases names, as the FMS field manager does, so that
        the keys of the dict are the names the model sees. Ignored when
        ``case_sensitive_keys`` is true.

        Returns:
            str: ``"upper"`` or ``"lower"``.
        """
        return "upper"

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        """The text of a new entry of each kind the format can write.

        Keyed by container rule name and entry category, e.g. ``("block", "key_value")``.
        The value is ordinary text with ``{key}``, ``{keys:SEP}`` and ``{value}`` markers.
        ``{key}`` is the key of a plain entry. ``{keys:SEP}`` is every key of a chain,
        parted by SEP, as the Fortran ``a%b = 1`` is written ``{keys:%} = {value}``, so one
        template serves a key path of any depth. A list template has two ``{value}``
        markers, because the text between them is the element separator, which longer lists
        repeat. A block is written empty, its contents being added afterwards.

        A new entry is first written as a copy of the nearest entry of its kind, which keeps
        its spacing, its punctuation and the notation of its values: a minimal diff matters
        more than any declared convention. A template is for the first entry of a kind,
        which has none to copy. The whitespace the line of its first key starts with is its
        own indentation, which that of the entry above replaces where the file shows one.

        Returns:
            Mapping[tuple[str, str], str]: The templates; empty by default.
        """
        return {}

    @property
    def block_rules(self) -> Sequence[str]:
        """Names of the rules that hold the contents of a block.

        Nearly always just ``"block"``. A format with more than one kind of block declares
        them all, most-addable first: the first name is the one new entries are written
        into, and a block held by any of the others has no body to write one in. A block
        written one component per line, rather than as a delimited body, is one: a key
        added to it is written by the block holding it, as a line carrying the whole key
        path.

        Returns:
            Sequence[str]: The rule names.
        """
        return (BLOCK_RULE,)

    @property
    def repeated_blocks(self) -> str:
        """What it means for a file to write the same block twice.

        ``"merge"``, the default, reads the two as one block holding the entries of both. A
        key assigned in both keeps the last value, which is what a format uses repetition to
        express when it uses it at all.

        ``"separate"`` reads them as a list of blocks, one per occurrence. A format whose
        reader scans forward and stops at the first block of the name needs this: a program
        reading the block once sees the first occurrence, one reading it twice sees each in
        turn, and neither of those is the merge. Keeping the occurrences is the only reading
        all of them can be recovered from.

        This applies only to the first of ``block_rules``. A block held by any other rule
        merges whatever the policy, because repeating one of those means something else: a
        block spelled one component per line makes ``a%b`` and ``a%c`` two writings of the
        single key ``a``, and merging them is what reads them correctly.

        Returns:
            str: ``"merge"`` or ``"separate"``.
        """
        return "merge"

    @property
    def value_rule_priority(self) -> Sequence[str]:
        """Order in which value-type rules are tried when writing a value for a new key.

        Several value types accept the same Python type, so the order decides how a value is
        written: whether a ``bool`` becomes ``.true.`` or ``True``, or a ``float`` uses a
        double-precision exponent. Only rules the grammar admits are considered.

        Returns:
            Sequence[str]: The rule names, most preferred first.
        """
        return VALUE_RULE_PRIORITY

    def parse(self, stream: str) -> Config:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            Config: instance of the Config class storing the parsed data.
        """
        ctx = ParseContext(
            compile_grammar(self.grammar),
            self.case_sensitive_keys,
            self.entry_templates,
            tuple(self.value_rule_priority),
            tuple(self.block_rules),
            self.repeated_blocks,
            self.key_case,
        )

        # Parse text. Here we add a newline character to simplify the writting of the
        # grammars, as otherwise one would have to explicitly take into account the case
        # where the text does no end with a newline. The start symbol has to be named
        # explicitly, because the parser has more than one.
        tree = ctx.lark.parse(stream + "\n", start=START_RULE)

        AddParent().visit(tree)
        return Config(ConfigStore(tree, ctx))
