# Grammars

These modules say what the grammar of a format has to declare, and what it is compiled into. They
come first in the package: they import nothing from the other groups. `grammar_contract` names the
rules that every grammar must use, `grammar_values` maps the value-type rules to Python values, and
`grammar_compiled` builds and caches what a parsed configuration needs in order to edit itself.
[`UnsupportedEntryError`][access.config.grammar_values.UnsupportedEntryError], which is also defined
in `grammar_values`, is documented with the [configurations](../configurations.md).

::: access.config.grammar_contract

::: access.config.grammar_values
    options:
      filters: ["!^_", "!^UnsupportedEntryError$"]

::: access.config.grammar_compiled
