# Configurations

These are the types that a parser returns. [`Config`][access.config.config_dict.Config] is a `dict`
and [`ConfigList`][access.config.config_dict.ConfigList] is a `list`, so you read them as you read
the builtins. Every write is also made to the parsed file, so that `str()` writes the file back with
the change. [Editing](../user-guide/editing.md) explains what you can change, and how.

::: access.config.config_dict

::: access.config.grammar_values.UnsupportedEntryError
