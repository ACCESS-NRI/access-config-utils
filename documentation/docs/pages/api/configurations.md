# Configurations

What a parser returns. [`Config`][access.config.config_dict.Config] is a `dict` and
[`ConfigList`][access.config.config_dict.ConfigList] a `list`, so reading them is reading the
builtins; every write is also made to the parsed file, so that `str()` writes it back with the
change. [Editing](../user-guide/editing.md) explains what can be changed and how.

::: access.config.config_dict

::: access.config.grammar_values.UnsupportedEntryError
