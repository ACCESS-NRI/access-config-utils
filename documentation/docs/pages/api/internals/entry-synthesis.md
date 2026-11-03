# Entry synthesis

These modules write the text of an entry that the file does not have. A new entry is never built as
tree nodes: the modules produce its text, which Lark then parses. `entry_fragments` holds the
template type and compiles the text that a parser declares. `entry_style` reads from the file how
its entries are written. `entry_render` chooses a template and writes the candidates. [How parsing
and editing work](../../developer-guide/parsers.md#adding-a-key) follows a key through them.

::: access.config.entry_fragments

::: access.config.entry_style

::: access.config.entry_render
