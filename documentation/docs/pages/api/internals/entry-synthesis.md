# Entry synthesis

Writing the text of an entry the file does not have yet. A new entry is never built as tree
nodes: these modules produce its text, which Lark then parses. `entry_fragments` holds the
template type and compiles the text a parser declares, `entry_style` reads from the file how
its entries are written, and `entry_render` chooses a template and writes the candidates.
[How parsing and editing work](../../developer-guide/parsers.md#adding-a-key) follows a key
through them.

::: access.config.entry_fragments

::: access.config.entry_style

::: access.config.entry_render
