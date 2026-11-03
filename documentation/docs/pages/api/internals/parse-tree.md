# Parse trees

Reading a Lark parse tree and changing one. `tree_navigation` only reads a tree, and decides
where a new entry belongs; `tree_edits` only changes one, without leaving it in a shape the
grammar could not have produced; `tree_reader` turns a tree into one reference per key, and
stops there: building a `Config` from the references is left to `config_dict`.

::: access.config.tree_navigation

::: access.config.tree_edits

::: access.config.tree_reader
