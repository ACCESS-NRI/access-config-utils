# Editing

These modules hold the operations behind a `Config`. They are the last group before the data that
the caller sees. `list_layout` writes the text of a list whose length changes, and `config_insert`
adds an entry. `config_resize` writes a list of another length into its entry. `config_store` holds
the tree behind one configuration, and keeps it in step with the dict.

::: access.config.list_layout

::: access.config.config_insert

::: access.config.config_resize

::: access.config.config_store
