# API reference

This reference is generated from the docstrings in the source code. Its pages follow the way the
package is organised: the configuration files, then the parallel layouts, and then the internals
of the parsers, for developers.

| Page | Modules | What they cover |
|---|---|---|
| [Configurations](configurations.md) | `config_dict`, `grammar_values` | The dict, list and record a file is read as, and the error raised for a value a format cannot write |
| [Parsers](parsers.md) | `parser`, `fortran_nml`, `mom6_input`, `nuopc_config`, `nuopc_runseq`, `field_table` | The base class of the parsers built on a grammar, and the parser of each format |
| [YAML and JSON](yaml-json.md) | `yaml_config`, `json_config` | The parsers that wrap ruamel.yaml and `json` |
| [Components and layouts](parallel-components.md) | `parallel_component`, `parallel_domain`, `parallel_mpi_grid` | Component trees, the layouts found for them, domains and grids of ranks |
| [Allocation strategies](allocation-strategies.md) | `parallel_allocation_strategies` | How the cores of a parent are divided among its sub-components |
| [Constraints](constraints.md) | `parallel_constraints` | The ready-made rules for filtering layouts |
| [Layout search](layout-search.md) | `parallel_layouts` | `iter_layouts`, which runs a search |
| [Grammars](internals/grammar.md) | `grammar_contract`, `grammar_values`, `grammar_compiled` | What a grammar declares, and what it is compiled into |
| [Parse trees](internals/parse-tree.md) | `tree_navigation`, `tree_edits`, `tree_reader` | Reading a parse tree, changing one, and turning one into a reference per key |
| [Entry synthesis](internals/entry-synthesis.md) | `entry_fragments`, `entry_style`, `entry_render` | The text of an entry the file does not have yet |
| [Editing](internals/editing.md) | `list_layout`, `config_insert`, `config_resize`, `config_store` | Adding an entry, resizing a list, and the tree behind a configuration |

All module names are relative to `access.config`.

## What to import from where

The parsers, and the types a parsed configuration is made of, are imported from the package
itself:

```python
from access.config import (
    Config,
    ConfigList,
    ConfigParser,
    ConfigRecord,
    FieldTableParser,
    FortranNMLParser,
    JSONParser,
    MOM6InputParser,
    NUOPCParser,
    NUOPCRunSeqParser,
    UnsupportedEntryError,
    YAMLParser,
)
```

Everything else is imported from the module that defines it, such as `ConfigBlockList`, the
blocks of a namelist group written more than once, and the whole of the layout search:

```python
from access.config.config_dict import ConfigBlockList
from access.config.parallel_allocation_strategies import FixedAllocation, FreeAllocation, RootAllocation
from access.config.parallel_component import CoreSharing, ParallelComponent
from access.config.parallel_constraints import SubdomainAspectRatioConstraint
from access.config.parallel_domain import Domain
from access.config.parallel_layouts import iter_layouts
```

The internals are documented for developers. They are not a stable interface, and may change from
one version to the next.
