# Getting started

In this tutorial we install access-config-utils, edit a Fortran namelist, and run a first search for
parallelisation layouts. You can follow it on a laptop.

## Installation

access-config-utils needs Python 3.11 or later. You can install it with pip, with conda or from
source:

=== "pip"

    ```bash
    pip install access-config-utils
    ```

=== "conda"

    ```bash
    conda install -c accessnri -c conda-forge access-config-utils
    ```

=== "From source"

    ```bash
    git clone https://github.com/ACCESS-NRI/access-config-utils.git
    cd access-config-utils
    pip install .
    ```

Then check that the package can be imported:

```bash
python -c "import access.config; print(access.config.__version__)"
```

This prints the version you have just installed.

## Editing a configuration file

We start with a configuration file. The namelist below is the `input.nml` of an ACCESS-OM3
configuration, which sets up MOM6 and the FMS infrastructure it runs on:

```fortran title="input.nml"
&MOM_input_nml
    output_directory = './'
    restart_input_dir = './'
    restart_output_dir = './'
    parameter_filename = 'MOM_input', 'MOM_override'
/

&fms_nml
    clock_grain = 'ROUTINE'
    clock_flags = 'NONE'
    domains_stack_size = 5000000
    stack_size = 0
/

&diag_manager_nml
    flush_nc_files = .true.
/

&mpp_io_nml
    cf_compliance = .true.
/
```

Save it as `input.nml` in the current directory, or adapt the path in the examples below to
where your file is. We first read the file with the parser for Fortran namelists. The result is a
`Config`, which is a `dict` of the namelist groups, and each group is a `dict` of its variables:

```python
from pathlib import Path

from access.config import FortranNMLParser

config = FortranNMLParser().parse(Path("input.nml").read_text())
print(config["fms_nml"]["domains_stack_size"])
print(config["mom_input_nml"]["parameter_filename"])
```

The values are Python values, and a list in the file is a Python list. This prints:

```text
5000000
['MOM_input', 'MOM_override']
```

Fortran does not tell upper case from lower case in names, so neither does the configuration:
`config["FMS_NML"]` is the same group.

You edit the configuration as you would edit a dict. Here we change two values and add a new
variable. We then write the configuration back to the file with `str`, and print the file:

```python
config["fms_nml"]["domains_stack_size"] = 8000000
config["fms_nml"]["clock_grain"] = "LOOP"
config["diag_manager_nml"]["max_files"] = 100
Path("input.nml").write_text(str(config))
print(Path("input.nml").read_text())
```

The file comes back with our edits, and everything else as it was:

```text
&MOM_input_nml
    output_directory = './'
    restart_input_dir = './'
    restart_output_dir = './'
    parameter_filename = 'MOM_input', 'MOM_override'
/

&fms_nml
    clock_grain = 'LOOP'
    clock_flags = 'NONE'
    domains_stack_size = 8000000
    stack_size = 0
/

&diag_manager_nml
    flush_nc_files = .true.
    max_files = 100
/

&mpp_io_nml
    cf_compliance = .true.
/
```

Note that the new string kept the quotes of the one it replaced. The new variable, `max_files`,
was written after the last variable of its group, with the same indentation.
[Configuration files](user-guide/reading-and-writing.md) goes further: it covers the other
formats, lists, blocks, and what cannot be written.

## Searching for layouts

The second part of the package searches for parallelisation layouts. A parallelisation layout, or
layout for short, says how the cores of a job are shared among the components of a model. The
model is described as a tree of components. Ours is a coupled model with two components, an
atmosphere and an ocean, each with a grid of its own. We count its layouts on 8 cores:

```python
from access.config.parallel_component import ParallelComponent
from access.config.parallel_domain import Domain
from access.config.parallel_layouts import iter_layouts

model = ParallelComponent(
    "model",
    subcomponents=(
        ParallelComponent("atm", domain=Domain((192, 144))),
        ParallelComponent("ocn", domain=Domain((360, 300))),
    ),
)
print(sum(1 for _ in iter_layouts(model, 8)))
```

This prints:

```text
112
```

`iter_layouts` finds 112 layouts on 8 cores. They are all the ways of dividing the cores between
the two components, idle cores included, combined with all the grids of MPI ranks that each
component can arrange its share in. That is a lot, so we narrow the search down with an
allocation strategy. Here it gives the atmosphere 4 cores, and the ocean at least 4:

```python
from access.config.parallel_allocation_strategies import FixedAllocation, FreeAllocation, RootAllocation

allocations = RootAllocation(
    subcomponents={"atm": FixedAllocation(4), "ocn": FreeAllocation(min_cores=4)}
)
for layout in iter_layouts(model, 8, allocations=allocations):
    print([(sub.name, sub.n_cores, sub.decomposition.grid.shape) for sub in layout.sub_layouts])
```

Now only nine layouts are left:

```text
[('atm', 4, (1, 4)), ('ocn', 4, (1, 4))]
[('atm', 4, (1, 4)), ('ocn', 4, (2, 2))]
[('atm', 4, (1, 4)), ('ocn', 4, (4, 1))]
[('atm', 4, (2, 2)), ('ocn', 4, (1, 4))]
[('atm', 4, (2, 2)), ('ocn', 4, (2, 2))]
[('atm', 4, (2, 2)), ('ocn', 4, (4, 1))]
[('atm', 4, (4, 1)), ('ocn', 4, (1, 4))]
[('atm', 4, (4, 1)), ('ocn', 4, (2, 2))]
[('atm', 4, (4, 1)), ('ocn', 4, (4, 1))]
```

In each of them, a component runs its 4 cores as 4 MPI ranks, with one thread each, in a grid of
1×4, 2×2 or 4×1 ranks. [Parallelisation layouts](user-guide/layouts/components.md) explains how
to describe a model and how its cores can be allocated. It also shows how to rule out the layouts
that a study does not want, such as grids whose subdomains are far from square.

## Where to go next

- The [User guide](user-guide/index.md) covers each format and the layout search in full.
- The [Developer guide](developer-guide/index.md) explains how the package works and how to extend
  it.
- The [API reference](api/index.md) documents every class and function.
