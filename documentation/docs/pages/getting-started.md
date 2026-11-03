# Getting started

This page installs access-config-utils, edits a Fortran namelist, and runs a first layout
search. Nothing here needs more than a laptop.

## Installation

access-config-utils needs Python 3.11 or later.

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

Check that the package can be imported:

```bash
python -c "import access.config; print(access.config.__version__)"
```

## Editing a configuration file

The namelist below is the `input.nml` of an ACCESS-OM3 configuration, which sets up MOM6 and
the FMS infrastructure it runs on:

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

A parser reads the text of the file into a `Config`, which is a `dict`: here, of the namelist
groups, each a `dict` of its variables. The values are Python values, and a list of them is a
list:

```python
from pathlib import Path

from access.config import FortranNMLParser

config = FortranNMLParser().parse(Path("input.nml").read_text())
print(config["fms_nml"]["domains_stack_size"])
print(config["mom_input_nml"]["parameter_filename"])
```

```text
5000000
['MOM_input', 'MOM_override']
```

Fortran does not tell upper case from lower case in names, so neither does the configuration:
`config["FMS_NML"]` is the same group.

The configuration is edited as a dict is. Writing it out with `str` gives the file back with the
edits made, and everything else as it was:

```python
config["fms_nml"]["domains_stack_size"] = 8000000
config["fms_nml"]["clock_grain"] = "LOOP"
config["diag_manager_nml"]["max_files"] = 100
Path("input.nml").write_text(str(config))
print(Path("input.nml").read_text())
```

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

The new string kept the quotes of the one it replaced, and the new variable, `max_files`, was
written after the last variable of its group and indented like it.
[Configuration files](user-guide/reading-and-writing.md) goes further: the other formats, lists,
blocks, and what cannot be written.

## Searching for layouts

A model is described as a tree of components. Here a coupled model has two, an atmosphere and an
ocean, each with a grid of its own:

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

```text
112
```

On 8 cores, `iter_layouts` finds 112 layouts: every way of dividing the cores between the two
components, idle cores included, combined with every grid of MPI ranks each component's share can
be arranged in. An allocation strategy narrows the search down. Here it gives the atmosphere 4
cores, and the ocean at least 4:

```python
from access.config.parallel_allocation_strategies import FixedAllocation, FreeAllocation, RootAllocation

allocations = RootAllocation(
    subcomponents={"atm": FixedAllocation(4), "ocn": FreeAllocation(min_cores=4)}
)
for layout in iter_layouts(model, 8, allocations=allocations):
    print([(sub.name, sub.n_cores, sub.decomposition.grid.shape) for sub in layout.sub_layouts])
```

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

Each component runs its 4 cores as 4 MPI ranks, one thread each, in a grid of 1×4, 2×2 or 4×1
ranks. [Parallel layouts](user-guide/layouts/components.md) explains how to describe a model, how
its cores can be allocated, and how to rule out the layouts a study does not want, such as grids
whose subdomains are far from square.

## Where to go next

- The [User guide](user-guide/index.md) covers each format and the layout search in full.
- The [Developer guide](developer-guide/index.md) explains how the package works and how to extend
  it.
- The [API reference](api/index.md) documents every class and function.
