# User guide

access-config-utils has two parts, which do not depend on each other: parsers for the
configuration files of the ACCESS models, and a search for the parallel layouts a model can run
in. Each has its own section of this guide.

## Configuration files

A parser reads the text of a configuration file into a dict, which is edited like any other and
written back out with everything the edits did not touch as it was.

- [Reading and writing](reading-and-writing.md): parsing a file, what its values and keys are
  read as, and writing it back.
- [Editing](editing.md): changing values and lists, adding and deleting keys, and what cannot be
  written.
- A page for each format, saying how it reads and what is particular to it:
  [Fortran namelists](formats/fortran-namelists.md), [MOM6 parameter files](formats/mom6.md),
  [NUOPC run configurations](formats/nuopc-runconfig.md),
  [NUOPC run sequences](formats/nuopc-runseq.md), [FMS field tables](formats/field-tables.md),
  and [YAML and JSON](formats/yaml-json.md).

## Parallel layouts

A layout search finds every way a number of cores can be shared among the components of a model,
as MPI ranks, OpenMP threads and process grids, that the model can run and a study asks for.

- [Components and layouts](layouts/components.md): describing a model as a tree of components,
  and what a layout of it holds.
- [Allocating cores](layouts/allocations.md): how many cores each component may receive, and how
  many threads it may run.
- [Constraints](layouts/constraints.md): ruling out the layouts a study does not want.
- [Searching for layouts](layouts/searching.md): running a search, using its results, and keeping
  it small.
