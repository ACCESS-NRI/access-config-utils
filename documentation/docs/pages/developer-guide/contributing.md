# Contributing

Contributions are welcome, as issues and pull requests on
[GitHub](https://github.com/ACCESS-NRI/access-config-utils). For a significant change, discuss it
in an issue before opening a pull request.

## Development installation

Work in a virtual environment, with the package installed in editable mode together with its
development and test dependencies:

```bash
git clone https://github.com/ACCESS-NRI/access-config-utils.git
cd access-config-utils
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[devel,test]"
pre-commit install
```

## Tests

Run the tests with:

```bash
python -m pytest
```

Coverage is measured on every run and reported in the terminal and in `htmlcov/`. It is 100% of
the lines of `src/access/config`, and a change keeps it there, error paths included. While
working on one module, a run of its tests alone is much faster:

```bash
python -m pytest tests/test_parallel_layouts.py -q --no-cov
```

Finish with a full run all the same: the grammars share `config.lark`, and the constraints share
their interfaces, so a change in one module often breaks another's tests.

### One test file per module

`tests/` mirrors `src/`: `tests/test_<module>.py` for each module, and each test file takes its
module to 100% coverage **on its own**. That is stricter than the total the ordinary report shows,
so check it for the module that changed:

```bash
m=config_store
python -m pytest tests/test_$m.py -o addopts="" --cov=access.config.$m \
       --cov-report=term-missing --cov-fail-under=100
```

Two files are exceptions, because something has to show that the pieces still fit together:
`test_parser.py` drives the whole parser stack through `parse` and the mapping interface, and
`test_parallel_layouts.py` runs real searches of real component trees. With the tests of the
format modules, they are what shows that real files still round-trip byte for byte.

### Fakes and mocks

A module's tests replace what it imports from *other modules of the package*, and nothing else:

- Lark's `Tree`, `Token`, `Lark` and `Reconstructor` are always the real classes. Building a
  `Tree("key_value", [...])` to hand to a function is building its input, not mocking it, and it
  is how the awkward branches are reached.
- Mocks use `unittest.mock`, not pytest's `monkeypatch`: `patch.object`, and `patch.multiple` for
  several collaborators at once.
- Patch the module that imports a name, not the one that defines it: every import in the package
  is `from access.config.X import name`, so each importer has its own reference to it.
- A fixture whose fakes depend on arguments known only later enters its patches on a
  `contextlib.ExitStack` it holds, since a `with patch(...)` inside the function it returns would
  unwind at once.
- `Mock(spec=SomeDataclass)` sees methods and properties, but not the fields declared without a
  default, which are only annotations: use a `MagicMock` and assign what the code reads.

`tests/conftest.py` holds what more than one file needs: the hand-written stateful fakes
`FakeStore` and `FakeContext`, builders of tree nodes, the compiled grammars of the shipped
parsers, and the tables of the text each parser writes for a new entry.

### Checking that a test can fail

Coverage says that a line ran, not that a test would notice it change. To check that a test
binds, break the code it covers and see that test fail. Then clear the bytecode:

```bash
find . -name __pycache__ -type d -not -path "./.git/*" | xargs rm -rf
```

Python checks a `.pyc` file against the size and modification time of its source, so an edit of
the same length undone within the same second leaves the broken code loaded.

### Doctests

The `Examples:` sections of the `parallel_*` modules are doctests, run with the tests by
`tests/test_parallel_doctests.py`. Those of the other modules are not run by it; after changing
one, run them all:

```bash
python -m pytest --doctest-modules src/access/config/ --no-cov
```

## Code style

The code is checked and formatted with [ruff](https://docs.astral.sh/ruff/), through the
pre-commit hooks, which `pre-commit install` runs on every commit. Lines of code have up to 120
characters, and docstrings and comments up to 92. The complexity check, `C901`, is enforced:
split a function rather than silencing it. Every source file starts with the two-line Apache-2.0
header of the others, every new module with `from __future__ import annotations`, and everything
is annotated, tests included (`-> None`).

## Docstrings

The docstrings follow the Google style, and the [API reference](../api/index.md) is generated from
them by [mkdocstrings](https://mkdocstrings.github.io/):

- The sections are `Args:`, `Returns:`, `Yields:`, `Raises:` and `Examples:`, in that order, each
  after a blank line, with their entries indented one level and their continuations two.
- Types are repeated in the docstring, spelled as the annotation spells them:
  `name (Type): Description.` under `Args:`, `Type: Description.` under `Returns:` and `Yields:`,
  and `ExceptionType: When.` under `Raises:`.
- A constructor's arguments, dataclass fields included, are documented on the class, never on
  `__init__`. An attribute that is not an argument gets a `#` comment where it is defined.
- How much to write depends on where it is. The public API, the `parallel_*` modules and every
  module docstring get all the sections. A function internal to the parser half keeps its summary,
  its explanation, `Raises:` and `Examples:`, and leaves out `Args:`, `Returns:` and `Yields:`. A
  private helper gets a summary and at most two more lines. The reasons for a design are written
  once, in the module docstring or in a comment beside the code they explain.
- The markup is reStructuredText's inline markup: double backticks for code, `*emphasis*` for a
  parameter's name, `**bold**` for emphasis, and no Sphinx roles.
- The text is rendered as Markdown, so a list is preceded by a blank line, a block of code is
  indented and introduced with a single colon, never `::`, and a list of terms is a bullet list,
  `- **term**: definition`, never a reStructuredText definition list.

The documentation is built in strict mode, so a docstring that mkdocstrings cannot read, such as
an argument documented under a name the function does not have, fails the build.

## Documentation

This site is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from
the ACCESS-NRI documentation template,
[documentation-infra](https://github.com/ACCESS-NRI/documentation-infra), and published on Read
the Docs, which `.readthedocs.yaml` configures. Everything else is in `documentation/`:

- `mkdocs.yml`, the configuration, including the navigation;
- `docs/pages/`, the pages, one Markdown file each, the API reference in `docs/pages/api/`;
- `abbreviations.md`, the glossary behind the tooltips of abbreviations;
- `requirements.txt`, the packages the build needs.

The rest of `documentation/` is kept as the template has it, so that it can be updated from there:
`hooks/`, `overrides/`, the stylesheets, scripts, fonts and logos under `docs/`, `requirements.txt`
and `.readthedocs.yaml`.

The pages of the API reference are each a run of `::: access.config.<module>` blocks, one page per
concept. A new module needs a block on the right page, and a row in the table of the
[API reference](../api/index.md); a new page needs an entry in the `nav` of `mkdocs.yml`, or the
strict build fails.

To preview the site while editing it, install the requirements in an environment of their own, and
serve it:

```bash
pip install -r documentation/requirements.txt
mkdocs serve -f documentation/mkdocs.yml
```

The continuous integration builds the site with warnings treated as errors. To check that before
pushing, run the same build:

```bash
READTHEDOCS_CANONICAL_URL=https://access-config-utils.readthedocs.io/latest/ \
    mkdocs build --strict -f documentation/mkdocs.yml
```

The variable stands in for the one Read the Docs sets, without which the build refuses to run. The
date at the foot of each page comes from its git history, so a page not yet committed makes the
strict build warn.

The examples on the pages are meant to be run: the output shown after each is what it printed.
Read the Docs builds a preview of the site for every pull request, and adds a link to it to the
pull request's description; the `Check links` workflow checks the links of the preview.

## Comparing with f90nml

`scripts/run_f90nml_tests.py` runs the test suite of [f90nml](https://github.com/marshallward/f90nml),
a widely used namelist parser, with its readers replaced by `FortranNMLParser`, and reports where
the two read a file differently. It runs the suite twice, as it is and with the readers replaced,
and blames only the difference on this parser, sorting each failure by its cause: a file that
cannot be read, values read differently, a file one refuses and the other does not, or a test of
f90nml's writer or API rather than of reading. It is a diagnostic rather than a test, and exits 0
unless it cannot run:

```bash
pip download f90nml --no-binary :all: --no-deps
tar xzf f90nml-*.tar.gz
python scripts/run_f90nml_tests.py --source f90nml-<version>
```

`-v` lists the tests and files in each group, and `--json PATH` also writes the full report to a
file.

## Before committing

Run these from the root of the repository, and stop at the first that fails:

```bash
ruff format .
ruff check .
python -m pytest
```

`pre-commit run --all-files` runs the formatting checks the continuous integration runs. After
changing a docstring or a page, build the documentation strictly as well, as above. A change to
something `AGENTS.md`, the README, the task guides in `skills/` or these pages state, a name, a
count or a rule, updates them in the same change: nothing checks them.

## Releases

A release is made by tagging a commit of the main branch with a tag starting with `v`, such as
`v0.6`, and pushing the tag. The `CD` workflow then publishes the package on
[PyPI](https://pypi.org/project/access-config-utils/) and on the
[accessnri conda channel](https://anaconda.org/accessnri/access-config-utils), and Read the Docs
builds the documentation of the new version.
