---
name: check
description: Run the format, lint and test gate for access-config-utils before committing or opening a PR. Use when asked to check, verify, validate or lint the code, or when finishing a change and wanting to confirm CI will pass.
---

# Pre-commit check

Run these from the repository root, in order, and stop at the first failure.

```shell
ruff format .
ruff check .
python -m pytest
```

This mirrors CI (`.github/workflows/ci.yml`): the `formatting` job runs `pre-commit`, which is
ruff-check plus ruff-format; the `build` job runs `python -m pytest` on Python 3.11, 3.12 and 3.13;
and the `docs` job builds the documentation strictly.

Equivalently, for the formatting half: `pre-commit run --all-files`.

## Notes

- `pyproject.toml` sets pytest `addopts` to always collect coverage (`term`, `html`, `xml`), so
  plain `python -m pytest` already produces a coverage report — do not add `--cov` flags.
- While iterating, `python -m pytest tests/test_<module>.py -q --no-cov` is much faster. Always
  finish with a full unflagged run, because the shared `config.lark` and the shared constraint
  interfaces mean a change in one module frequently breaks another module's tests.
- Coverage is currently 100% across `src/access/config`, and **each module also reaches 100%
  from its own test file alone**, in both halves — a stricter property the aggregate report cannot
  show. If you touched one, check it in isolation:

  ```shell
  m=config_store   # the module you changed
  python -m pytest tests/test_$m.py -o addopts="" --cov=access.config.$m \
         --cov-report=term-missing --cov-fail-under=100
  ```

  A line covered only from another file is a missing test in this one. The two integration
  suites, `test_parser.py` and `test_parallel_layouts.py`, drive other modules too, but that
  does not count towards those modules' own 100%.
- The `Examples:` blocks of the `parallel_*` modules run as doctests within `python -m pytest`;
  those in `entry_fragments.py` and `entry_render.py` do not. After touching either of those, run
  `python -m pytest --doctest-modules src/access/config/ --no-cov`.
- If you broke code on purpose to check that a test binds, clear the bytecode afterwards:
  `find . -name __pycache__ -type d -not -path "./.git/*" | xargs rm -rf` (`AGENTS.md` says why).
- `ruff format .` rewrites files. Run it before `ruff check .` so the check sees the formatted
  result, and re-read any file you had open before editing it further.
- Line length is 120 and the enabled rules include `C901` (complexity). If `C901` fires, prefer
  splitting the function; only use `# noqa: C901`, with a reason, when the complexity is
  irreducible. The package has no such site today.
- Docstrings and comments wrap at 92 columns (`W505`), narrower than the 120 for code. This is
  the check that fires most often on new tests; `ruff format` will not fix it for you.
- New files need the two-line Apache-2.0 header
  (`# Copyright 2025 ACCESS-NRI and contributors...` / `# SPDX-License-Identifier: Apache-2.0`).
  Neither ruff nor CI enforces this — check by eye.
- After changing a docstring, or anything under `documentation/`, build the documentation as CI
  does, with the commands under Commands in `AGENTS.md`: the build fails on a docstring griffe
  cannot read, a page missing from the navigation, and a broken link or anchor. It needs its own
  environment, `pip install -r documentation/requirements.txt`, and warns about the date of a page
  not yet committed.
- If the change alters something `AGENTS.md`, `README.md`, a guide in `skills/` or a page under
  `documentation/docs/pages/` states — a name, a count, a rule — update it in the same change.
  Nothing checks these, and they drift.

## Reporting

State the actual outcome: number of tests passed/failed and whether lint was clean. If anything
fails, show the relevant output rather than summarising it away.
