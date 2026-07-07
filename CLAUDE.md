# Ersilia Python Package — Developer Guide

This is the developer guide for a Python package built from the Ersilia Open Source Initiative's package template. The rules below apply to any code, docs, or release work in this repository.

## Working with the user

- **Ask, don't assume.** For any non-trivial decision — which approach to take, what to name something, whether to add a dependency, how to handle an ambiguous case — use the `AskUserQuestion` tool BEFORE editing. A couple of short questions up front beat a wrong-direction change.
- **Plans are mandatory.** Anything beyond a one-line fix or pure read-only investigation must go through plan mode. Be insistent: if invoked outside plan mode for non-trivial work, propose a plan in chat and stop until the user confirms. Do not skip planning to "save time".
- **Surface uncertainty.** When you have multiple reasonable options or are unsure about intent, name them and ask. Don't pick silently.

## Package layout

- **Rename the package folder.** As soon as real code lands, rename `src/my_package/` to the actual package name (snake_case, matching `[project].name` in `pyproject.toml`). Never leave `my_package` in place once the template is being used.
- **Remove `core.py` if untouched.** The templated `src/<package>/core.py` exists only to make the layout valid. If it has not been modified, delete it rather than letting placeholder code linger.
- **Favour submodules.** Group code into submodules (`io/`, `utils/`, `cli/`, ...) instead of a single flat file. Avoid a flat namespace.
- **Keep public APIs small.** Ersilia packages are thought of as simple APIs and CLIs. Avoid over-parametrising function signatures; a few well-chosen arguments are better than many optional knobs.

## Code style and quality

- **Run ruff before every commit.** `ruff check` and `ruff format` must both pass.
- **Docstrings: NumPy convention.** Write succinct NumPy-style docstrings for every public class, function, and method. For private helpers, only add a docstring when the intent isn't obvious from the name and signature.
- **Keep code, docstrings, and docs aligned.** Revisit docstrings, the README, and any `docs/` files periodically — fix drift as you see it, not in a separate cleanup pass.

## Logging

Use the Ersilia logging pattern: a module-level singleton built on stdlib `logging` + Rich's `RichHandler`, exposing the usual levels plus a `success()` method. See [`ersilia/utils/logging.py`](https://github.com/ersilia-os/ersilia) as the reference implementation; [`lazy-qsar`](https://github.com/ersilia-os/lazy-qsar) is an acceptable `loguru`-based alternative if richer formatting is needed.

```python
from <package>.utils.logging import logger

logger.info("Loaded %d molecules", len(df))
logger.success("Model saved → %s", model_dir)
logger.warning("Skipping invalid SMILES: %s", smi)
```

Import the singleton everywhere — do not call `logging.getLogger(...)` directly in feature code.

## CLI (optional)

- **Use Click.** If the package exposes a CLI, build it with [Click](https://click.palletsprojects.com/), organised as `src/<package>/cli/commands/` with one file per command and a small `create_cli.py` that registers them. This mirrors [`ersilia-os/ersilia`](https://github.com/ersilia-os/ersilia).
- **Document commands as a table.** In the README, list CLI commands in a compact two-column table (command → one-line description). Do not write extensive prose for each flag — that belongs in `--help`.

## Tests

- **Smoke-test the user-facing API/CLI.** High-level tests that exercise the documented entry points catch regressions where they actually matter. Skip exhaustive unit-test coverage of internals.
- **Keep `tests/` lean.** Transient `pytest` files are fine during development, but delete them once the code they exercised has stabilised. A small, curated test suite is better than a large pile of mostly-redundant tests.

## Dependencies and packaging

- **Pin exact versions.** Use `==X.Y.Z` for every entry in `pyproject.toml` (and any other requirements file the user adds). No floors (`>=`), no ranges.
- **Evaluate every new dependency.** Adding a library is a long-term cost. Prefer the standard library or an existing transitive dependency; only add a new package when the benefit is clear and the alternative would be substantial code.
- **Keep `pyproject.toml` in sync with the package.** When code starts (or stops) importing something, update `pyproject.toml` in the same commit. The project name, version, and dependency list must always reflect the current state of `src/`.

## Data with eosvc

- `data/` is gitignored on purpose. Do not commit datasets, model artefacts, or large binaries to git.
- Use [`eosvc`](https://github.com/ersilia-os) to back `data/` with an S3 bucket when the package needs reproducible inputs or outputs across machines.

## README guidelines

- **Be brutally brief.** The README should answer "what is this and how do I use it" and nothing else. Aim for a screen or two. Long-form content belongs in `docs/`.
- **Never use the package name as the H1 title.** For example, a package named `lazy-qsar` should not have `# lazy-qsar` at the top — write a short descriptive title instead (e.g. `# Lazy QSAR modelling for small molecules`).
- **CLI commands as a table.** If the package ships a CLI, document its commands in one small table; don't reproduce `--help` output in Markdown.
- **No AI-style filler.** Skip generic "Installation / Contributing / License / Acknowledgements" boilerplate unless the project actually has something to say about it.

## Versioning and releases

- **Semantic versioning only.** Versions are `vMAJOR.MINOR.PATCH` (e.g. `v0.1.0`). Do not use date-based, build-number, or other schemes.
- **PyPI releases via GitHub Actions.** When the user is ready to publish to PyPI, add a GitHub Action that triggers on release (not on every push). The git tag, GitHub release name, and `[project].version` in `pyproject.toml` must all match — release is blocked otherwise.

## Ersilia ecosystem

- Be aware of Ersilia's codebase in [GitHub](https://github.com/ersilia-os). Ersilia develops many tools.
- Ersilia maintains a set of skills in [`ersilia-skills`](https://github.com/ersilia-os/ersilia-skills). That repo is updated independently — check it for the current list before assuming a skill is or isn't available, and use a skill instead of writing the same logic from scratch when one fits.