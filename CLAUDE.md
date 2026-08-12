# olinda — developer guide

Olinda distils a slow chemistry model into one fast, self-describing ONNX file. A teacher model is
scored once over a 1.36M-compound reference library; a compact gradient-boosting student learns to
reproduce it; everything fuses into a single `model.onnx` that runs on onnxruntime alone.

This guide covers what is non-obvious about working here. The Ersilia house rules apply
([`eos-python-package`](https://github.com/ersilia-os/eos-python-package)); what follows is where this
repository differs, and why.

## The three models, and their letters

The same names appear in the code, the run directory, the diagrams and the README. Learn them first —
almost every module refers to them.

| | |
|---|---|
| **S** | the **surrogate**, distilled from the teacher's soft labels |
| **H** | the **hard-label model**, trained on real measurements |
| **H_S** | **H** carried onto **S**'s scale by an isotonic map |
| **T** | predicted **1-NN Tanimoto** to the labelled set |
| **a** | the blend weight, `a = a_max · ramp(T)` |

The shipped prediction is `(1 − a) · S + a · H_S`. `S`, `H_S` and `a` stay **internal tensors** in the
fused graph: a column declares exactly one output, its prediction. That is deliberate — the wiring has
changed once already, and callers must not be able to depend on it.

## Layout

Package code lives in `olinda/`, **not** `src/olinda/`. Don't "fix" this.

| | |
|---|---|
| `olinda/cli/` | one module per command in `commands/`, assembled by `create_cli.py` |
| `olinda/train/` | the boosting engines, tuning, per-column training |
| `olinda/hard/` | the hard-label head: `labels`, `gate`, `train`, and `layout` for the on-disk names |
| `olinda/export/` | the fuse: stage graphs, metadata, parity checking, `build_bundle` |
| `olinda/report/` | `olinda validate` and its figures |
| `olinda/data/` | the reference library and the run's split |
| `olinda/console.py` | user-facing output — see below |
| `olinda/utils/logging.py` | the levelled logger — see below |

## The lazy-import contract

This is the constraint most likely to be broken by an innocent-looking edit.

numpy, pandas, xgboost, rdkit and matplotlib cost about **12 seconds** to import. `olinda --help` must
not pay that, and a base install does not even have most of them. So:

> **A module under `olinda/cli/` may import, at module scope, only `__future__`, the standard library,
> `click`/`rich_click`, and other `olinda.cli.*` modules.** Nothing from `olinda.*` outside
> `olinda/cli/` — not even `olinda.console`. Every such import goes inside the command body.

No exceptions, because a rule with exceptions is one nobody applies. `create_cli.py` imports all ten
command modules to register them, so a single slip lands on every invocation.

Two tests enforce it, and they fail differently on purpose: `tests/test_cli_surface.py` parses the AST
and names the offending file; `tests/test_inference_install.py` checks `sys.modules` from a clean
subprocess, which is the only way to catch an import that a dev machine happens to satisfy.

Commands are registered with `add_command` rather than `@cli.command`, so no command module imports the
group. That keeps the import graph acyclic even though `fit` imports five of its siblings — it drives
the pipeline by calling their callbacks directly.

## Output: two channels, and they are not interchangeable

- **`olinda/console.py`** — everything a user reads. Rules, panels, live tables, progress bars, `echo`.
  Not levelled and not silenceable; it is presentation.
- **`olinda/utils/logging.py`** — the loguru singleton (`from olinda.utils.logging import logger`),
  with the usual levels plus `success()`.

**The console handler is attached at `WARNING`.** `logger.debug`, `logger.info` and `logger.success`
therefore print nothing. That is intentional — status belongs to `console` — but it means routing
user-visible output through `logger.info` silently deletes it. Both write through the same Rich
`Console`, so they interleave correctly with live regions.

## Install tiers

`pip install olinda` runs models: the CLI, `olinda predict`, `OlindaArtifact`. `[report]` adds
`olinda validate` and its figures. `[train]` adds the boosting stack and includes `[report]`.

A command needing an absent extra must **refuse and name it** — `require_train_extra` /
`require_report_extra` — never die on whichever heavy import came first. CI installs each tier
separately and exercises it, so the boundaries are real.

## Conventions

- **ruff** is the canonical Ersilia config, with two documented deviations in `ruff.toml`
  (`target-version = "py311"`, and `preview` scoped under `[lint]`). `ruff check` and `ruff format`
  must both pass; `.pre-commit-config.yaml` runs them at the same pinned version as CI.
- **Docstrings** are NumPy convention. `D101`/`D102` are enforced. Say *why* rather than restating the
  signature — `olinda/data/reference.py` is the exemplar.
- **Dependencies** are pinned exactly. `train`'s self-reference `olinda[report]==1.1.0` must be bumped
  in lockstep with `[project].version`.
- **`git blame`** — run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once, or the
  whole-repo reformat owns every line.
- **Tests** smoke-test the documented entry points. They run real fits on a synthetic 240-row library,
  so the whole suite is ~25 seconds; keep it that way.

## Data

`data/` is gitignored and backed by [`eosvc`](https://github.com/ersilia-os/eosvc); `access.json` is
local. The reference library (`~/.olinda/erl0_morgan.h5`, 2.8 GB) is fetched by `olinda setup` from a
public bucket. Never commit datasets, model artefacts or `.onnx` files.

## Things that look wrong and are not

- `example/run_stepwise.sh` has no `export` branch. Correct: `learn-hard` builds the artifact and
  `clean` moves it out; `export` only exists to rebuild one.
- `MODEL_NAME = "model.onnx"` is defined in three places. Left alone deliberately — importing it across
  those modules would add edges for a thirteen-character string.
- The `archive/master-v2-refactor` tag points at an **orphan root commit** on no branch. That tag is the
  only thing keeping it reachable, so it is not semver and must not be deleted.
- Tag `v1.0.0` sits on a 2022 commit with a published release. This is why the current version is 1.1.0.
