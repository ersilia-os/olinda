# olinda — developer guide

Olinda distils a slow chemistry model into one fast, self-describing ONNX file. A teacher model is
scored once over a 1.36M-compound reference library; a compact gradient-boosting student learns to
reproduce it; everything fuses into a single `model.onnx` that runs on onnxruntime, with SMILES
featurized by RDKit compiled to WebAssembly.

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
| `olinda/featurizer/` | SMILES → qmfp bytes: the spec, the byte codec, and the wasmtime loader for `rdkit_qmfp.wasm` |
| `wasm/` | the C++ wrapper and build script that compile RDKit to that WebAssembly module |
| `olinda/console.py` | user-facing output — see below |
| `olinda/utils/logging.py` | the levelled logger — see below |

## The lazy-import contract

This is the constraint most likely to be broken by an innocent-looking edit.

numpy, pandas, xgboost and matplotlib cost about **12 seconds** to import. `olinda --help` must
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

`pip install olinda` runs models from Python — `OlindaArtifact` — on numpy, onnxruntime and wasmtime
alone: no rdkit, no pandas, no CLI stack. `[cli]` adds the `olinda` command (click, rich, loguru, tqdm);
`[report]` adds `olinda validate`, its figures and pandas, and includes `[cli]`; `[train]` adds the
boosting stack and includes `[report]`. `[dev]` carries rdkit, but only as a test oracle.

On a base install the `olinda` script still exists (entry points cannot be optional), so
`olinda/_entry.py` answers with the extra to install, in plain stdlib.

A command needing an absent extra must **refuse and name it** — `require_train_extra` /
`require_report_extra` — never die on whichever heavy import came first. CI installs each tier
separately and exercises it, so the boundaries are real.

## Conventions

- **ruff** is the canonical Ersilia config, with two documented deviations in `ruff.toml`
  (`target-version = "py311"`, and `preview` scoped under `[lint]`). `ruff check` and `ruff format`
  must both pass; `.pre-commit-config.yaml` runs them at the same pinned version as CI.
- **Docstrings** are NumPy convention. `D101`/`D102` are enforced. Say *why* rather than restating the
  signature — `olinda/data/reference.py` is the exemplar.
- **Dependencies** are pinned exactly. The self-references `olinda[cli]==…` (in `report`) and
  `olinda[report]==…` (in `train`) must be bumped in lockstep with `[project].version`.
- **One RDKit release everywhere**: `RDKIT_TAG` in `wasm/build.sh`, the `rdkit` pin in `[dev]`, and the
  RDKit.js version any browser runtime uses. `tests/test_qmfp.py` fails if the first two disagree.
- **`git blame`** — run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once, or the
  whole-repo reformat owns every line.
- **Tests** smoke-test the documented entry points. They run real fits on a synthetic 240-row library,
  so the whole suite is ~25 seconds; keep it that way.

## Data

`data/` is gitignored and backed by [`eosvc`](https://github.com/ersilia-os/eosvc); `access.json` is
local. The reference library (`~/.olinda/erl0_qmfp_r3.h5`, 2.8 GB) is fetched by `olinda setup` from a
public bucket and built by `scripts/compute_qmfp_library.py`. Never commit datasets, model artefacts,
`.onnx` files or the compiled `.wasm`.

## Things that look wrong and are not

- `example/run_stepwise.sh` has no `export` branch. Correct: `learn-hard` builds the artifact and
  `clean` moves it out; `export` only exists to rebuild one.
- `MODEL_NAME = "model.onnx"` is defined in three places. Left alone deliberately — importing it across
  those modules would add edges for a thirteen-character string.
- The `archive/v2-refactor` tag points at an **orphan root commit** on no branch — a squashed snapshot
  of an abandoned v2 rewrite. That tag is the only thing keeping it reachable, so it is deliberately not
  semver and must not be deleted. (It was `archive/master-v2-refactor` until 2026-08-13; there is no
  `master` branch and never was.)
- Tag `v1.0.0` sits on a 2022 commit with a published release. This is why the 1.x line started at 1.1.0.
- A model's input is **packed bytes**, `uint8 (N, 1024)`, not a fingerprint. It is exactly what RDKit
  MinimalLib's `get_morgan_fp_as_bytes` and RDKit.js return (qmfp: Morgan with count simulation), and a
  decoder inside the graph turns it into the `(N, 2048)` floats the heads read. Runtimes in any language
  pass those bytes straight through and hold no fingerprint logic, so they cannot drift apart.
- `olinda/featurizer/rdkit_qmfp.wasm` is gitignored, yet the package cannot featurize without it. It is
  RDKit compiled from `wasm/` by `wasm/build.sh` (inside `emscripten/emsdk`); CI builds or restores it
  and packs it into the wheel. In a fresh checkout, run the build once (the command is at the top of the
  script) or download the `rdkit-qmfp-wasm` artifact from a CI run.
- Models carry **golden vectors** instead of an RDKit version pin. On load, `OlindaArtifact` re-featurizes
  and re-scores them: that, not a version string, is the evidence a runtime reproduces the model.
