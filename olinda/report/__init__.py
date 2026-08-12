"""Validate a finished ``model.onnx`` against labelled data and report on it.

`learn-soft` scores each column on the run's own validation split, but those numbers describe raw
predictions taken *before* the isotonic correction is fitted, and they disappear when `clean` runs.
This module answers a different question: given this artifact and *this* data, how does it do?

Because the input is arbitrary held-out data it is also the honest place to measure the calibrated
model — the correction was fitted on the run's validation rows, and these rows are new.

    from olinda.report import validate_model
    validate_model("model.onnx", soft_labels="heldout.csv", out_dir="report/")

Needs the ``[report]`` extra — onnx to read the graph, stylia/matplotlib to draw. Deliberately not the
base install: running a model should never drag in a plotting stack. Every one of those imports is
made inside a function, so importing this module costs nothing and :func:`require_report_extra` gives
one clear error instead of an ImportError from somewhere deep in the call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPORT_NAME = "report.html"
METRICS_NAME = "metrics.json"
TABLE_NAME = "performance_table.csv"


def require_report_extra() -> None:
  """Raise a single actionable error if the reporting dependencies are absent."""
  missing = []
  for module, why in (("onnx", "read the model graph"), ("stylia", "draw the figures")):
    try:
      __import__(module)
    except ImportError:
      missing.append(f"{module} (to {why})")
  if missing:
    raise RuntimeError(
      "olinda validate needs the reporting extra — missing: "
      + ", ".join(missing)
      + '.\nInstall it with:  pip install "olinda[report]"'
    )


def _match_columns(value_cols, tasks) -> tuple[dict, list]:
  """``({task: source column}, ignored)`` using the same naming rules ``prepare`` applies.

  ``match_hard_columns`` raises when *any* column fails to match, which is right for `prepare` — a
  stray column there is a mistake worth stopping for. A validation file is different: it may carry an
  id, a comment, or five assays of which this model predicts one. So match column by column and let
  the rest go, reporting what was ignored rather than refusing the whole file.
  """
  from olinda.data.reference import match_hard_columns

  matched: dict[str, str] = {}
  ignored, conflicts = [], []
  for col in value_cols:
    try:
      pair = match_hard_columns(list(tasks), [col])  # {source: task}
    except ValueError:
      ignored.append(col)
      continue
    for source, task in pair.items():
      if task in matched:
        conflicts.append(f"{matched[task]!r} and {source!r} both name task {task!r}")
      matched[task] = source
  if conflicts:
    raise ValueError("ambiguous columns — " + "; ".join(conflicts))
  return matched, ignored


def _read_labels(path, artifact_columns) -> tuple[list[str], dict, list]:
  """Read a SMILES + values file and map its value columns onto the model's task names."""
  import pandas as pd

  from olinda.predict import read_smiles

  path = Path(path)
  suffix = path.suffix.lower()
  if suffix in (".parquet", ".pq"):
    frame = pd.read_parquet(str(path))
  else:
    frame = pd.read_csv(str(path), sep="\t" if suffix == ".tsv" else ",")
  read_smiles(path)  # reuse its validation of the smiles column, and its error message

  value_cols = [c for c in frame.columns if c != "smiles"]
  if not value_cols:
    raise ValueError(f"{path} has a smiles column but no values to compare against")
  matched, ignored = _match_columns(value_cols, artifact_columns)
  if not matched:
    raise ValueError(
      f"none of {value_cols} match this model's tasks {list(artifact_columns)} — "
      "columns are matched by name, allowing a suffix"
    )
  smiles = frame["smiles"].astype(str).tolist()
  return smiles, {task: frame[src].to_numpy() for task, src in matched.items()}, ignored


def _library_overlap(smiles) -> float | None:
  """Fraction of *smiles* that appear in the reference library, or ``None`` if it is not installed.

  Validating on the library the model was distilled from is a legitimate question ("how well did it
  fit?") but not a measure of generalisation, so the report says which one you asked.
  """
  from olinda.data.fetch import MORGAN_FINGERPRINTS_FILENAME, OLINDA_HOME

  path = Path(OLINDA_HOME) / MORGAN_FINGERPRINTS_FILENAME
  if not path.exists():
    return None
  try:
    import h5py

    with h5py.File(path, "r") as f:
      raw = f["input"][:]
  except (OSError, KeyError):
    return None
  library = {s.decode() if isinstance(s, bytes) else str(s) for s in raw}
  return float(np.mean([s in library for s in smiles])) if smiles else 0.0


def _hard_head_warning(kind: str, artifact, n_rows: int) -> list[str]:
  """Flag the easy mistake of scoring a ground-truth head on the labels it was trained on.

  The artifact records how many compounds each hard head saw but not *which* — so identity cannot be
  proven from the ``.onnx`` alone. A matching row count is strong enough evidence to say so plainly;
  otherwise just state the training size and let the reader judge. Without this, handing `validate`
  the same measurements you trained on returns a near-perfect AUROC with nothing to suggest it is
  memorisation.
  """
  if kind != "hard":
    return []
  notes = []
  for column in artifact.metadata.get("columns", []):
    trained_on = (column.get("hard") or {}).get("n_train")
    if not trained_on:
      continue
    if trained_on == n_rows:
      notes.append(
        f"{column['name']}: this model's ground-truth head was trained on {trained_on:,} compounds "
        f"and you passed exactly {n_rows:,} — if these are the same measurements, the ranking "
        "metrics are in-sample and will look far better than the model generalises."
      )
    else:
      notes.append(
        f"{column['name']}: the ground-truth head was trained on {trained_on:,} compounds; any of "
        "those present here are scored in-sample."
      )
  return notes


def _predict(artifact, smiles, echo=None) -> "dict[str, np.ndarray]":
  import warnings

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # unparseable SMILES are counted, not warned twice
    frame = artifact.run(smiles, progress=False)
  if echo:
    echo(f"scored [bold]{len(smiles):,}[/] compounds", "run")
  return {c: frame[c].to_numpy() for c in artifact.columns}


def validate_model(
  model_onnx: str | Path,
  *,
  soft_labels: str | Path | None = None,
  hard_labels: str | Path | None = None,
  out_dir: str | Path = "report",
) -> dict:
  """Score an artifact against labelled data and write a report directory.

  Parameters
  ----------
  model_onnx : str or Path
      A fused ``model.onnx`` (a run directory containing one also works).
  soft_labels : str or Path, optional
      SMILES plus teacher values — any size, any order. Produces correlation and residual
      diagnostics.
  hard_labels : str or Path, optional
      SMILES plus binary labels. Produces ROC, precision–recall and enrichment. Note this scores the
      model's **blended** output, which is what ``predict`` emits, not the ground-truth head alone.
  out_dir : str or Path
      Directory to write ``report.html``, ``metrics.json``, ``performance_table.csv``, ``png/`` and
      ``pdf/`` into.

  Returns
  -------
  dict
      The same structure written to ``metrics.json``.
  """
  import json
  from datetime import datetime, timezone

  from olinda.artifact import OlindaArtifact
  from olinda.console import echo
  from olinda.metrics import binary_metrics, json_safe, regression_metrics
  from olinda.report import html as html_mod
  from olinda.report import plots
  from olinda.report.internals import describe_graph

  require_report_extra()
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  artifact = OlindaArtifact(model_onnx)

  report: dict = {
    "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model": {**artifact.describe(), "backend": artifact.metadata.get("backend")},
    "notes": [],
    "figures": {"soft": [], "hard": [], "internals": []},
  }

  for kind, source in (("soft", soft_labels), ("hard", hard_labels)):
    if source is None:
      continue
    smiles, values, ignored = _read_labels(source, artifact.columns)
    echo(f"{kind} labels · [bold]{len(smiles):,}[/] compounds · {', '.join(values)}", "run")
    if ignored:
      report["notes"].append(
        f"{kind} labels: ignored {', '.join(map(str, ignored))} — no task of this model is named that"
      )
    predicted = _predict(artifact, smiles, echo)

    overlap = _library_overlap(smiles)
    if overlap is not None and overlap > 0.5:
      report["notes"].append(
        f"{overlap:.0%} of the {kind}-label compounds are in the reference library this model was "
        "distilled from, so these numbers describe fit, not generalisation."
      )
    report["notes"].extend(_hard_head_warning(kind, artifact, len(smiles)))

    entry = {"file": str(source), "n": len(smiles), "metrics": {}}
    n_bad = 0
    for task, truth in values.items():
      pred = predicted[task]
      truth = np.asarray(truth, dtype=np.float64)
      ok = np.isfinite(pred) & np.isfinite(truth)
      n_bad = max(n_bad, int((~np.isfinite(pred)).sum()))
      y, p = truth[ok], pred[ok]
      if len(y) == 0:
        report["notes"].append(f"no comparable rows for {task} in the {kind} labels — skipped")
        continue

      if kind == "soft":
        m = regression_metrics(y, p)
        entry["metrics"][task] = m
        for name, draw, square in (
          (
            "correlation",
            lambda ax, st, y=y, p=p, m=m, t=task: plots.correlation(ax, st, y, p, metrics=m, task=t),
            True,
          ),
          ("residuals", lambda ax, st, y=y, p=p: plots.residual_hist(ax, st, y, p), False),
          ("residual_structure", lambda ax, st, y=y, p=p: plots.residuals_vs_pred(ax, st, y, p), True),
          ("calibration", lambda ax, st, y=y, p=p: plots.calibration_bins(ax, st, y, p), True),
          ("residual_qq", lambda ax, st, y=y, p=p: plots.qq_residuals(ax, st, y, p), True),
        ):
          fig = plots.render(draw, out_dir, f"{task}_{name}", square=square)
          if fig:
            report["figures"]["soft"].append(fig)
      else:
        binary = np.unique(y)
        if not set(binary.tolist()) <= {0.0, 1.0}:
          report["notes"].append(f"{task}: hard labels are not 0/1, skipping the ranking metrics")
          continue
        m = binary_metrics(y, p)
        entry["metrics"][task] = m
        for name, draw, square in (
          ("roc", lambda ax, st, y=y, p=p, m=m: plots.roc(ax, st, y, p, metrics=m), True),
          (
            "precision_recall",
            lambda ax, st, y=y, p=p, m=m: plots.precision_recall(ax, st, y, p, metrics=m),
            True,
          ),
          ("enrichment", lambda ax, st, y=y, p=p: plots.enrichment(ax, st, y, p), False),
          ("score_by_class", lambda ax, st, y=y, p=p: plots.score_by_class(ax, st, y, p), False),
        ):
          fig = plots.render(draw, out_dir, f"{task}_{name}", square=square)
          if fig:
            report["figures"]["hard"].append(fig)

    entry["n_unparseable"] = n_bad
    report[kind] = entry

  # The artifact's own calibration curves — available with no data at all.
  try:
    internals = describe_graph(artifact.path)
  except (ValueError, OSError) as exc:
    report["notes"].append(f"could not read the graph internals: {exc}")
    internals = {}
  report["internals"] = {
    task: {k: v for k, v in info.items() if k in ("id", "n_trees", "n_nodes")}
    for task, info in internals.items()
  }
  for task, info in internals.items():
    for stage, curve, label, xlabel in (
      ("soft_calibration", info["soft_calibration"], "Surrogate correction", "Raw student output"),
      ("hard_calibration", info["hard_calibration"], "Ground truth → teacher", "G probability"),
    ):
      fig = plots.render(
        lambda ax, st, c=curve, t=f"{label} · {task}", x=xlabel: plots.calibration_map(
          ax, st, c, title=t, xlabel=x
        ),
        out_dir,
        f"{task}_{stage}",
      )
      if fig:
        report["figures"]["internals"].append(fig)

  (out_dir / METRICS_NAME).write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
  _write_table(out_dir / TABLE_NAME, report)
  html_mod.write_report(out_dir, report)
  return report


def _write_table(path: Path, report: dict) -> None:
  """One row per task per label kind, for a spreadsheet or a CI check."""
  import csv

  rows = []
  for kind in ("soft", "hard"):
    for task, m in (report.get(kind) or {}).get("metrics", {}).items():
      flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
      flat.update({f"ef_{k}": v for k, v in (m.get("enrichment") or {}).items()})
      rows.append({"task": task, "labels": kind, **flat})
  if not rows:
    return
  fields = list(dict.fromkeys(k for r in rows for k in r))
  with open(path, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
