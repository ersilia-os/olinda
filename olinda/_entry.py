"""Console-script entry point.

The base install carries only what inference needs, so the CLI's dependencies may be absent. Fail
with an instruction rather than an ImportError traceback.
"""

from __future__ import annotations


def main() -> None:
  try:
    from olinda.cli import cli
  except ImportError as exc:  # pragma: no cover - exercised only on a base install
    missing = getattr(exc, "name", None) or "a training dependency"
    raise SystemExit(
      f"The olinda CLI needs the training dependencies (missing: {missing}).\n"
      "This looks like an inference-only install. Install the full stack with:\n\n"
      "    pip install 'olinda[train]'\n\n"
      "Running distilled models needs no extras — see olinda.OnnxArtifact."
    ) from exc
  cli()
