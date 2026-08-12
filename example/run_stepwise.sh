#!/usr/bin/env bash
#
# The same distillation as run.sh, but one step at a time — so you can stop after the surrogate,
# look at what it learned, and only then decide to add the hard-label head.
#
#   ./run_stepwise.sh prepare    read the teacher + hard columns, plan each column's split
#   ./run_stepwise.sh tune       optional Optuna pass (single-column runs only)
#   ./run_stepwise.sh soft       train the surrogate; fuses a working soft-only model.onnx
#   ./run_stepwise.sh check      metrics, plots and a smoke prediction  <- stop here and look
#   ./run_stepwise.sh hard       train + calibrate the hard-label head, re-fuse
#   ./run_stepwise.sh finish     move the model out and delete the run folder
#
# Runs on the full 1.36M-compound library by default, which is what makes the numbers worth reading.
# For a fast plumbing check, shrink it:  SAMPLES=5000 ./run_stepwise.sh prepare
#
# Note: --hard-labels is passed at PREPARE, not at learn-hard — that step reads the hard.h5 that
# prepare wrote. Declaring them costs nothing if you never run `hard`.

set -euo pipefail
cd "$(dirname "$0")"

# Inputs live in data/, everything the run produces goes to results/.
DATA=${DATA:-data}
OUT=${OUT:-results}

SOFT=${SOFT:-$DATA/erl0_eos3804_v1.csv}          # teacher values over the reference library
HARD=${HARD:-$DATA/eos3804_hard.csv}     # your own measurements
MODEL=${MODEL:-$OUT/eos3804.onnx}                # the artifact you keep
RUN=${MODEL%.onnx}                               # working folder — `clean` derives it from MODEL
SMOKE=${SMOKE:-$DATA/erl0_smiles_first10k.csv}   # a small SMILES file for the check step

mkdir -p "$OUT"

SAMPLE_ARG=""
[ -n "${SAMPLES:-}" ] && SAMPLE_ARG="--max-samples ${SAMPLES}"

die() { echo "  ✖ $*" >&2; exit 1; }
run() { echo "  ▪ $*"; "$@"; }

case "${1:-}" in

  prepare)
    [ -f "$SOFT" ] || die "no teacher file at $SOFT"
    [ -f "$HARD" ] || die "no hard-label file at $HARD"
    [ -e "$RUN/manifest.json" ] && die "$RUN is already prepared — delete it to start over"
    # shellcheck disable=SC2086
    run olinda prepare --soft-labels "$SOFT" --hard-labels "$HARD" --model-dir "$RUN" $SAMPLE_ARG
    ;;

  tune)
    [ -f "$RUN/manifest.json" ] || die "run '$0 prepare' first"
    run olinda tune --model-dir "$RUN" --trials "${TRIALS:-100}"
    ;;

  soft)
    [ -f "$RUN/manifest.json" ] || die "run '$0 prepare' first"
    run olinda learn-soft --model-dir "$RUN" --num-boost-round "${ROUNDS:-10000}"
    echo "  ✓ $RUN/model.onnx is a complete soft-only model — try '$0 check'"
    ;;

  check)
    [ -f "$RUN/model.onnx" ] || die "no model yet — run '$0 soft' first"
    python - "$RUN" "$SMOKE" <<'PY'
import json, sys, time
from pathlib import Path
from olinda import OlindaArtifact

run, smoke = Path(sys.argv[1]), Path(sys.argv[2])

print("\n  validation metrics, per column")
for col in sorted(run.glob("columns/*/val_metrics.json")):
    m = json.loads(col.read_text())
    name = json.loads((col.parent / "train_meta.json").read_text()).get("column", col.parent.name)
    print(f"    {name}")
    print(f"      R² {m['r2']:+.4f}   Spearman {m['spearman']:+.4f}   RMSE {m['rmse']:.5f}"
          f"   top-decile RMSE {m['top_decile_rmse']:.5f}   (n={m['n']:,})")
    plot = col.parent / "val_true_pred.png"
    if plot.exists():
        print(f"      plot: {plot}")

model = OlindaArtifact(run / "model.onnx")
print("\n  the fused artifact says about itself")
for k, v in model.describe().items():
    print(f"    {k}: {v}")

if smoke.exists():
    import pandas as pd
    q = pd.read_csv(smoke)["smiles"].tolist()
    t0 = time.time()
    df = model.run(q, progress=False)
    print(f"\n  smoke prediction · {len(df):,} molecules in {time.time() - t0:.1f}s")
    print("   ", df.describe().loc[["min", "mean", "max"]].to_string().replace("\n", "\n    "))
    bad = int(df[model.columns].isna().any(axis=1).sum())
    print(f"    unparseable SMILES: {bad}")
else:
    print(f"\n  (no {smoke} — skipping the smoke prediction)")
PY
    echo
    echo "  happy with that? '$0 hard' adds the hard-label head."
    ;;

  hard)
    [ -f "$RUN/model.onnx" ] || die "run '$0 soft' first — learn-hard needs a trained surrogate"
    ls "$RUN"/columns/*/hard.h5 >/dev/null 2>&1 || die "no hard labels prepared — re-run '$0 prepare' with $HARD"
    run olinda learn-hard --model-dir "$RUN"
    echo "  ✓ re-fused with the blend — '$0 check' again to compare"
    ;;

  finish)
    [ -f "$RUN/model.onnx" ] || die "nothing to finish — run '$0 soft' first"
    run olinda clean --model-onnx "$MODEL"
    echo "  ✓ $MODEL is the whole deliverable; $RUN/ is gone"
    ;;

  *)
    sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
