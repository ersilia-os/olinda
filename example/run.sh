#!/usr/bin/env bash
#
# Inputs live in data/, everything the run produces goes to results/.
#
#   ./run.sh          the real thing — distil from the full reference library, predict over all of it
#   ./run.sh check    a fast plumbing check — 20k reference compounds, 10k at predict
#
# `--max-samples` bounds every step that sweeps the library, learn-hard included, so check costs
# minutes rather than the best part of an hour. 20k rather than a token 1k because the applicability
# gate and the blend ceiling need enough neighbours to earn a hard head worth looking at.
#
# `check` writes to results/check/ so it can never overwrite a real run.
# The validate step needs the reporting extra:  pip install "olinda[report]"
#
# Columns are named explicitly below. olinda would find them anyway (a `smiles`/`input` column, then
# the values after it), but naming them means a reordered or extra column in a regenerated file is
# an error rather than a model quietly trained on the wrong thing.

set -euo pipefail
cd "$(dirname "$0")"

SOFT=data/erl0_eos3804_v1.csv               # teacher values over the reference library
HARD=data/eos3804_hard.csv          # your own measurements
PRED=data/prediction_set.csv                # the compounds you actually want scored
LIBRARY=data/erl0_smiles.csv                # the whole library, scored for good measure
HELDOUT=data/eos3804_prediction_set.csv     # the teacher's own values on PRED — 93% outside the library

SOFT_COL=abaumannii_inhibition_probability  # the teacher's column, in SOFT and HELDOUT
HARD_COL=abaumannii_inhibition              # the measured column, in HARD

OUT=results
SAMPLE_ARG=""

case "${1:-}" in
  "")
    ;;
  check)
    OUT=results/check
    LIBRARY=data/erl0_smiles_first10k.csv
    SAMPLE_ARG="--max-samples 20000"
    ;;
  *)
    echo "usage: $0 [check]" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT"

# shellcheck disable=SC2086
olinda fit --soft-labels "$SOFT" --hard-labels "$HARD" --model-onnx "$OUT/model_eos3804.onnx" \
  --soft-label-columns "$SOFT_COL" --hard-label-columns "$HARD_COL" $SAMPLE_ARG
olinda predict --model-onnx "$OUT/model_eos3804.onnx" --input "$PRED" --output "$OUT/output_eos3804.csv"
olinda predict --model-onnx "$OUT/model_eos3804.onnx" --input "$LIBRARY" --output "$OUT/output_erl0_eos3804.csv"

# The soft labels here are held out — only 7% of these compounds are in the reference library the
# student trained on, so the correlation is an honest read. The hard labels are NOT: fit trained on
# them, so treat the ROC/PR numbers as optimistic until you score measurements the model hasn't seen.
olinda validate --model-onnx "$OUT/model_eos3804.onnx" \
  --soft-labels "$HELDOUT" --soft-label-columns "$SOFT_COL" \
  --hard-labels "$HARD" --hard-label-columns "$HARD_COL" \
  --output-dir "$OUT/report"
