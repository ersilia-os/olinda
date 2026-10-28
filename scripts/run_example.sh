#!/usr/bin/env bash
#
# Run the worked example (example/run.sh) from anywhere, fetching its inputs first if they are missing.
#
#   scripts/run_example.sh          the real thing — distil from the full reference library
#   scripts/run_example.sh check    a fast plumbing check (20k reference compounds)
#
# The inputs live in data/ (eosvc, public bucket). They are downloaded with plain curl, so eosvc is not
# needed, and anything already present is skipped. Needs `pip install "olinda[train]"`.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=https://eosvc-public.s3.eu-central-1.amazonaws.com/olinda/data

INPUTS=(
  ersilia_reference_library_v0.csv
  example/erl0_eos3804_v1.csv
  example/eos3804_hard.csv
  example/eos3804_prediction_set.csv
  example/erl0_smiles_first10k.csv
  example/prediction_set.csv
)

for f in "${INPUTS[@]}"; do
  dest="$ROOT/data/$f"
  [ -s "$dest" ] && continue
  echo "  ▪ downloading $f"
  mkdir -p "$(dirname "$dest")"
  curl -fL --progress-bar -o "$dest.tmp" "$BASE/$f"
  mv "$dest.tmp" "$dest"
done

command -v olinda >/dev/null || { echo "  ✖ olinda is not installed: pip install \"olinda[train]\"" >&2; exit 1; }

exec "$ROOT/example/run.sh" "$@"
