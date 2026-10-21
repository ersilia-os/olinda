"""Compute qmfp fingerprints for the reference library and store them in HDF5.

The library is featurized exactly as every olinda model featurizes its input: through
:class:`olinda.featurizer.QmfpFeaturizer`, RDKit MinimalLib compiled to WebAssembly. The ``data``
matrix holds the decoded slot counts (0-15) as ``uint8``, which is what training reads;
``valid`` marks the rows RDKit could parse; ``input`` keeps the SMILES in library order, so teacher
files computed against the old library stay row-aligned.

Run from the repository root::

    python scripts/compute_qmfp_library.py --input reference_library_smiles.csv

Before publishing a library, cross-check a sample against full RDKit (needs ``rdkit``, the ``dev``
extra), which proves the WebAssembly build still matches it::

    python scripts/compute_qmfp_library.py --input ... --check-rdkit 100000
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from olinda.featurizer import QMFP_SPEC, QmfpFeaturizer
from olinda.utils.logging import logger

_CHUNK = 4096


def _featurize(smiles: list[str]):
    """One chunk in a worker: decoded slot values and validity. Each worker builds its own instance."""
    packed, valid = QmfpFeaturizer(QMFP_SPEC).featurize_bytes(smiles)
    from olinda.featurizer.codec import decode

    return decode(packed), valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input", required=True, help="CSV with the reference library's SMILES."
    )
    p.add_argument(
        "--output",
        default=str(Path.home() / ".olinda" / "erl0_qmfp_r3.h5"),
        help="Destination HDF5 file (default: ~/.olinda/erl0_qmfp_r3.h5, where `olinda prepare` looks).",
    )
    p.add_argument(
        "--smiles-col", default=None, help="SMILES column (default: the first column)."
    )
    p.add_argument(
        "--njobs", type=int, default=os.cpu_count() or 1, help="Worker processes."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the number of compounds (development).",
    )
    p.add_argument(
        "--check-rdkit",
        type=int,
        default=0,
        metavar="N",
        help="Compare N random rows with full RDKit and exit non-zero on any difference.",
    )
    return p.parse_args()


def _check_rdkit(path: Path, n: int) -> int:
    """Compare a random sample of the written library with the RDKit oracle; return the mismatch count."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from olinda.featurizer.codec import decode
    from tests.rdkit_oracle import RDKitOracle

    with h5py.File(path, "r") as h5:
        total = h5["data"].shape[0]
        idx = np.sort(
            np.random.default_rng(0).choice(total, size=min(n, total), replace=False)
        )
        smiles = h5["input"].asstr()[idx]
        data, valid = h5["data"][idx], h5["valid"][idx]
    ref, ref_valid = RDKitOracle(QMFP_SPEC).count_bytes(list(smiles))
    bad = (decode(ref) != data).any(axis=1) | (ref_valid != valid)
    for i in np.flatnonzero(bad)[:10]:
        logger.error(f"mismatch at row {idx[i]}: {smiles[i]}")
    return int(bad.sum())


def main() -> None:
    args = parse_args()
    import pandas as pd

    df = pd.read_csv(args.input)
    col = args.smiles_col or df.columns[0]
    smiles = df[col].astype(str).tolist()[: args.limit]
    n = len(smiles)
    if n == 0:
        logger.error("No SMILES found in the input file.")
        raise SystemExit(1)

    featurizer = QmfpFeaturizer(QMFP_SPEC)
    slots = int(QMFP_SPEC["slots"])
    logger.info(
        f"{n:,} SMILES from {args.input} · qmfp radius {QMFP_SPEC['morgan']['radius']} · {slots} slots · "
        f"RDKit {featurizer.rdkit_version} (WebAssembly)"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks = [smiles[i : i + _CHUNK] for i in range(0, n, _CHUNK)]
    with h5py.File(out, "w") as h5:
        data = h5.create_dataset("data", shape=(n, slots), dtype="uint8")
        valid = h5.create_dataset("valid", shape=(n,), dtype="bool")
        h5.create_dataset(
            "input",
            data=np.array(smiles, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        h5.attrs["fingerprint"] = json.dumps(QMFP_SPEC)
        h5.attrs["rdkit_version"] = featurizer.rdkit_version
        h5.attrs["layout"] = "decoded_uint8"
        h5.attrs["n_compounds"] = n

        start = 0
        with Pool(args.njobs) as pool, tqdm(total=n, unit="mol", desc="qmfp") as bar:
            for x, v in pool.imap(_featurize, chunks):
                data[start : start + len(v)] = x
                valid[start : start + len(v)] = v
                start += len(v)
                bar.update(len(v))
        n_invalid = int(n - np.count_nonzero(valid[:]))
        h5.attrs["n_invalid"] = n_invalid

    logger.info(
        f"data: ({n:,}, {slots}) uint8, {n * slots / 1e9:.2f} GB · {n_invalid:,} invalid SMILES"
    )
    logger.success(f"qmfp library written to {out}")

    if args.check_rdkit:
        bad = _check_rdkit(out, args.check_rdkit)
        if bad:
            logger.error(
                f"{bad} of {min(args.check_rdkit, n)} rows differ from full RDKit"
            )
            raise SystemExit(1)
        logger.success(
            f"{min(args.check_rdkit, n):,} sampled rows identical to full RDKit"
        )


if __name__ == "__main__":
    main()
