"""Compute Morgan **count** fingerprints for the reference library and store them in HDF5.

Uses the same definition as the Ersilia model ``eos5axz``: RDKit's ``MorganGenerator`` count
fingerprint (``radius=3``, ``fpSize=2048``), folded to ``fpSize`` bins with each count clipped at 255.
Output is ``erl0_morgan.h5`` — a ``data`` matrix and a UTF-8 string ``input`` (SMILES) dataset — the
reference-library descriptor set olinda trains on. Counts are 0–255, so ``data`` is stored as
``uint8`` (4× smaller than float32, lossless); the training pipeline casts it to float32 on read.

Run with::

    python scripts/compute_morgan_fingerprints.py
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from olinda.utils.logging import logger

RDLogger.DisableLog("rdApp.*")

# Per-process fingerprint generator (MorganGenerator is not picklable, so each worker builds its own).
_GEN = None
_NBITS = None


def _init(radius: int, nbits: int) -> None:
    global _GEN, _NBITS
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    _NBITS = nbits


def _fingerprint(smiles: str) -> np.ndarray:
    """Folded Morgan count fingerprint as uint8 (counts clipped at 255); zeros for invalid SMILES."""
    arr = np.zeros(_NBITS, dtype=np.uint8)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return arr
    for i, c in _GEN.GetCountFingerprint(mol).GetNonzeroElements().items():
        arr[i] = 255 if c > 255 else c
    return arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/ersilia_reference_library_v0.csv",
        help="CSV with the reference library.",
    )
    p.add_argument(
        "--output",
        default=str(Path.home() / ".olinda" / "erl0_morgan.h5"),
        help="Destination HDF5 file (default: ~/.olinda/erl0_morgan.h5, where `olinda prepare` looks).",
    )
    p.add_argument("--smiles-col", default="smiles", help="Name of the SMILES column.")
    p.add_argument(
        "--radius", type=int, default=3, help="Morgan radius (eos5axz uses 3)."
    )
    p.add_argument(
        "--nbits",
        type=int,
        default=2048,
        help="Fingerprint size / folded bins (eos5axz uses 2048).",
    )
    p.add_argument(
        "--njobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel worker processes.",
    )
    p.add_argument(
        "--batch-size", type=int, default=8192, help="Rows written per HDF5 slice."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the number of compounds (development).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd

    df = pd.read_csv(args.input, usecols=[args.smiles_col])
    smiles = df[args.smiles_col].astype(str).tolist()
    if args.limit is not None:
        smiles = smiles[: args.limit]
    n = len(smiles)
    if n == 0:
        logger.error("No SMILES found in the input file.")
        raise SystemExit(1)
    logger.info(f"Loaded {n} SMILES from {args.input}")
    logger.info(
        f"Morgan count fingerprints: radius={args.radius} nbits={args.nbits} (counts clipped at 255)"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as h5:
        data = h5.create_dataset("data", shape=(n, args.nbits), dtype="uint8")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("input", data=np.array(smiles, dtype=object), dtype=str_dtype)
        h5.attrs["rdkit_version"] = rdkit.__version__
        h5.attrs["fingerprint"] = "morgan_count"
        h5.attrs["radius"] = args.radius
        h5.attrs["nbits"] = args.nbits
        h5.attrs["count_clip"] = 255
        h5.attrs["n_compounds"] = n

        with logger.stage("Computing Morgan count fingerprints"):
            buf: list[np.ndarray] = []
            start = 0
            with (
                Pool(
                    processes=args.njobs,
                    initializer=_init,
                    initargs=(args.radius, args.nbits),
                ) as pool,
                tqdm(total=n, unit="mol", desc="morgan") as bar,
            ):
                for arr in pool.imap(_fingerprint, smiles, chunksize=1000):
                    buf.append(arr)
                    if len(buf) >= args.batch_size:
                        data[start : start + len(buf)] = np.vstack(buf)
                        start += len(buf)
                        bar.update(len(buf))
                        buf = []
                if buf:
                    data[start : start + len(buf)] = np.vstack(buf)
                    bar.update(len(buf))

    nbytes = n * args.nbits  # uint8 = 1 byte/element
    logger.info(
        f"data matrix: shape=({n}, {args.nbits}), dtype=uint8, size={nbytes / 1e9:.2f} GB"
    )
    logger.success(f"Morgan count fingerprints written to {out_path}")


if __name__ == "__main__":
    main()
