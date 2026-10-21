"""The public inference API: load a distilled ``model.onnx`` and predict from SMILES.

An olinda artifact is a single self-describing file. Everything needed to run it — the fingerprint
spec, the golden vectors that prove a runtime reproduces it, the task names and their outputs — travels
inside the file's ``metadata_props``, so the ``.onnx`` is the only input required:

    from olinda import OlindaArtifact

    model = OlindaArtifact("model.onnx")
    model.columns          # ['abaumannii_inhibition_probability', ...]
    model.trained_at       # '2026-08-11T09:14:00+00:00'
    y = model.run(["CCO", "c1ccccc1"])   # float64 array, one row per SMILES, one column per task

Inference needs numpy, onnxruntime and wasmtime, nothing else. SMILES are featurized by RDKit compiled
to WebAssembly (:mod:`olinda.featurizer`), so there is no rdkit package to install or to match.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

MODEL_NAME = "model.onnx"

# The metadata major version this code reads. Kept here, next to the reader, rather than imported from
# olinda.export: the export module belongs to the [train] extra and inference must not reach into it.
SUPPORTED_SCHEMA_MAJOR = 1

_BATCH = 4096


class GoldenMismatch(RuntimeError):
    """Raised, in strict mode, when this runtime does not reproduce a model's golden vectors."""


class GoldenMismatchWarning(RuntimeWarning):
    """Warned when this runtime does not reproduce a model's golden vectors."""


def _featurize_batch(featurizer, smiles: list[str]):
    """Featurize one batch in a worker process. Module-level so the pool can pickle a reference to it."""
    return featurizer.featurize_bytes(smiles)


def _as_smiles_list(smiles) -> list[str]:
    """Normalise user input to a list of SMILES strings, refusing the shapes that would mislead.

    A bare ``"CCO"`` is the trap worth guarding: a string is a perfectly good sequence of characters, so
    it would featurize as three one-atom molecules and return three rows without complaint. Being handed
    one molecule instead of a list is a natural mistake, and a wrong answer is a far worse outcome than
    an exception.
    """
    if isinstance(smiles, (str, bytes)):
        raise TypeError(
            f"expected a sequence of SMILES, got a single string {smiles!r:.40} — iterating it would score "
            f"each character as a molecule. Pass a list: [{smiles!r:.40}]"
        )
        # (a set would scramble the row order against the caller's input, so it is not accepted either)
    if isinstance(smiles, (dict, set, frozenset)):
        raise TypeError(
            f"expected an ordered sequence of SMILES, got {type(smiles).__name__} — order matters"
        )
    try:
        return [str(s) for s in smiles]
    except TypeError as exc:
        raise TypeError(
            f"expected a sequence of SMILES, got {type(smiles).__name__}"
        ) from exc


class _Progress:
    """A minimal stderr progress bar.

    Hand-rolled rather than reaching for rich or tqdm: the inference install is deliberately limited to
    numpy, onnxruntime and wasmtime, and a progress bar is not worth widening that. Silent unless stderr
    is a terminal, so redirected output and notebooks stay clean.
    """

    WIDTH = 28

    def __init__(
        self, total: int, label: str = "olinda", enabled: bool | None = None
    ) -> None:
        self.total = max(1, int(total))
        self.label = label
        if enabled is None:
            enabled = sys.stderr.isatty() and total > _BATCH
        self.enabled = bool(enabled)
        self._done = 0

    def advance(self, n: int) -> None:
        if not self.enabled:
            return
        self._done = min(self.total, self._done + n)
        frac = self._done / self.total
        filled = int(round(frac * self.WIDTH))
        bar = "█" * filled + "░" * (self.WIDTH - filled)
        sys.stderr.write(
            f"\r  {self.label} ▕{bar}▏ {frac:>4.0%} · {self._done:,}/{self.total:,}"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\r" + " " * (self.WIDTH + 40) + "\r")
            sys.stderr.flush()

    def __enter__(self) -> _Progress:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class OlindaArtifact:
    """A distilled olinda model, loaded from its ``model.onnx``.

    Parameters
    ----------
    path : str or Path
        The ``model.onnx`` itself, or a run directory containing one.
    strict : bool, optional
        Raise :class:`GoldenMismatch` instead of warning when this runtime does not reproduce the
        model's golden vectors (default ``False``).
    check_golden : bool, optional
        Featurize and score the golden vectors on load (default ``True``). They are the evidence that
        predictions here match the ones the model was built to make; skip them only to save the few
        milliseconds they take.

    Attributes
    ----------
    columns : list of str
        The task names this model predicts, in output order.
    golden : dict or None
        The result of the golden check: ``{"n", "bytes_differ", "outputs_differ"}``, or ``None`` when
        it was skipped.
    path : Path
        Where the artifact was loaded from.
    """

    def __init__(
        self, path: str | Path, *, strict: bool = False, check_golden: bool = True
    ) -> None:
        import onnxruntime as ort

        from olinda.featurizer import QmfpFeaturizer, n_input_bytes, validate_spec

        path = Path(path)
        if path.is_dir():
            path = path / MODEL_NAME
        if not path.exists():
            raise FileNotFoundError(f"no olinda artifact at {path}")

        self.path = path
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

        props = self._session.get_modelmeta().custom_metadata_map
        if "olinda.schema_version" not in props:
            if "olinda" in props:
                raise ValueError(
                    f"{path} was built by olinda 1.x, which fed Morgan count fingerprints from the rdkit "
                    "package. This olinda reads qmfp models only: rebuild it with `olinda fit`."
                )
            raise ValueError(
                f"{path} carries no olinda metadata — it may not be an olinda artifact."
            )
        self.schema_version = props["olinda.schema_version"]
        if int(self.schema_version.split(".")[0]) != SUPPORTED_SCHEMA_MAJOR:
            # Checked rather than assumed: a file written to another layout would otherwise be parsed as
            # though it were this one and go wrong field by field.
            raise ValueError(
                f"{path} declares metadata schema {self.schema_version}, but this olinda reads "
                f"{SUPPORTED_SCHEMA_MAJOR}.x. Install the olinda that wrote it, or rebuild the model."
            )

        self.fingerprint = validate_spec(json.loads(props["olinda.fingerprint"]))
        self.metadata = json.loads(props["olinda.model"])
        self._golden = json.loads(props.get("olinda.golden", "null"))
        self._rdkit_version = props.get("olinda.rdkit_version")

        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        width = model_input.shape[-1]
        if model_input.type != "tensor(uint8)" or width != n_input_bytes(
            self.fingerprint
        ):
            raise ValueError(
                f"{path} takes {model_input.type} of width {width}, not the uint8 "
                f"({n_input_bytes(self.fingerprint)},) qmfp bytes its metadata describes"
            )
        self._output_names = [o.name for o in self._session.get_outputs()]

        # A single-task model is just the one-column case, so nothing downstream distinguishes them.
        self._columns = self.metadata.get("columns") or []
        if not self._columns:
            raise ValueError(
                f"{path} records no columns — rebuild it with `olinda fit`."
            )
        # Fail here rather than with a bare KeyError deep inside run() if the embedded column list and
        # the graph disagree — which is what a hand-edited or mismatched artifact looks like.
        missing = [
            c["output"] for c in self._columns if c["output"] not in self._output_names
        ]
        if missing:
            raise ValueError(
                f"{path} declares output(s) {missing} that the graph does not produce; "
                f"it exposes {self._output_names}. Rebuild it with `olinda fit`."
            )

        self._featurizer = QmfpFeaturizer(self.fingerprint)
        have = self._featurizer.rdkit_version
        if self._rdkit_version and have != self._rdkit_version:
            warnings.warn(
                f"{path.name} was featurized with RDKit {self._rdkit_version}; this olinda computes "
                f"fingerprints with RDKit {have}. The golden check below says whether that matters.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.golden = None
        if check_golden and self._golden:
            self.golden = self._check_golden(strict)

    def _run_bytes(self, packed: np.ndarray) -> dict:
        return dict(
            zip(self._output_names, self._session.run(None, {self._input_name: packed}))
        )

    def _check_golden(self, strict: bool) -> dict:
        """Re-score the golden vectors; warn, or raise in strict mode, if anything differs."""
        from olinda.golden import check_golden

        result = check_golden(self._golden, self._run_bytes, self._featurizer)
        bad_bytes, bad_out = result["bytes_differ"], result["outputs_differ"]
        if bad_bytes or bad_out:
            if bad_bytes:
                what = (
                    f"{len(bad_bytes)} of {result['n']} golden molecules featurize differently "
                    f"(e.g. {bad_bytes[:3]}): this runtime's RDKit does not reproduce the "
                    "fingerprints the model was trained on, so its predictions may differ."
                )
            else:
                what = (
                    f"{len(bad_out)} of {result['n']} golden molecules featurize identically but "
                    f"score differently (e.g. {bad_out[:3]}): the graph is not running as it did "
                    "when the model was built."
                )
            if strict:
                raise GoldenMismatch(f"{self.path.name}: {what}")
            warnings.warn(
                f"{self.path.name}: {what}", GoldenMismatchWarning, stacklevel=3
            )
        return result

    # ── what the artifact says about itself ────────────────────────────────────

    @property
    def columns(self) -> list[str]:
        """The task names this model predicts, in output order."""
        return [c["name"] for c in self._columns]

    @property
    def n_columns(self) -> int:
        """How many tasks this model predicts."""
        return len(self._columns)

    @property
    def trained_at(self) -> str | None:
        """UTC timestamp of the fuse that produced this artifact."""
        return self.metadata.get("trained_at")

    @property
    def olinda_version(self) -> str | None:
        """The olinda release that fused this artifact, if it recorded one."""
        return self.metadata.get("olinda_version")

    @property
    def rdkit_version(self) -> str | None:
        """The RDKit release that featurized the model's training data and golden vectors."""
        return self._rdkit_version

    @property
    def featurizer_rdkit_version(self) -> str:
        """The RDKit release that featurizes SMILES here, compiled into olinda's WebAssembly module."""
        return self._featurizer.rdkit_version

    def _column(self, task: str) -> dict:
        for c in self._columns:
            if c["name"] == task:
                return c
        raise KeyError(
            f"{task!r} is not a task of this model; it predicts {self.columns}"
        )

    def heads_for(self, task: str) -> list[dict]:
        """The models behind one task, in the order they were added — the surrogate first.

        Each entry carries its own ``role``, ``task``, ``source``, ``training`` and ``metrics``, so asking
        "is this head a classifier?" is a lookup rather than a guess from which metrics are present. A
        soft-only column has one head; a blended one has two.
        """
        return [dict(h) for h in (self._column(task).get("heads") or [])]

    def task_for(self, task: str) -> dict:
        """``{"type", "range", "units"}`` for what one task predicts."""
        return dict(self._column(task).get("task") or {})

    def roles_for(self, task: str) -> list[str]:
        """Just the head roles behind one task, e.g. ``["soft", "hard"]``."""
        return [h.get("role") for h in (self._column(task).get("heads") or [])]

    @property
    def has_hard(self) -> bool:
        """True if any task blends in a hard-label head, so predictions use measured data.

        Derived from the heads rather than stored: a boolean beside the list it summarises is a second
        source of truth, and the two can disagree.
        """
        return any("hard" in self.roles_for(c["name"]) for c in self._columns)

    @property
    def n_features(self) -> int:
        """Fingerprint width the heads read: the qmfp slot count."""
        return int(self.fingerprint["slots"])

    def to_json(self, indent: int = 2) -> str:
        """Everything the file records about itself, as a JSON string.

        The raw ``olinda.*`` metadata, parsed, rather than the curated view :meth:`describe` returns —
        useful for inspecting or logging exactly what was embedded at fuse time.
        """
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "fingerprint": self.fingerprint,
                "rdkit_version": self._rdkit_version,
                "golden": self._golden,
                "model": self.metadata,
            },
            indent=indent,
            sort_keys=False,
            default=str,
        )

    def describe(self) -> dict:
        """A compact summary of the artifact, suitable for logging or display."""
        if self.golden is None:
            golden = "skipped"
        elif self.golden["bytes_differ"] or self.golden["outputs_differ"]:
            n_bad = len(self.golden["bytes_differ"]) + len(
                self.golden["outputs_differ"]
            )
            golden = f"fail: {n_bad}/{self.golden['n']}"
        else:
            golden = "pass"
        return {
            "path": str(self.path),
            "producer": self.metadata.get("producer", "olinda"),
            "olinda_version": self.olinda_version,
            "trained_at": self.trained_at,
            "rdkit_version": self.rdkit_version,
            "featurizer_rdkit_version": self.featurizer_rdkit_version,
            "n_features": self.n_features,
            "columns": self.columns,
            "has_hard": self.has_hard,
            "golden": golden,
        }

    # ── inference ──────────────────────────────────────────────────────────────

    def featurize(self, smiles) -> tuple[np.ndarray, np.ndarray]:
        """``(uint8 bytes (N, B), bool valid (N,))``: exactly what the graph takes, and which rows parsed."""
        return self._featurizer.featurize_bytes(_as_smiles_list(smiles))

    def _featurized_batches(self, smiles: list[str], batch_size: int, workers: int):
        """Yield ``(batch, (bytes, valid))`` in input order, featurizing ahead in *workers* processes.

        Featurization costs more than the graph on a one- or two-column model. With workers, the next
        batches are featurized while the caller scores the current one. At most ``workers + 1`` batches
        are in flight, so memory stays bounded on any input size.
        """
        batches = [
            smiles[start : start + batch_size]
            for start in range(0, len(smiles), batch_size)
        ]
        if workers <= 1 or len(batches) < 2:
            for batch in batches:
                yield batch, self._featurizer.featurize_bytes(batch)
            return

        from collections import deque
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(workers, len(batches))) as pool:
            pending: deque = deque()
            queue = iter(batches)
            for batch in queue:
                pending.append(
                    (batch, pool.submit(_featurize_batch, self._featurizer, batch))
                )
                if len(pending) > workers:
                    break
            while pending:
                batch, future = pending.popleft()
                upcoming = next(queue, None)
                if upcoming is not None:
                    pending.append(
                        (
                            upcoming,
                            pool.submit(_featurize_batch, self._featurizer, upcoming),
                        )
                    )
                yield batch, future.result()

    def run(
        self,
        smiles,
        batch_size: int = _BATCH,
        progress: bool | None = None,
        workers: int | None = 1,
    ) -> np.ndarray:
        """Predict for a list of SMILES. This is the only way to run the model.

        Parameters
        ----------
        smiles : sequence of str
            The molecules to score.
        batch_size : int, optional
            Rows per forward pass. Bounds memory on large inputs; does not change results.
        progress : bool, optional
            Show a progress bar on stderr. Defaults to showing one for multi-batch inputs on a terminal,
            so redirected output and notebooks stay clean.
        workers : int or None, optional
            Processes that featurize ahead of the graph; ``None`` means one per CPU. Does not change
            results. Defaults to 1 because worker processes re-import the calling script on macOS and
            Windows, so a script that passes more needs an ``if __name__ == "__main__":`` guard.
            ``olinda predict`` uses every CPU.

        Returns
        -------
        numpy.ndarray
            ``float64 (N, n_columns)``: one row per SMILES, in input order, and **one column per task**
            in :attr:`columns` order. For a blended column the value already folds in the hard-label
            model at the weight it earned. The graph declares nothing else: the pieces behind that
            number are internal to it by design.

        Notes
        -----
        Molecules RDKit cannot parse come back as ``NaN``, and a single warning names how many. Their
        fingerprint would be all-zero bytes, which the graph happily scores, so without this an
        unparseable input would return a confident-looking number.
        """
        import os

        smiles = _as_smiles_list(smiles)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        workers = (os.cpu_count() or 1) if workers is None else int(workers)

        wanted = [c["output"] for c in self._columns]
        out = np.empty((len(smiles), len(wanted)), dtype=np.float64)
        n_invalid = 0
        start = 0
        with _Progress(len(smiles), enabled=progress) as bar:
            for batch, (packed, valid) in self._featurized_batches(
                smiles, batch_size, workers
            ):
                scored = self._run_bytes(packed)
                block = np.column_stack(
                    [np.asarray(scored[n], dtype=np.float64).ravel() for n in wanted]
                )
                block[~valid] = np.nan
                n_invalid += int((~valid).sum())
                out[start : start + len(batch)] = block
                start += len(batch)
                bar.advance(len(batch))

        if n_invalid:
            warnings.warn(
                f"{n_invalid} of {len(smiles)} input SMILES could not be parsed; their predictions are NaN",
                RuntimeWarning,
                stacklevel=2,
            )
        return out

    def __len__(self) -> int:
        return self.n_columns

    def __repr__(self) -> str:
        return f"OlindaArtifact({self.path.name!r}, columns={self.columns}, trained_at={self.trained_at!r})"
