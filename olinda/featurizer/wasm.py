"""RDKit MinimalLib as WebAssembly, run under wasmtime: SMILES in, packed Morgan count bytes out.

olinda never imports RDKit. ``rdkit_qmfp.wasm`` (built from ``wasm/`` by ``wasm/build.sh``) is RDKit's
own SMILES parser and Morgan generator (MinimalLib, as in RDKit.js), compiled to a standalone module that
exports plain C functions. It is one platform-independent file, so the wheel stays pure Python.

One instance per process, built on first use. A wasmtime store is not thread-safe, so calls are
serialised with a lock; parallelism comes from processes, each with its own instance.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import numpy as np

WASM_PATH = Path(__file__).with_name("rdkit_qmfp.wasm")

_lock = threading.Lock()
_module = None


class _Module:
    """The instantiated module and the few exports olinda calls."""

    def __init__(self, path: Path):
        import wasmtime

        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} is missing from the olinda install. It is built from wasm/ by "
                "wasm/build.sh and packed into the wheel; a source checkout needs that build once."
            )
        config = wasmtime.Config()
        # Compiled machine code is cached on disk, so later processes start in milliseconds. An
        # unwritable cache directory only costs start-up time.
        with contextlib.suppress(Exception):
            config.cache = True
        engine = wasmtime.Engine(config)
        module = wasmtime.Module(engine, path.read_bytes())
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        # Emscripten's standalone mode tells the host when memory grows; there is nothing to do.
        linker.define_func(
            "env",
            "emscripten_notify_memory_growth",
            wasmtime.FuncType([wasmtime.ValType.i32()], []),
            lambda _: None,
        )
        self.store = wasmtime.Store(engine)
        self.store.set_wasi(wasmtime.WasiConfig())
        exports = linker.instantiate(self.store, module).exports(self.store)
        # A WASI reactor: run the C++ static constructors (RDKit's periodic table among them) once.
        exports["_initialize"](self.store)
        self._memory = exports["memory"]
        self._malloc = exports["malloc"]
        self._free = exports["free"]
        self._batch = exports["qmfp_featurize_batch"]
        self.rdkit_version = self._read_cstr(exports["qmfp_rdkit_version"](self.store))

    def _read_cstr(self, ptr: int) -> str:
        raw = bytes(self._memory.read(self.store, ptr, ptr + 64))
        return raw.split(b"\0", 1)[0].decode()

    def _put(self, data: bytes) -> int:
        ptr = self._malloc(self.store, max(1, len(data)))
        self._memory.write(self.store, data, ptr)
        return ptr

    def featurize(
        self, smiles: list[str], mol_details: str, fp_details: str, nbytes: int
    ):
        """``(uint8 (N, nbytes), bool (N,))`` for SMILES already known to be non-blank and NUL-free."""
        n = len(smiles)
        buf = b"".join(s.encode() + b"\0" for s in smiles)
        ptrs = [
            self._put(buf),
            self._put(mol_details.encode() + b"\0"),
            self._put(fp_details.encode() + b"\0"),
        ]
        p_out = self._malloc(self.store, n * nbytes)
        p_valid = self._malloc(self.store, n)
        try:
            p_smiles, p_mol, p_fp = ptrs
            rc = self._batch(
                self.store, p_smiles, n, p_mol, p_fp, p_out, nbytes, p_valid
            )
            if rc != 0:
                raise RuntimeError(
                    f"qmfp_featurize_batch returned {rc}: the fingerprint options "
                    f"{fp_details} do not give {nbytes}-byte fingerprints"
                )
            out = np.frombuffer(
                bytes(self._memory.read(self.store, p_out, p_out + n * nbytes)),
                np.uint8,
            ).reshape(n, nbytes)
            valid = np.frombuffer(
                bytes(self._memory.read(self.store, p_valid, p_valid + n)), np.uint8
            ).astype(bool)
        finally:
            for ptr in (*ptrs, p_out, p_valid):
                self._free(self.store, ptr)
        return out, valid


def module() -> _Module:
    """This process's instance, built on first use."""
    global _module
    with _lock:
        if _module is None:
            _module = _Module(WASM_PATH)
        return _module


def featurize(smiles: list[str], mol_details: str, fp_details: str, nbytes: int):
    """Featurize one batch through this process's instance; see :meth:`_Module.featurize`."""
    m = module()
    with _lock:
        return m.featurize(smiles, mol_details, fp_details, nbytes)


def rdkit_version() -> str:
    """The RDKit release compiled into the module."""
    return module().rdkit_version
