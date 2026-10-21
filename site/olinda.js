// olinda in JavaScript: load a model.onnx, featurize SMILES, predict, and check the golden vectors.
//
// A port of what olinda's Python does at inference time, for the browser (and Node, where CI checks it
// against Python). No DOM here: app.js is the page. The featurizer is the same rdkit_qmfp.wasm the
// Python package ships, RDKit compiled to WebAssembly, so the bytes it produces are the bytes the
// model was trained on.

import { readOnnxMetadata } from "./onnxmeta.js";

const SUPPORTED_SCHEMA_MAJOR = 1;

// ── the featurizer: rdkit_qmfp.wasm ─────────────────────────────────────────────────────────────

// The module is a WASI reactor. It never reads files or the environment, so WASI is stubbed: every
// call reports success with nothing to do. Emscripten's memory-growth notice needs no answer either.
function wasiStubs(getMemory) {
  const zero = (ptr) => new DataView(getMemory().buffer).setUint32(ptr, 0, true);
  return {
    wasi_snapshot_preview1: {
      fd_write: (_fd, _iov, _n, nwritten) => (zero(nwritten), 0),
      fd_read: (_fd, _iov, _n, nread) => (zero(nread), 0),
      fd_close: () => 0,
      fd_seek: () => 0,
      environ_sizes_get: (count, size) => (zero(count), zero(size), 0),
      environ_get: () => 0,
      clock_time_get: (_id, _precision, out) => {
        new DataView(getMemory().buffer).setBigUint64(out, BigInt(Date.now()) * 1000000n, true);
        return 0;
      },
      proc_exit: (code) => {
        throw new Error(`rdkit_qmfp.wasm exited with ${code}`);
      },
    },
    env: { emscripten_notify_memory_growth: () => {} },
  };
}

export async function loadFeaturizer(wasmBytes) {
  let memory;
  const { instance } = await WebAssembly.instantiate(wasmBytes, wasiStubs(() => memory));
  const ex = instance.exports;
  memory = ex.memory;
  ex._initialize(); // C++ static constructors: RDKit's periodic table among them
  const cstr = (ptr) => {
    const bytes = new Uint8Array(memory.buffer, ptr);
    return new TextDecoder().decode(bytes.subarray(0, bytes.indexOf(0)));
  };
  return { ex, memory: () => memory, rdkitVersion: cstr(ex.qmfp_rdkit_version()) };
}

function put(fz, bytes) {
  const ptr = fz.ex.malloc(Math.max(1, bytes.length));
  new Uint8Array(fz.memory().buffer, ptr, bytes.length).set(bytes);
  return ptr;
}

const utf8 = (s) => new TextEncoder().encode(s);

function featurizeGroup(fz, smiles, molDetails, fpDetails, nbytes) {
  const n = smiles.length;
  const parts = smiles.map((s) => utf8(s + "\0"));
  const buf = new Uint8Array(parts.reduce((k, p) => k + p.length, 0));
  parts.reduce((k, p) => (buf.set(p, k), k + p.length), 0);
  const pS = put(fz, buf);
  const pM = put(fz, utf8(molDetails + "\0"));
  const pF = put(fz, utf8(fpDetails + "\0"));
  const pOut = fz.ex.malloc(n * nbytes);
  const pValid = fz.ex.malloc(Math.max(1, n));
  try {
    const rc = fz.ex.qmfp_featurize_batch(pS, n, pM, pF, pOut, nbytes, pValid);
    if (rc !== 0) throw new Error(`qmfp_featurize_batch returned ${rc} for options ${fpDetails}`);
    const mem = fz.memory().buffer; // read after the call: memory may have grown
    return {
      bytes: new Uint8Array(mem, pOut, n * nbytes).slice(),
      valid: Array.from(new Uint8Array(mem, pValid, n), (v) => v === 1),
    };
  } finally {
    for (const p of [pS, pM, pF, pOut, pValid]) fz.ex.free(p);
  }
}

// A port of QmfpFeaturizer.featurize_bytes (olinda/featurizer/__init__.py): every row comes back, in
// order; blank SMILES are invalid; explicit hydrogen atoms are parsed with the spec's other options.
export function featurize(fz, spec, smiles) {
  const nbytes = spec.slots / 2;
  const fpDetails = JSON.stringify({ ...spec.morgan, countEncoding: spec.encoding });
  const explicitH = new RegExp(spec.explicit_h_regex);
  const bytes = new Uint8Array(smiles.length * nbytes);
  const valid = new Array(smiles.length).fill(false);
  const groups = { plain: [], h: [] };
  smiles.forEach((s, i) => {
    if (s.trim() && !s.includes("\0")) groups[explicitH.test(s) ? "h" : "plain"].push(i);
  });
  for (const [kind, idx] of Object.entries(groups)) {
    if (!idx.length) continue;
    const molDetails = JSON.stringify(kind === "h" ? spec.get_mol_explicit_h : spec.get_mol);
    const out = featurizeGroup(fz, idx.map((i) => smiles[i]), molDetails, fpDetails, nbytes);
    idx.forEach((row, j) => {
      bytes.set(out.bytes.subarray(j * nbytes, (j + 1) * nbytes), row * nbytes);
      valid[row] = out.valid[j];
    });
  }
  return { bytes, valid, nbytes };
}

// ── the model's own description ─────────────────────────────────────────────────────────────────

// onnxruntime-web does not expose metadata_props, so they are read from the protobuf directly.
export function readMetadata(onnxBytes) {
  const props = readOnnxMetadata(onnxBytes);
  if (!("olinda.schema_version" in props)) {
    if ("olinda" in props) {
      throw new Error(
        "this model was built by olinda 1.2 or earlier, which fed Morgan count fingerprints from the " +
          "rdkit package. This page reads qmfp models only: rebuild it with `olinda fit`.",
      );
    }
    throw new Error("this file carries no olinda metadata — it may not be an olinda model.");
  }
  const schema = props["olinda.schema_version"];
  if (Number(schema.split(".")[0]) !== SUPPORTED_SCHEMA_MAJOR) {
    throw new Error(`the model declares metadata schema ${schema}; this page reads ${SUPPORTED_SCHEMA_MAJOR}.x`);
  }
  const spec = JSON.parse(props["olinda.fingerprint"]);
  if (spec.type !== "qmfp" || spec.encoding !== "count4") {
    throw new Error(`unsupported fingerprint ${spec.type}/${spec.encoding}: this page reads qmfp count4`);
  }
  return {
    schemaVersion: schema,
    spec,
    model: JSON.parse(props["olinda.model"]),
    golden: props["olinda.golden"] ? JSON.parse(props["olinda.golden"]) : null,
    rdkitVersion: props["olinda.rdkit_version"] || null,
  };
}

// ── inference ───────────────────────────────────────────────────────────────────────────────────

async function scoreBytes(ort, session, bytes, n, nbytes) {
  const feeds = { [session.inputNames[0]]: new ort.Tensor("uint8", bytes, [n, nbytes]) };
  const out = await session.run(feeds);
  const scored = {};
  for (const name of session.outputNames) scored[name] = Array.from(out[name].data, Number);
  return scored;
}

// One row per SMILES, one value per column, NaN where RDKit could not parse the molecule: its
// fingerprint would be all zeros, which the graph scores happily, so the number would mislead.
export async function predict(ort, session, fz, meta, smiles, batchSize = 4096) {
  const columns = meta.model.columns.map((c) => ({ name: c.name, output: c.output }));
  const rows = [];
  for (let start = 0; start < smiles.length; start += batchSize) {
    const batch = smiles.slice(start, start + batchSize);
    const { bytes, valid, nbytes } = featurize(fz, meta.spec, batch);
    const scored = await scoreBytes(ort, session, bytes, batch.length, nbytes);
    batch.forEach((s, i) => {
      rows.push({ smiles: s, values: columns.map((c) => (valid[i] ? scored[c.output][i] : NaN)) });
    });
  }
  return { columns: columns.map((c) => c.name), rows };
}

// ── the golden check (a port of olinda/golden.py) ───────────────────────────────────────────────

async function digest(row) {
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", row));
  return Array.from(hash.subarray(0, 8), (b) => b.toString(16).padStart(2, "0")).join("");
}

const isclose = (a, b, atol, rtol) =>
  (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= atol + rtol * Math.abs(b);

// Featurize and score the golden molecules again. Bytes that differ are a featurizer problem; bytes
// that agree with predictions that do not are a graph problem. Both lists empty means this page
// reproduces the model as it was built.
export async function goldenCheck(ort, session, fz, meta) {
  const g = meta.golden;
  if (!g) return { n: 0, bytesDiffer: [], outputsDiffer: [], skipped: true };
  const { bytes, valid, nbytes } = featurize(fz, meta.spec, g.smiles);
  const scored = await scoreBytes(ort, session, bytes, g.smiles.length, nbytes);
  const bytesDiffer = [];
  const outputsDiffer = [];
  for (let i = 0; i < g.smiles.length; i++) {
    const d = await digest(bytes.subarray(i * nbytes, (i + 1) * nbytes));
    if (valid[i] !== g.valid[i] || d !== g.fp_sha256[i]) {
      bytesDiffer.push(g.smiles[i]);
      continue;
    }
    for (const [name, want] of Object.entries(g.outputs)) {
      const got = valid[i] ? scored[name][i] : NaN;
      const w = want[i] === null ? NaN : want[i];
      if (!isclose(got, w, g.atol ?? 1e-5, g.rtol ?? 1e-5)) {
        outputsDiffer.push(g.smiles[i]);
        break;
      }
    }
  }
  return { n: g.smiles.length, bytesDiffer, outputsDiffer, skipped: false };
}
