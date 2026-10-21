// CI parity check for the web page: run site/olinda.js in Node on a model and compare with Python.
//
//   node site/test.mjs MODEL.onnx SMILES.txt PYTHON_PREDICTIONS.csv
//
// SMILES.txt has one SMILES per line; PYTHON_PREDICTIONS.csv is `olinda predict`'s output for the same
// SMILES. Fails unless the golden check passes and every prediction agrees to 1e-6, with empty cells
// (NaN) in the same rows. Not deployed.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";

import { goldenCheck, loadFeaturizer, predict, readMetadata } from "./olinda.js";

const here = dirname(fileURLToPath(import.meta.url));
const [modelPath, smilesPath, pythonCsv] = process.argv.slice(2);

const fz = await loadFeaturizer(readFileSync(join(here, "rdkit_qmfp.wasm")));
const modelBytes = readFileSync(modelPath);
const meta = readMetadata(modelBytes);
const session = await ort.InferenceSession.create(modelBytes);

const golden = await goldenCheck(ort, session, fz, meta);
console.log(
  `RDKit ${fz.rdkitVersion} (model: ${meta.rdkitVersion}) · golden ${golden.n} molecules · ` +
    `${golden.bytesDiffer.length} bytes differ · ${golden.outputsDiffer.length} outputs differ`,
);

// Every line is a row, blank ones included (they must come back NaN); only the final newline is dropped.
const smiles = readFileSync(smilesPath, "utf8").replace(/\n$/, "").split("\n");
const t0 = performance.now();
const { columns, rows } = await predict(ort, session, fz, meta, smiles);
const seconds = (performance.now() - t0) / 1000;

const lines = readFileSync(pythonCsv, "utf8").trim().split(/\r?\n/);
const header = lines[0].split(",");
const want = lines.slice(1).map((l) => l.split(","));
let worst = 0;
let nanMismatch = 0;
rows.forEach((r, i) => {
  if (want[i][0] !== r.smiles) throw new Error(`row ${i}: SMILES differ (${want[i][0]} vs ${r.smiles})`);
  columns.forEach((c, j) => {
    const cell = want[i][header.indexOf(c)];
    const w = cell === "" ? NaN : Number(cell);
    const got = r.values[j];
    if (Number.isNaN(w) !== Number.isNaN(got)) nanMismatch++;
    else if (!Number.isNaN(w)) worst = Math.max(worst, Math.abs(w - got));
  });
});
console.log(
  `${rows.length} SMILES × ${columns.length} columns in ${seconds.toFixed(2)} s · ` +
    `max |Δ| vs Python ${worst.toExponential(2)} · NaN mismatches ${nanMismatch}`,
);

const ok = !golden.skipped && !golden.bytesDiffer.length && !golden.outputsDiffer.length;
if (!ok || worst > 1e-6 || nanMismatch) {
  console.error("FAIL: the web page does not reproduce olinda's predictions");
  process.exit(1);
}
console.log("OK: the web page reproduces olinda's Python predictions");
