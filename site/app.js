// The page: read the model and the SMILES, run olinda.js, and show a table.

import { goldenCheck, loadFeaturizer, predict, readMetadata } from "./olinda.js";

const $ = (id) => document.getElementById(id);
const status = (text) => ($("status").textContent = text);

let ort;
let fz;
let loaded = null; // { file, meta, session, golden } for the model currently chosen
let example = null; // the example model, once chosen, as a File like an uploaded one

async function init() {
  ort = await import("./ort/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("./ort/", location.href).href;
  // GitHub Pages cannot enable cross-origin isolation, which threads need; one thread is enough here.
  ort.env.wasm.numThreads = 1;
  fz = await loadFeaturizer(await (await fetch("rdkit_qmfp.wasm")).arrayBuffer());
  status(`Ready. RDKit ${fz.rdkitVersion} (WebAssembly). Choose a model.onnx, or use the example.`);
  $("run").disabled = false;
  $("example").disabled = false;
}

async function loadModel(file) {
  if (loaded && loaded.file === file) return loaded;
  status(`Reading ${file.name}…`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const meta = readMetadata(bytes);
  const session = await ort.InferenceSession.create(bytes);
  status(`Checking ${file.name} against its golden molecules…`);
  const golden = await goldenCheck(ort, session, fz, meta);
  loaded = { file, meta, session, golden };
  return loaded;
}

function goldenText(golden) {
  if (golden.skipped) return "golden check skipped (the model has no golden vectors)";
  const bad = golden.bytesDiffer.length + golden.outputsDiffer.length;
  if (!bad) return `golden check passed (${golden.n} molecules)`;
  return (
    `GOLDEN CHECK FAILED: ${golden.bytesDiffer.length} of ${golden.n} molecules featurize differently ` +
    `and ${golden.outputsDiffer.length} score differently, so these predictions may not match the model's`
  );
}

const fmt = (v) => (Number.isNaN(v) ? "" : String(v));

function render(columns, rows) {
  const table = $("results");
  table.replaceChildren();
  const head = table.insertRow();
  for (const name of ["smiles", ...columns]) {
    const th = document.createElement("th");
    th.textContent = name;
    head.appendChild(th);
  }
  for (const r of rows) {
    const tr = table.insertRow();
    tr.insertCell().textContent = r.smiles;
    for (const v of r.values) tr.insertCell().textContent = fmt(v);
  }
  table.hidden = false;

  const quote = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const csv = [["smiles", ...columns].join(","), ...rows.map((r) => [quote(r.smiles), ...r.values.map(fmt)].join(","))];
  const link = $("download");
  if (link.href) URL.revokeObjectURL(link.href);
  link.href = URL.createObjectURL(new Blob([csv.join("\n") + "\n"], { type: "text/csv" }));
  link.hidden = false;
}

async function useExample() {
  $("example").disabled = true;
  try {
    const bytes = await (await fetch("example.onnx")).arrayBuffer();
    example = new File([bytes], "example.onnx");
    $("model").value = ""; // the example replaces any uploaded file
    status("Example model selected (count_sum). Press Predict.");
  } catch (err) {
    status(`Error loading the example model: ${err.message}`);
  } finally {
    $("example").disabled = false;
  }
}

async function run() {
  const file = $("model").files[0] || example;
  if (!file) return status("Choose a model.onnx, or use the example model.");
  const smiles = $("smiles").value.replace(/\n$/, "").split("\n").map((s) => s.trim());
  $("run").disabled = true;
  try {
    const { meta, session, golden } = await loadModel(file);
    status(`Predicting ${smiles.length} SMILES…`);
    const t0 = performance.now();
    const { columns, rows } = await predict(ort, session, fz, meta, smiles);
    const invalid = rows.filter((r) => r.values.some(Number.isNaN)).length;
    render(columns, rows);
    status(
      `${rows.length} SMILES in ${((performance.now() - t0) / 1000).toFixed(2)} s` +
        (invalid ? ` (${invalid} could not be parsed: empty cells)` : "") +
        ` · ${goldenText(golden)} · model featurized with RDKit ${meta.rdkitVersion}, this page uses ${fz.rdkitVersion}`,
    );
  } catch (err) {
    status(`Error: ${err.message}`);
  } finally {
    $("run").disabled = false;
  }
}

$("run").addEventListener("click", run);
$("example").addEventListener("click", useExample);
$("model").addEventListener("change", () => {
  loaded = null;
  example = null; // an uploaded file replaces the example
});
init().catch((err) =>
  status(
    `Error loading the page: ${err.message}. The featurizer needs WebAssembly exception handling: ` +
      "Chrome 137+, Firefox 131+ or Safari 18.4+. If your browser is recent, reload the page.",
  ),
);
