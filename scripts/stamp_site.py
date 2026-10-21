"""Stamp the assembled web page so a browser can never mix files from two deploys.

GitHub Pages serves every file with ``cache-control: max-age=600`` and offers no way to change it, so
for ten minutes after a deploy a browser may pair a fresh ``index.html`` with a cached ``app.js`` from
the deploy before. The page then half-works: the example button, for one, stayed disabled.

The fix is to make every reference build-specific. ``index.html`` loads ``app.js?v=<build>``, and so on
down the chain of imports and fetches, so each copy of ``index.html``, cached or not, only ever loads
files from its own build. onnxruntime-web's files go in a directory named for their version.

Fails if any reference it expects is missing, so an edit to the page cannot silently drop the stamp.

    python scripts/stamp_site.py SITE_DIR BUILD ORT_VERSION
"""

import sys
from pathlib import Path


def stamp(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"{path.name}: expected exactly one {old!r}, found {count}"
            )
        text = text.replace(old, new)
    path.write_text(text)


def main() -> None:
    site, build, ort_version = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    v = f"?v={build}"
    stamp(site / "index.html", [('src="app.js"', f'src="app.js{v}"')])
    stamp(
        site / "app.js",
        [
            ('from "./olinda.js"', f'from "./olinda.js{v}"'),
            (
                'import("./ort/ort.wasm.min.mjs")',
                f'import("./ort/{ort_version}/ort.wasm.min.mjs")',
            ),
            (
                'new URL("./ort/", location.href)',
                f'new URL("./ort/{ort_version}/", location.href)',
            ),
            ('fetch("rdkit_qmfp.wasm")', f'fetch("rdkit_qmfp.wasm{v}")'),
            ('fetch("example.onnx")', f'fetch("example.onnx{v}")'),
        ],
    )
    stamp(site / "olinda.js", [('from "./onnxmeta.js"', f'from "./onnxmeta.js{v}"')])
    print(f"stamped {site} with build {build} and onnxruntime-web {ort_version}")


if __name__ == "__main__":
    main()
