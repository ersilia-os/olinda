"""`olinda library` dumps the reference library's SMILES as a one-column CSV.

The contract is narrow and entirely about fidelity: same molecules, same order, one column named
`smiles`. Order is the part worth testing — a teacher file has to be row-aligned with the library, so
a CSV that quietly reordered or deduplicated would produce a file `prepare` accepts and trains
wrongly from.
"""

from __future__ import annotations

import csv

from tests.conftest import plain, run_cli, write_library


def _read(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], [r[0] for r in rows[1:]]


def test_library_writes_one_smiles_column_in_library_order(library, tmp_path):
    """Every molecule, once, in the h5's own order, under a single `smiles` header."""
    home, smiles, _ = library
    out = tmp_path / "reference.csv"

    result = run_cli(["library", "-o", out])
    assert result.exit_code == 0, plain(result)

    header, written = _read(out)
    assert header == ["smiles"], f"expected a lone `smiles` header, got {header}"
    assert written == list(smiles), "the CSV must preserve the library's order exactly"


def test_library_matches_the_h5_it_read(library, tmp_path):
    """Read the source directly and compare, so the test does not just trust the fixture's copy."""
    import h5py

    home, _, _ = library
    out = tmp_path / "reference.csv"
    assert run_cli(["library", "-o", out]).exit_code == 0

    with h5py.File(home / "erl0_morgan.h5", "r") as handle:
        expected = list(handle["input"].asstr())

    _, written = _read(out)
    assert written == expected


def test_library_refuses_when_the_library_is_not_downloaded(tmp_path, monkeypatch):
    """Point at an empty home: the refusal must say to run `olinda setup`, not raise an OSError."""
    import olinda.data.fetch as fetch

    monkeypatch.setattr(fetch, "OLINDA_HOME", tmp_path / "empty")
    result = run_cli(["library", "-o", tmp_path / "out.csv"])

    assert result.exit_code != 0
    assert "olinda setup" in plain(result), plain(result)
    assert not (tmp_path / "out.csv").exists(), "nothing should be written on refusal"


def test_library_creates_missing_parent_directories(library, tmp_path):
    """`-o results/nested/reference.csv` should work without the caller pre-making the folders."""
    out = tmp_path / "results" / "nested" / "reference.csv"
    assert run_cli(["library", "-o", out]).exit_code == 0
    assert out.exists()


def test_library_handles_a_single_row_library(tmp_path, monkeypatch):
    """A one-compound library still gets a header and exactly one data row, not an off-by-one."""
    import olinda.data.fetch as fetch

    home = tmp_path / "tiny"
    smiles, _ = write_library(home, n_rows=1)
    monkeypatch.setattr(fetch, "OLINDA_HOME", home)

    out = tmp_path / "one.csv"
    assert run_cli(["library", "-o", out]).exit_code == 0

    header, written = _read(out)
    assert header == ["smiles"]
    assert written == list(smiles) and len(written) == 1
