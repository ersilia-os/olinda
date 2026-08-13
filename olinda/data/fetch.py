"""Download reference data artefacts from public S3.

Single source of truth for the download URLs, local filenames, and home directory of the
data artefacts olinda fetches at setup time. Mirrors the lazyqsar convention: everything is
stored under ``~/.olinda`` and downloaded with a small ``urlretrieve``-based helper.

The reference-library Morgan descriptors (``erl0_morgan.h5``) live in the public ``eosvc-public``
bucket, so they download over plain HTTPS with no credentials and without eosvc installed.
"""

from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

from olinda.console import echo, success

# Local directory where all downloaded data artefacts are stored.
OLINDA_HOME = Path.home() / ".olinda"

_S3_BASE = "https://eosvc-public.s3.eu-central-1.amazonaws.com"
_S3_PREFIX = "olinda/data"

# Reference-library descriptors: Morgan count fingerprints (also producible locally with
# scripts/compute_morgan_fingerprints.py). This is the only representation olinda uses.
MORGAN_FINGERPRINTS_FILENAME = "erl0_morgan.h5"
MORGAN_FINGERPRINTS_URL = f"{_S3_BASE}/{_S3_PREFIX}/{MORGAN_FINGERPRINTS_FILENAME}"


def _safe_download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically, with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=dest.name) as bar:

            def _hook(blocks, block_size, total):
                if total and total > 0:
                    bar.total = total
                bar.update(blocks * block_size - bar.n)

            urlretrieve(url, tmp, reporthook=_hook)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def download_morgan_fingerprints(target_dir: str | None = None) -> Path | None:
    """Download the reference-library Morgan count-fingerprint HDF5 (``erl0_morgan.h5``) — best-effort.

    If it hasn't been published to S3 yet (or the download fails), this warns and returns ``None`` rather
    than raising, so ``olinda setup`` still succeeds; you can always generate it locally with
    ``scripts/compute_morgan_fingerprints.py``.
    """
    base_dir = Path(target_dir) if target_dir else OLINDA_HOME
    dest = base_dir / MORGAN_FINGERPRINTS_FILENAME
    if dest.exists():
        echo(f"already present · [dim]{dest}[/]", "info")
        return dest
    try:
        echo(f"downloading {MORGAN_FINGERPRINTS_FILENAME} → [dim]{dest}[/]", "run")
        _safe_download(MORGAN_FINGERPRINTS_URL, dest)
        success(f"reference library ready → [dim]{dest}[/]")
        return dest
    except Exception as exc:
        echo(
            f"reference library not downloaded ({type(exc).__name__}); "
            "generate locally with scripts/compute_morgan_fingerprints.py",
            "warning",
        )
        return None
