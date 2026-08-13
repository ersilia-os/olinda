"""The base install must be enough to run a distilled model — and to run it from the CLI.

``pip install olinda`` promises numpy, pandas, rdkit and onnxruntime plus the console shell, and
nothing else. One module-scope ``import h5py`` anywhere in ``olinda/__init__ → artifact → featurizer``
would break that, and the rest of the suite would never notice, because the dev environment has the
training stack installed.

So this file must stay importable and passable *without* the training extras: no h5py, no xgboost, no
matplotlib at module scope, and every check that inspects ``sys.modules`` runs in a subprocess — under
pytest the other test modules have already imported the world by collection time.
"""

from __future__ import annotations

import subprocess
import sys

# What a base install does not have. Reaching for any of these on the inference path is the bug.
# The console shell (click, rich, rich-click, loguru, tqdm) is deliberately absent from this list: it
# is part of the base install, because `olinda predict` needs a terminal and nothing else.
_EXTRA_ONLY = (
    "lightgbm",
    "xgboost",
    "h5py",
    "lazyqsar",
    "onnx",
    "onnxmltools",
    "optuna",
    "stylia",
    "matplotlib",
)


def _python(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_the_public_api_pulls_in_nothing_heavy():
    code = (
        "import sys;"
        "from olinda import OlindaArtifact, RDKitVersionMismatch;"
        f"bad=[m for m in {_EXTRA_ONLY!r} if m in sys.modules];"
        "print(','.join(bad))"
    )
    assert _python(code) == "", "importing olinda reached for an extras-only module"


def test_featurizing_needs_only_rdkit_and_numpy():
    """Featurization is the one heavy thing inference does in Python — it must stay on the light path."""
    code = (
        "import sys;"
        "from olinda.featurizer import MorganCountFeaturizer;"
        "x = MorganCountFeaturizer().transform(['CCO', 'c1ccccc1']);"
        f"bad=[m for m in {_EXTRA_ONLY!r} if m in sys.modules];"
        "print(f'{x.shape[0]}x{x.shape[1]}|' + ','.join(bad))"
    )
    assert _python(code) == "2x2048|"


def test_the_cli_starts_without_any_extra():
    """`olinda --help` is the first thing anyone runs; on a base install it must render, not refuse.

    It used to refuse: the console shell lived in the [report] extra while the console script was
    installed unconditionally, so `pip install olinda` shipped an `olinda` command that could not
    start. Importing the CLI is the whole check — it is what was failing.
    """
    code = (
        "import sys;"
        "from olinda.cli import cli;"
        f"bad=[m for m in {_EXTRA_ONLY!r} if m in sys.modules];"
        "print(','.join(bad))"
    )
    assert _python(code) == "", "importing the CLI reached for an extras-only module"


def test_a_distilling_command_names_the_extra_it_needs():
    """With the CLI in base, `olinda fit` is reachable without the training stack — so it must explain.

    The guard used to live at the console-script entry point, where it caught the missing CLI itself.
    Now that the CLI always imports, the refusal has to come from the commands, or a user on a base
    install gets a bare `ModuleNotFoundError: h5py` from somewhere deep in a run.
    """
    from olinda.train import _TRAIN_MODULES, require_train_extra

    code = (
        "import sys;"
        # Poison every training module so the guard fires in an environment that actually has them.
        f"sys.modules.update({{m: None for m, _ in {list(_TRAIN_MODULES)!r}}});"
        "from olinda.cli import cli;"
        "from click.testing import CliRunner;"
        "r = CliRunner().invoke(cli, ['fit', '-s', 'x.csv', '-m', 'y.onnx']);"
        "print(r.exit_code);"
        "print(r.output)"
    )
    out = _python(code)
    assert out.splitlines()[0] != "0", "fit must refuse without the training extra"
    assert "olinda[train]" in out, (
        f"the refusal must name the extra to install; got:\n{out}"
    )
    assert "Traceback" not in out

    # And the guard tells the truth about whatever environment it is in. This file runs in both: the
    # `test` job has the training stack, where it must stay quiet, and `inference-install` does not,
    # where it must raise and name the extra. Asserting either one unconditionally fails the other job
    # — which it did, on `main`, for as long as this line read `require_train_extra()`.
    try:
        require_train_extra()
    except RuntimeError as exc:
        assert "olinda[train]" in str(exc), (
            f"the guard raised without naming the extra: {exc}"
        )
