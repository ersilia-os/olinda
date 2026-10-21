"""The base install must be enough to run a distilled model, and nothing more.

``pip install olinda`` promises numpy, onnxruntime and wasmtime. That is the whole inference path:
SMILES are featurized by RDKit compiled to WebAssembly, so there is no rdkit, no pandas, and no CLI
stack. One module-scope ``import pandas`` anywhere in ``olinda/__init__ → artifact → featurizer``
would break that, and the rest of the suite would never notice, because the dev environment has every
extra installed.

So this file must stay importable and passable *without* the extras, and every check that inspects
``sys.modules`` runs in a subprocess — under pytest the other test modules have already imported the
world by collection time. CI runs it on a base install too (``inference-install``), with a model built
in a separate training environment and passed in through ``OLINDA_FIXTURE_MODEL``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# What a base install does not have. Reaching for any of these on the inference path is the bug.
_CLI = ("click", "rich_click", "rich", "loguru", "tqdm")
_TRAIN_AND_REPORT = (
    "rdkit",
    "pandas",
    "pyarrow",
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
_EXTRA_ONLY = _CLI + _TRAIN_AND_REPORT


def _python(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _leaked(modules) -> str:
    return (
        f"bad=[m for m in {tuple(modules)!r} if m in sys.modules];print(','.join(bad))"
    )


def test_importing_the_public_api_pulls_in_nothing_from_the_extras():
    code = "import sys;from olinda import OlindaArtifact, GoldenMismatch;" + _leaked(
        _EXTRA_ONLY
    )
    assert _python(code) == "", "importing olinda reached for an extras-only module"


def test_featurizing_does_not_import_rdkit():
    """Featurization is RDKit compiled to WebAssembly: the rdkit package must never be touched."""
    code = (
        "import sys;"
        "from olinda.featurizer import QmfpFeaturizer;"
        "b, v = QmfpFeaturizer().featurize_bytes(['CCO', 'c1ccccc1', 'not_a_smiles']);"
        "print(f'{b.shape[0]}x{b.shape[1]}|{v.tolist()}|', end='');"
        + _leaked(_EXTRA_ONLY)
    )
    assert _python(code) == "3x1024|[True, True, False]|"


@pytest.fixture
def model_path(request):
    """A model.onnx: the one CI built elsewhere, or the session's own (which needs the training stack)."""
    given = os.environ.get("OLINDA_FIXTURE_MODEL")
    if given:
        return given
    for module in ("xgboost", "h5py", "onnx"):
        pytest.importorskip(module)
    return str(request.getfixturevalue("artifact"))


def test_running_a_model_imports_nothing_from_the_extras(model_path):
    """Load, golden-check and score: the whole inference path, on numpy, onnxruntime and wasmtime."""
    code = (
        "import sys;"
        "from olinda import OlindaArtifact;"
        f"m = OlindaArtifact({model_path!r});"
        "y = m.run(['CCO', 'c1ccccc1']);"
        "print(f\"{y.shape[0]}|{m.describe()['golden']}|\", end='');"
        + _leaked(_EXTRA_ONLY)
    )
    assert _python(code) == "2|pass|"


def test_without_the_cli_extra_the_command_names_it():
    """`pip install olinda` installs an `olinda` script it cannot run; it must say what to install.

    Entry points cannot be optional, so on a base install the script exists and the CLI does not. The
    answer has to come from the entry point itself, in plain stdlib, rather than as a traceback.
    """
    code = (
        "import sys;"
        # Poison the CLI stack so this holds in an environment that actually has it.
        f"sys.modules.update({{m: None for m in {_CLI!r}}});"
        "from olinda._entry import main;"
        "\ntry:\n    main()\nexcept SystemExit as e:\n    print(e.code)"
    )
    out = _python(code)
    assert 'pip install "olinda[cli]"' in out
    assert "Traceback" not in out


def test_the_cli_pulls_in_nothing_from_training_or_reporting():
    """`olinda --help` on a [cli] install must not reach for the training or reporting stacks."""
    pytest.importorskip("rich_click")
    code = "import sys;from olinda.cli import cli;" + _leaked(_TRAIN_AND_REPORT)
    assert _python(code) == "", "importing the CLI reached for an extras-only module"


def test_a_distilling_command_names_the_extra_it_needs():
    """On a [cli] install `olinda fit` is reachable without the training stack, so it must explain.

    Otherwise a user gets a bare `ModuleNotFoundError: h5py` from somewhere deep in a run.
    """
    pytest.importorskip("rich_click")
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

    # And the guard tells the truth about whatever environment it is in: quiet with the training stack,
    # raising and naming the extra without it. Asserting either one unconditionally fails the other job.
    try:
        require_train_extra()
    except RuntimeError as exc:
        assert "olinda[train]" in str(exc), (
            f"the guard raised without naming the extra: {exc}"
        )
