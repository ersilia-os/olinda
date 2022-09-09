"""Olinda."""

from importlib.metadata import PackageNotFoundError, version  # type: ignore
from warnings import filterwarnings

from olinda.distillation import distill  # noqa: F401

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

filterwarnings(action="ignore", category=DeprecationWarning, module="tensorboard")
