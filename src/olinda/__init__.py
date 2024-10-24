"""Olinda."""

from importlib.metadata import PackageNotFoundError, version  # type: ignore
from warnings import filterwarnings

from olinda import distillation  # noqa: F401

# Version
__version__ = "0.3.0"

filterwarnings(action="ignore", category=DeprecationWarning, module="tensorboard")
