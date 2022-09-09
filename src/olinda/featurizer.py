"""Featurizer for SMILES."""

import joblib

from olinda.utils import get_package_root_path


class Featurizer:
    def __init__(self: "Featurizer", mode: str) -> None:
        """Init

        Args:
            mode (str): Type of featurizer to use.
        """
        if mode == "flat2grid":
            self.transformer = joblib.load(get_package_root_path() / "flat2grid.joblib")
            self.name = "flat2grid"
