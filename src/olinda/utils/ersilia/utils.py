"""Utilities."""

from typing import Any, Callable
from ersilia import ErsiliaModel

def run_ersilia_api_in_context(model_id: str) -> Callable:
    """Utility function to execute Ersilia API.

    Args:
        model_id (str): Ersilia model hub ID.

    Returns:
        Callable: Util function.
    """

    def execute(x: Any) -> Any:       
        with ErsiliaModel(model_id) as em_api:
            tmp = em_api.run(x, output="pandas")
            ###WIP: Hardcoded for eos97yu but need general output adapter
            return tmp['logPe']
    return execute

