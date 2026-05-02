from metrics.PerformanceMetric import PerformanceMetric
from clients.client import Client

from typing import Any, Dict, List, Literal

AlgorithmOptions = Literal[
    "fedAvg",
    "fedPer",
    "perFedAvg",
]

class ModelPerformanceMetric(PerformanceMetric):
    """
    A class representing a model performance metric.
    """
    def __init__(self, name: str):
        self.name = name

    def measure(
        self,
        clients: List[Client],
        algorithm: AlgorithmOptions,
        context: Dict[str, Any] | None = None,
    ) -> Any:
        """
        Compute the performance metric for the given clients and algorithm.
        """
        raise NotImplementedError("Subclasses should implement this method.")
