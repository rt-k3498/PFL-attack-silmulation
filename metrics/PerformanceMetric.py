from typing import Any, Dict, Tuple
import numpy as np
from attacks.attack import Attack

class PerformanceMetric:
    def __init__(self, name: str):
        self.name = name

    def measure(
        self,
        used_in_training_data: Tuple[np.ndarray, np.ndarray],
        attack: Attack,
        context: Dict[str, Any] | None = None,
    ): 
        raise NotImplementedError("Subclasses must implement this method")
