import numpy as np
from typing import Any, List, Dict, Tuple
from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric

class MSE_metric(PerformanceMetric):
    def __init__(self):
        super().__init__("MSE_metric")

    @staticmethod
    def _as_batch(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            return arr[np.newaxis, ...]
        return arr

    def measure(
        self,
        used_in_training_data: Tuple[np.ndarray, np.ndarray],
        attack: Attack,
        context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, float]]:
        reconstructed_input = self._as_batch(attack.reconstructed_input)
        x, _y = used_in_training_data
        actual_input = self._as_batch(x)

        results = []
        for sample_index in range(min(len(actual_input), len(reconstructed_input))):
            input_mse = float(np.mean((reconstructed_input[sample_index] - actual_input[sample_index]) ** 2))
            results.append({"input_mse": input_mse})
        return results
