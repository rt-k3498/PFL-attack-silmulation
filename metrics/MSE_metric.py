import numpy as np
from typing import List, Dict, Tuple
from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric

class MSE_metric(PerformanceMetric):
    def __init__(self):
        super().__init__("MSE_metric")

    def measure(self, used_in_training_data: List[Tuple[np.ndarray, np.ndarray]], attack: Attack) -> List[Dict[str, float]]:
        results = []
        reconstructed_input = np.asarray(attack.reconstructed_input, dtype=np.float32)
        for (x, _y) in used_in_training_data:
            actual_input = np.asarray(x, dtype=np.float32)
            input_mse = float(np.mean((reconstructed_input - actual_input) ** 2))
            results.append({"input_mse": input_mse})
        return results
