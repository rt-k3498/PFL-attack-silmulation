import numpy as np
import tensorflow as tf
from typing import Any, List, Dict, Tuple
from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric

class SSIM_metric(PerformanceMetric):
    def __init__(self, max_val: float = 1.0, filter_size: int = 11):
        super().__init__("SSIM_metric")
        self.max_val = max_val
        self.filter_size = filter_size

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
            ssim = tf.image.ssim(
                tf.convert_to_tensor(reconstructed_input[sample_index:sample_index + 1], dtype=tf.float32),
                tf.convert_to_tensor(actual_input[sample_index:sample_index + 1], dtype=tf.float32),
                max_val=self.max_val,
                filter_size=self.filter_size,
            )
            results.append({"input_ssim": float(tf.reduce_mean(ssim).numpy())})
        return results
