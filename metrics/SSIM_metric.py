import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric

class SSIM_metric(PerformanceMetric):
    def __init__(self, max_val: float = 1.0, filter_size: int = 11):
        super().__init__("SSIM_metric")
        self.max_val = max_val
        self.filter_size = filter_size

    def measure(self, used_in_training_data: List[Tuple[np.ndarray, np.ndarray]], attack: Attack) -> List[Dict[str, float]]:
        results = []
        reconstructed_input = tf.convert_to_tensor(attack.reconstructed_input, dtype=tf.float32)
        for (x, _y) in used_in_training_data:
            actual_input = tf.convert_to_tensor(x, dtype=tf.float32)
            ssim = tf.image.ssim(
                reconstructed_input,
                actual_input,
                max_val=self.max_val,
                filter_size=self.filter_size,
            )
            input_ssim = float(tf.reduce_mean(ssim).numpy())
            results.append({"input_ssim": input_ssim})
        return results
