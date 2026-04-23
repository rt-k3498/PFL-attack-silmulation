import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric

class PSNR_metric(PerformanceMetric):
    def __init__(self, max_val: float = 1.0):
        super().__init__("PSNR_metric")
        self.max_val = max_val

    def measure(self, used_in_training_data: List[Tuple[np.ndarray, np.ndarray]], attack: Attack) -> List[Dict[str, float]]:
        results = []    
        reconstructed_input = tf.convert_to_tensor(attack.reconstructed_input, dtype=tf.float32)
        for (x, _y) in used_in_training_data:
            actual_input = tf.convert_to_tensor(x, dtype=tf.float32)
            psnr = tf.image.psnr(reconstructed_input, actual_input, max_val=self.max_val)
            input_psnr = float(tf.reduce_mean(psnr).numpy())
            results.append({"input_psnr": input_psnr})
        return results
