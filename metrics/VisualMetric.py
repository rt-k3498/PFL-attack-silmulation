import os
import hashlib
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric


class VisualMetric(PerformanceMetric):
    def __init__(
        self,
        true_dir: str = "./results/true_images",
        reconstructed_dir: str = "./results/reconstructed_images",
    ):
        super().__init__("VisualMetric")
        self.true_dir = true_dir
        self.reconstructed_dir = reconstructed_dir
        self.tag = ""
        os.makedirs(self.true_dir, exist_ok=True)
        os.makedirs(self.reconstructed_dir, exist_ok=True)

    @staticmethod
    def _to_uint8_png_bytes(x_01: np.ndarray) -> bytes:
        arr = np.asarray(x_01, dtype=np.float64)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return tf.io.encode_png(tf.convert_to_tensor(arr)).numpy()

    def measure(
        self,
        used_in_training_data: List[Tuple[np.ndarray, np.ndarray]],
        attack: Attack,
    ) -> List[Dict[str, str]]:
        results = []
        reconstructed = np.asarray(attack.reconstructed_input, dtype=np.float32)
        tag = self.tag or "untagged"
        for idx, (x, _y) in enumerate(used_in_training_data):
            x_np = np.asarray(x, dtype=np.float32)
            img_hash = hashlib.md5(x_np.tobytes()).hexdigest()[:8]

            true_path = os.path.join(self.true_dir, f"image_{img_hash}.png")
            if not os.path.exists(true_path):
                tf.io.write_file(true_path, self._to_uint8_png_bytes(x_np))

            recon_path = os.path.join(
                self.reconstructed_dir,
                f"{tag}__image_{img_hash}__s{idx}.png",
            )
            tf.io.write_file(recon_path, self._to_uint8_png_bytes(reconstructed))

            results.append({
                "visual_true_path": true_path,
                "visual_reconstructed_path": recon_path,
            })
        return results
