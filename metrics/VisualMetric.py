import hashlib
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Any, Dict, List, Tuple
from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric


class VisualMetric(PerformanceMetric):
    def __init__(
        self,
        true_dir: str = "./results/true_images",
        reconstructed_dir: str = "./results/reconstructed_images",
    ):
        super().__init__("VisualMetric")
        self.true_dir = Path(true_dir)
        self.reconstructed_dir = Path(reconstructed_dir)
        self.tag = ""
        self.true_dir.mkdir(parents=True, exist_ok=True)
        self.reconstructed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_uint8_png_bytes(x_01: np.ndarray) -> bytes:
        arr = np.asarray(x_01, dtype=np.float64)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return tf.io.encode_png(tf.convert_to_tensor(arr)).numpy()

    @staticmethod
    def _as_batch(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            return arr[np.newaxis, ...]
        return arr

    @staticmethod
    def _decode_labels(value: Any) -> List[int]:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return [int(arr)]
        if arr.ndim == 1:
            return [int(np.argmax(arr))]
        return [int(label) for label in np.argmax(arr, axis=1)]

    @staticmethod
    def _safe_part(value: Any) -> str:
        value = str(value)
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_") or "value"

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _base_name(
        self,
        context: Dict[str, Any],
        sample_index: int,
        label: int,
        image: np.ndarray,
    ) -> str:
        img_hash = hashlib.md5(np.asarray(image, dtype=np.float32).tobytes()).hexdigest()[:8]
        parts = [
            self._safe_part(context.get("algorithm", "algorithm")),
            self._safe_part(context.get("attack", "attack")),
            f"run_{context.get('run', 0)}",
            f"round_{context.get('communication_round', 0)}",
            f"client_{context.get('client_id', 0)}",
            f"sample_{sample_index}",
            f"label_{label}",
            img_hash,
        ]
        return "__".join(parts)

    def measure(
        self,
        used_in_training_data: Tuple[np.ndarray, np.ndarray],
        attack: Attack,
        context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, str]]:
        context = context or {}
        results = []
        reconstructed = self._as_batch(attack.reconstructed_input)
        x, y = used_in_training_data
        x_np = self._as_batch(x)
        labels = self._decode_labels(y)

        for sample_index in range(min(len(x_np), len(reconstructed), len(labels))):
            label = labels[sample_index]
            base_name = self._base_name(context, sample_index, label, x_np[sample_index])
            true_path = self._unique_path(self.true_dir / f"actual__{base_name}.png")
            reconstructed_path = self._unique_path(self.reconstructed_dir / f"reconstructed__{base_name}.png")

            tf.io.write_file(str(true_path), self._to_uint8_png_bytes(x_np[sample_index]))
            tf.io.write_file(str(reconstructed_path), self._to_uint8_png_bytes(reconstructed[sample_index]))

            results.append({
                "actual_image_path": str(true_path),
                "reconstructed_image_path": str(reconstructed_path),
            })
        return results
