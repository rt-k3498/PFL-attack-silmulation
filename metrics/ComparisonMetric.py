import numpy as np
from typing import Any, Dict, List, Tuple

from attacks.attack import Attack
from metrics.PerformanceMetric import PerformanceMetric


class ComparisonMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("ComparisonMetric")

    @staticmethod
    def _as_batch(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.reshape((1,))
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

    def measure(
        self,
        used_in_training_data: Tuple[np.ndarray, np.ndarray],
        attack: Attack,
        context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        actual_input, actual_label = used_in_training_data
        actual_input = self._as_batch(actual_input)
        reconstructed_input = self._as_batch(attack.reconstructed_input)
        actual_labels = self._decode_labels(actual_label)
        reconstructed_labels = self._decode_labels(attack.reconstructed_label)

        row_count = min(
            len(actual_input),
            len(reconstructed_input),
            len(actual_labels),
            len(reconstructed_labels),
        )
        return [
            {
                "real_input_value": actual_input[idx],
                "reconstructed_input_value": reconstructed_input[idx],
                "actual_label": actual_labels[idx],
                "reconstructed_label": reconstructed_labels[idx],
            }
            for idx in range(row_count)
        ]
