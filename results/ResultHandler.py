import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


class ResultHandler:
    fieldnames: Sequence[str] = ()

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_run_index = 0
        self._next_index = 0
        self._initialize_csv()

    def set_run_index(self, run_index: int) -> None:
        self.current_run_index = run_index

    def _initialize_csv(self) -> None:
        with self.csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def _append_row(self, row: Dict[str, Any]) -> None:
        row = {field: row.get(field, "") for field in self.fieldnames}
        row["index"] = self._next_index
        self._next_index += 1
        with self.csv_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)

    @staticmethod
    def _serialize(value: Any) -> Any:
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return value

    @classmethod
    def _normalize_metric_row(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        aliases = {
            "real_input_value": "real input value",
            "reconstructed_input_value": "reconstructed input value",
            "actual_label": "actual label",
            "reconstructed_label": "reconstructed label",
            "input_mse": "input mse",
            "input_psnr": "input psnr",
            "input_ssim": "input ssim",
            "actual_image_path": "actual image path",
            "reconstructed_image_path": "reconstructed image path",
            "visual_true_path": "actual image path",
            "visual_reconstructed_path": "reconstructed image path",
            "test_input": "test input",
            "test_label": "test label",
            "predicted_label": "predicted label",
            "output_crossentropy": "output crossentropy",
        }
        normalized = {}
        for key, value in row.items():
            normalized_key = aliases.get(key, key)
            normalized[normalized_key] = cls._serialize(value)
        return normalized

    @staticmethod
    def _measure(metric: Any, *args: Any, context: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        try:
            result = metric.measure(*args, context=context)
        except TypeError:
            result = metric.measure(*args)
        return list(result or [])

    @staticmethod
    def _merge_rows(metric_results: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        metric_results = list(metric_results)
        row_count = max((len(rows) for rows in metric_results), default=0)
        merged = []
        for row_idx in range(row_count):
            row: Dict[str, Any] = {}
            for rows in metric_results:
                if row_idx < len(rows):
                    row.update(rows[row_idx])
            merged.append(row)
        return merged


class AttackResultHandler(ResultHandler):
    fieldnames = [
        "index",
        "run",
        "communication_round",
        "client_id",
        "sample_index",
        "algorithm",
        "attack",
        "real input value",
        "reconstructed input value",
        "actual label",
        "reconstructed label",
        "input mse",
        "input psnr",
        "input ssim",
        "actual image path",
        "reconstructed image path",
    ]

    def __init__(
        self,
        metrics: Sequence[Any] | None = None,
        csv_path: str | Path = "results/attack_results/raw_results.csv",
        specific_folder: str | Path | None = None,
    ):
        self.metrics = list(metrics or [])
        csv_path = Path(csv_path)
        if specific_folder is not None:
            csv_path = csv_path.parent / specific_folder / csv_path.name
        super().__init__(csv_path)

    def append_attack_results(
        self,
        algorithm: Any,
        attack: Any,
        communication_round: int,
        client_id: int,
        used_in_training_data: Any,
    ) -> None:
        context = {
            "run": self.current_run_index,
            "communication_round": communication_round,
            "client_id": client_id,
            "algorithm": getattr(algorithm, "name", algorithm.__class__.__name__),
            "attack": getattr(attack, "name", attack.__class__.__name__),
        }
        metric_results = [
            self._measure(metric, used_in_training_data, attack, context=context)
            for metric in self.metrics
        ]

        for sample_index, metric_row in enumerate(self._merge_rows(metric_results)):
            row = {
                "run": self.current_run_index,
                "communication_round": communication_round,
                "client_id": client_id,
                "sample_index": sample_index,
                "algorithm": context["algorithm"],
                "attack": context["attack"],
            }
            row.update(self._normalize_metric_row(metric_row))
            self._append_row(row)


class AlgorithmResultHandler(ResultHandler):
    fieldnames = [
        "index",
        "run",
        "source_attack",
        "client_id",
        "evaluation_iteration",
        "sample_index",
        "algorithm",
        "test input",
        "test label",
        "predicted label",
        "output crossentropy",
    ]

    def __init__(
        self,
        metrics: Sequence[Any] | None = None,
        csv_path: str | Path = "results/algorithm_results/raw_results.csv",
        specific_folder: str | Path | None = None,
    ):
        self.metrics = list(metrics or [])
        csv_path = Path(csv_path)
        if specific_folder is not None:
            csv_path = csv_path.parent / specific_folder / csv_path.name
        super().__init__(csv_path)

    def _prepare_metric_context(self, clients: Sequence[Any], algorithm_option: str) -> Dict[str, Any]:
        for metric in self.metrics:
            prepare = getattr(metric, "prepare_evaluation_context", None)
            if callable(prepare):
                return dict(prepare(clients, algorithm_option))
        return {}

    def _model_context_rows(
        self,
        clients: Sequence[Any],
        metric_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        num_iterations = metric_context.get("num_iterations")
        batch_size = metric_context.get("batch_size")
        if num_iterations is None or batch_size is None:
            for metric in self.metrics:
                num_iterations = num_iterations or getattr(metric, "num_iterations", None)
                batch_size = batch_size or getattr(metric, "batch_size", None)
        if num_iterations is None or batch_size is None:
            return []

        rows = []
        for client in clients:
            for evaluation_iteration in range(int(num_iterations)):
                for sample_index in range(int(batch_size)):
                    rows.append({
                        "client_id": client.id,
                        "evaluation_iteration": evaluation_iteration,
                        "sample_index": sample_index,
                    })
        return rows

    def append_algorithm_results(
        self,
        algorithm: Any,
        clients: Sequence[Any],
        source_attack: Any = None,
    ) -> None:
        algorithm_option = getattr(algorithm, "model_metric_option")
        metric_context = self._prepare_metric_context(clients, algorithm_option)
        metric_results = [
            self._measure(metric, clients, algorithm_option, context=metric_context)
            for metric in self.metrics
        ]
        context_rows = self._model_context_rows(clients, metric_context)
        merged_rows = self._merge_rows(metric_results)

        for row_idx, metric_row in enumerate(merged_rows):
            context_row = context_rows[row_idx] if row_idx < len(context_rows) else {}
            row = {
                "run": self.current_run_index,
                "source_attack": getattr(source_attack, "name", "") if source_attack else "",
                "algorithm": getattr(algorithm, "name", algorithm.__class__.__name__),
                **context_row,
            }
            row.update(self._normalize_metric_row(metric_row))
            self._append_row(row)
