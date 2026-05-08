from typing import Any, Dict, List, Literal

import numpy as np
import tensorflow as tf

from clients.client import Client
from data.data import CIFAR10Data
from metrics.ModelPerformanceMetric import AlgorithmOptions, ModelPerformanceMetric
from models.model import Model


SettingOptions = Literal[
    "num_iterations",
    "batch_size",
    "number_of_clients",
    "adaptation_alpha",
    "loss_function",
]
Settings = Dict[SettingOptions, Any]


class PredictionComparison(ModelPerformanceMetric):
    def __init__(self, seed: int = 0, settings: Settings = {}):
        super().__init__("PredictionComparison")
        self.data = CIFAR10Data(seed=seed, train=False)
        self.num_classes = len(self.data._CIFAR_10_CLASSES)
        self.num_iterations = settings.get("num_iterations", 1)
        self.batch_size = settings.get("batch_size", 10)
        self.adaptation_alpha = settings.get("adaptation_alpha", 0.1)
        self.loss_function = settings.get("loss_function", tf.keras.losses.CategoricalCrossentropy())
        self.test_x, self.test_y = self.data.get_structured_x_y(
            batch_size=self.batch_size,
            number_of_batches=self.num_classes * self.num_iterations,
        )

    def prepare_evaluation_context(
        self,
        clients: List[Client],
        algorithm: AlgorithmOptions,
    ) -> Dict[str, Any]:
        match algorithm:
            case "fedAvg" | "fedPer":
                models = [client.get_model() for client in clients]
            case "perFedAvg":
                models = [self._adapt_per_fed_avg_model(client) for client in clients]
            case _:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        return {
            "models": models,
            "num_iterations": self.num_iterations,
            "batch_size": self.batch_size,
        }

    def measure(
        self,
        clients: List[Client],
        algorithm: AlgorithmOptions,
        context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        context = context or self.prepare_evaluation_context(clients, algorithm)
        match algorithm:
            case "fedAvg" | "fedPer" | "perFedAvg":
                return self._measure_predictions(clients, context)
            case _:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _test_batch_indicies(self, client: Client, iteration: int) -> List[int]:
        return [iteration * self.num_classes + label_class for label_class in client.get_label_classes()]

    @staticmethod
    def _decode_labels(value: Any) -> List[int]:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return [int(arr)]
        if arr.ndim == 1:
            return [int(np.argmax(arr))]
        return [int(label) for label in np.argmax(arr, axis=1)]

    def _adapt_per_fed_avg_model(self, client: Client) -> Model:
        model = client.get_model().clone()
        x, y = client.random_samples(1)

        with tf.GradientTape() as tape:
            y_pred = model.model(x, training=True)
            loss = self.loss_function(y, y_pred)
        gradients = tape.gradient(loss, model.model.trainable_variables)
        updated_weights = [
            weight if gradient is None else weight - self.adaptation_alpha * gradient
            for weight, gradient in zip(model.model.trainable_variables, gradients)
        ]
        model.set_weights(updated_weights)
        return model

    def _measure_predictions(
        self,
        clients: List[Client],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        models = context.get("models") or [client.get_model() for client in clients]
        results = []
        for model, client in zip(models, clients):
            for iteration in range(self.num_iterations):
                batch_indices = self._test_batch_indicies(client, iteration)
                x = tf.concat([self.test_x[i] for i in batch_indices], axis=0)
                y = tf.concat([self.test_y[i] for i in batch_indices], axis=0)
                y_pred = model.model(x, training=False)
                test_labels = self._decode_labels(y)
                predicted_labels = self._decode_labels(y_pred)

                x_np = np.asarray(x, dtype=np.float32)
                for sample_index in range(min(len(x_np), len(test_labels), len(predicted_labels))):
                    results.append({
                        "test_input": x_np[sample_index],
                        "test_label": test_labels[sample_index],
                        "predicted_label": predicted_labels[sample_index],
                    })
        return results
