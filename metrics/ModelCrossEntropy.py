from metrics.ModelPerformanceMetric import ModelPerformanceMetric, AlgorithmOptions
from data.data import CIFAR10Data
from clients.client import Client
from models.model import Model

from typing import Dict, Any, List, Literal
import tensorflow as tf


SettingOptions = Literal[
    "num_iterations", # per client model accuracy
    "batch_size",
    "number_of_clients",
    "adaptation_alpha",
    "loss_function",
]
Settings = Dict[SettingOptions, Any]

class ModelCrossEntropy(ModelPerformanceMetric):

    def __init__(self, seed: int = 0, settings: Settings = {}):
        super().__init__("ModelCrossEntropy")
        self.data = CIFAR10Data(seed=seed, train=False)
        self.num_classes = len(self.data._CIFAR_10_CLASSES)
        self.num_iterations = settings.get("num_iterations", 1)
        self.batch_size = settings.get("batch_size", 10)
        self.adaptation_alpha = settings.get("adaptation_alpha", 0.1)
        self.loss_function = settings.get("loss_function", tf.keras.losses.CategoricalCrossentropy())
        self.test_x, self.test_y = self.data.get_structured_x_y(
            batch_size=self.batch_size,
            number_of_batches=self.num_classes*self.num_iterations,
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
    ) -> List[Dict[str, float]]:
        """
        Compute per-sample cross-entropy loss for the selected algorithm.
        """
        context = context or self.prepare_evaluation_context(clients, algorithm)
        match algorithm:
            case "fedAvg":
                return self.fedAvg_measure(clients, context)
            case "fedPer":
                return self.fedPer_measure(clients, context)
            case "perFedAvg":
                return self.perFedAvg_measure(clients, context)
            case _:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _test_batch_indicies(self, client: Client, iteration: int) -> List[int]:
        return [iteration * self.num_classes + label_class for label_class in client.get_label_classes()]

    def _evaluate_model_on_client_batches(self, model: Model, client: Client) -> List[Dict[str, float]]:
        results = []
        for iteration in range(self.num_iterations):
            batch_indicies = self._test_batch_indicies(client, iteration)
            x = tf.concat([self.test_x[i] for i in batch_indicies], axis=0)
            y = tf.concat([self.test_y[i] for i in batch_indicies], axis=0)
            y_pred = model.model(x, training=False)
            batch_loss = tf.keras.losses.categorical_crossentropy(y, y_pred)
            for value in batch_loss.numpy().tolist():
                results.append({"output_crossentropy": float(value)})

        if not results:
            raise ValueError("No test samples available for ModelCrossEntropy evaluation")
        return results

    def _measure_client_losses(
        self,
        clients: List[Client],
        context: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        if not clients:
            raise ValueError("ModelCrossEntropy requires at least one client")

        evaluated_models = context.get("models") or [client.get_model() for client in clients]
        results = []
        for model, client in zip(evaluated_models, clients):
            results.extend(self._evaluate_model_on_client_batches(model, client))
        return results

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


    def fedAvg_measure(
        self,
        clients: List[Client],
        context: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        """
        Output per-sample cross-entropy loss for FedAvg.
        """
        return self._measure_client_losses(clients, context)
    
    def fedPer_measure(
        self,
        clients: List[Client],
        context: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        """
        Output per-sample cross-entropy loss for FedPer.
        """
        return self._measure_client_losses(clients, context)
    
    def perFedAvg_measure(
        self,
        clients: List[Client],
        context: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        """
        Output per-sample cross-entropy loss after one Per-FedAvg adaptation step.
        """
        return self._measure_client_losses(clients, context)

        
