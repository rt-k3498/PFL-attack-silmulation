from typing import Dict, Literal, Any, List, Callable
import numpy as np
import tensorflow as tf
from rich import print

from models.model import Model
from clients.client import Client
from attacks.attack import Attack
from data.data import CIFAR10Data
from metrics.PerformanceMetric import PerformanceMetric
from metrics.ModelPerformanceMetric import ModelPerformanceMetric

SettingOptions = Literal[
    "communication_rounds",
    "client_training_rounds",
    "alpha",
    "client_training_batch_size",
    "client_training_epochs",
    "loss_function",
    "metrics",
]
Settings = Dict[SettingOptions, Any]

class FedAvg:

    def __init__(self, model: Model, clients: List[Client], seed: int, settings: Settings = {}):
        self.name = "FedAvg"
        self.model_metric_option = "fedAvg"
        self.clients = clients
        self.seed = seed
        self.model = model
        self.communication_rounds = settings.get("communication_rounds", 1)
        self.client_training_rounds = settings.get("client_training_rounds", 1)
        self.alpha = settings.get("alpha", 0.1)
        self.client_training_batch_size = settings.get("client_training_batch_size", 1)
        self.client_training_epochs = settings.get("client_training_epochs", 1)
        self.loss_function = settings.get("loss_function", tf.keras.losses.MeanSquaredError())
        self.metrics = settings.get("metrics", ["accuracy"])

    def aggregate(self) -> None:
        """
        Aggregate the client models into the global model.
        """
        weights = [client.get_weights() for client in self.clients]
        if not weights:
            return
        averaged = [np.mean(np.stack(ws, axis=0), axis=0) for ws in zip(*weights)]
        self.model.set_weights(averaged)

    def client_training_algorithm(self, model: Model, client: Client) -> Model:
        """
        Standard SGD local training: sample a batch, compute gradients, update weights.
        """
        for _ in range(self.client_training_rounds):
            x, y = client.get_sample()
            with tf.GradientTape() as tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            gradients = tape.gradient(loss, model.model.trainable_variables)
            new_weights = [
                weight - self.alpha * gradient
                for weight, gradient in zip(model.model.trainable_variables, gradients)
            ]
            model.set_weights(new_weights)

        return model

    def run(
        self,
        attack: Attack = None,
        attack_performance_metrics: List[PerformanceMetric] = None,
        model_performance_metrics: List[ModelPerformanceMetric] = None,
        result_handlers: List[Any] = None,
    ) -> list:
        """
        Run the FedAvg algorithm.
        """
        attack_results = []
        performance_results = []
        result_handlers = result_handlers or []
        attack_result_handlers = [
            handler for handler in result_handlers
            if hasattr(handler, "append_attack_results")
        ]
        algorithm_result_handlers = [
            handler for handler in result_handlers
            if hasattr(handler, "append_algorithm_results")
        ]

        for client in self.clients:
            client.set_training_algorithm(self.client_training_algorithm)

        for idx in range(self.communication_rounds):
            print(f"[yellow]Communication round {idx + 1}/{self.communication_rounds}[/yellow]")

            clients_data = {}

            print("[blue]Training clients...[/blue]")

            for client in self.clients:
                client.clear_training_data()
                client.set_model(self.model.clone())
                client.train()
                weights = client.get_weights()
                data = client.get_data_used_for_training()[-1]
                clients_data[client.id] = (weights, data)

            if attack:
                print(f"[blue]Running {attack.name} attack...[/blue]")
                for client_id in clients_data:
                    attack.run(self.model, clients_data[client_id][0], {"learning_rate": self.alpha, "num_classes": len(CIFAR10Data._CIFAR_10_CLASSES)})
                    for handler in attack_result_handlers:
                        handler.append_attack_results(
                            self,
                            attack,
                            idx + 1,
                            client_id,
                            clients_data[client_id][1],
                        )
                    if attack_performance_metrics:
                        for performance_metric in attack_performance_metrics:
                            result = performance_metric.measure(clients_data[client_id][1], attack)
                            attack_results.append({"client_id": client_id, "performance_metric": performance_metric.name, "result": result})

                print("[blue]Aggregating client models...[/blue]")

            self.aggregate()

        print("[green]FedAvg completed.[/green]")
        
        for client in self.clients:
            client.set_model(self.model.clone())

        for handler in algorithm_result_handlers:
            handler.append_algorithm_results(self, self.clients, source_attack=attack)
        
        if model_performance_metrics:
            print("[blue]Evaluating final trained global model...[/blue]")
            for performance_metric in model_performance_metrics:
                result = performance_metric.measure(self.clients, self.model_metric_option)
                performance_results.append({"performance_metric": performance_metric.name, "result": result})

        return {"attack_results": attack_results, "performance_results": performance_results}
