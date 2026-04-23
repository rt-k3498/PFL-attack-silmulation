from typing import Dict, Literal, Any, List, Callable
import numpy as np
import tensorflow as tf

from models.model import Model
from clients.client import Client
from attacks.attack import Attack
from data.data import CIFAR10Data
from metrics.PerformanceMetric import PerformanceMetric

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
        self.clients = clients
        self.init_data = CIFAR10Data(seed=seed).get_x_y(1, 1)
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
        weights = [client.get_model().get_weights() for client in self.clients]
        if not weights:
            return
        averaged = [np.mean(np.stack(ws, axis=0), axis=0) for ws in zip(*weights)]
        self.model.set_weights(averaged)

    def client_training_algorithm(self, model: Model, client: Client) -> Model:
        """
        Standard SGD local training: sample a batch, compute gradients, update weights.
        """
        for _ in range(self.client_training_rounds):
            x, y = client.sample(self.client_training_batch_size)
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

    def run(self, attack: Attack = None, performance_metrics: List[PerformanceMetric] = None) -> list:
        """
        Run the FedAvg algorithm.
        """
        self.model.model(self.init_data[0][0])
        results = []

        for client in self.clients:
            client.set_training_algorithm(self.client_training_algorithm)

        for communication_round in range(self.communication_rounds):

            clients_data = {}

            for client in self.clients:
                client.clear_training_data()
                client.set_model(self.model.clone())
                client.train()
                weights = client.get_weights()
                data = client.get_data_used_for_training()
                clients_data[client.id] = (weights, data)

            if attack:
                for client_id in clients_data:
                    attack.run(self.model, clients_data[client_id][0], {"learning_rate": self.alpha, "num_classes": len(CIFAR10Data._CIFAR_10_CLASSES)})
                    if performance_metrics:
                        for performance_metric in performance_metrics:
                            result = performance_metric.measure(clients_data[client_id][1], attack)
                            results.append({"client_id": client_id, "performance_metric": performance_metric.name, "result": result})

            self.aggregate()

        return results
