from typing import Dict, Literal, Any, List, Callable
import numpy as np
import tensorflow as tf

from models.model import Model
from clients.client import Client
from attacks.attack import Attack
from data.data import Data  # used in __init__ for init_data

SettingOptions = Literal[
    "communication_rounds", 
    "client_adaptation_rounds",
    "client_training_rounds", 
    "alpha", 
    "beta", 
    "client_training_batch_size", 
    "client_training_epochs",
    "loss_function", 
    "metrics",
    "local_training_approximation",
    "reuse_data_batches",
]
Settings = Dict[SettingOptions, Any]

class PerFedAvg:
    local_training_approximation_options = ["HF", "FO"]
    
    def __init__(self, model: Model, clients: List[Client], settings: Settings = {}):
        self.clients = clients
        self.init_data = Data().get_x_y(1, 1) # (x, y)
        self.model = model
        self.communication_rounds = settings.get("communication_rounds", 1)
        self.client_adaptation_rounds = settings.get("client_adaptation_rounds", 1)
        self.client_training_rounds = settings.get("client_training_rounds", 1)
        self.alpha = settings.get("alpha", 0.1)
        self.beta = settings.get("beta", 0.1)
        self.client_training_batch_size = settings.get("client_training_batch_size", 1)
        self.client_training_epochs = settings.get("client_training_epochs", 0)
        self.loss_function = settings.get("loss_function", tf.keras.losses.MeanSquaredError())
        self.metrics = settings.get("metrics", ["accuracy"])
        self.local_training_approximation = settings.get("local_training_approximation", "FO")
        self.reuse_data_batches = settings.get("reuse_data_batches", False)

        if self.local_training_approximation not in self.local_training_approximation_options:
            raise Exception("Invalid local training approximation")


    def aggregate(self) -> None:
        """
        Aggregate the client models into the global model.
        """
        weights = [client.get_model().get_weights() for client in self.clients]
        if not weights:
            return
        averaged = [np.mean(np.stack(ws, axis=0), axis=0) for ws in zip(*weights)]
        self.model.set_weights(averaged)

    def _HVP_training_algorithm(self, model: Model, client: Client) -> Model:
        pass

    def _HF_training_algorithm(self, model: Model, client: Client) -> Model:
        pass

    def _FO_training_algorithm(self, model: Model, client: Client) -> Model:
        if self.reuse_data_batches:
            for _ in range(self.client_training_rounds):
                original_weights = model.get_weights()
                x, y = client.sample(self.client_training_batch_size)
                for _ in range(self.client_adaptation_rounds):
                    with tf.GradientTape() as tape:
                        y_pred = model.model(x, training=True)
                        loss = self.loss_function(y, y_pred)
                    gradients = tape.gradient(loss, model.model.trainable_variables)
                    new_weights = [weight - self.alpha * gradient for weight, gradient in zip(model.model.trainable_variables, gradients)]
                    model.set_weights(new_weights)

                with tf.GradientTape() as tape:
                    y_pred = model.model(x, training=True)
                    loss = self.loss_function(y, y_pred)
                gradients = tape.gradient(loss, model.model.trainable_variables)
                meta_weights = [w_orig - self.beta * gradient for w_orig, gradient in zip(original_weights, gradients)]
                model.set_weights(meta_weights)
        else:
            for _ in range(self.client_training_rounds):
                original_weights = model.get_weights()
                for _ in range(self.client_adaptation_rounds):
                    x, y = client.sample(self.client_training_batch_size)
                    with tf.GradientTape() as tape:
                        y_pred = model.model(x, training=True)
                        loss = self.loss_function(y, y_pred)
                    gradients = tape.gradient(loss, model.model.trainable_variables)
                    new_weights = [weight - self.alpha * gradient for weight, gradient in zip(model.model.trainable_variables, gradients)]
                    model.set_weights(new_weights)

                x, y = client.sample(self.client_training_batch_size)
                with tf.GradientTape() as tape:
                    y_pred = model.model(x, training=True)
                    loss = self.loss_function(y, y_pred)
                gradients = tape.gradient(loss, model.model.trainable_variables)
                meta_weights = [w_orig - self.beta * gradient for w_orig, gradient in zip(original_weights, gradients)]
                model.set_weights(meta_weights)

        return model

    def client_training_algorithm(self, model: Model, client: Client) -> Model:
        """
        Runs the local training algorithm given a model, data and settings.
        """
        if self.local_training_approximation == "HF":
            return self._HF_training_algorithm(model, client)
        elif self.local_training_approximation == "FO":
            return self._FO_training_algorithm(model, client)
        else:
            raise Exception("Invalid local training approximation")


    def run(self, attack: Attack = None, performance_metrics: List[Callable] = None) -> list:
        """
        Run the PerFedAvg algorithm.
        """

        self.model.model(self.init_data[0][0])
        results = []

        for client in self.clients:
            client.set_training_algorithm(self.client_training_algorithm)

        for communication_round in range(self.communication_rounds):
            print(f"Communication round {communication_round + 1} of {self.communication_rounds}")

            for client in self.clients:
                client.clear_training_data()
                client.set_model(Model.clone(self.model))
                client.train()
            print(f"Clients model training completed")

            if attack:
                for client in self.clients:
                    attack.run(self.model, client.get_model(), {"learning_rate": self.beta})
                    if performance_metrics:
                        for performance_metric in performance_metrics:
                            result = performance_metric(client.get_data_used_for_training(), attack)
                            results.append(result)

            self.aggregate()
            print(f"Clients update aggregation completed")

        return results
