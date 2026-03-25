from typing import Dict, Literal, Any, List, Callable
import numpy as np
import tensorflow as tf

from models.model import Model
from clients.client import Client
from attacks.attack import Attack
from data.data import Data

SettingOptions = Literal[
    "communication_rounds", 
    "client_adaptation_rounds",
    "client_training_rounds", 
    "alpha", 
    "beta", 
    "client_training_batch_size", 
    "client_training_epochs",
    "loss_function", 
    "adaptation_optimizer",
    "meta_optimizer",
    "metrics",
    "local_training_approximation",
]
Settings = Dict[SettingOptions, Any]

class PerFedAvg:
    local_training_approximation_options = ["HF", "FO"]
    
    def __init__(self, model: Model, clients: List[Client], settings: Settings = {}):
        self.clients = clients
        self.init_data = Data().get_x_y(1, 1) # (x, y)
        self.model = model
        self.communication_rounds = settings["communication_rounds"] if settings.get("communication_rounds", False) else 1
        self.client_adaptation_rounds = settings["client_adaptation_rounds"] if settings.get("client_adaptation_rounds", False) else 1
        self.client_training_rounds = settings["client_training_rounds"] if settings.get("client_training_rounds", False) else 1
        self.alpha = settings["alpha"] if settings.get("alpha", False) else 0.1
        self.beta = settings["beta"] if settings.get("beta", False) else 0.1
        self.client_training_batch_size = settings["client_training_batch_size"] if settings.get("client_training_batch_size", False) else 1
        self.client_training_epochs = settings["client_training_epochs"] if settings.get("client_training_epochs", False) else 1
        self.loss_function = settings["loss_function"] if settings.get("loss_function", False) else tf.keras.losses.MeanSquaredError()
        self.adaptation_optimizer = settings["adaptation_optimizer"] if settings.get("adaptation_optimizer", False) else tf.keras.optimizers.SGD(learning_rate=0.1)
        self.meta_optimizer = settings["meta_optimizer"] if settings.get("meta_optimizer", False) else tf.keras.optimizers.SGD(learning_rate=0.1)
        self.metrics = settings["metrics"] if settings.get("metrics", False) else ["accuracy"]
        self.local_training_approximation = settings["local_training_approximation"] if settings.get("local_training_approximation", False) else "FO"

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

    def _HF_training_algorithm(self, model: Model, data: Data) -> Model:
        pass

    def _FO_training_algorithm(self, model: Model, data: Data) -> Model:
        for _ in range(self.client_training_rounds):
            for _ in range(self.client_adaptation_rounds):
                # adapatation step
                x, y = data.get_x_y(self.client_training_batch_size, 1)
                x = x[0]
                y = y[0]
                with tf.GradientTape() as tape:
                    y_pred = model.model(x, training=True)
                    loss = self.loss_function(y, y_pred)
                gradients = tape.gradient(loss, model.model.trainable_variables)
                new_weights = [weight - self.alpha * gradient for weight, gradient in zip(model.model.trainable_variables, gradients)]

            model.set_weights(new_weights)
            # meta-training step
            x, y = data.get_x_y(self.client_training_batch_size, 1)
            x = x[0]
            y = y[0]
            with tf.GradientTape() as tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            gradients = tape.gradient(loss, model.model.trainable_variables)
            new_weights = [weight - self.beta * gradient for weight, gradient in zip(model.model.trainable_variables, gradients)]
            model.set_weights(new_weights)
            
        return model

    def client_training_algorithm(self, model: Model, data: Data) -> Model:
        """
        Runs the local training algorithm given a model, data and settings.
        """
        if self.local_training_approximation == "HF":
            return self._HF_training_algorithm(model, data)
        elif self.local_training_approximation == "FO":
            return self._FO_training_algorithm(model, data)
        else:
            raise Exception("Invalid local training approximation")


    def run(self, attack:Attack = None):
        """
        Run the PerFedAvg algorithm.
        """

        self.model.model(self.init_data[0][0])

        for client in self.clients:
            client.set_training_algorithm(self.client_training_algorithm)
        
        for communication_round in range(self.communication_rounds):
            print(f"Communication round {communication_round + 1} of {self.communication_rounds}")

            for client in self.clients:
                client.set_model(Model.clone(self.model))
                client.train()
            print(f"Clients model training completed")
            
            # Apply attack to clients

            self.aggregate()
            print(f"Clients update aggregation completed")
            
