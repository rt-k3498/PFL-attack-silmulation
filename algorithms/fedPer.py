from typing import Dict, Literal, Any, List, Callable
import numpy as np
import tensorflow as tf

from models.model import Model
from clients.client import Client
from attacks.attack import Attack
from data.data import Data

SettingOptions = Literal[
    "communication_rounds", 
    "client_training_rounds", 
    "alpha",
    "K_p", # number of model personalized layers
    "K_b", # number of model base layers
    "client_training_batch_size", 
    "client_training_epochs",
    "loss_function",
    "metrics",
]
Settings = Dict[SettingOptions, Any]

class FedPer:

    def __init__(self, model: Model, clients: List[Client], settings: Settings = {}):
        self.clients = clients
        self.init_data = Data().get_x_y(1, 1) # (x, y)
        self.model = model
        self.K_p = settings.get("K_p", 1)
        self.K_b = settings.get("K_b", len(self.model.layers) - self.K_p)
        self.communication_rounds = settings.get("communication_rounds", 1)
        self.client_training_rounds = settings.get("client_training_rounds", 1)
        self.alpha = settings.get("alpha", 0.1)
        self.client_training_batch_size = settings.get("client_training_batch_size", 1)
        self.client_training_epochs = settings.get("client_training_epochs", 1)
        self.loss_function = settings.get("loss_function", tf.keras.losses.MeanSquaredError())
        self.metrics = settings.get("metrics", ["accuracy"])


    def run(self):
        pass