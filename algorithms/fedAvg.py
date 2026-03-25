from typing import List, Dict, Literal, Any
from models.model import Model
from clients.client import Client
from attacks.attack import Attack
from data.data import Data

SettingOptions = Literal["num_communication_rounds", "num_client_SGD_rounds", "learning_rate", "client_training_batch_size", "client_training_epochs"]
Settings = Dict[SettingOptions, Any]

class FedAvg:
    def __init__(self, model: Model, clients: List[Client], settings: Settings):
        self.model = model
        self.clients = clients
        self.settings = settings

    def run(self):
        pass