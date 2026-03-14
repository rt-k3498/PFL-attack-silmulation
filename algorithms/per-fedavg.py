from typing import Dict, Literal, Any, List

from models.model import Model
from client import Client
from attacks.attack import Attack

SettingOptions = Literal["num_communication_rounds", "num_adaptation_rounds", "num_SGD_rounds", "loss_function"]
Settings = Dict[SettingOptions, Any]

class PerFedAvg:
    counter = 0
    
    def __init__(self, model: Model, clients: List[Client], settings: Settings):
        if PerFedAvg.counter > 0:
            raise Exception("PerFedAvg can only be initialized once")
        PerFedAvg.counter += 1
        self.model = model
        self.clients = clients
        self.num_communication_rounds = settings["num_communication_rounds"] if settings.get("num_communication_rounds", False) else 5
        self.num_adaptation_rounds = settings["num_adaptation_rounds"] if settings.get("num_adaptation_rounds", False) else 1
        self.num_SGD_rounds = settings["num_SGD_rounds"] if settings.get("num_SGD_rounds", False) else 5
        self.loss_function = settings["loss_function"] if settings.get("loss_function", False) else PerFedAvg.loss_function()

    @staticmethod
    def loss_function():
        """
        Loss function for the PerFedAvg algorithm.
        """


    def run(self, attack:Attack):
        """
        Run the PerFedAvg algorithm.
        """
        self.model.set_initial_weights()
        
        for communication_round in range(self.num_communication_rounds):
            print(f"Communication round {communication_round + 1} of {self.num_communication_rounds}")

            map(lambda client: client.train(Model.clone(self.model), {"num_adaptation_rounds": self.num_adaptation_rounds, "num_SGD_rounds": self.num_SGD_rounds, "loss_function": self.loss_function}), self.clients)
            print(f"Clients models trained")
            
            # Apply attack to clients

            new_model = Model.aggregate_weights(self.clients)
            print(f"Weights aggregated")
            
            self.model = new_model


