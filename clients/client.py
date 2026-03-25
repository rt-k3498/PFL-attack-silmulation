from typing import List, Dict, Literal, Any, Callable
from models.model import Model
from data.data import Data

ClientTrainingFunction = Callable[[Model, Data], Model]

class Client:
    def __init__(self, id: str, data: Data):
        self.id = id
        self.data = data
        self.model = None
        self.training_algorithm = None

    def set_model(self, model: Model) -> None:
        self.model = model

    def get_model(self) -> Model:
        if self.model is None:
            raise Exception("Model not set")
        return self.model

    def set_training_algorithm(self, training_algorithm: ClientTrainingFunction) -> None:
        self.training_algorithm = training_algorithm
    
    def get_data(self) -> Data:
        return self.data

    def train(self) -> None:
        if self.model is None:
            raise Exception("Model not set")
        if self.training_algorithm is None:
            raise Exception("Training algorithm not set")

        self.model = self.training_algorithm(self.model, self.data)