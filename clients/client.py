from __future__ import annotations
from typing import List, Dict, Literal, Any, Callable, Tuple
import tensorflow as tf
from models.model import Model
from data.data import Data

ClientTrainingFunction = Callable[[Model, "Client"], Model]

class Client:
    def __init__(self, id: str, data: Data, batch_size: int = 10):
        self.id = id
        x, y = data.get_x_y(batch_size, 1)
        self.data_x = x[0]
        self.data_y = y[0]
        self.used_in_training_data: List[Tuple[tf.Tensor, tf.Tensor]] = []
        self.model = None
        self.training_algorithm = None

    def sample(self, n: int) -> Tuple[tf.Tensor, tf.Tensor]:
        indices = tf.random.shuffle(tf.range(tf.shape(self.data_x)[0]))[:n]
        x = tf.gather(self.data_x, indices)
        y = tf.gather(self.data_y, indices)
        self.used_in_training_data.append((x, y))
        return x, y

    def get_data_used_for_training(self) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        return self.used_in_training_data

    def clear_training_data(self) -> None:
        self.used_in_training_data = []

    def set_model(self, model: Model) -> None:
        self.model = model

    def get_model(self) -> Model:
        if self.model is None:
            raise Exception("Model not set")
        return self.model

    def set_training_algorithm(self, training_algorithm: ClientTrainingFunction) -> None:
        self.training_algorithm = training_algorithm

    def train(self) -> None:
        if self.model is None:
            raise Exception("Model not set")
        if self.training_algorithm is None:
            raise Exception("Training algorithm not set")

        self.model = self.training_algorithm(self.model, self)