from __future__ import annotations
from typing import List, Dict, Literal, Any, Callable, Tuple
import tensorflow as tf
from models.model import Model
from data.data import CIFAR10Data
import numpy as np

ClientTrainingFunction = Callable[[Model, "Client"], Model]

class Client:
    def __init__(self, id: str, data: CIFAR10Data, seed: int, batch_size: int = 10):
        self.id = id
        self.seed = seed
        x, y = data.get_x_y(batch_size, 1)
        self.data_x = x[0]
        self.data_y = y[0]
        self.used_in_training_data: List[Tuple[tf.Tensor, tf.Tensor]] = []
        self.model = None
        self.training_algorithm = None
        self.send_partial_layers: bool = False
        self.send_first_n_layers: int | None = None
        self._send_first_n_layers: int | None = None
        self.store_last_n_layers: int | None = None
        self._store_last_n_layers: int | None = None
        self._last_n_layer_weights: List[type(np.array)] | None = None

    def sample(self, n: int) -> Tuple[tf.Tensor, tf.Tensor]:
        indices = tf.random.shuffle(tf.range(tf.shape(self.data_x)[0]), seed=self.seed)[:n]
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
        if self.send_partial_layers:
            self._last_n_layer_weights = self.model.get_weights()[-self._store_last_n_layers:]

    def get_model(self) -> Model:
        if self.model is None:
            raise Exception("Model not set")
        return self.model

    def clear_model(self) -> None:
        self.model = None

    def reinitialize_client(self) -> None:
        self.clear_model()
        self.clear_training_data()
        self.remove_partial_layer_rule()

    def set_training_algorithm(self, training_algorithm: ClientTrainingFunction) -> None:
        self.training_algorithm = training_algorithm

    def train(self) -> None:
        if self.model is None:
            raise Exception("Model not set")
        if self.training_algorithm is None:
            raise Exception("Training algorithm not set")

        if self.send_partial_layers:
            self.model.set_weights(self.model.get_weights()[:self._send_first_n_layers] + self._last_n_layer_weights)

        self.model = self.training_algorithm(self.model, self)

        if self.send_partial_layers:
            self._last_n_layer_weights = self.model.get_weights()[-self._store_last_n_layers:]

    def get_weights(self) -> List[type(np.array)]:
        if self.send_partial_layers is False:
            return self.model.get_weights()
        return self.model.get_weights()[:self._send_first_n_layers]

    def set_partial_layer_rule(self, total_layers: int, store_last_n_layers: int) -> None:
        self.send_partial_layers = True
        self.store_last_n_layers = store_last_n_layers
        self._store_last_n_layers = store_last_n_layers*2
        self.send_first_n_layers = total_layers - store_last_n_layers
        self._send_first_n_layers = self.send_first_n_layers*2

    def remove_partial_layer_rule(self) -> None:
        self.send_partial_layers = False
        self.store_last_n_layers = None
        self.send_first_n_layers = None

