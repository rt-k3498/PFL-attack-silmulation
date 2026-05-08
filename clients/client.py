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
        self.label_classes: List[int] | None = None
        self.seed = seed
        x, y = data.get_x_y(batch_size, 1)
        self.data_x = x[0]
        self.data_y = y[0]
        self.used_in_training_data: List[Tuple[tf.Tensor, tf.Tensor]] = []
        self.model: Model | None = None
        self.training_algorithm = None
        self.send_partial_layers: bool = False
        self.send_first_n_layers: int | None = None
        self._send_first_n_layers: int | None = None
        self.store_last_n_layers: int | None = None
        self._store_last_n_layers: int | None = None
        self._last_n_layer_weights: List[np.array] | None = None
        self._get_sample_index = 0

    def set_label_classes(self, label_classes: List[int]) -> None:
        self.label_classes = label_classes

    def get_label_classes(self) -> List[int]:
        if self.label_classes is None:
            try:
                return [int(self.id)]
            except (TypeError, ValueError) as exc:
                raise Exception("Label classes not set") from exc
        return self.label_classes

    def random_samples(self, n: int) -> Tuple[tf.Tensor, tf.Tensor]:
        indices = tf.random.shuffle(tf.range(tf.shape(self.data_x)[0]), seed=self.seed)[:n]
        x = tf.gather(self.data_x, indices)
        y = tf.gather(self.data_y, indices)
        self.used_in_training_data.append((x, y))
        return x, y
    
    def get_sample(self)-> Tuple[tf.Tensor, tf.Tensor]:
        if self._get_sample_index >= tf.shape(self.data_x)[0]:
            raise Exception("No more samples available")
        x = tf.gather(self.data_x, [self._get_sample_index])
        y = tf.gather(self.data_y, [self._get_sample_index])
        self.used_in_training_data.append((x, y))
        self._get_sample_index += 1
        return x, y

    def get_data_used_for_training(self) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        return self.used_in_training_data

    def clear_training_data(self) -> None:
        self.used_in_training_data = []

    def set_model(self, model: Model) -> None:
        if self.model: 
            if self.send_partial_layers:
                if self._last_n_layer_weights:
                    self.model.set_weights(model.get_weights()[:self._send_first_n_layers] + self._last_n_layer_weights)
                    return
                self.model.set_weights(model.get_weights())
                self._last_n_layer_weights = self.model.get_weights()[-self._store_last_n_layers:]
            else: 
                self.model.set_weights(model.get_weights())
        else:
            self.model = model
            if self.send_partial_layers:
                self._last_n_layer_weights = self.model.get_weights()[-self._store_last_n_layers:]

    def get_model(self) -> Model:
        if self.model is None:
            raise Exception("Model not set")
        return self.model

    @staticmethod
    def _normalize_sample_tensor(values: Any, sample_rank: int) -> tf.Tensor:
        if isinstance(values, (list, tuple)):
            if not values:
                raise ValueError("Client data cannot be empty")
            tensors = [tf.convert_to_tensor(value) for value in values]
            first_rank = tensors[0].shape.rank
            if first_rank == sample_rank:
                return tf.stack(tensors, axis=0)
            return tf.concat(tensors, axis=0)

        tensor = tf.convert_to_tensor(values)
        if tensor.shape.rank == sample_rank:
            return tensor[tf.newaxis, ...]
        return tensor
    
    def set_data(self, data_x: Any, data_y: Any) -> None:
        data_x = self._normalize_sample_tensor(data_x, sample_rank=3)
        data_y = self._normalize_sample_tensor(data_y, sample_rank=1)

        if tf.shape(data_x)[0] != tf.shape(data_y)[0]:
            raise ValueError("data_x and data_y must contain the same number of samples")

        x_y = list(zip(tf.unstack(data_x), tf.unstack(data_y)))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(x_y)

        x, y = zip(*x_y)
        self.data_x = tf.stack(x, axis=0)
        self.data_y = tf.stack(y, axis=0)
        self._get_sample_index = 0

    def clear_model(self) -> None:
        self.model = None

    def reinitialize_client(self) -> None:
        self.clear_model()
        self.clear_training_data()
        self.remove_partial_layer_rule()
        self._get_sample_index = 0

    def set_training_algorithm(self, training_algorithm: ClientTrainingFunction) -> None:
        self.training_algorithm = training_algorithm

    def train(self) -> None:
        if self.model is None:
            raise Exception("Model not set")
        if self.training_algorithm is None:
            raise Exception("Training algorithm not set")

        self.model = self.training_algorithm(self.model, self)

        if self.send_partial_layers:
            self._last_n_layer_weights = self.model.get_weights()[-self._store_last_n_layers:]

    def get_weights(self) -> List[np.array]:
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
        self._store_last_n_layers = None
        self.send_first_n_layers = None
        self._send_first_n_layers = None
