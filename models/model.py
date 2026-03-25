from __future__ import annotations 
import tensorflow as tf
import numpy as np
from typing import List, Dict, Any, Callable, Union, Literal

SettingOptions = Literal["layers"]
Settings = Dict[SettingOptions, Any]

class Model: 
    def __init__(self, settings: Settings = {}):
        self.layers: List[tf.keras.layers.Layer] = settings.get("layers", [])
        self.model: tf.keras.Model = Model.build_model({"layers": self.layers})

    @staticmethod
    def build_model(settings: Settings) -> tf.keras.Model:
        model = tf.keras.Sequential(settings["layers"])
        return model

    def get_weights(self) -> List[type(np.array)]:
        return self.model.get_weights()

    def set_weights(self, weights: List[type(np.array)]) -> None:
        self.model.set_weights(weights)

    def clone(self) -> Model:
        new_model = Model({"layers": self.layers})
        new_model.model = tf.keras.models.clone_model(self.model)
        new_model.model.set_weights(self.get_weights())
        return new_model
