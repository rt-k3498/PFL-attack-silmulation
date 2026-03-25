import tensorflow as tf
from typing import Literal, Dict, Any

from models.CNN import CNN

SettingOptions = Literal["layers"]
Settings = Dict[SettingOptions, Any]

class LeNet(CNN):

    def __init__(self, settings: Settings = {}):
        settings["layers"] = [
            tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), activation='relu', strides=2, padding='valid'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), activation='relu', strides=2, padding='valid'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='relu'),
            tf.keras.layers.Dense(84, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
        super().__init__(settings)
