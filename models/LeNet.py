import tensorflow as tf
from typing import Literal, Dict, Any

from models.CNN import CNN

SettingOptions = Literal["layers"]
Settings = Dict[SettingOptions, Any]

class LeNet(CNN): # LeNet-5

    def __init__(self, seed: int, settings: Settings = {}):
        self.seed = seed
        settings["layers"] = [
            tf.keras.layers.Input(shape=(32, 32, 3)),

            # C1
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                activation='sigmoid',
                padding='valid',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
            ),

            # S2
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),

            # C3
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                activation='sigmoid',
                padding='valid',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
            ),

            # S4
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),

            # C5
            tf.keras.layers.Conv2D(
                filters=120,
                kernel_size=(5, 5),
                activation='sigmoid',
                padding='valid',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
            ),

            tf.keras.layers.Flatten(),

            # F6
            tf.keras.layers.Dense(84, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),

            # Output
            tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        ]
        super().__init__(settings)
