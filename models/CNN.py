from typing import Literal, Dict, Any
import tensorflow as tf 
#tf.keras.layers.Layer
#tf.keras.layers.Conv2D
#tf.keras.layers.MaxPooling2D
#tf.keras.layers.Conv3D
#tf.keras.layers.MaxPooling3D
#tf.keras.optimizers.Adam
#tf.keras.optimizers.SGD
#tf.keras.optimizers.RMSprop
#tf.keras.metrics.Accuracy
#tf.keras.metrics.Precision
#tf.keras.metrics.Recall
#tf.keras.activations.relu
#tf.keras.activations.sigmoid
#tf.keras.activations.tanh
#tf.keras.activations.softmax
#etc

from models.model import Model

SettingOptions = Literal["layers"]
Settings = Dict[SettingOptions, Any]

class CNN(Model):

    def __init__(self, settings: Settings = {}):
        super().__init__(settings)
        

