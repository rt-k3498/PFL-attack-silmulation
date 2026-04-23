from attacks.attack import Attack
from models.model import Model
from typing import Dict, Literal, Any, List, Tuple
import numpy as np
import tensorflow as tf

SettingOptions = Literal["max_iterations", "min_loss", "min_loss_change", "step_size", "alg_loss_function"]
Settings = Dict[SettingOptions, Any]

ProtocolInfoOptions = Literal[
    "learning_rate", 
    "batch_size",
    "epochs",
    "optimizer",
    "class_mapping",
]
ProtocolInfo = Dict[ProtocolInfoOptions, Any]

class iDLG(Attack):

    def __init__(self, seed: int, settings: Settings = {}):
        self.seed = seed
        super().__init__("iDLG")
        self.max_iterations = settings.get("max_iterations", 1000)
        self.min_loss = settings.get("min_loss", 10**-5)
        self.min_loss_change = settings.get("min_loss_change", 10**-6)
        self.alg_loss_function = settings.get("alg_loss_function", tf.keras.losses.SparseCategoricalCrossentropy())
        self.step_size = settings.get("step_size", 0.1)

        self.reconstructed_input = None
        self.reconstructed_label = None

    def reconstruct_label(self, global_model: Model, client_weights: List[type(np.array)], learning_rate: float) -> int:
        global_model = global_model.model
        global_last_layer_weights = np.array(global_model.layers[-1].get_weights()[0]) #only the weights, not the bias
        client_last_layer_weights = np.array(client_weights[-1]) #only the weights, not the bias

        client_last_layer_gradients = (client_last_layer_weights - global_last_layer_weights)/(-1 * learning_rate)

        return np.argmin(np.sum(client_last_layer_gradients, axis=0))

    def infer_input_dimension(self, model: Model) -> Tuple:
        return model.model.input_shape[1:]

    def reconstruct_input(self, client_gradients: list, input_dimensions: Tuple, label: int, model: Model) -> np.ndarray:
        model = model.model
        dummy_data = tf.Variable(tf.random.normal((1,) + input_dimensions, dtype=tf.float32, seed=self.seed))
        label_tensor = tf.constant([label], dtype=tf.int32)
        curr_loss = float('inf')
        prev_loss = 0
        iterations = 0

        while (iterations < self.max_iterations
               and abs(curr_loss - prev_loss) > self.min_loss_change
               and curr_loss > self.min_loss):
            iterations += 1
            prev_loss = curr_loss

            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as inner_tape:
                    y_pred = model(dummy_data, training=False)
                    loss = self.alg_loss_function(label_tensor, y_pred)
                dummy_gradients = inner_tape.gradient(loss, model.trainable_variables)

                gradient_loss = tf.add_n([
                    tf.reduce_sum(tf.square(dg - rg))
                    for dg, rg in zip(dummy_gradients, client_gradients)
                ])

            grad_on_dummy = outer_tape.gradient(gradient_loss, dummy_data)
            dummy_data.assign(dummy_data - self.step_size * grad_on_dummy)
            curr_loss = gradient_loss.numpy()

        return dummy_data.numpy()


    def run(self, global_model: Model, client_weights: List[type(np.array)], other: ProtocolInfo) -> None:
        input_dimensions = self.infer_input_dimension(global_model)
        reconstructed_label = self.reconstruct_label(global_model, client_weights, other["learning_rate"])

        client_gradients = [
            tf.constant((gw - cw) / other["learning_rate"])
            for gw, cw in zip(global_model.get_weights(), client_weights)
        ]

        reconstructed_input = self.reconstruct_input(
            client_gradients, input_dimensions, reconstructed_label, global_model
        )

        self.reconstructed_input = reconstructed_input
        self.reconstructed_label = reconstructed_label