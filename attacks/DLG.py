from attacks.attack import Attack
from typing import Dict, Literal, Any, List, Tuple
import tensorflow as tf
import numpy as np
from models.model import Model
import tensorflow_probability as tfp

SettingOptions = Literal[
    "max_iterations",
    "num_correction_pairs",
    "tolerance",
    "f_relative_tolerance",
    "max_line_search_iterations",
]
Settings = Dict[SettingOptions, Any]

ProtocolInfoOptions = Literal[
    "learning_rate", 
    "batch_size",
    "epochs",
    "optimizer",
    "class_mapping",
    "num_classes",
]
ProtocolInfo = Dict[ProtocolInfoOptions, Any]

class DLG(Attack):
    def __init__(self, seed: int, settings: Settings = {}):
        super().__init__("DLG")
        self.seed = seed
        self.max_iterations = settings.get("max_iterations", 1000)

        # LBFGS-specific knobs. Defaults track plgp-main/src/idlg_modified.py
        # (history_size=100, tolerance_grad=1e-9, tolerance_change=1e-11).
        self.num_correction_pairs = settings.get("num_correction_pairs", 200)
        self.tolerance = settings.get("tolerance", 1e-9)
        self.f_relative_tolerance = settings.get("f_relative_tolerance", 1e-11)
        self.max_line_search_iterations = settings.get("max_line_search_iterations", 50)

        self.reconstructed_input = None
        self.reconstructed_label = None

    def loss_function(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        y_pred_safe = tf.clip_by_value(y_pred, 1e-12, 1.0 - 1e-12)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred_safe))

    def infer_input_dimension(self, model: Model) -> Tuple:
        return model.model.input_shape[1:]

    def reconstruct_input_and_label(self, client_gradients: list, first_n_layers: int, input_dimensions: Tuple, num_classes: int, model: Model) -> Tuple[np.ndarray, int]:
        keras_model = model.model
        image_shape = (1,) + tuple(input_dimensions)
        n_pixels = int(np.prod(input_dimensions))

        # Flat optimization vector: [raw_image_logits, raw_label_logits].
        # Using the same seeded tf.random.uniform source as the old code so the
        # _SeededAttack adapter in simulation.py keeps producing matched trials.
        init = tf.random.uniform(
            [n_pixels + num_classes], dtype=tf.float32, seed=self.seed
        )

        @tf.function
        def value_and_grad(flat):
            with tf.GradientTape() as outer:
                outer.watch(flat)
                # Slicing must happen inside the outer tape so the tape
                # records the split and can backprop through it; otherwise
                # raw_x/raw_y look like disconnected constants and the
                # outer gradient comes back None.
                raw_x = flat[:n_pixels]
                raw_y = flat[n_pixels:]
                # Sigmoid reparametrization keeps pixels in (0,1) smoothly,
                # replacing the old in-loop tf.clip_by_value projection.
                dummy_x = tf.reshape(tf.sigmoid(raw_x), image_shape)
                dummy_y_softmax = tf.nn.softmax(raw_y)[tf.newaxis, :]
                with tf.GradientTape() as inner:
                    inner.watch(flat)
                    y_pred = keras_model(dummy_x, training=False)
                    loss = self.loss_function(y_pred, dummy_y_softmax)
                dummy_gradients = inner.gradient(loss, keras_model.trainable_variables)

                gradient_loss = tf.add_n([
                    tf.reduce_sum(tf.square(dg - rg))
                    for dg, rg in zip(dummy_gradients[:first_n_layers], client_gradients)
                ])
            g = outer.gradient(gradient_loss, flat)
            return gradient_loss, g

        results = tfp.optimizer.lbfgs_minimize(
            value_and_grad,
            initial_position=init,
            max_iterations=self.max_iterations,
            num_correction_pairs=self.num_correction_pairs,
            tolerance=self.tolerance,
            f_relative_tolerance=self.f_relative_tolerance,
            max_line_search_iterations=self.max_line_search_iterations,
        )

        raw_final = results.position
        dummy_input_data = tf.reshape(tf.sigmoid(raw_final[:n_pixels]), image_shape)
        dummy_label_softmax = tf.nn.softmax(raw_final[n_pixels:])[tf.newaxis, :]

        return dummy_input_data.numpy(), dummy_label_softmax.numpy()

    def run(self, global_model: Model, client_weights: List[type(np.array)], other: ProtocolInfo) -> None:
        input_dimensions = self.infer_input_dimension(global_model)
        first_n_layers = len(client_weights)

        client_gradients = [
            tf.constant((gw - cw) / other["learning_rate"])
            for gw, cw in zip(global_model.get_weights(), client_weights)
        ]

        reconstructed_input, reconstructed_label = self.reconstruct_input_and_label(
            client_gradients, first_n_layers, input_dimensions, other["num_classes"], global_model
        )

        self.reconstructed_input = reconstructed_input
        self.reconstructed_label = reconstructed_label
