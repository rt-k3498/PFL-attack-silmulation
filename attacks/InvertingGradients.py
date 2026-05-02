import tensorflow as tf
from typing import List, Dict, Any, Tuple, Literal
from models.model import Model
from attacks.attack import Attack
import numpy as np

SettingOptions = Literal[
    "init_step_size",
    "final_step_size",
    "max_iterations",
    "alpha",
    "use_signed_adam",
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

class InvertingGradients(Attack):
    def __init__(self, seed: int, settings: Settings = {}):
        super().__init__("InvertingGradients")
        self.seed = seed
        self.max_iterations = settings.get("max_iterations", 1000)
        self.init_step_size = settings.get("init_step_size", 0.1)
        self.final_step_size = settings.get("final_step_size", 0.001)
        self.alpha = settings.get("alpha", 0.1)
        self.use_signed_adam = settings.get("use_signed_adam", True)

    def _make_optimizer(self) -> tf.keras.optimizers.Optimizer:
        learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.init_step_size,
            decay_steps=self.max_iterations,
            alpha=self.final_step_size / self.init_step_size,
        )
        return tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)


    def infer_input_dimension(self, model: Model) -> Tuple:
        return model.model.input_shape[1:]

    def loss_function(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        y_pred_safe = tf.clip_by_value(y_pred, 1e-12, 1.0 - 1e-12)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred_safe))

    def reconstruct_input_and_label(self, client_gradients: list, first_n_layers: int, input_dimensions: Tuple, num_classes: int, model: Model) -> Tuple[np.ndarray, int]:
        model = model.model
        raw_image = tf.Variable(
            tf.random.uniform((1,) + input_dimensions, dtype=tf.float32, seed=self.seed)
        )
        dummy_label = tf.Variable(
            tf.random.uniform((1, num_classes), dtype=tf.float32, seed=self.seed)
        )
        optimizer = self._make_optimizer()

        client_flat = tf.concat([tf.reshape(g, [-1]) for g in client_gradients], axis=0)
        client_norm = tf.norm(client_flat)

        iterations = 0
        while (iterations < self.max_iterations):
            iterations += 1
            with tf.GradientTape() as outer_tape:
                dummy_data = tf.sigmoid(raw_image)
                with tf.GradientTape() as inner_tape:
                    y_pred = model(dummy_data, training=False)
                    dummy_y_softmax = tf.nn.softmax(dummy_label, axis=1)
                    loss = self.loss_function(y_pred, dummy_y_softmax)
                gradients = inner_tape.gradient(loss, model.trainable_variables)

                dummy_flat = tf.concat(
                    [tf.reshape(g, [-1]) for g in gradients[:first_n_layers]], axis=0
                )
                num = tf.reduce_sum(dummy_flat * client_flat)
                den = tf.norm(dummy_flat) * client_norm + 1e-12
                cos_loss = 1.0 - num / den

                tv = tf.reduce_sum(tf.image.total_variation(dummy_data))
                gradient_loss = cos_loss + self.alpha * tv

            grads = outer_tape.gradient(gradient_loss, [raw_image, dummy_label])
            if self.use_signed_adam:
                grads = [tf.sign(g) for g in grads]
            optimizer.apply_gradients(zip(grads, [raw_image, dummy_label]))

        return tf.sigmoid(raw_image).numpy(), tf.nn.softmax(dummy_label, axis=1).numpy()


    def run(self, global_model: Model, client_weights: List[np.array], other: ProtocolInfo) -> None:
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
