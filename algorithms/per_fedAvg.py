from typing import Dict, Literal, Any, List, Callable
import numpy as np
import tensorflow as tf
from rich import print

from models.model import Model
from clients.client import Client
from attacks.attack import Attack
from data.data import CIFAR10Data  
from metrics.PerformanceMetric import PerformanceMetric
from metrics.ModelPerformanceMetric import ModelPerformanceMetric

SettingOptions = Literal[
    "communication_rounds", 
    "client_adaptation_rounds",
    "client_training_rounds", 
    "alpha", 
    "beta", 
    "client_training_batch_size", 
    "client_training_epochs",
    "loss_function", 
    "metrics",
    "local_training_approximation",
    "hf_delta",
]
Settings = Dict[SettingOptions, Any]

class PerFedAvg:
    local_training_approximation_options = ["HF", "FO", "HVP"]
    
    def __init__(self, model: Model, clients: List[Client], seed: int, settings: Settings = {}):
        self.clients = clients
        self.model = model
        self.communication_rounds = settings.get("communication_rounds", 1)
        self.client_adaptation_rounds = settings.get("client_adaptation_rounds", 1)
        self.client_training_rounds = settings.get("client_training_rounds", 1)
        self.alpha = settings.get("alpha", 0.1)
        self.beta = settings.get("beta", 0.1)
        self.client_training_batch_size = settings.get("client_training_batch_size", 1)
        self.client_training_epochs = settings.get("client_training_epochs", 0)
        self.loss_function = settings.get("loss_function", tf.keras.losses.MeanSquaredError())
        self.metrics = settings.get("metrics", ["accuracy"])
        self.local_training_approximation = settings.get("local_training_approximation", "FO")
        self.name = f"Per-FedAvg({self.local_training_approximation})"
        self.model_metric_option = "perFedAvg"
        self.hf_delta = settings.get("hf_delta", 1e-2)

        if self.local_training_approximation not in self.local_training_approximation_options:
            raise Exception("Invalid local training approximation")


    def aggregate(self) -> None:
        """
        Aggregate the client models into the global model.
        """
        weights = [client.get_weights() for client in self.clients]
        if not weights:
            return
        averaged = [np.mean(np.stack(ws, axis=0), axis=0) for ws in zip(*weights)]
        self.model.set_weights(averaged)

    def _HVP_training_algorithm(self, model: Model, client: Client) -> Model:
        """
        Per-FedAvg meta-update using the exact Hessian-vector product via
        nested GradientTape. Reuses one (x, y) batch per training round
        (matches reuse_data_batches=True semantics).

        theta <- theta - beta * (v - alpha * H_f(theta) * v),
        where v = grad f(theta - alpha * grad f(theta); x, y).
        """
        for _ in range(self.client_training_rounds):
            original_weights = model.get_weights()
            x, y = client.get_sample()

            with tf.GradientTape() as inner_tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            inner_grads = inner_tape.gradient(loss, model.model.trainable_variables)

            adapted_weights = [
                w - self.alpha * g.numpy() for w, g in zip(original_weights, inner_grads)
            ]
            model.set_weights(adapted_weights)
            with tf.GradientTape() as adapted_tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            v = adapted_tape.gradient(loss, model.model.trainable_variables)
            v_const = [tf.constant(vi.numpy()) for vi in v]

            model.set_weights(original_weights)
            with tf.GradientTape() as outer_tape:
                with tf.GradientTape() as hv_inner_tape:
                    y_pred = model.model(x, training=True)
                    loss = self.loss_function(y, y_pred)
                g_vars = hv_inner_tape.gradient(loss, model.model.trainable_variables)
                gv_dot = tf.add_n([
                    tf.reduce_sum(gi * vi) for gi, vi in zip(g_vars, v_const)
                ])
            Hv = outer_tape.gradient(gv_dot, model.model.trainable_variables)

            meta_weights = [
                w - self.beta * (vi.numpy() - self.alpha * hvi.numpy())
                for w, vi, hvi in zip(original_weights, v_const, Hv)
            ]
            model.set_weights(meta_weights)

        return model

    def _HF_training_algorithm(self, model: Model, client: Client) -> Model:
        """
        Per-FedAvg meta-update using a finite-difference Hessian-free
        approximation of the Hessian-vector product. Reuses one (x, y) batch
        per training round.

        Hv approx = (grad f(theta + delta * v) - grad f(theta - delta * v)) / (2 * delta)
        theta <- theta - beta * (v - alpha * Hv)
        """
        for _ in range(self.client_training_rounds):
            original_weights = model.get_weights()
            x, y = client.get_sample()

            with tf.GradientTape() as inner_tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            inner_grads = inner_tape.gradient(loss, model.model.trainable_variables)

            adapted_weights = [
                w - self.alpha * g.numpy() for w, g in zip(original_weights, inner_grads)
            ]
            model.set_weights(adapted_weights)
            with tf.GradientTape() as adapted_tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            v = adapted_tape.gradient(loss, model.model.trainable_variables)
            v_np = [vi.numpy() for vi in v]

            plus_weights = [w + self.hf_delta * vi for w, vi in zip(original_weights, v_np)]
            model.set_weights(plus_weights)
            with tf.GradientTape() as plus_tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            g_plus = plus_tape.gradient(loss, model.model.trainable_variables)
            g_plus_np = [gpi.numpy() for gpi in g_plus]

            minus_weights = [w - self.hf_delta * vi for w, vi in zip(original_weights, v_np)]
            model.set_weights(minus_weights)
            with tf.GradientTape() as minus_tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            g_minus = minus_tape.gradient(loss, model.model.trainable_variables)
            g_minus_np = [gmi.numpy() for gmi in g_minus]

            hv_approx = [
                (gp - gm) / (2.0 * self.hf_delta) for gp, gm in zip(g_plus_np, g_minus_np)
            ]

            meta_weights = [
                w - self.beta * (vi - self.alpha * hvi)
                for w, vi, hvi in zip(original_weights, v_np, hv_approx)
            ]
            model.set_weights(meta_weights)

        return model

    def _FO_training_algorithm(self, model: Model, client: Client) -> Model:
        for _ in range(self.client_training_rounds):
            original_weights = model.get_weights()
            x, y = client.get_sample()
            for _ in range(self.client_adaptation_rounds):
                with tf.GradientTape() as tape:
                    y_pred = model.model(x, training=True)
                    loss = self.loss_function(y, y_pred)
                gradients = tape.gradient(loss, model.model.trainable_variables)
                new_weights = [weight - self.alpha * gradient for weight, gradient in zip(model.model.trainable_variables, gradients)]
                model.set_weights(new_weights)

            with tf.GradientTape() as tape:
                y_pred = model.model(x, training=True)
                loss = self.loss_function(y, y_pred)
            gradients = tape.gradient(loss, model.model.trainable_variables)
            meta_weights = [w_orig - self.beta * gradient for w_orig, gradient in zip(original_weights, gradients)]
            model.set_weights(meta_weights)
        return model

    def client_training_algorithm(self, model: Model, client: Client) -> Model:
        """
        Runs the local training algorithm given a model, data and settings.
        """
        if self.local_training_approximation == "HF":
            return self._HF_training_algorithm(model, client)
        elif self.local_training_approximation == "FO":
            return self._FO_training_algorithm(model, client)
        elif self.local_training_approximation == "HVP":
            return self._HVP_training_algorithm(model, client)
        else:
            raise Exception("Invalid local training approximation")


    def run(
        self,
        attack: Attack = None,
        attack_performance_metrics: List[PerformanceMetric] = None,
        model_performance_metrics: List[ModelPerformanceMetric] = None,
        result_handlers: List[Any] = None,
    ) -> list:
        """
        Run the PerFedAvg algorithm.
        """

        attack_results = []
        performance_results = []
        result_handlers = result_handlers or []
        attack_result_handlers = [
            handler for handler in result_handlers
            if hasattr(handler, "append_attack_results")
        ]
        algorithm_result_handlers = [
            handler for handler in result_handlers
            if hasattr(handler, "append_algorithm_results")
        ]

        for client in self.clients:
            client.set_training_algorithm(self.client_training_algorithm)

        for idx in range(self.communication_rounds):
            print(f"[yellow]Communication round {idx + 1}/{self.communication_rounds}[/yellow]")

            clients_data = {}

            print("[blue]Training clients...[/blue]")

            for client in self.clients:
                client.clear_training_data()
                client.set_model(self.model.clone())
                client.train()
                weights = client.get_weights()
                data = client.get_data_used_for_training()[-1]
                clients_data[client.id] = (weights, data)

            if attack:
                print(f"[blue]Running {attack.name} attack...[/blue]")
                for client_id in clients_data:
                    attack.run(self.model, clients_data[client_id][0], {"learning_rate": self.beta, "num_classes": len(CIFAR10Data._CIFAR_10_CLASSES)})
                    for handler in attack_result_handlers:
                        handler.append_attack_results(
                            self,
                            attack,
                            idx + 1,
                            client_id,
                            self.clients[client_id].get_label_classes(),
                            clients_data[client_id][1],
                        )
                    if attack_performance_metrics:
                        for performance_metric in attack_performance_metrics:
                            result = performance_metric.measure(clients_data[client_id][1], attack)
                            attack_results.append({"client_id": client_id, "performance_metric": performance_metric.name, "result": result})

            print("[blue]Aggregating client models...[/blue]")
            
            self.aggregate()

        print("[green]PerFedAvg completed.[/green]")

        for client in self.clients:
            client.set_model(self.model.clone())

        for handler in algorithm_result_handlers:
            handler.append_algorithm_results(self, self.clients, source_attack=attack)

        if model_performance_metrics:
            print("[blue]Evaluating final trained global model...[/blue]")
            for performance_metric in model_performance_metrics:
                result = performance_metric.measure(self.clients, self.model_metric_option)
                performance_results.append({"performance_metric": performance_metric.name, "result": result})

        return {"attack_results": attack_results, "performance_results": performance_results}
