from data.data import CIFAR10Data
from clients.client import Client
from models.LeNet import LeNet
from algorithms.per_fedAvg import PerFedAvg
from algorithms.fedAvg import FedAvg
from algorithms.fedPer import FedPer
from attacks.DLG import DLG
from attacks.InvertingGradients import InvertingGradients
from metrics.ComparisonMetric import ComparisonMetric
from metrics.MSE_metric import MSE_metric
from metrics.PSNR_metric import PSNR_metric
from metrics.SSIM_metric import SSIM_metric
from metrics.VisualMetric import VisualMetric
from metrics.ModelCrossEntropy import ModelCrossEntropy
from metrics.PredictionComparison import PredictionComparison
from results.ResultHandler import AlgorithmResultHandler, AttackResultHandler

from rich import print
import numpy as np
import tensorflow as tf
import random

seed = 42

def reseed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def trial_seed(run: int) -> int:
    return seed + 1000 * run

class _SeededAttack:
    """
    Adapter that reseeds right before delegating to the underlying attack's
    run(). This makes the attack's internal random draws (e.g. DLG's dummy
    input/label init) depend only on the trial id (i, j), not on the amount
    of TF random state consumed by whichever algorithm ran beforehand. That
    is what lets us fairly compare algorithms/variants on the same trial.
    """

    def __init__(self, inner, trial_seed_value: int):
        self._inner = inner
        self._trial_seed = trial_seed_value

    def run(self, *args, **kwargs):
        reseed(self._trial_seed)
        return self._inner.run(*args, **kwargs)

    @property
    def reconstructed_input(self):
        return self._inner.reconstructed_input

    @property
    def reconstructed_label(self):
        return self._inner.reconstructed_label

    @property
    def name(self):
        return self._inner.name


reseed(seed)

config = {
    "batch_size": 10, # number of images per client (should match the number of communication rounds for simplicity, since each client will be sampled once per round)
    "client_training_batch_size": 1, # batch size for client training (should match the number of images per client for simplicity, so that each client trains on all their data each round)
    "num_runs": 3, # number of runs (complete executions of all algorithms and attacks, for averaging simulations)
    "ds": CIFAR10Data(seed=seed),
    "x_data_list": None,
    "y_data_list": None,
    "local_training_rounds": 1, # number of local training rounds per client
    "communication_rounds": None, # number of communication rounds (calls to algo.run())
    "num_clients": 10, # number of clients (and thus number of reconstructed images per algo/attack/run)
    "loss_function": tf.keras.losses.CategoricalCrossentropy(),
    "num_final_model_evaluation_iterations": 1, # number of batches to evaluate the final model on (for ModelCrossEntropy metric)
    "labels_per_client": 1, # number of unique labels per client (for non-iid data partitioning)
}

classes_per_run = len(CIFAR10Data._CIFAR_10_CLASSES)
if config["num_clients"] * config["labels_per_client"] != classes_per_run:
    raise ValueError("num_clients * labels_per_client must cover each CIFAR-10 class exactly once")

config["communication_rounds"] = config["batch_size"] * config["labels_per_client"]
config["x_data_list"], config["y_data_list"] = config["ds"].get_structured_x_y(config["batch_size"], classes_per_run*config["num_runs"])

model_performance_metric_settings = {
    "batch_size": config["batch_size"],
    "num_iterations": config["num_final_model_evaluation_iterations"],
    "number_of_clients": config["num_clients"],
    "loss_function": config["loss_function"],
    "adaptation_alpha": 0.1,
}

attack_result_handler = AttackResultHandler(metrics=[
    ComparisonMetric(),
    MSE_metric(),
    PSNR_metric(),
    SSIM_metric(),
])
algorithm_result_handler = AlgorithmResultHandler(metrics=[
    PredictionComparison(seed=seed, settings=model_performance_metric_settings),
    ModelCrossEntropy(seed=seed, settings=model_performance_metric_settings),
])
result_handlers = [attack_result_handler, algorithm_result_handler]

algos = {
    "FedAvg": lambda model, clients: FedAvg(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "client_training_batch_size": config["client_training_batch_size"],
        "loss_function": config["loss_function"],
    }),
    "FedPer(K_p=1)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "alpha": 0.1,
        "K_p": 1, 
        "client_training_batch_size": config["client_training_batch_size"],
        "loss_function": config["loss_function"],
    }),
    "FedPer(K_p=2)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "alpha": 0.1,
        "K_p": 2,
        "client_training_batch_size": config["client_training_batch_size"],
        "loss_function": config["loss_function"],
    }),
    "FedPer(K_p=3)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "alpha": 0.1,
        "K_p": 3,
        "client_training_batch_size": config["client_training_batch_size"],
        "loss_function": config["loss_function"],
    }),
    "FedPer(K_p=4)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "alpha": 0.1,
        "K_p": 4,
        "client_training_batch_size": config["client_training_batch_size"],
        "loss_function": config["loss_function"],
    }),
    "FedPer(K_p=5)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "alpha": 0.1,
        "K_p": 5,
        "client_training_batch_size": config["client_training_batch_size"],
        "loss_function": config["loss_function"],
    }),
    "Per-FedAvg(FO)": lambda model, clients: PerFedAvg(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "client_adaptation_rounds": 1,
        "client_training_batch_size": config["client_training_batch_size"],
        "reuse_data_batches": True,
        "local_training_approximation": "FO",
        "loss_function": config["loss_function"],
    }),
    "Per-FedAvg(HF)": lambda model, clients: PerFedAvg(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "client_adaptation_rounds": 1,
        "client_training_batch_size": config["client_training_batch_size"],
        "reuse_data_batches": True,
        "local_training_approximation": "HF",
        "loss_function": config["loss_function"],
    }),
    "Per-FedAvg(HVP)": lambda model, clients: PerFedAvg(model, clients, seed=seed, settings={
        "communication_rounds": config["communication_rounds"],
        "client_training_rounds": config["local_training_rounds"],
        "client_adaptation_rounds": 1,
        "client_training_batch_size": config["client_training_batch_size"],
        "reuse_data_batches": True,
        "local_training_approximation": "HVP",
        "loss_function": config["loss_function"],
    }),
}

attacks = {
    "DLG": lambda: DLG(seed=seed, settings={
        "max_iterations": 300,
        "f_relative_tolerance": 1.25e-12,
        "max_line_search_iterations": 300,
        "num_correction_pairs": 175,
        "tolerance": 1e-15,
    }),
    "InvertingGradients": lambda: InvertingGradients(seed=seed, settings={
        "max_iterations": 300,
        "init_step_size": 0.1,
        "final_step_size": 0.09,
        "alpha": 6.3e-13,
    }),
}


total = config["num_runs"] * len(algos) * len(attacks)
done = 0

clients = [Client(id=i, data=config["ds"], seed=seed, batch_size=config["batch_size"]*config["labels_per_client"]) for i in range(config["num_clients"])]

client_labels_assignment = {client.id: [] for client in clients}

for client in clients:
    for i in range(config["labels_per_client"]):
        client_labels_assignment[client.id].append((client.id + (i * config["num_clients"])))
    client.set_label_classes(client_labels_assignment[client.id])

del client_labels_assignment

for run in range(config["num_runs"]): 
    for handler in result_handlers:
        handler.set_run_index(run)

    client_x_data_assignment = {client.id: [] for client in clients}
    client_y_data_assignment = {client.id: [] for client in clients}

    for client in clients:
        for i in client.get_label_classes():
            client_x_data_assignment[client.id].append(config["x_data_list"][i + (classes_per_run * run)])
            client_y_data_assignment[client.id].append(config["y_data_list"][i + (classes_per_run * run)])

        client.set_data(client_x_data_assignment[client.id], client_y_data_assignment[client.id])
    
    del client_x_data_assignment
    del client_y_data_assignment

    for _algo_name, make_algo in algos.items():
        for _attack_name, make_attack in attacks.items():
            done += 1

            ts = trial_seed(run)
            reseed(ts)

            model = LeNet(seed=seed)
            
            for client in clients: 
                client.reinitialize_client()

            attack = _SeededAttack(make_attack(), ts)
            algo = make_algo(model, clients)
            print(f"[cyan][{done}/{total}] run={run+1} algo={algo.name} attack={attack.name}[/cyan]")
            algo.run(
                attack,
                result_handlers=result_handlers,
            )
