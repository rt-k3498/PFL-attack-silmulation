from data.data import CIFAR10Data
from clients.client import Client
from models.LeNet import LeNet
from algorithms.per_fedAvg import PerFedAvg
from algorithms.fedAvg import FedAvg
from algorithms.fedPer import FedPer
from attacks.DLG import DLG
from metrics.MSE_metric import MSE_metric
from metrics.PSNR_metric import PSNR_metric
from metrics.SSIM_metric import SSIM_metric
from metrics.VisualMetric import VisualMetric

import csv
from rich import print
import numpy as np
import tensorflow as tf
import random
from tabulate import tabulate
from collections import defaultdict
from typing import Dict, Tuple

seed = 50
I = 10 # number of images
J = 1 # number of runs

def reseed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def trial_seed(i: int, j: int) -> int:
    return seed + 1000 * j + i

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

ds = CIFAR10Data(seed=seed)
x_data_list, y_data_list = ds.get_x_y(1, I)

mse_metric = MSE_metric()
psnr_metric = PSNR_metric()
ssim_metric = SSIM_metric()
visual_metric = VisualMetric()


def _safe_tag(s: str) -> str:
    for ch in (" ", "/", "(", ")", ",", "="):
        s = s.replace(ch, "_")
    return s

loss_function = tf.keras.losses.CategoricalCrossentropy()

algos = {
    "FedAvg": lambda model, clients: FedAvg(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "client_training_batch_size": 1,
        "loss_function": loss_function,
    }),
    "FedPer(K_p=1)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "alpha": 0.1,
        "K_p": 2, 
        "client_training_batch_size": 1,
        "loss_function": loss_function,
    }),
    "FedPer(K_p=2)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "alpha": 0.1,
        "K_p": 2,
        "client_training_batch_size": 1,
        "loss_function": loss_function,
    }),
    "FedPer(K_p=3)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "alpha": 0.1,
        "K_p": 3,
        "client_training_batch_size": 1,
        "loss_function": loss_function,
    }),
    "FedPer(K_p=4)": lambda model, clients: FedPer(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "alpha": 0.1,
        "K_p": 4,
        "client_training_batch_size": 1,
        "loss_function": loss_function,
    }),
    "Per-FedAvg(FO)": lambda model, clients: PerFedAvg(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "client_adaptation_rounds": 1,
        "client_training_batch_size": 1,
        "reuse_data_batches": True,
        "local_training_approximation": "FO",
        "loss_function": loss_function,
    }),
    "Per-FedAvg(HF)": lambda model, clients: PerFedAvg(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "client_adaptation_rounds": 1,
        "client_training_batch_size": 1,
        "reuse_data_batches": True,
        "local_training_approximation": "HF",
        "loss_function": loss_function,
    }),
    "Per-FedAvg(HVP)": lambda model, clients: PerFedAvg(model, clients, seed=seed, settings={
        "communication_rounds": 1,
        "client_training_rounds": 1,
        "client_adaptation_rounds": 1,
        "client_training_batch_size": 1,
        "reuse_data_batches": True,
        "local_training_approximation": "HVP",
        "loss_function": loss_function,
    }),
}

attacks = {
    "DLG": lambda: DLG(seed=seed, settings={
        "max_iterations": 500,
    }),
}


results = {}
total = J * I * len(algos) * len(attacks)
done = 0
for j in range(J):
    for i in range(I):
        for algo_name, make_algo in algos.items():
            for attack_name, make_attack in attacks.items():
                done += 1
                print(f"[cyan][{done}/{total}] run={j+1} image={i} algo={algo_name} attack={attack_name}[/cyan]")

                ts = trial_seed(i, j)
                reseed(ts)

                model = LeNet(seed=seed)
                client = Client(id=1, data=ds, seed=seed, batch_size=1)
                client.data_x = x_data_list[i]
                client.data_y = y_data_list[i]

                attack = _SeededAttack(make_attack(), ts)
                algo = make_algo(model, [client])
                visual_metric.tag = _safe_tag(f"{algo_name}__{attack_name}__i{i}__j{j}")
                results[(algo_name, attack_name, i, j)] = algo.run(
                    attack, performance_metrics=[mse_metric, psnr_metric, ssim_metric, visual_metric]
                )


rows = []
for (algo_name, attack_name, i, j), metric_entries in results.items():
    merged: Dict[Tuple[int, int], Dict[str, float]] = {}
    for entry in metric_entries:
        client_id = entry["client_id"]
        for sample_idx, sample in enumerate(entry["result"]):
            key = (client_id, sample_idx)
            if key not in merged:
                merged[key] = {"Client": client_id}
            merged[key].update(sample)

    for (client_id, _sample_idx), sample in merged.items():
        rows.append({
            "Algorithm":  algo_name,
            "Attack":     attack_name,
            "Image":      i,
            "Run":        j,
            "Client":     client_id,
            "Input MSE":  sample.get("input_mse"),
            "Input PSNR": sample.get("input_psnr"),
            "Input SSIM": sample.get("input_ssim"),
        })

algo_order = {name: idx for idx, name in enumerate(algos.keys())}
attack_order = {name: idx for idx, name in enumerate(attacks.keys())}

rows.sort(key=lambda r: (
    algo_order[r["Algorithm"]],
    attack_order[r["Attack"]],
    r["Image"],
    r["Run"],
    r["Client"],
))

output_lines = []

def emit(line: str = "") -> None:
    print(line)
    output_lines.append(line)

emit()
emit("=" * 80)
emit("Raw results (one row per reconstructed sample)")
emit("=" * 80)
emit(tabulate(rows, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))


per_image_agg = defaultdict(lambda: {"mse_sum": 0.0, "psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0})
for r in rows:
    key = (r["Algorithm"], r["Attack"], r["Image"])
    per_image_agg[key]["mse_sum"]  += r["Input MSE"]
    per_image_agg[key]["psnr_sum"] += r["Input PSNR"]
    per_image_agg[key]["ssim_sum"] += r["Input SSIM"]
    per_image_agg[key]["count"]    += 1

per_image_rows = [
    {
        "Algorithm":      algo_name,
        "Attack":         attack_name,
        "Image":          image_idx,
        "Avg Input MSE":  v["mse_sum"]  / v["count"],
        "Avg Input PSNR": v["psnr_sum"] / v["count"],
        "Avg Input SSIM": v["ssim_sum"] / v["count"],
        "N":              v["count"],
    }
    for (algo_name, attack_name, image_idx), v in per_image_agg.items()
]

per_image_rows.sort(key=lambda r: (
    algo_order[r["Algorithm"]],
    attack_order[r["Attack"]],
    r["Image"],
))

emit()
emit("=" * 80)
emit("Averages per (Algorithm, Attack, Image) - mean over runs (j)")
emit("=" * 80)
emit(tabulate(per_image_rows, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))


overall_agg = defaultdict(lambda: {"mse_sum": 0.0, "psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0})
for r in rows:
    key = (r["Algorithm"], r["Attack"])
    overall_agg[key]["mse_sum"]  += r["Input MSE"]
    overall_agg[key]["psnr_sum"] += r["Input PSNR"]
    overall_agg[key]["ssim_sum"] += r["Input SSIM"]
    overall_agg[key]["count"]    += 1

overall_rows = [
    {
        "Algorithm":      algo_name,
        "Attack":         attack_name,
        "Avg Input MSE":  v["mse_sum"]  / v["count"],
        "Avg Input PSNR": v["psnr_sum"] / v["count"],
        "Avg Input SSIM": v["ssim_sum"] / v["count"],
        "N":              v["count"],
    }
    for (algo_name, attack_name), v in overall_agg.items()
]

overall_rows.sort(key=lambda r: (
    algo_order[r["Algorithm"]],
    attack_order[r["Attack"]],
))

emit()
emit("=" * 80)
emit("Concise averages per (Algorithm, Attack) - mean over runs (j) and images (i)")
emit("=" * 80)
emit(tabulate(overall_rows, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))

raw_fieldnames = [
    "Algorithm",
    "Attack",
    "Image",
    "Run",
    "Client",
    "Input MSE",
    "Input PSNR",
    "Input SSIM",
]
with open("./results/raw_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open("./results/results.txt", "w") as f:
    f.write("\n".join(output_lines) + "\n")


