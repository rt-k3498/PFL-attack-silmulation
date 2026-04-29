from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from attacks.DLG import DLG
from attacks.InvertingGradients import InvertingGradients
from clients.client import Client
from data.data import CIFAR10Data
from metrics.MSE_metric import MSE_metric
from metrics.PSNR_metric import PSNR_metric
from metrics.SSIM_metric import SSIM_metric
from models.LeNet import LeNet


SEED = 50
IMAGE_COUNT = 1
ROUNDS = 3
VALUES_PER_PARAMETER = 3
MAX_ITERATIONS = 300
OUTPUT_DIR = Path("results/hyperparameters")
ATTACK_CHOICES = ("DLG", "InvertingGradients", "all")

DLG_INITIAL_GRID = {
    "num_correction_pairs": [50, 100, 150],
    "max_line_search_iterations": [100, 200, 300],
    "tolerance": [ 1e-18, 1e-15, 1e-12],
    "f_relative_tolerance": [ 1e-18, 1e-15, 1e-12],
}

INVERTING_GRADIENTS_INITIAL_GRID = {
    "init_step_size": [ 0.01, 0.1, 1.0],
    "final_step_size": [ 0.001, 0.01, 0.1],
    "alpha": [ 1e-14, 1e-13, 1e-12],
}

INTEGER_PARAMETERS = {
    "num_correction_pairs",
    "max_line_search_iterations",
}
POSITIVE_FLOAT_FLOOR = float(np.nextafter(0.0, 1.0))


class _SeededAttack:
    """Reseed directly before attack reconstruction for fair candidate comparison."""

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


def reseed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def trial_seed(image_idx: int) -> int:
    return SEED + image_idx


def with_max_iterations(settings: Dict[str, Any]) -> Dict[str, Any]:
    full_settings = dict(settings)
    full_settings["max_iterations"] = MAX_ITERATIONS
    return full_settings


def make_attack(attack_name: str, settings: Dict[str, Any]):
    if attack_name == "DLG":
        return DLG(seed=SEED, settings=with_max_iterations(settings))
    if attack_name == "InvertingGradients":
        return InvertingGradients(seed=SEED, settings=with_max_iterations(settings))
    raise ValueError(f"Unknown attack: {attack_name}")


def prepare_fedavg_trials(x_data_list, y_data_list) -> List[Dict[str, Any]]:
    """Prepare the deterministic one-client, one-sample FedAvg update per image."""
    data = CIFAR10Data(seed=SEED)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    trials = []

    for image_idx, (x_data, y_data) in enumerate(zip(x_data_list, y_data_list)):
        ts = trial_seed(image_idx)
        reseed(ts)

        global_model = LeNet(seed=SEED)
        global_model.model(x_data)

        client = Client(id=1, data=data, seed=SEED, batch_size=1)
        client.data_x = x_data
        client.data_y = y_data
        client.clear_training_data()

        local_model = global_model.clone()
        client.set_model(local_model)

        train_x, train_y = client.sample(1)
        with tf.GradientTape() as tape:
            y_pred = local_model.model(train_x, training=True)
            loss = loss_function(train_y, y_pred)
        gradients = tape.gradient(loss, local_model.model.trainable_variables)
        new_weights = [
            weight - 0.1 * gradient
            for weight, gradient in zip(local_model.model.trainable_variables, gradients)
        ]
        local_model.set_weights(new_weights)

        trials.append({
            "image_idx": image_idx,
            "trial_seed": ts,
            "global_model": global_model,
            "client_weights": client.get_weights(),
            "used_training_data": client.get_data_used_for_training(),
            "client_id": client.id,
        })

    return trials


def make_grid(settings_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    names = list(settings_grid.keys())
    candidates = []
    for values in itertools.product(*(settings_grid[name] for name in names)):
        settings = dict(zip(names, values))
        if settings.get("final_step_size", 0) > settings.get("init_step_size", math.inf):
            continue
        candidates.append(settings)
    return candidates


def refined_scale(previous_values: List[Any], count: int) -> float:
    if count != 3:
        raise ValueError("Grid refinement requires VALUES_PER_PARAMETER == 3")

    sorted_values = sorted(float(v) for v in previous_values)
    if len(sorted_values) != 3:
        raise ValueError("Grid refinement requires exactly 3 values per parameter")

    gaps = [
        sorted_values[idx + 1] - sorted_values[idx]
        for idx in range(len(sorted_values) - 1)
    ]
    current_scale = sum(gaps) / len(gaps)
    return current_scale / 2.0


def float_refined_values(best: float, previous_values: List[Any], count: int) -> List[float]:
    if count == 1:
        return [max(float(best), POSITIVE_FLOAT_FLOOR)]

    scale = refined_scale(previous_values, count)
    values = [float(best) - scale, float(best), float(best) + scale]
    values[0] = max(values[0], POSITIVE_FLOAT_FLOOR)
    return values


def int_refined_values(best: int, previous_values: List[Any], count: int) -> List[int]:
    if count == 1:
        return [max(1, int(best))]

    scale = refined_scale(previous_values, count)
    rounded = [max(1, int(round(value))) for value in (best - scale, best, best + scale)]
    values = sorted(set(rounded))
    if len(values) != 3:
        raise ValueError(
            f"Integer refinement for best={best} and previous_values={previous_values} "
            "did not produce 3 distinct values after rounding"
        )
    return values


def refine_grid(
    previous_grid: Dict[str, List[Any]],
    best_settings: Dict[str, Any],
    count: int,
) -> Dict[str, List[Any]]:
    refined = {}
    for name, previous_values in previous_grid.items():
        best = best_settings[name]
        if name in INTEGER_PARAMETERS:
            refined[name] = int_refined_values(int(best), previous_values, count)
        else:
            refined[name] = float_refined_values(float(best), previous_values, count)
    return refined


def run_candidate_on_image(
    attack_name: str,
    settings: Dict[str, Any],
    trial: Dict[str, Any],
) -> Dict[str, float]:
    ts = trial["trial_seed"]
    reseed(ts)

    attack = _SeededAttack(make_attack(attack_name, settings), ts)
    attack.run(
        trial["global_model"],
        trial["client_weights"],
        {"learning_rate": 0.1, "num_classes": len(CIFAR10Data._CIFAR_10_CLASSES)},
    )

    merged = {"Client": trial["client_id"]}
    for metric in [MSE_metric(), PSNR_metric(), SSIM_metric()]:
        for sample in metric.measure(trial["used_training_data"], attack):
            merged.update(sample)

    return {
        "Client": merged["Client"],
        "Input MSE": merged.get("input_mse"),
        "Input PSNR": merged.get("input_psnr"),
        "Input SSIM": merged.get("input_ssim"),
    }


def run_candidate(
    attack_name: str,
    round_idx: int,
    candidate_idx: int,
    settings: Dict[str, Any],
    trials: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for trial in trials:
        metric_row = run_candidate_on_image(
            attack_name,
            settings,
            trial,
        )
        rows.append({
            "Attack": attack_name,
            "Round": round_idx,
            "Candidate": candidate_idx,
            **settings,
            "Image": trial["image_idx"],
            **metric_row,
        })
    return rows


def aggregate_candidate_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(lambda: {
        "mse_sum": 0.0,
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "count": 0,
        "settings": {},
    })

    for row in rows:
        key = (row["Attack"], row["Round"], row["Candidate"])
        grouped[key]["mse_sum"] += row["Input MSE"]
        grouped[key]["psnr_sum"] += row["Input PSNR"]
        grouped[key]["ssim_sum"] += row["Input SSIM"]
        grouped[key]["count"] += 1
        grouped[key]["settings"] = {
            k: v
            for k, v in row.items()
            if k not in {
                "Attack",
                "Round",
                "Candidate",
                "Image",
                "Client",
                "Input MSE",
                "Input PSNR",
                "Input SSIM",
            }
        }

    summary_rows = []
    for (attack_name, round_idx, candidate_idx), values in grouped.items():
        count = values["count"]
        summary_rows.append({
            "Attack": attack_name,
            "Round": round_idx,
            "Candidate": candidate_idx,
            **values["settings"],
            "Avg Input MSE": values["mse_sum"] / count,
            "Avg Input PSNR": values["psnr_sum"] / count,
            "Avg Input SSIM": values["ssim_sum"] / count,
            "N": count,
        })

    summary_rows.sort(key=lambda r: (r["Attack"], r["Round"], r["Candidate"]))
    return summary_rows


def best_summary_row(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return min(summary_rows, key=lambda row: row["Avg Input MSE"])


def settings_from_summary_row(row: Dict[str, Any], parameter_names: Iterable[str]) -> Dict[str, Any]:
    return {name: row[name] for name in parameter_names}


def run_attack_search(
    attack_name: str,
    initial_grid: Dict[str, List[Any]],
    trials: List[Dict[str, Any]],
    rounds: int,
    values_per_parameter: int,
    image_count: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grid = initial_grid
    all_rows = []
    round_best_rows = []

    for round_idx in range(1, rounds + 1):
        candidates = make_grid(grid)
        round_rows = []
        total = len(candidates)

        for candidate_idx, settings in enumerate(candidates, start=1):
            print(
                f"[attack={attack_name} images={image_count}] "
                f"round {round_idx}/{rounds} candidate {candidate_idx}/{total}: {settings}"
            )
            candidate_rows = run_candidate(
                attack_name,
                round_idx,
                candidate_idx,
                settings,
                trials,
            )
            round_rows.extend(candidate_rows)
            all_rows.extend(candidate_rows)

        round_summary = aggregate_candidate_rows(round_rows)
        best_row = best_summary_row(round_summary)
        round_best_rows.append(best_row)

        if round_idx < rounds:
            best_settings = settings_from_summary_row(best_row, grid.keys())
            grid = refine_grid(grid, best_settings, values_per_parameter)

    all_summary_rows = aggregate_candidate_rows(all_rows)
    return all_rows, all_summary_rows, round_best_rows


def write_outputs(
    output_dir: Path,
    raw_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    round_best_rows: List[Dict[str, Any]],
    image_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Attack",
        "Round",
        "Candidate",
        "num_correction_pairs",
        "max_line_search_iterations",
        "tolerance",
        "f_relative_tolerance",
        "init_step_size",
        "final_step_size",
        "alpha",
        "Image",
        "Client",
        "Input MSE",
        "Input PSNR",
        "Input SSIM",
    ]

    with (output_dir / "raw_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(raw_rows)

    output_lines = []

    def emit(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    emit()
    emit("=" * 80)
    emit("Raw hyperparameter results (one row per reconstructed sample)")
    emit("=" * 80)
    emit(tabulate(raw_rows, headers="keys", tablefmt="fancy_grid", floatfmt=".6g"))

    emit()
    emit("=" * 80)
    emit(f"Average per candidate - mean over {image_count} images")
    emit("=" * 80)
    emit(tabulate(summary_rows, headers="keys", tablefmt="fancy_grid", floatfmt=".6g"))

    emit()
    emit("=" * 80)
    emit("Best setting per round - selected by lowest average input MSE")
    emit("=" * 80)
    emit(tabulate(round_best_rows, headers="keys", tablefmt="fancy_grid", floatfmt=".6g"))

    final_best_by_attack = []
    for attack_name in ["DLG", "InvertingGradients"]:
        attack_rows = [row for row in round_best_rows if row["Attack"] == attack_name]
        if attack_rows:
            final_best_by_attack.append(best_summary_row(attack_rows))

    emit()
    emit("=" * 80)
    emit("Final best setting per attack")
    emit("=" * 80)
    emit(tabulate(final_best_by_attack, headers="keys", tablefmt="fancy_grid", floatfmt=".6g"))

    with (output_dir / "results.txt").open("w") as f:
        f.write("\n".join(output_lines) + "\n")


def smoke_grid(grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    return {name: [values[len(values) // 2]] for name, values in grid.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FedAvg gradient inversion hyperparameter searches for DLG and InvertingGradients."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one image, one round, and one candidate per attack for verification.",
    )
    parser.add_argument(
        "--attack",
        choices=ATTACK_CHOICES,
        default="all",
        help="Choose which attack search to run.",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=IMAGE_COUNT,
        help="Number of images to evaluate in non-smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for raw_results.csv and results.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reseed(SEED)

    if args.image_count < 1:
        raise ValueError("--image-count must be at least 1")

    image_count = 1 if args.smoke else args.image_count
    rounds = 1 if args.smoke else ROUNDS
    values_per_parameter = 1 if args.smoke else VALUES_PER_PARAMETER

    dlg_grid = smoke_grid(DLG_INITIAL_GRID) if args.smoke else DLG_INITIAL_GRID
    inverting_grid = (
        smoke_grid(INVERTING_GRADIENTS_INITIAL_GRID)
        if args.smoke
        else INVERTING_GRADIENTS_INITIAL_GRID
    )

    data = CIFAR10Data(seed=SEED)
    x_data_list, y_data_list = data.get_x_y(batch_size=1, number_of_batches=image_count)
    trials = prepare_fedavg_trials(x_data_list, y_data_list)

    raw_rows = []
    summary_rows = []
    round_best_rows = []

    attack_grids = [
        ("DLG", dlg_grid),
        ("InvertingGradients", inverting_grid),
    ]
    if args.attack != "all":
        attack_grids = [entry for entry in attack_grids if entry[0] == args.attack]

    selected_attacks = ", ".join(name for name, _ in attack_grids)
    print(
        f"Starting hyperparameter search: attacks={selected_attacks} "
        f"images={image_count} rounds={rounds} smoke={args.smoke}"
    )

    for attack_name, grid in attack_grids:
        attack_raw_rows, attack_summary_rows, attack_best_rows = run_attack_search(
            attack_name,
            grid,
            trials,
            rounds,
            values_per_parameter,
            image_count,
        )
        raw_rows.extend(attack_raw_rows)
        summary_rows.extend(attack_summary_rows)
        round_best_rows.extend(attack_best_rows)

    raw_rows.sort(key=lambda r: (r["Attack"], r["Round"], r["Candidate"], r["Image"]))
    summary_rows.sort(key=lambda r: (r["Attack"], r["Round"], r["Candidate"]))
    round_best_rows.sort(key=lambda r: (r["Attack"], r["Round"]))

    write_outputs(Path(args.output_dir), raw_rows, summary_rows, round_best_rows, image_count)


if __name__ == "__main__":
    main()
