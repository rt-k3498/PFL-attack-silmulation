import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_RESULTS_CSV = PROJECT_ROOT / "results" / "attack_results" / "raw_results.csv"
ALGORITHM_RESULTS_CSV = PROJECT_ROOT / "results" / "algorithm_results" / "raw_results.csv"


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it to use the plot_* helpers."
        ) from exc
    return plt


def _maybe_dataframe(rows: List[Dict[str, Any]]):
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return rows
    return pd.DataFrame(rows)


def _records(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        return data.to_dict("records")
    return list(data)


def _to_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _load_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def combine_csvs(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
) -> Path:
    """
    Read every CSV in `input_paths`, take the union of their column headers
    (preserving first-seen order), concatenate all rows, and write the result
    to `output_path`. Missing columns are written as empty strings. Returns
    the resolved output path.
    """
    fieldnames: List[str] = []
    seen: set[str] = set()
    all_rows: List[Dict[str, str]] = []

    for path in input_paths:
        rows = _load_csv(path)
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        all_rows.extend(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return output_path


def load_attack_results(path: str | Path = ATTACK_RESULTS_CSV):
    return _maybe_dataframe(_load_csv(path))


def load_algorithm_results(path: str | Path = ALGORITHM_RESULTS_CSV):
    return _maybe_dataframe(_load_csv(path))


def attack_summary(data: Any = None):
    records = _records(load_attack_results() if data is None else data)
    grouped: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {
        "rows": 0,
        "input mse": [],
        "input psnr": [],
        "input ssim": [],
    })

    for row in records:
        key = (row.get("algorithm"), row.get("attack"))
        grouped[key]["rows"] += 1
        for metric in ("input mse", "input psnr", "input ssim"):
            value = _to_float(row.get(metric))
            if value is not None:
                grouped[key][metric].append(value)

    summary = []
    for (algorithm, attack), values in grouped.items():
        mse = values["input mse"]
        psnr = values["input psnr"]
        ssim = values["input ssim"]
        summary.append({
            "algorithm": algorithm,
            "attack": attack,
            "rows": values["rows"],
            "mean_mse": _mean(mse),
            "min_mse": min(mse) if mse else None,
            "max_mse": max(mse) if mse else None,
            "mean_psnr": _mean(psnr),
            "min_psnr": min(psnr) if psnr else None,
            "max_psnr": max(psnr) if psnr else None,
            "mean_ssim": _mean(ssim),
            "min_ssim": min(ssim) if ssim else None,
            "max_ssim": max(ssim) if ssim else None,
        })

    summary.sort(key=lambda row: (row["algorithm"] or "", row["attack"] or ""))
    return _maybe_dataframe(summary)


def algorithm_summary(data: Any = None):
    records = _records(load_algorithm_results() if data is None else data)
    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "rows": 0,
        "output crossentropy": [],
        "correct": 0,
        "predictions": 0,
    })

    for row in records:
        key = row.get("algorithm")
        grouped[key]["rows"] += 1
        loss = _to_float(row.get("output crossentropy"))
        if loss is not None:
            grouped[key]["output crossentropy"].append(loss)

        test_label = row.get("test label")
        predicted_label = row.get("predicted label")
        if test_label not in (None, "") and predicted_label not in (None, ""):
            grouped[key]["predictions"] += 1
            grouped[key]["correct"] += int(str(test_label) == str(predicted_label))

    summary = []
    for algorithm, values in grouped.items():
        losses = values["output crossentropy"]
        predictions = values["predictions"]
        summary.append({
            "algorithm": algorithm,
            "rows": values["rows"],
            "mean_crossentropy": _mean(losses),
            "min_crossentropy": min(losses) if losses else None,
            "max_crossentropy": max(losses) if losses else None,
            "accuracy": values["correct"] / predictions if predictions else None,
        })

    summary.sort(key=lambda row: row["algorithm"] or "")
    return _maybe_dataframe(summary)


def attack_metric_by_round(data: Any = None, metric: str = "input mse"):
    records = _records(load_attack_results() if data is None else data)
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in records:
        value = _to_float(row.get(metric))
        if value is None:
            continue
        key = (row.get("algorithm"), row.get("attack"), row.get("communication_round"))
        grouped[key].append(value)

    rows = [
        {
            "algorithm": algorithm,
            "attack": attack,
            "communication_round": int(round_idx),
            metric: _mean(values),
        }
        for (algorithm, attack, round_idx), values in grouped.items()
    ]
    rows.sort(key=lambda row: (row["algorithm"] or "", row["attack"] or "", row["communication_round"]))
    return _maybe_dataframe(rows)


def plot_attack_metric(
    data: Any = None,
    metric: str = "input mse",
    title: str | None = None,
):
    plt = _load_pyplot()
    metric_map = {
        "input mse": "mean_mse",
        "input psnr": "mean_psnr",
        "input ssim": "mean_ssim",
    }
    value_column = metric_map.get(metric, metric)
    records = _records(attack_summary(data))
    algorithms = sorted({row["algorithm"] for row in records})
    attacks = sorted({row["attack"] for row in records})

    fig, axis = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(len(attacks), 1)
    positions = list(range(len(algorithms)))
    for attack_idx, attack in enumerate(attacks):
        values_by_algorithm = {
            row["algorithm"]: row.get(value_column)
            for row in records
            if row["attack"] == attack
        }
        offsets = [pos + attack_idx * width for pos in positions]
        values = [values_by_algorithm.get(algorithm, 0) for algorithm in algorithms]
        axis.bar(offsets, values, width=width, label=attack)

    axis.set_title(title or f"Mean {metric} by algorithm and attack")
    axis.set_xlabel("Algorithm")
    axis.set_ylabel(metric)
    axis.set_xticks([pos + width * (len(attacks) - 1) / 2 for pos in positions])
    axis.set_xticklabels(algorithms, rotation=45, ha="right")
    axis.legend(title="Attack")
    fig.tight_layout()
    return axis


def plot_attack_metric_by_round(data: Any = None, metric: str = "input mse"):
    plt = _load_pyplot()
    records = _records(attack_metric_by_round(data, metric))
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["algorithm"], row["attack"])].append(row)

    fig, axis = plt.subplots(figsize=(12, 5))
    for (algorithm, attack), rows in grouped.items():
        rows.sort(key=lambda row: row["communication_round"])
        axis.plot(
            [row["communication_round"] for row in rows],
            [row[metric] for row in rows],
            marker="o",
            label=f"{algorithm} / {attack}",
        )
    axis.set_title(f"Mean {metric} by communication round")
    axis.set_xlabel("Communication round")
    axis.set_ylabel(metric)
    axis.legend()
    fig.tight_layout()
    return axis


def plot_algorithm_crossentropy(data: Any = None):
    plt = _load_pyplot()
    records = _records(algorithm_summary(data))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        [row["algorithm"] for row in records],
        [row["mean_crossentropy"] for row in records],
    )
    axis.set_title("Mean output crossentropy by algorithm")
    axis.set_xlabel("Algorithm")
    axis.set_ylabel("Output crossentropy")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return axis


def plot_algorithm_accuracy(data: Any = None):
    plt = _load_pyplot()
    records = _records(algorithm_summary(data))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        [row["algorithm"] for row in records],
        [row["accuracy"] for row in records],
    )
    axis.set_title("Prediction accuracy by algorithm")
    axis.set_xlabel("Algorithm")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return axis
