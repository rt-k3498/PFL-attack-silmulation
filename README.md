# PFL-attack-silmulation

## Setup

Install the simulation dependencies from the project root:

```bash
make install
```

## Main simulation

Run the main simulation from the project root:

```bash
make run
```

## Hyperparameter search

`hyperparameter_simulation.py` runs a FedAvg-based gradient inversion hyperparameter search for the `DLG` and `InvertingGradients` attacks. For each attack, it:

1. Builds deterministic one-client reconstruction trials from sampled CIFAR-10 images.
2. Evaluates every parameter combination in the current grid on the selected images.
3. Scores each candidate by averaging reconstruction metrics across those images.
4. Chooses the candidate with the lowest average input MSE.
5. Refines the next round around that winner by shrinking the parameter scale.

The search is attack-specific, so you can run `DLG`, `InvertingGradients`, or both in sequence.

### Refinement rule

For non-smoke runs, refinement is a 3-point search:

- each parameter must have exactly 3 values in the current grid
- the current scale is the average adjacent gap across those 3 sorted values
- the next scale is half of that current scale
- the next 3 values are `[best - new_scale, best, best + new_scale]`
- integer parameters are rounded to the nearest whole number and clamped to stay positive
- positive float parameters are clamped to a tiny positive floor if needed

### Commands

Run only DLG:

```bash
python hyperparameter_simulation.py --attack DLG --output-dir results/hyperparameters/dlg
```

Run only InvertingGradients:

```bash
python hyperparameter_simulation.py --attack InvertingGradients --output-dir results/hyperparameters/inverting_gradients
```

Run both attacks:

```bash
python hyperparameter_simulation.py --attack all
```

Override image count:

```bash
python hyperparameter_simulation.py --attack DLG --image-count 1
```

Run a quick verification pass:

```bash
python hyperparameter_simulation.py --smoke --attack DLG
```

Write results to a separate directory:

```bash
python hyperparameter_simulation.py --attack DLG --output-dir results/hyperparameters/dlg
```

### CLI options

- `--attack {DLG,InvertingGradients,all}` chooses which attack search to run
- `--image-count N` sets how many images are evaluated in non-smoke runs
- `--smoke` forces a one-image, one-round, one-candidate verification run
- `--output-dir PATH` chooses where `raw_results.csv` and `results.txt` are written

### Output files

Each run writes:

- `raw_results.csv`: one row per reconstructed sample
- `results.txt`: the formatted console summary with raw rows, per-candidate averages, and best settings

If you want separate stored results for different attacks, use different `--output-dir` values for each run so the files do not overwrite each other.
    
