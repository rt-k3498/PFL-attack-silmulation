# Hyperparameter Optimization Summary: Calibrating Gradient Inversion Attacks

## 1. Executive Summary

`hyperparameter_simulation.py` is a calibration experiment for the attack side of the project. Its role is not to compare federated learning algorithms directly. Instead, it tunes the reconstruction hyperparameters of the two gradient inversion attacks used later in the main simulation:

- `DLG`
- `InvertingGradients`

The experiment asks:

> Under a controlled one-client FedAvg reconstruction setting, which attack hyperparameters produce the most accurate image reconstruction?

The selected attack settings are then fed into `simulations/simulation.py`, where the attacks are used as fixed evaluation instruments against the full algorithm set:

- `FedAvg`
- `FedPer(K_p=1..5)`
- `Per-FedAvg(FO)`
- `Per-FedAvg(HF)`
- `Per-FedAvg(HVP)`

This separation is methodologically important. The hyperparameter optimization stage makes the attack implementations competitive before the main privacy comparison begins. Otherwise, a poor reconstruction result in the main simulation might reflect weak attack settings rather than genuine privacy robustness of the learning algorithm.

In short:

```text
hyperparameter_simulation.py tunes the attacks
simulations/simulation.py uses the tuned attacks to evaluate FL/PFL privacy leakage
```

## 2. Scientific Purpose

Gradient inversion attacks are optimization procedures. Their success depends strongly on hyperparameters such as:

- optimizer step size
- stopping tolerances
- line-search budget
- L-BFGS history size
- image regularization strength

If these values are arbitrary, the main experiment becomes hard to interpret. A weak attack configuration can make an algorithm look private even when a better-tuned attack could reconstruct the data.

The hyperparameter experiment therefore serves three scientific purposes:

1. **Attack calibration**: find strong settings for each reconstruction attack.
2. **Fair comparison**: use fixed, preselected attack settings when comparing learning algorithms.
3. **Justification**: provide empirical evidence for why the attack settings in `simulation.py` were chosen.

The experiment is best viewed as a pre-study or calibration phase. It chooses the attack instruments used in the main privacy experiment.

## 3. Position in the Overall Project

The project contains two experimental layers.

### Layer 1: Attack Hyperparameter Optimization

File:

```text
hyperparameter_simulation.py
```

Purpose:

```text
Tune DLG and InvertingGradients on a simple FedAvg reconstruction task.
```

Output:

```text
results/hyperparameters/dlg/raw_results.csv
results/hyperparameters/dlg/results.txt
results/hyperparameters/inverting_gradients/raw_results.csv
results/hyperparameters/inverting_gradients/results.txt
```

### Layer 2: Main Privacy-Utility Simulation

File:

```text
simulations/simulation.py
```

Purpose:

```text
Use the tuned attacks to compare privacy leakage across FedAvg, FedPer, and Per-FedAvg variants.
```

Output:

```text
results/attack_results/raw_results.csv
results/algorithm_results/raw_results.csv
results/true_images/
results/reconstructed_images/
```

The connection is direct: the attack settings selected in the hyperparameter stage appear as the fixed attack configurations in the main simulation.

## 4. Main Design Principle

The hyperparameter experiment intentionally uses a simpler setting than the full simulation.

Instead of testing all algorithms, all clients, all rounds, and all personalized variants, it focuses on a minimal reconstruction case:

```text
one CIFAR-10 image -> one client -> one FedAvg-style local update -> one attack reconstruction
```

This is a clean calibration environment because (Reasoning):

- the attacks were tuned to fedavg case as the actual experiment was a controlled comparison between the effectiveness of the attacks in a tradietional FL setting vs PFL setting, and the FedAvg case is the most traditional FL setting. Tuning on FedAvg therefore ensures that the attacks are strong in the setting they were originally designed for, which provides a solid baseline for comparison when they are later applied to the more complex PFL settings, and also enables a fair comparison of the attacks' performance in their intended context before testing their effectivness across different algorithms. hence attributing the results in the main simulation to the algorithm's privacy structure and design.
- a single image was used to keep computational costs lower.
- the same target can be reused across many candidate hyperparameter settings
- reconstruction quality can be compared directly by MSE, PSNR, and SSIM

The calibrated attack is then moved into the harder main simulation.

## 5. Experimental Constants

The top-level constants in `hyperparameter_simulation.py` define the search:

| Constant | Value | Meaning |
|---|---:|---|
| `SEED` | 50 | base seed for data, model, and attack initialization |
| `IMAGE_COUNT` | 1 | default number of CIFAR-10 images used for non-smoke search |
| `ROUNDS` | 3 | number of grid-refinement rounds / searching rounds |
| `VALUES_PER_PARAMETER` | 3 | three-point grid per parameter per round |
| `MAX_ITERATIONS` | 300 | reconstruction iterations for every attack candidate |
| `OUTPUT_DIR` | `results/hyperparameters` | default output folder |

The choice of `MAX_ITERATIONS = 300` is especially important because the main simulation also uses 300 reconstruction iterations. Thus the tuning is performed under the same reconstruction budget later used in the main comparison.

## 6. Data Used in the Hyperparameter Search

The search uses CIFAR-10 through `CIFAR10Data`.

Data preprocessing is the same basic preprocessing used elsewhere in the project:

- load CIFAR-10 via TensorFlow Datasets
- normalize image pixels to `[0, 1]`
- one-hot encode labels over 10 classes

The hyperparameter search uses:

```python
data.get_x_y(batch_size=1, number_of_batches=image_count)
```

Conceptually, this means:

```text
sample image_count single-image batches from CIFAR-10
```

With the default `IMAGE_COUNT = 1`, every hyperparameter candidate is evaluated on one reconstructed image.

If `--image-count N` is supplied, each candidate is evaluated on `N` independent one-image reconstruction trials, and the metrics are averaged across those images.

## 7. Trial Construction

The function:

```python
prepare_fedavg_trials(x_data_list, y_data_list)
```

constructs the deterministic reconstruction targets.

For each selected image:

1. Compute a trial seed:

```text
trial_seed(image_idx) = SEED + image_idx
```

2. Reseed NumPy, TensorFlow, and Python `random`.
3. Create a fresh LeNet model.
4. Create a single client.
5. Assign the selected CIFAR-10 image and label to that client.
6. Clone the global model into a local model.
7. Perform one local supervised training step using categorical cross-entropy.
8. Store:

- image index
- trial seed
- global model before training
- client weights after the one-step update
- the private image and label used for training
- client id

The stored object is a complete reconstruction trial:

```text
known global model + submitted client weights + known learning rate
```

This is precisely what a gradient inversion attack needs.

## 9. Model Used in the Search

The search uses the same model family as the main simulation:

```text
LeNet
```

The architecture is:

| Stage | Layer | Details |
|---|---|---|
| input | Input | `(32, 32, 3)` |
| C1 | Conv2D | 6 filters, 5x5, sigmoid |
| S2 | AveragePooling2D | 2x2 |
| C3 | Conv2D | 16 filters, 5x5, sigmoid |
| S4 | AveragePooling2D | 2x2 |
| C5 | Conv2D | 120 filters, 5x5, sigmoid |
| flatten | Flatten | vectorization |
| F6 | Dense | 256 units, sigmoid |
| F7 | Dense | 128 units, sigmoid |
| output | Dense | 10 units, softmax |

This matters because gradient inversion hyperparameters are not fully architecture-independent. A setting tuned on a shallow linear model would not necessarily transfer to this CNN. By tuning on LeNet, the calibration stage is aligned with the main simulation.

## 10. Attack Knowledge Assumptions

For every candidate setting, the attack is given:

- the global model before local training
- the client model weights after local training
- learning rate `0.1`
- number of classes `10`

The attack then computes an observed gradient-like signal:

```text
observed_gradient = (global_weight - client_weight) / 0.1
```

This is a white-box gradient inversion setting. The attacker knows the model architecture, the current parameters, and the learning rate. This is standard for evaluating gradient leakage in federated learning because the server usually has the global model and receives client updates.

## 11. Reproducibility and Seeding

The script defines:

```python
reseed(seed)
```

which sets:

- NumPy random seed
- TensorFlow random seed
- Python `random` seed

It also wraps each attack in `_SeededAttack`, which reseeds immediately before reconstruction.

This matters because DLG and InvertingGradients both initialize dummy images and labels randomly. Without reseeding before every candidate, two candidates might differ partly because they got different random initializations, not because their hyperparameters are better.

The design objective is:

```text
same image + same model + same attack initialization + different hyperparameters
```

That makes candidate comparisons cleaner.

## 12. Candidate Evaluation Workflow

The core evaluation function is:

```python
run_candidate_on_image(attack_name, settings, trial)
```

For one attack candidate and one image trial, it:

1. Reseeds to the trial seed.
2. Constructs the attack object with the candidate settings.
3. Runs the attack against the stored global model and client weights.
4. Measures reconstruction quality using:

- MSE
- PSNR
- SSIM

5. Returns one metric row:

```text
Client, Input MSE, Input PSNR, Input SSIM
```

The next layer:

```python
run_candidate(...)
```

applies the same candidate to every selected image trial.

If `image_count = 1`, one candidate produces one row.

If `image_count = N`, one candidate produces `N` rows, and later aggregation averages over those `N` rows.

## 13. Objective Function

The primary optimization objective is:

```text
minimize average input MSE
```

MSE is computed as:

```text
mean((reconstructed_image - true_image)^2)
```

over normalized `[0, 1]` pixels.

The script also reports:

- average PSNR
- average SSIM

but candidate selection is explicitly based on the lowest average MSE.

This means the optimization criterion is pixel fidelity, not perceptual similarity or label accuracy.

That is a defensible choice for attack calibration because MSE is:

- direct
- stable
- differentiable in spirit, even though it is used only for evaluation here
- easy to compare across candidates

PSNR and SSIM provide complementary interpretive context:

- lower MSE should usually correspond to higher PSNR
- higher SSIM indicates more structural visual similarity

## 14. Grid Search Strategy

The search is a multi-round, coarse-to-fine grid search.

Each attack starts with an initial grid. In each round:

1. Generate all candidate combinations.
2. Evaluate each candidate on the selected image trials.
3. Aggregate metrics by candidate.
4. Choose the candidate with the lowest average MSE.
5. Refine the grid around that winner.

This is repeated for `ROUNDS = 3`.

The design is local search over a small parameter grid rather than exhaustive global optimization. It is practical because attack reconstruction is expensive.

## 15. Refinement Rule

The refinement rule assumes each parameter has exactly 3 values.

For each parameter:

1. Sort the previous three values.
2. Compute the average adjacent gap.
3. Halve that gap.
4. Build a new three-value grid around the previous best:

```text
[best - new_scale, best, best + new_scale]
```

For positive floating-point parameters:

```text
values are clamped to a tiny positive floor
```

For integer parameters:

```text
values are rounded and clamped to at least 1
```

The effect is a zoom-in procedure:

```text
round 1: broad search
round 2: narrower search around round 1 best
round 3: narrower search around round 2 best
```

This gives a computationally affordable approximation to hyperparameter optimization.

## 16. DLG Search Space

The DLG initial grid is:

```python
DLG_INITIAL_GRID = {
    "num_correction_pairs": [50, 100, 150],
    "max_line_search_iterations": [100, 200, 300],
    "tolerance": [1e-18, 1e-15, 1e-12],
    "f_relative_tolerance": [1e-18, 1e-15, 1e-12],
}
```

DLG always also receives:

```text
max_iterations = 300
```

### 16.1 `num_correction_pairs`

This controls the memory/history size used by L-BFGS.

Higher values allow L-BFGS to store more curvature information. This can improve optimization quality but increases memory and computation.

In DLG, the optimization problem is high-dimensional:

```text
32 * 32 * 3 image variables + 10 label variables = 3082 variables
```

More correction pairs can help the optimizer navigate this landscape.

### 16.2 `max_line_search_iterations`

L-BFGS uses line search to choose a step size along a search direction.

This parameter controls how much effort is spent finding an acceptable step at each iteration.

Too small a value can cause premature or poor steps. Too large a value increases runtime.

### 16.3 `tolerance`

This is an L-BFGS convergence threshold. Smaller values demand tighter convergence before stopping.

In a reconstruction setting, overly loose tolerance may stop before the dummy image has matched the observed gradient closely.

### 16.4 `f_relative_tolerance`

This controls stopping based on relative objective improvement.

If the objective stops improving enough, L-BFGS may terminate. Tuning this parameter matters because gradient matching losses can have long, slow improvement phases.

## 17. InvertingGradients Search Space

The InvertingGradients initial grid is:

```python
INVERTING_GRADIENTS_INITIAL_GRID = {
    "init_step_size": [0.01, 0.1, 1.0],
    "final_step_size": [0.001, 0.01, 0.1],
    "alpha": [1e-14, 1e-13, 1e-12],
}
```

InvertingGradients always also receives:

```text
max_iterations = 300
```

### 17.1 `init_step_size`

This is the initial learning rate for Adam.

If it is too small, reconstruction may barely move from initialization within 300 iterations. If it is too large, optimization may be unstable.

### 17.2 `final_step_size`

This is the final learning rate target in the cosine decay schedule.

The attack uses:

```python
tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=init_step_size,
    decay_steps=max_iterations,
    alpha=final_step_size / init_step_size,
)
```

Thus the optimizer gradually decays from `init_step_size` toward `final_step_size`.

The grid generator excludes candidates where:

```text
final_step_size > init_step_size
```

This prevents schedules that increase rather than decay.

### 17.3 `alpha`

In InvertingGradients, `alpha` weights the total variation regularization term:

```text
loss = cosine_gradient_loss + alpha * total_variation(image)
```

This is not the FedAvg learning rate. It is the image prior strength.

A larger `alpha` encourages smoother reconstructions. A smaller `alpha` makes the attack rely almost entirely on gradient direction matching.

## 18. DLG Attack Objective

DLG reconstructs the private image and label by matching full gradients.

The optimization variables are:

```text
dummy image logits
dummy label logits
```

The dummy image is converted to valid pixel space by sigmoid:

```text
dummy_image = sigmoid(raw_image)
```

The dummy label is converted to a soft label by softmax:

```text
dummy_label = softmax(raw_label)
```

DLG computes the gradient of cross-entropy loss for the dummy image and dummy label. It then minimizes:

```text
sum over layers ||dummy_gradient_l - observed_gradient_l||^2
```

The intuition is:

> If a dummy image produces the same model gradient as the real private image, then the dummy image is likely close to the real private image.

The tuned DLG hyperparameters are therefore optimization-control parameters for the L-BFGS solver.

## 19. InvertingGradients Attack Objective

InvertingGradients reconstructs the private image and label by matching gradient direction rather than squared gradient magnitude.

It flattens the observed gradients and dummy gradients, then computes:

```text
cosine_loss = 1 - dot(dummy_gradient, observed_gradient)
                / (||dummy_gradient|| * ||observed_gradient||)
```

It adds a total variation prior:

```text
gradient_loss = cosine_loss + alpha * total_variation(dummy_image)
```

The optimization variables are:

- raw image tensor
- dummy label logits

The optimizer is Adam with a cosine-decayed learning rate schedule.

The tuned InvertingGradients hyperparameters therefore control:

- optimization speed
- final step refinement
- strength of image smoothness prior

## 20. Metrics Used for Candidate Ranking

The hyperparameter search uses three reconstruction metrics.

### 20.1 Input MSE

File:

```text
metrics/MSE_metric.py
```

Definition:

```text
mean((reconstructed_input - actual_input)^2)
```

Lower is better.

This is the selection criterion.

### 20.2 Input PSNR

File:

```text
metrics/PSNR_metric.py
```

Definition:

```text
tf.image.psnr(reconstructed, actual, max_val=1.0)
```

Higher is better.

PSNR is a signal-quality metric derived from reconstruction error.

### 20.3 Input SSIM

File:

```text
metrics/SSIM_metric.py
```

Definition:

```text
tf.image.ssim(reconstructed, actual, max_val=1.0, filter_size=11)
```

Higher is better.

SSIM captures structural similarity and can better reflect whether the reconstruction is visually recognizable.

## 21. Output Format

For each run, the script writes:

```text
raw_results.csv
results.txt
```

### 21.1 Raw CSV

The CSV fields are:

```text
Attack
Round
Candidate
num_correction_pairs
max_line_search_iterations
tolerance
f_relative_tolerance
init_step_size
final_step_size
alpha
Image
Client
Input MSE
Input PSNR
Input SSIM
```

DLG rows populate the L-BFGS fields. InvertingGradients rows populate the Adam and regularization fields.

### 21.2 Text Summary

`results.txt` contains:

1. raw hyperparameter rows
2. average per candidate
3. best setting per round
4. final best setting per attack

The final best row is selected from the best rows of the refinement rounds, again by lowest average MSE.

## 22. Current Saved Search Outputs

The current workspace contains:

| File | Rows including header | Data rows |
|---|---:|---:|
| `results/hyperparameters/dlg/raw_results.csv` | 244 | 243 |
| `results/hyperparameters/inverting_gradients/raw_results.csv` | 58 | 57 |

With `image_count = 1`, each data row corresponds to one candidate evaluated on one reconstructed image.

DLG has 243 rows because:

```text
3 rounds * 81 candidates per round * 1 image = 243 rows
```

The DLG grid has four parameters with three values each:

```text
3^4 = 81 candidates per round
```

InvertingGradients has fewer candidates because candidates with:

```text
final_step_size > init_step_size
```

are filtered out.

## 23. DLG Saved Best Settings

The saved DLG search reports the following best setting per round:

| Round | Candidate | `num_correction_pairs` | `max_line_search_iterations` | `tolerance` | `f_relative_tolerance` | Avg MSE | Avg PSNR | Avg SSIM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 78 | 150 | 300 | `1e-15` | `1e-12` | 0.011621 | 19.3476 | 0.794811 |
| 2 | 69 | 175 | 300 | `1e-15` | `1.25e-12` | 0.0115721 | 19.3659 | 0.795122 |
| 3 | 8 | 162 | 275 | `6.37499e-14` | `1.25e-12` | 0.0115835 | 19.3616 | 0.794995 |

The final best setting is the round 2 setting:

```python
DLG(
    max_iterations=300,
    num_correction_pairs=175,
    max_line_search_iterations=300,
    tolerance=1e-15,
    f_relative_tolerance=1.25e-12,
)
```

This matches the DLG settings used in `simulations/simulation.py`:

```python
"DLG": lambda: DLG(seed=seed, settings={
    "max_iterations": 300,
    "f_relative_tolerance": 1.25e-12,
    "max_line_search_iterations": 300,
    "num_correction_pairs": 175,
    "tolerance": 1e-15,
})
```

This is the cleanest example of the hyperparameter experiment feeding directly into the main simulation.

## 24. InvertingGradients Saved Best Settings

The saved InvertingGradients search reports the following best setting per round:

| Round | Candidate | `init_step_size` | `final_step_size` | `alpha` | Avg MSE | Avg PSNR | Avg SSIM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 0.1 | 0.1 | `1e-12` | 0.00899388 | 20.4605 | 0.804132 |
| 2 | 1 | 0.1 | 0.07525 | `7.525e-13` | 0.00828956 | 20.8147 | 0.818488 |
| 3 | 7 | 0.1 | 0.087625 | `6.2875e-13` | 0.0080191 | 20.9587 | 0.816465 |

The final best setting is:

```python
InvertingGradients(
    max_iterations=300,
    init_step_size=0.1,
    final_step_size=0.087625,
    alpha=6.2875e-13,
)
```

The main simulation uses a rounded nearby setting:

```python
"InvertingGradients": lambda: InvertingGradients(seed=seed, settings={
    "max_iterations": 300,
    "init_step_size": 0.1,
    "final_step_size": 0.09,
    "alpha": 6.3e-13,
})
```

So the main simulation appears to use the hyperparameter-search result rounded to simpler numeric values:

```text
0.087625 -> 0.09
6.2875e-13 -> 6.3e-13
```

The conceptual transfer is still clear: the search found that a relatively high and nearly flat Adam step schedule with a very small total-variation coefficient was strongest under the calibration setting.

## 25. Why the Search Uses MSE as the Selection Criterion

Selecting by MSE makes the attack calibration conservative in a useful way.

If an attack has low MSE, the reconstructed image is numerically close to the true private image. In gradient inversion research, this is a strong privacy failure signal because the attacker has recovered substantial pixel-level information.

SSIM and PSNR are still reported because MSE alone can miss perceptual structure. However, using all metrics as a multi-objective optimization problem would require defining tradeoffs:

- Is a small MSE improvement worth a lower SSIM?
- Is a higher PSNR always visually better?
- Should label recovery be weighted?

The project avoids these complications by using MSE as the primary objective and treating PSNR/SSIM as supporting evidence.

## 26. Why This Supports the Main Simulation

The main simulation wants to compare algorithms, not tune attacks separately for each algorithm.

If every algorithm had separately tuned attack settings, then the privacy comparison would become entangled with per-algorithm attack calibration. Instead, this project uses one tuned DLG configuration and one tuned InvertingGradients configuration across all algorithms.

That makes the main experiment easier to interpret:

```text
Given the same attack implementation and same calibrated attack settings,
which learning algorithm leaks more under reconstruction?
```

The hyperparameter stage therefore establishes the attack settings before the main comparison, reducing the risk of post-hoc tuning.

## 27. How the Tuned Attacks Are Used in `simulation.py`

In `simulations/simulation.py`, the tuned attacks are placed in the `attacks` dictionary:

```python
attacks = {
    "DLG": lambda: DLG(... tuned settings ...),
    "InvertingGradients": lambda: InvertingGradients(... tuned settings ...),
}
```

The simulation then loops over:

```text
run
algorithm
attack
```

For each algorithm-attack pair:

1. initialize a fresh LeNet model
2. initialize 10 clients with structured CIFAR-10 data
3. run 5 communication rounds
4. after each client local update, run the tuned attack
5. record reconstruction metrics and saved images
6. aggregate client updates
7. evaluate final model utility

The tuned attack settings therefore determine how strong the reconstruction attempt is for every algorithm in the main comparison.

## 28. Relationship to FedPer and Per-FedAvg

The hyperparameter search tunes on FedAvg, but the main simulation applies the tuned attacks to FedPer and Per-FedAvg too.

This is a reasonable design choice for an initial thesis experiment because:

- FedAvg gives the cleanest one-step gradient inversion target.
- The attacks themselves are generic gradient/update inversion procedures.
- Tuning separately on every algorithm could bias the comparison.
- The same fixed settings provide a common attack baseline.

There is also an important limitation:

```text
The FedAvg-tuned hyperparameters may not be globally optimal for FedPer partial updates or Per-FedAvg meta-updates.
```

This does not invalidate the experiment, but it should shape the wording of conclusions.

A careful interpretation is:

> The main simulation compares algorithms under a common FedAvg-calibrated attack configuration.

Not:

> The main simulation proves each algorithm is secure against its individually strongest possible attack.

## 29. Methodological Strengths

The hyperparameter optimization stage has several strengths.

### 29.1 It Separates Calibration from Evaluation

The attack settings are selected before the main FL/PFL comparison. This is good experimental hygiene.

### 29.2 It Uses the Same Architecture

The search uses LeNet, the same model used in the main simulation.

### 29.3 It Uses the Same Dataset Family

Both calibration and main simulation use CIFAR-10.

### 29.4 It Uses the Same Reconstruction Budget

Both stages use:

```text
max_iterations = 300
```

This ensures that tuned settings are appropriate for the runtime budget used in the main experiment.

### 29.5 It Controls Randomness

The `_SeededAttack` wrapper makes candidate comparisons less sensitive to random initialization differences.

### 29.6 It Reports Multiple Metrics

Although selection uses MSE, the output includes PSNR and SSIM, giving a richer reconstruction-quality picture.

## 30. Methodological Limitations

The hyperparameter optimization also has limitations that should be acknowledged.

### 30.1 Default Image Count Is Very Small

The default `IMAGE_COUNT = 1` means the saved searches optimize on one image.

This can overfit attack settings to that image. A stronger calibration would use more images and report mean plus variance.

### 30.2 Selection Is Based Only on MSE

MSE is useful, but it is not the only meaningful reconstruction metric.

An attack setting with slightly worse MSE but better SSIM might produce more recognizable images. The current selection rule does not account for that.

### 30.3 FedAvg Calibration May Not Be Optimal for All Algorithms

FedPer exposes partial gradients. Per-FedAvg exposes meta-update directions. Their inversion landscapes may differ from FedAvg.

The current design intentionally uses common attack settings, which supports fair comparison, but it may not represent each algorithm's worst-case tuned attack.

### 30.4 Search Space Is Manually Chosen

The initial grids encode assumptions about plausible ranges.

If the true best settings lie outside those ranges, the search will not find them.

### 30.5 Three-Round Refinement Can Find Local Winners

The coarse-to-fine grid search refines around the best current candidate. If round 1 selects a locally good region that is not globally best, later rounds continue searching near that local region.

This is a reasonable engineering compromise, but not a guarantee of global optimality.

## 31. Interpretation of Current Best Results

### 31.1 DLG

The DLG search found a best MSE around:

```text
0.01157
```

with:

```text
PSNR about 19.37
SSIM about 0.795
```

The selected settings favor:

- large L-BFGS memory/history (`num_correction_pairs = 175`)
- large line-search budget (`max_line_search_iterations = 300`)
- tight convergence tolerance (`tolerance = 1e-15`)
- relative tolerance near `1.25e-12`

Interpretation:

> DLG benefits from a fairly aggressive and high-budget L-BFGS configuration under this reconstruction task.

### 31.2 InvertingGradients

The InvertingGradients search found a best MSE around:

```text
0.00802
```

with:

```text
PSNR about 20.96
SSIM about 0.816
```

The selected settings favor:

- initial step size `0.1`
- final step size close to the initial step size
- very small total variation weight

Interpretation:

> In this calibration case, InvertingGradients reconstructs best when Adam keeps taking relatively large steps across the 300-iteration budget, while total variation regularization remains very weak.

### 31.3 Comparison Between Attacks in the Calibration Case

On the saved one-image calibration outputs, InvertingGradients achieved lower MSE than DLG:

```text
DLG best MSE:                 about 0.01157
InvertingGradients best MSE:  about 0.00802
```

This does not prove InvertingGradients is always stronger, but it suggests it was stronger on the specific calibration image and search space used here.

The main simulation is needed to compare the attacks across:

- algorithms
- rounds
- clients
- class-structured client data
- partial FedPer updates
- Per-FedAvg meta-updates

## 32. Recommended Thesis Wording

A strong and precise explanation would be:

> Before running the full PFL privacy comparison, I calibrated the two gradient inversion attacks on a controlled one-client FedAvg reconstruction task. For each candidate hyperparameter setting, the attack attempted to reconstruct a private CIFAR-10 image from a known LeNet model update. Candidates were evaluated by pixel MSE, PSNR, and SSIM, and the search selected the setting with the lowest average MSE. The selected DLG and InvertingGradients settings were then fixed and used unchanged in the main simulation across FedAvg, FedPer, and Per-FedAvg variants. This procedure separates attack calibration from algorithm comparison and reduces the chance that observed privacy differences are artifacts of arbitrary attack hyperparameters.

## 33. Final Conceptual Summary

The hyperparameter optimization experiment is not an auxiliary script in the trivial sense. It is the calibration stage that defines the strength of the adversary used in the main experiment.

Its logic is:

```text
1. Construct a simple, deterministic private-image leakage case.
2. Try many DLG and InvertingGradients settings.
3. Measure how accurately each setting reconstructs the private image.
4. Select the best setting by lowest MSE.
5. Use those tuned settings in the full FL/PFL simulation.
```

The main simulation can then make a cleaner claim:

```text
privacy leakage is compared across algorithms under fixed, empirically tuned attack configurations
```

That is the key justification for this experiment.

