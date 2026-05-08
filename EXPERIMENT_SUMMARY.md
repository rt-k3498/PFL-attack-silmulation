# Experiment Summary: Personalized Federated Learning Under Gradient Inversion Attacks

## 1. Executive Summary

This project simulates a privacy attack experiment for federated learning and personalized federated learning (PFL). The core question is:

> When clients train locally and send model updates to a server, how much private training data can an attacker reconstruct from those updates, and how does this leakage change across FedAvg, FedPer, and Per-FedAvg variants?

The main experiment is implemented in `simulations/simulation.py`. It compares:

- `FedAvg`
- `FedPer(K_p=1)` through `FedPer(K_p=5)`
- `Per-FedAvg(FO)`
- `Per-FedAvg(HF)`
- `Per-FedAvg(HVP)`

against two gradient inversion attacks:

- `DLG`
- `InvertingGradients`

The dataset is CIFAR-10. Each client receives a small, class-structured local dataset. During each communication round, every client trains on exactly one local sample, sends a model update, and the attack attempts to reconstruct that sample from the update before aggregation occurs.

The experiment records two categories of results:

- Attack privacy results: reconstructed input, reconstructed label, MSE, PSNR, SSIM, and saved image pairs.
- Algorithm utility results: predicted labels and output cross-entropy on structured CIFAR-10 test batches.

The experiment is best understood as a controlled privacy-utility comparison across FL/PFL algorithms under single-sample gradient/update inversion.

## 3. High-Level Experimental Design

The experiment has the following structure:

1. Load CIFAR-10.
2. Build deterministic, structured client datasets.
3. For each run:
4. For each algorithm:
5. For each attack:
6. Reinitialize a fresh LeNet model.
7. Reinitialize all clients with the same local data assignment.
8. Run the algorithm for all communication rounds.
9. In every round, after every client trains, attack the submitted client update.
10. Aggregate client updates into the global model.
11. After training, evaluate final model behavior on structured CIFAR-10 test data.
12. Write privacy and utility results to CSV files and reconstructed image folders.

- Reasoning: This design allows a direct comparison of how much information about the private training sample can be reconstructed from the client update across different algorithms and attacks, while also measuring the final utility of the trained models. the number of iterations (for error bars) comes from the simulation/experiment structure itself. This experimental structure mimics a standard FL training loop with an attack inserted before aggregation. 

With the current configuration:

| Quantity | Value |
|---|---:|
| random seed | 42 |
| number of runs | 1 |
| number of clients | 10 |
| local dataset size per client | 5 images |
| communication rounds | 5 |
| local training rounds per client per communication round | 1 |
| algorithms | 9 |
| attacks | 2 |
| attack reconstructions per algorithm-attack pair | 50 |
| total attack rows in combined run | 900 |
| final utility rows in combined run | 900 |

The count of 900 attack rows comes from:

```text
1 run * 9 algorithms * 2 attacks * 5 communication rounds * 10 clients = 900
```
- That is:
    - 50 reconstructions per algorithm-attack pair (5 communicaton rounds * 10 clients)
    - 5 reconstructions per cifar-10 class (1 client per class, 5 communication rounds)

The count of 900 utility rows comes from:

```text
1 run * 9 algorithms * 2 source attacks * 10 clients * 1 test batch/client * 5 samples/batch = 900
```
- That is:
    - 50 utility rows per algorithm (10 clients * 1 test batch/client * 5 samples/batch)
    - 5 utility rows per cifar-10 class (1 client per class, 1 test batch/client, 5 samples/batch)

The utility rows are duplicated by `source_attack` because the algorithm is rerun once in the DLG trial and once in the InvertingGradients trial. The attack does not feed back into training, so `source_attack` is mostly provenance for which run produced the utility measurements.

The utility measures are done after training completes, so they reflect the final model state after all communication rounds.

## 4. Dataset Construction

The data module is `data/data.py`.

`CIFAR10Data` loads CIFAR-10 through TensorFlow Datasets:

```python
tfds.load("cifar10", split=split, as_supervised=True, data_dir="./data/public")
```

Images are normalized to `[0, 1]`:

```python
x_normalized = x / 255.0
```

Labels are one-hot encoded with depth 10.

### Structured Client Data

The main simulation uses:

```python
get_structured_x_y(batch_size=5, number_of_batches=num_clients * num_runs)
```

This method constructs one batch at a time, where batch `i` contains samples from class:

```text
i mod 10
```

For the current run with 10 clients and 1 run:

| Client | Local CIFAR-10 class |
|---:|---:|
| 0 | class 0 |
| 1 | class 1 |
| 2 | class 2 |
| 3 | class 3 |
| 4 | class 4 |
| 5 | class 5 |
| 6 | class 6 |
| 7 | class 7 |
| 8 | class 8 |
| 9 | class 9 |

Each client therefore holds 5 images from one class. This is a strongly non-IID label-partitioned setup.

### Important Detail: `batch_size` Means Local Dataset Size Here

The configuration calls this field `batch_size`, but in the current training loop it functions primarily as the number of images assigned to each client.

Each communication round calls:

```python
client.get_sample()
```

That returns one sample at a time, not a batch of 5. Since there are 5 communication rounds and 5 local images per client, each client consumes exactly one new local image per round.

So the effective local update is a single-sample update:

```text
one client, one image, one label, one local gradient/update per round
```

This design is favorable for reconstruction attacks because there is no multi-sample gradient mixing inside the update.

## 5. Client Behavior

The client abstraction is `clients/client.py`.

Each client stores:

- `id`
- local `data_x`
- local `data_y`
- current local `model`
- local training function
- a log of data used during training
- optional partial-layer sharing rules for FedPer

The main simulation overwrites the random client data initialized in the constructor with structured data:

```python
client.set_data(config["x_data_list"][client.id], config["y_data_list"][client.id])
```

`set_data` shuffles the client's 5 local samples using NumPy RNG seeded with the same seed. Then `get_sample()` returns samples sequentially.

- Reasoning: for extra randomness in the order of local samples, but still deterministic and comparable across runs, we shuffle the local dataset with a seeded RNG. This means the same 5 images are used for each client across runs, but they are consumed in a different order than they were loaded. (to prevent any bias that may be introduced from the original order of the CIFAR-10 dataset, which is sorted by class) - marjinal not worth mentioning.

During a communication round:

1. The server clones the current global model.
2. The client receives the clone. (so that the server model remains unchanged upong training and each client starts from the same global state without rewriting the global model)
3. The client trains locally.
4. The client records which sample was used. (to provide the attack with the true input and label for evaluation)
5. The server reads the client's submitted weights.
6. The attack attempts reconstruction from the difference between global and client weights.

## 6. Model Architecture

The model is `models/LeNet.py`, implemented as a Keras `Sequential` model wrapped by the project `Model` class.

It is a LeNet-style CNN adapted to CIFAR-10:

| Stage | Layer | Details |
|---|---|---|
| input | Input | `(32, 32, 3)` |
| C1 | Conv2D | 6 filters, 5x5, sigmoid |
| S2 | AveragePooling2D | 2x2, stride 2 |
| C3 | Conv2D | 16 filters, 5x5, sigmoid |
| S4 | AveragePooling2D | 2x2, stride 2 |
| C5 | Conv2D | 120 filters, 5x5, sigmoid |
| flatten | Flatten | vectorization |
| F6 | Dense | 256 units, sigmoid |
| F7 | Dense | 128 units, sigmoid |
| output | Dense | 10 units, softmax |

Trainable layers are:

1. Conv2D 6
2. Conv2D 16
3. Conv2D 120
4. Dense 256
5. Dense 128
6. Dense 10

- Reasoning: the reason for the LeNet architecture is that it is a simple, well-known CNN that is small enough to train quickly and run gradient inversion attacks on without excessive computational cost. It is adapted to CIFAR-10 by using 3-channel input. the Le-Net5 model has 3 convolutional layers and 2 dense layers, which gives a total of 5 trainable layers. The devation from this (3 Cov layers and 3 Dense layers) is to increase the total number of trainable layers to 6, which allows for more interesting FedPer configurations where some layers are shared and some are private. The use of sigmoid activations is a design choice that may affect the attack performance, as it can lead to different gradient distributions compared to ReLU. It was chosen to provide a non-linear function behavior similar to the tanh activation used in the original LeNet5 model. The final output layer uses softmax because this is a multi-class classification problem with 10 classes.

Each trainable layer contributes two weight arrays: kernel and bias. Thus the full model has 12 trainable weight arrays.

The initializers use `GlorotUniform(seed=42)`, making fresh model creation deterministic.

## 7. Algorithms Compared

### 7.1 FedAvg

Implementation: `algorithms/fedAvg.py`

FedAvg is the baseline federated averaging method.

For every communication round:

1. Each client receives a clone of the current global model.
2. Each client trains locally for `client_training_rounds = 1`.
3. The client uses one sample:

```python
x, y = client.get_sample()
```

4. The local model performs one gradient descent step:

```text
theta_client = theta_global - alpha * grad L(theta_global; x, y)
```

5. The server collects all client weights.
6. The server averages all client weights elementwise:

```text
theta_global_next = mean(theta_client_1, ..., theta_client_N)
```

- Reasoning: there is no weighting applied to the client updates in the current implementation, so each client contributes equally to the global update regardless of local dataset size or other factors. The learning rate `alpha` is a hyperparameter that controls the step size of the local update. The loss function is categorical cross-entropy, which is standard for multi-class classification problems like CIFAR-10 (and the attacks are designed to work with models trained with Cross-Entropy loss and one-hot labels). 

- Justification: there was no hyperparameter tuning for the alpha (step size) or the number of local training rounds or the number of communication rounds. The focus was on the attacks and the utility of the FL algorithms was only a measure to find the relative utility differences between the algorithms to extract structural differences. hence there was no FL hyperparameter tuning. and since they were kept the same across all algorithms, the relative differences in utility can be attributed to the algorithmic differences rather than hyperparameter differences. The same applies to the number of communication rounds and local training rounds. the number of communication rounds were used to generate multiple attack reconstructions per algorithm-attack pair, and the number of local training rounds was set to 1 to create a strong attack setting where each update is based on a single sample (as this removes the mixing effect of multiple samples in a batch or multiple local steps, making it easier for the attack to reconstruct the individual sample and hence attributing the effectiveness to the algorithmic structure and not the averaging effects).

Current settings:

| Setting | Value |
|---|---:|
| communication rounds | 5 |
| local training rounds | 1 |
| learning rate `alpha` | 0.1 default |
| loss | categorical cross-entropy |

### 7.2 FedPer

Implementation: `algorithms/fedPer.py`

FedPer separates model layers into:

- shared base layers
- private personalized head layers

The parameter `K_p` controls how many final trainable layers are kept private on each client.

The experiment tests:

```text
K_p = 1, 2, 3, 4, 5
```

- Reasoning: the K_p values of 1, 2, 3, 4, 5 are done to explore a range of personalization levels, from minimal (only the final layer is private) to maximal (all but one layer are private). This allows us to observe how the amount of shared information (and thus potential leakage) changes as we increase the number of private layers. The choice of 5 as the maximum K_p is because the model has 6 trainable layers, so K_p=5 means only one layer is shared, which is an extreme case that can provide insights into the limits of privacy protection through personalization. and due to the 3 Conv layers and 3 Dense layers these values also allow us to explore any potential differences in privacy leakage between convolutional and dense layers when they are shared or kept private.

Since LeNet has 6 trainable layers, this means:

| Variant | Private trainable layers | Shared trainable layers |
|---|---:|---:|
| FedPer(K_p=1) | 1 | 5 |
| FedPer(K_p=2) | 2 | 4 |
| FedPer(K_p=3) | 3 | 3 |
| FedPer(K_p=4) | 4 | 2 |
| FedPer(K_p=5) | 5 | 1 |

At the weight-array level, each trainable layer has kernel and bias, so `K_p` private trainable layers correspond to `2 * K_p` private arrays.

During communication:

1. The client receives the shared/global model.
2. The client preserves its private final layers across rounds.
3. The client trains the whole local model on one sample.
4. The client sends only the shared base-layer weights.
5. The attack receives only the shared part of the update.
6. The server averages only the shared weights.
7. Private layers are never aggregated globally.

This is the project's main experiment for testing whether personalization layers reduce gradient inversion leakage. Intuitively, larger `K_p` means fewer shared gradients are visible to the server-attacker.

### 7.3 Per-FedAvg

Implementation: `algorithms/per_fedAvg.py`

Per-FedAvg is a meta-learning style personalized federated algorithm. Instead of merely learning a global model, it learns an initialization that can adapt quickly to each client.

The experiment tests three local update approximations:

- `Per-FedAvg(FO)`: first-order approximation
- `Per-FedAvg(HF)`: Hessian-free finite-difference approximation
- `Per-FedAvg(HVP)`: exact Hessian-vector product using nested gradient tapes

- Reasoning: three variants of Per-FedAvg are tested to explore how different levels of approximation in the meta-update affect both privacy leakage and model utility. The first-order variant is the simplest and most computationally efficient, while the Hessian-free and HVP variants incorporate second-order information that may lead to better personalization but also potentially more complex updates that could affect the attack's ability to reconstruct the training sample. By comparing these three variants, we can gain insights into the trade-offs between computational complexity, personalization effectiveness, and vulnerability to gradient inversion attacks.

Current settings:

| Setting | Value |
|---|---:|
| communication rounds | 5 |
| client training rounds | 1 |
| client adaptation rounds | 1 |
| alpha | 0.1 default |
| beta | 0.1 default |
| loss | categorical cross-entropy |

#### First-Order Per-FedAvg

For one client and one sample:

1. Start from original global weights `theta`.
2. Compute an adapted model:

```text
theta_adapted = theta - alpha * grad L(theta; x, y)
```

3. Compute the gradient at the adapted model.
4. Update the original model with the meta-gradient:

```text
theta_client = theta - beta * grad L(theta_adapted; x, y)
```

In this code, the same sample is reused for adaptation and meta-update.

#### Hessian-Free Per-FedAvg

The Hessian-free version approximates the Hessian-vector product using finite differences:

```text
Hv approx = (grad f(theta + delta * v) - grad f(theta - delta * v)) / (2 * delta)
```

And delta was set at 1e-2 in the current implementation.

- Justification: the delta was not optimized. It was set to a small value that is commonly used for finite difference approximations in numerical analysis. 

The local update is:

```text
theta_client = theta - beta * (v - alpha * Hv)
```

where `v` is the gradient after the inner adaptation step.

#### Exact HVP Per-FedAvg

The HVP version computes the Hessian-vector product directly with nested `tf.GradientTape`.

Conceptually, it uses the same update form:

```text
theta_client = theta - beta * (v - alpha * H_f(theta) * v)
```

but computes `H_f(theta) * v` through automatic differentiation rather than finite differences.

#### Privacy Meaning of Per-FedAvg Updates

For Per-FedAvg, the attack does not see a simple SGD update. It sees a meta-update. The attack code reconstructs a gradient-like quantity as:

```text
(global_weights - client_weights) / beta
```

This approximates the meta-gradient update rather than the ordinary supervised gradient. Because the local update still depends on a single private sample in the current setup, including these personalization methods aims to test if the meta-gradients can still leak information about that sample. and the use of all three different Per-FedAvg variants allows us to see if the complexity of the meta-update affects the attack's ability to reconstruct the training sample.

## 8. Attack Model

The attacker is effectively an honest-but-curious server or observer that knows:

- the current global model before aggregation
- the client-submitted model weights or partial weights
- the learning rate used to produce the update
- the model architecture
- the number of classes

For FedAvg and FedPer, the attack reconstructs client gradients as:

```text
client_gradient = (global_weight - client_weight) / learning_rate
```

This is valid for a one-step gradient descent update:

```text
client_weight = global_weight - learning_rate * gradient
```

For FedPer, only the shared layer weights are available, so the attack matches only the visible prefix of model gradients.

For Per-FedAvg, the same computation is applied using `beta`, so the recovered object is the meta-update direction.

Attacks run before aggregation in every communication round. Therefore, the attacker targets individual client updates, not an already averaged global update.

## 9. Attack 1: DLG

Implementation: `attacks/DLG.py`

DLG stands for Deep Leakage from Gradients. The core idea is:

1. Initialize dummy image logits and dummy label logits randomly.
2. Pass the dummy image through the known model.
3. Compute the dummy supervised loss with the dummy soft label.
4. Compute dummy gradients with respect to model parameters.
5. Optimize the dummy image and label so that dummy gradients match the observed client gradients.

The optimization objective is a sum of squared gradient differences:

```text
min_dummy_x,dummy_y sum_l || grad_l(dummy_x, dummy_y) - observed_grad_l ||^2
```

The implementation uses:

- `tfp.optimizer.lbfgs_minimize`
    - Reasoning: The DLG paper uses L-BFGS for the optimization, and this implementation follows that choice to stay faithful to the original method. This optimizer is not in TensorFlow core, so we use the TensorFlow Probability implementation. 
- sigmoid reparameterization for image pixels
    - Reasoning: the sigmoid reparameterization ensures that the reconstructed pixel values remain in the valid range of [0, 1] during optimization. This is important because pixel values outside this range would not correspond to valid images and could lead to unrealistic reconstructions.
- softmax reparameterization for labels
- only the first `first_n_layers` gradient tensors, which matters for FedPer partial updates

Current main-simulation DLG settings:

| Setting | Value |
|---|---:|
| max_iterations | 300 |
| f_relative_tolerance | 1.25e-12 |
| max_line_search_iterations | 300 |
| num_correction_pairs | 175 |
| tolerance | 1e-15 |

The attack output is:

- `reconstructed_input`: shape `(1, 32, 32, 3)`
- `reconstructed_label`: shape `(1, 10)`, softmax probabilities

## 10. Attack 2: Inverting Gradients

Implementation: `attacks/InvertingGradients.py`

This attack optimizes a dummy image and label to align the dummy gradient direction with the observed client gradient direction.

The loss is:

```text
gradient_loss = cosine_distance(dummy_gradient, observed_gradient)
                + alpha * total_variation(dummy_image)
```

where:

```text
cosine_distance = 1 - dot(dummy_gradient, observed_gradient)
                    / (||dummy_gradient|| * ||observed_gradient||)
```

The total variation term biases the reconstruction toward spatially smoother images.

The implementation uses:

- Adam optimizer
- cosine-decayed learning rate
- sigmoid image reparameterization
- softmax label reparameterization
- optional signed gradients for the Adam update, enabled by default

Current main-simulation InvertingGradients settings:

| Setting | Value |
|---|---:|
| max_iterations | 300 |
| init_step_size | 0.1 |
| final_step_size | 0.09 |
| alpha | 6.3e-13 |
| use_signed_adam | True default |

Like DLG, it only compares the visible gradient tensors, which is important for FedPer.

## 11. Seeding and Fairness Controls

The simulation defines:

```python
seed = 42
```

and uses:

```python
reseed(seed)
```

to set:

- NumPy seed
- TensorFlow seed
- Python `random` seed

There is also a `_SeededAttack` adapter. Immediately before every attack execution, it reseeds the random number generators with a trial seed:

```text
trial_seed(run) = seed + 1000 * run
```

For the current single-run experiment, this is always:

```text
42
```

The intended purpose is fairness: attack random initialization should not depend on how much random state was consumed by the algorithm that happened to run before it.

This means DLG and InvertingGradients reconstructions are initialized deterministically and comparably across algorithm variants.

## 12. Metrics: Attack Reconstruction Quality

Attack results are handled by `AttackResultHandler` in `results/ResultHandler.py`.

The main simulation uses these attack metrics:

- `ComparisonMetric`
- `MSE_metric`
- `PSNR_metric`
- `SSIM_metric`
- `VisualMetric`

### 12.1 ComparisonMetric

Implementation: `metrics/ComparisonMetric.py`

Stores:

- actual input tensor
- reconstructed input tensor
- actual class label
- reconstructed class label

Labels are decoded by `argmax`.

This metric is not a scalar quality score; it provides the raw data needed for inspection and downstream analysis.

### 12.2 MSE

Implementation: `metrics/MSE_metric.py`

Computes:

```text
mean((reconstructed_input - actual_input)^2)
```

Lower is better.

Since images are normalized to `[0, 1]`, MSE is also on the normalized pixel scale.

### 12.3 PSNR

Implementation: `metrics/PSNR_metric.py`

Computes peak signal-to-noise ratio using:

```python
tf.image.psnr(..., max_val=1.0)
```

Higher is better.

Because images are normalized to `[0, 1]`, `max_val=1.0` is appropriate.

### 12.4 SSIM

Implementation: `metrics/SSIM_metric.py`

Computes structural similarity using:

```python
tf.image.ssim(..., max_val=1.0, filter_size=11)
```

Higher is better.

SSIM is more perceptual than MSE or PSNR because it measures local structural similarity.

### 12.5 VisualMetric

Implementation: `metrics/VisualMetric.py`

Writes PNG files for:

- true input image
- reconstructed input image

The file names encode:

- algorithm
- attack
- run
- communication round
- client id
- sample index
- label
- hash of the true image

Example path shape:

```text
results/true_images/dlg/actual__FedAvg__DLG__run_0__round_1__client_3__sample_0__label_3__23f48c2c.png
results/reconstructed_images/dlg/reconstructed__FedAvg__DLG__run_0__round_1__client_3__sample_0__label_3__23f48c2c.png
```

- Reasoning: all these metrics were used in comnination to have a non biased and whole picture of the attack performance. MSE and PSNR provide pixel-wise error and signal quality measures, while SSIM captures perceptual similarity. The visual metric allows for qualitative inspection of reconstructions, which can reveal details that scalar metrics might miss. By using all these metrics together, we can get a comprehensive understanding of how well the attacks are reconstructing the private training samples across different algorithms.

## 13. Metrics: Final Model Utility

Algorithm utility results are handled by `AlgorithmResultHandler`.

The main simulation uses:

- `PredictionComparison`
- `ModelCrossEntropy`

Both use CIFAR-10 test data with the same structured class batching pattern as the training data.

### 13.1 PredictionComparison

Implementation: `metrics/PredictionComparison.py`

For each client:

1. Select a structured test batch corresponding to that client id.
2. Run the client's final model on that batch.
3. Store:

- test input
- test label
- predicted label

Predictions are decoded with `argmax`.

The analysis utility later computes accuracy by comparing `test label` and `predicted label`.

### 13.2 ModelCrossEntropy

Implementation: `metrics/ModelCrossEntropy.py`

For each client and test sample, it computes categorical cross-entropy:

```text
CE(y_true, y_pred)
```

Lower cross-entropy means the model assigns higher probability to the correct label.

### 13.3 Per-FedAvg Evaluation Adaptation

For `perFedAvg`, both model utility metrics adapt each client's model before evaluation:

```text
theta_eval = theta_client - adaptation_alpha * grad L(theta_client; one random local sample)
```

with:

```text
adaptation_alpha = 0.1
```

This reflects the Per-FedAvg idea that the learned model is an initialization meant to adapt quickly to a client's data.

FedAvg and FedPer are evaluated without this extra adaptation step.

## 14. Result Files

The main combined simulation writes:

```text
results/attack_results/raw_results.csv
results/algorithm_results/raw_results.csv
```

The attack-specific simulation scripts write:

```text
results/attack_results/dlg/raw_results.csv
results/algorithm_results/dlg/raw_results.csv
results/attack_results/ig/raw_results.csv
results/algorithm_results/ig/raw_results.csv
```

Images are written under:

```text
results/true_images/
results/reconstructed_images/
```

The current workspace contains:

| File | Rows including header |
|---|---:|
| `results/attack_results/raw_results.csv` | 901 |
| `results/algorithm_results/raw_results.csv` | 901 |
| `results/attack_results/dlg/raw_results.csv` | 451 |
| `results/attack_results/ig/raw_results.csv` | 451 |
| `results/algorithm_results/dlg/raw_results.csv` | 451 |
| `results/algorithm_results/ig/raw_results.csv` | 451 |

So the combined result files contain 900 data rows each, and the split attack files contain 450 data rows each.

### Important Result Handler Behavior

`ResultHandler` opens CSV files in write mode during initialization:

```python
with self.csv_path.open("w", newline="") as file:
```

This means rerunning a simulation overwrites the corresponding raw CSV file.

PNG image files are protected by `_unique_path`, so duplicate image names receive suffixes rather than being overwritten.

## 15. Analysis Utilities

The file `result_analysis/analysis_utils.py` provides helper functions for loading, summarizing, combining, and plotting results.

Important functions:

- `combine_csvs(...)`
- `load_attack_results(...)`
- `load_algorithm_results(...)`
- `attack_summary(...)`
- `algorithm_summary(...)`
- `attack_metric_by_round(...)`
- `plot_attack_metric(...)`
- `plot_attack_metric_by_round(...)`
- `plot_algorithm_crossentropy(...)`
- `plot_algorithm_accuracy(...)`

The attack summary groups by:

```text
(algorithm, attack)
```

and reports:

- row count
- mean/min/max MSE
- mean/min/max PSNR
- mean/min/max SSIM

The algorithm summary groups by:

```text
algorithm
```

and reports:

- row count
- mean/min/max cross-entropy
- prediction accuracy

## 16. Hyperparameter Search Component

The file `hyperparameter_simulation.py` is a separate experiment for attack hyperparameter tuning.

Its documented purpose is to tune DLG and InvertingGradients parameters using a FedAvg-based reconstruction setup.

It performs:

1. A deterministic one-client FedAvg update on sampled CIFAR-10 images.
2. A grid search over attack hyperparameters.
3. Evaluation by MSE, PSNR, and SSIM.
4. Selection by lowest average input MSE.
5. Grid refinement around the best candidate for the next round.

It writes:

```text
results/hyperparameters/.../raw_results.csv
results/hyperparameters/.../results.txt
```

### Current Code Issue in Hyperparameter Search

The current `hyperparameter_simulation.py` calls:

```python
client.sample(1)
```

But `Client` currently defines:

- `random_samples(n)`
- `get_sample()`

and does not define `sample`.

So the hyperparameter script appears stale relative to the current `Client` API. It likely needs to be changed to `client.random_samples(1)` or `client.get_sample()` depending on whether random or sequential sampling is intended.

## 17. What the Experiment Is Measuring Scientifically

The experiment measures two related phenomena.

### 17.1 Privacy Leakage

The privacy question is:

> Given the model update sent by a client, can an attacker recover the private image and label that caused the update?

The attack result metrics quantify reconstruction success:

- MSE: pixel-wise error
- PSNR: signal quality
- SSIM: perceptual/structural similarity
- label agreement: whether reconstructed label argmax matches the true label
- visual inspection: saved true/reconstructed image pairs

Because each update is based on one image, successful reconstruction means the server can recover an individual training example.

### 17.2 Model Utility

The utility question is:

> How well does each algorithm's final client/global model perform on class-structured CIFAR-10 test samples?

The model utility metrics quantify:

- prediction correctness
- output cross-entropy

This lets the experiment compare privacy leakage against model performance.

For example, if FedPer with larger `K_p` leaks less because fewer layers are shared, one must also check whether its predictive utility decreases.

## 18. Expected Interpretive Axes

The most natural comparisons from the generated results are:

### FedAvg vs FedPer

FedAvg shares all trainable weights. FedPer shares only the base layers and keeps the last `K_p` trainable layers private.

A likely hypothesis is:

```text
As K_p increases, fewer gradients are visible, so reconstruction quality should degrade.
```

This should be tested against MSE, PSNR, SSIM, and label recovery.

### FedAvg vs Per-FedAvg

FedAvg exposes a simple supervised gradient update.

Per-FedAvg exposes a meta-update that includes adaptation behavior and, in HF/HVP variants, second-order information.

The privacy question is whether this meta-update is harder or easier to invert than the ordinary FedAvg update.

### DLG vs InvertingGradients

DLG matches gradients by squared Euclidean distance and uses L-BFGS.

InvertingGradients matches gradient direction with cosine distance, uses Adam, and adds total variation regularization.

The comparison asks which optimization objective is more effective under:

- full gradients
- partial gradients from FedPer
- meta-gradients from Per-FedAvg

### Round-by-Round Leakage

The attack runs after every communication round.

Round-by-round analysis can show whether reconstruction becomes easier or harder as the global model evolves.

The helper `attack_metric_by_round(...)` supports this analysis.

## 19. Methodological Caveats

### 19.1 One Run Is Not Enough for Statistical Claims

The current main configuration uses:

```text
num_runs = 1
```

This is useful for a controlled demonstration, but weak for final empirical claims. A thesis-quality comparison should report averages and variability over multiple runs and possibly multiple data partitions.

### 19.2 The Data Partition Is Extremely Structured

Each client receives images from exactly one class. This is a valid non-IID stress test, but it is not representative of all FL settings.

It can also simplify label inference because client id is correlated with label in the current run.

### 19.3 Each Update Uses One Sample

The attack setting is especially strong because each client update is based on a single sample.

In real FL, clients often train on mini-batches and multiple local steps. Larger batches mix gradients from multiple examples and can make exact reconstruction harder, although not impossible.

### 19.4 Attack Knows the Learning Rate and Architecture

The attack assumes knowledge of:

- architecture
- global weights
- client weights
- learning rate or meta learning rate
- number of classes

This is a standard white-box gradient inversion setting, but the assumption should be stated explicitly.

### 19.5 Algorithm Utility Is Recomputed Per Attack Trial

The algorithm result rows include `source_attack`, but the attack itself does not alter training. Thus utility differences between `source_attack=DLG` and `source_attack=InvertingGradients` should not be interpreted as caused by the attack.

They are separate reruns of the same algorithm configuration.

### 19.6 Importing Simulation Files Runs the Experiment

`simulations/simulation.py`, `simulations/dlg_sim.py`, and `simulations/ig_sim.py` execute at module import time. They do not use:

```python
if __name__ == "__main__":
```

This means importing these files from another script would start the full simulation.

### 19.7 Makefile Is Stale

`make run` currently points to:

```bash
python simulation.py
```

but the active file is:

```bash
simulations/simulation.py
```

The make target should likely be updated if the root script is intentionally removed.

### 19.8 Hyperparameter Script Uses an Old Client API

`hyperparameter_simulation.py` calls `client.sample(1)`, which is not currently defined.

This should be fixed before relying on the hyperparameter search path.

## 20. Conceptual Takeaway

This experiment is a privacy-utility study of personalized federated learning under gradient inversion.

The central mechanism is:

```text
private client sample -> local model update -> server-visible weights -> gradient inversion attack -> reconstructed sample
```

FedAvg exposes the full update. FedPer intentionally hides some final personalized layers. Per-FedAvg exposes a meta-learning update. The project then measures whether those algorithmic differences change the attacker's ability to recover the original CIFAR-10 image and label.

In computer science terms, the experiment is probing an information leakage channel: the client update is not merely an optimization signal, but also an encoded representation of the client's private data. The attacks try to decode that representation by solving an inverse problem.

The strongest statement supported by the current code is:

> Under a deterministic, one-sample-per-client-round CIFAR-10 setup, the project compares how much image and label information can be reconstructed from FedAvg, FedPer, and Per-FedAvg client updates using DLG and InvertingGradients.

For stronger empirical claims, the experiment should be repeated across more seeds, more runs, less deterministic client partitions, and more realistic local batch/multi-step training regimes.

