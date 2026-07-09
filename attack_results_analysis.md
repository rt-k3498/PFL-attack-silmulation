# Attack Results Analysis

This report explains the current attack results in `results/attack_results/raw_results.csv` and the way they are summarized in `result_analysis/result_analysis.ipynb`.

The notebook is mainly a plotting notebook. It loads:

- `results/attack_results/raw_results.csv`
- `results/algorithm_results/raw_results.csv`
- helper functions from `result_analysis/analysis_utils.py`

For the attack results, the notebook calls `attack_summary(attack_rows)` and plots:

- `input mse`
- `input psnr`
- `input ssim`
- `label_recover`

The interpretation below is based on the raw attack CSV, not only on the plots.

## What Each Attack Row Means

Each row in `results/attack_results/raw_results.csv` is one gradient inversion attempt against one client update.

The simulation setup is:

- `num_runs = 3`
- `num_clients = 5`
- `communication_rounds = batch_size * labels_per_client = 5 * 2 = 10`
- `algorithms = 9`
- `attacks = 2`

So the row count is:

```text
3 runs * 9 algorithms * 2 attacks * 10 rounds * 5 clients = 2700 attack rows
```

The raw file has exactly 2700 data rows.

Important columns:

| Column | Meaning |
| --- | --- |
| `run` | Simulation run index, currently 0, 1, 2. |
| `communication_round` | FL round being attacked, 1 through 10. |
| `client_id` | The attacked client. |
| `client_label_classes` | The two CIFAR-10 classes assigned to that client. |
| `algorithm` | FL/PFL algorithm used for the attacked update. |
| `attack` | Either `DLG` or `InvertingGradients`. |
| `real input value` | The actual training image used in the client update. |
| `reconstructed input value` | The image reconstructed by the attack. |
| `actual label` | True label of the training image. |
| `reconstructed label` | Label inferred by the attack. |
| `input mse` | Pixel-level mean squared error. Lower is better for the attacker. |
| `input psnr` | Peak signal-to-noise ratio. Higher is better for the attacker. |
| `input ssim` | Structural similarity. Higher is better for the attacker. |

The attack is run inside each algorithm before aggregation. For example, `FedAvg.run()` trains each client from the current global model, records the client's updated weights and the training sample, then runs the attack against the global model and the client weights. See `algorithms/fedAvg.py`, `algorithms/fedPer.py`, and `algorithms/per_fedAvg.py`.

This matters because the attack is not attacking the final model. It is attacking the per-round client update.

## Metric Interpretation

Use the metrics this way:

| Metric | Better attacker result | Meaning |
| --- | --- | --- |
| `input mse` | Lower | Reconstructed pixels are numerically closer to the real image. |
| `input psnr` | Higher | Same basic information as MSE, expressed as signal-to-noise. |
| `input ssim` | Higher | Reconstructed image has more similar visual structure. |
| `label_recover` | Higher | Reconstructed label equals actual label. |

`label_recover` is computed in `result_analysis/analysis_utils.py` by checking:

```text
actual label == reconstructed label
```

It is independent from image quality. A row can have a terrible image reconstruction but still recover the label correctly.

That separation is central to the current results.

## High-Level Result

The clearest reading is:

1. `FedAvg` is highly vulnerable to both attacks, especially `InvertingGradients`.
2. `FedPer` substantially reduces image reconstruction quality, but it does not eliminate label leakage.
3. `Per-FedAvg` creates a split result: `DLG` gets somewhat closer images but poor labels, while `InvertingGradients` gets labels almost perfectly but produces visually bad images.
4. PFL methods look better than `FedAvg` for pixel privacy, but not reliably better for label privacy.

Do not summarize the result as "PFL prevents gradient inversion." A more accurate statement is:

```text
In this setup, PFL usually makes image reconstruction worse, but labels can still leak strongly.
```

## Overall By Attack

Across all algorithms:

| attack | rows | mean_mse | median_mse | mean_psnr | mean_ssim | label_recovery |
| --- | --- | --- | --- | --- | --- | --- |
| DLG | 1350 | 0.1364 | 0.1049 | 10.447 | 0.138 | 0.654 |
| InvertingGradients | 1350 | 0.2047 | 0.2271 | 8.694 | 0.153 | 0.706 |

This aggregate table is useful but slightly misleading. `InvertingGradients` is excellent on `FedAvg`, very poor on Per-FedAvg image reconstruction, and strong on Per-FedAvg labels. Averaging all algorithms together hides that split.

The algorithm-by-attack table is more important.

## Main Summary Table

| algorithm | attack | rows | mean_mse | median_mse | mean_psnr | mean_ssim | label_recovery |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | DLG | 150 | 0.0308 | 0.0120 | 17.795 | 0.544 | 1.000 |
| FedAvg | InvertingGradients | 150 | 0.0052 | 0.0048 | 23.299 | 0.780 | 1.000 |
| FedPer(K_p=1) | DLG | 150 | 0.1691 | 0.1547 | 9.241 | 0.094 | 0.960 |
| FedPer(K_p=1) | InvertingGradients | 150 | 0.2253 | 0.2479 | 7.344 | 0.082 | 0.627 |
| FedPer(K_p=2) | DLG | 150 | 0.1747 | 0.1827 | 9.003 | 0.103 | 0.800 |
| FedPer(K_p=2) | InvertingGradients | 150 | 0.1938 | 0.2055 | 7.663 | 0.123 | 0.367 |
| FedPer(K_p=3) | DLG | 150 | 0.1664 | 0.1558 | 9.476 | 0.125 | 0.787 |
| FedPer(K_p=3) | InvertingGradients | 150 | 0.1810 | 0.1994 | 7.920 | 0.141 | 0.333 |
| FedPer(K_p=4) | DLG | 150 | 0.2115 | 0.2423 | 7.889 | 0.131 | 0.773 |
| FedPer(K_p=4) | InvertingGradients | 150 | 0.1967 | 0.2185 | 7.612 | 0.136 | 0.380 |
| FedPer(K_p=5) | DLG | 150 | 0.1986 | 0.2444 | 7.921 | 0.081 | 0.627 |
| FedPer(K_p=5) | InvertingGradients | 150 | 0.1443 | 0.1428 | 8.591 | 0.098 | 0.660 |
| Per-FedAvg(FO) | DLG | 150 | 0.0952 | 0.0828 | 10.822 | 0.053 | 0.540 |
| Per-FedAvg(FO) | InvertingGradients | 150 | 0.2981 | 0.2918 | 5.277 | 0.010 | 1.000 |
| Per-FedAvg(HF) | DLG | 150 | 0.0925 | 0.0823 | 10.897 | 0.053 | 0.200 |
| Per-FedAvg(HF) | InvertingGradients | 150 | 0.2988 | 0.2926 | 5.270 | 0.006 | 0.993 |
| Per-FedAvg(HVP) | DLG | 150 | 0.0891 | 0.0823 | 10.976 | 0.054 | 0.200 |
| Per-FedAvg(HVP) | InvertingGradients | 150 | 0.2988 | 0.2921 | 5.270 | 0.006 | 0.993 |

## FedAvg Interpretation

`FedAvg` is the easiest target.

Results:

| attack | mean_mse | mean_ssim | label_recovery |
| --- | --- | --- | --- |
| DLG | 0.0308 | 0.544 | 1.000 |
| InvertingGradients | 0.0052 | 0.780 | 1.000 |

This means both attacks recover the label perfectly, and `InvertingGradients` also recovers very close images.

Why this happens:

- The local update is a one-sample cross-entropy gradient.
- `FedAvg` sends the full model update.
- The final layer gradient contains a very direct label signal.
- The attacker sees enough gradient information to optimize both the dummy image and dummy label.

For a single sample with softmax cross-entropy, the output-layer bias gradient has the form:

```text
gradient_for_class_c = predicted_probability_c - 1[c == true_label]
```

The true class is the one with the distinctive negative component. This is why label recovery is often almost trivial for single-sample updates.

The current `FedAvg` result is therefore expected. It is not a suspicious result. It means your setup is very vulnerable to gradient inversion when the full one-sample update is exposed.

## FedPer Interpretation

`FedPer` has much worse image reconstruction than `FedAvg`.

Combined across both attacks:

| algorithm | rows | mean_mse | median_mse | mean_psnr | mean_ssim | label_recovery |
| --- | --- | --- | --- | --- | --- | --- |
| FedPer(K_p=1) | 300 | 0.1972 | 0.2288 | 8.293 | 0.088 | 0.793 |
| FedPer(K_p=2) | 300 | 0.1843 | 0.2003 | 8.333 | 0.113 | 0.583 |
| FedPer(K_p=3) | 300 | 0.1737 | 0.1936 | 8.698 | 0.133 | 0.560 |
| FedPer(K_p=4) | 300 | 0.2041 | 0.2254 | 7.750 | 0.133 | 0.577 |
| FedPer(K_p=5) | 300 | 0.1715 | 0.1541 | 8.256 | 0.089 | 0.643 |

Compared to `FedAvg`, these are bad reconstructions:

- `FedAvg` combined mean MSE: `0.0180`
- `FedPer` combined mean MSE range: `0.1715` to `0.2041`
- `FedAvg` combined mean SSIM: `0.662`
- `FedPer` combined mean SSIM range: `0.088` to `0.133`

So `FedPer` is clearly reducing pixel-level leakage in this experiment.

The reason is in the FedPer design and implementation:

- `FedPer` stores the last `K_p` layers locally.
- The client only sends the shared first layers.
- The attack therefore sees fewer layers as `K_p` increases.
- The attack also reconstructs using the global model, while the client gradient was produced using a personalized local head.

That last point is important. In `FedPer`, the gradient of the shared layers depends on the personalized later layers. But the attack computes dummy gradients using the global model. As the personalized head diverges from the global head, the attack target becomes less exact.

This explains why FedPer image reconstruction is poor.

### FedPer Labels

FedPer label recovery does not vanish:

| algorithm | DLG label recovery | InvertingGradients label recovery |
| --- | --- | --- |
| FedPer(K_p=1) | 0.960 | 0.627 |
| FedPer(K_p=2) | 0.800 | 0.367 |
| FedPer(K_p=3) | 0.787 | 0.333 |
| FedPer(K_p=4) | 0.773 | 0.380 |
| FedPer(K_p=5) | 0.627 | 0.660 |

This means hiding personalized layers helps, but it does not fully protect labels. Earlier shared-layer gradients still carry class information.

The label pattern is not monotonic in `K_p`, so do not claim "more personalized layers always reduce label recovery." The current data supports a weaker and safer claim:

```text
FedPer substantially reduces image reconstruction quality, while label leakage remains possible and varies by attack and K_p.
```

## Per-FedAvg Interpretation

Per-FedAvg has the most attack-dependent result.

| algorithm | attack | mean_mse | mean_ssim | label_recovery |
| --- | --- | --- | --- | --- |
| Per-FedAvg(FO) | DLG | 0.0952 | 0.053 | 0.540 |
| Per-FedAvg(FO) | InvertingGradients | 0.2981 | 0.010 | 1.000 |
| Per-FedAvg(HF) | DLG | 0.0925 | 0.053 | 0.200 |
| Per-FedAvg(HF) | InvertingGradients | 0.2988 | 0.006 | 0.993 |
| Per-FedAvg(HVP) | DLG | 0.0891 | 0.054 | 0.200 |
| Per-FedAvg(HVP) | InvertingGradients | 0.2988 | 0.006 | 0.993 |

This is the most important pattern in the attack results:

```text
For Per-FedAvg, InvertingGradients almost perfectly recovers labels but fails badly at image reconstruction.
```

That means the label channel and image channel are behaving differently.

Why this happens:

- Per-FedAvg updates are meta-updates, not ordinary one-step FedAvg gradients.
- The attack code still treats the client weight delta as if it were a normal gradient-like target.
- For image reconstruction, this mismatch hurts a lot.
- For label reconstruction, the class direction still survives strongly enough for `InvertingGradients`.

`InvertingGradients` uses cosine similarity between flattened gradients. Cosine matching is focused on direction, not exact scale. That can make it good at label inference even when it cannot reconstruct pixels.

`DLG`, by contrast, directly minimizes squared gradient differences. It is more sensitive to the exact gradient magnitude and to whether the dummy gradient is computed at the right model state. In this Per-FedAvg setting, that makes `DLG` weaker at labels, especially for HF and HVP.

The thesis-relevant point is:

```text
Poor image reconstruction does not imply label privacy.
```

Per-FedAvg is the clearest evidence of that in your data.

## Round Trends

The first and last communication round show how attacks change as training/personalization evolves:

| algorithm | attack | mse_r1 | mse_r10 | ssim_r1 | ssim_r10 | label_r1 | label_r10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | DLG | 0.0186 | 0.0523 | 0.633 | 0.445 | 1.000 | 1.000 |
| FedAvg | InvertingGradients | 0.0062 | 0.0044 | 0.774 | 0.785 | 1.000 | 1.000 |
| FedPer(K_p=1) | DLG | 0.0140 | 0.2070 | 0.679 | 0.020 | 1.000 | 1.000 |
| FedPer(K_p=1) | InvertingGradients | 0.0166 | 0.2333 | 0.545 | 0.032 | 1.000 | 0.600 |
| FedPer(K_p=2) | DLG | 0.0291 | 0.1836 | 0.667 | 0.027 | 1.000 | 1.000 |
| FedPer(K_p=2) | InvertingGradients | 0.0317 | 0.2021 | 0.488 | 0.082 | 1.000 | 0.267 |
| FedPer(K_p=3) | DLG | 0.0193 | 0.2329 | 0.693 | 0.015 | 1.000 | 1.000 |
| FedPer(K_p=3) | InvertingGradients | 0.0308 | 0.1780 | 0.513 | 0.104 | 1.000 | 0.200 |
| FedPer(K_p=4) | DLG | 0.0259 | 0.2324 | 0.688 | 0.069 | 1.000 | 1.000 |
| FedPer(K_p=4) | InvertingGradients | 0.0283 | 0.1892 | 0.552 | 0.103 | 1.000 | 0.200 |
| FedPer(K_p=5) | DLG | 0.0563 | 0.2663 | 0.299 | 0.059 | 1.000 | 0.800 |
| FedPer(K_p=5) | InvertingGradients | 0.1018 | 0.1436 | 0.181 | 0.096 | 1.000 | 0.600 |
| Per-FedAvg(FO) | DLG | 0.1034 | 0.0901 | 0.047 | 0.048 | 0.600 | 1.000 |
| Per-FedAvg(FO) | InvertingGradients | 0.2898 | 0.2973 | 0.015 | 0.013 | 1.000 | 1.000 |
| Per-FedAvg(HF) | DLG | 0.0758 | 0.0901 | 0.050 | 0.048 | 0.600 | 0.200 |
| Per-FedAvg(HF) | InvertingGradients | 0.2907 | 0.2958 | 0.012 | 0.010 | 0.933 | 1.000 |
| Per-FedAvg(HVP) | DLG | 0.0758 | 0.0901 | 0.050 | 0.048 | 0.600 | 0.200 |
| Per-FedAvg(HVP) | InvertingGradients | 0.2904 | 0.2959 | 0.012 | 0.010 | 0.933 | 1.000 |

The round trend says:

- `FedAvg + InvertingGradients` stays strong throughout training.
- `FedAvg + DLG` becomes worse by round 10 but still recovers labels perfectly.
- `FedPer` starts much easier to attack in round 1 and becomes much harder to reconstruct visually by round 10.
- `FedPer + InvertingGradients` label recovery drops sharply for `K_p=2`, `K_p=3`, and `K_p=4`.
- `Per-FedAvg + InvertingGradients` is consistently bad for images but consistently strong for labels.
- `Per-FedAvg + DLG` is mostly poor for labels except FO in round 10.

This supports the idea that personalization mainly damages image reconstruction over time. It does not reliably remove label leakage.

## Label Confusion Patterns

The label recovery table hides an important failure mode: many bad attack cases collapse reconstructed labels toward class `6`.

| algorithm | attack | label_recovery | top_reconstructed_labels | top_wrong_mappings |
| --- | --- | --- | --- | --- |
| FedAvg | DLG | 1.000 | 5:15, 6:15, 7:15 | none |
| FedAvg | InvertingGradients | 1.000 | 5:15, 6:15, 7:15 | none |
| FedPer(K_p=1) | DLG | 0.960 | 6:21, 5:15, 7:15 | 1->6:3, 4->6:3 |
| FedPer(K_p=1) | InvertingGradients | 0.627 | 6:71, 5:15, 8:15 | 4->6:15, 7->6:12, 9->6:12 |
| FedPer(K_p=2) | DLG | 0.800 | 6:45, 7:15, 9:15 | 1->6:6, 3->6:6, 4->6:6 |
| FedPer(K_p=2) | InvertingGradients | 0.367 | 6:110, 1:9, 0:9 | 2->6:15, 4->6:15, 3->6:13 |
| FedPer(K_p=3) | DLG | 0.787 | 6:47, 7:15, 9:15 | 1->6:6, 3->6:6, 4->6:6 |
| FedPer(K_p=3) | InvertingGradients | 0.333 | 6:115, 1:9, 5:6 | 4->6:15, 2->6:14, 3->6:14 |
| FedPer(K_p=4) | DLG | 0.773 | 6:40, 9:22, 7:15 | 0->9:7, 1->6:6, 3->6:6 |
| FedPer(K_p=4) | InvertingGradients | 0.380 | 6:108, 1:9, 5:6 | 2->6:15, 4->6:15, 9->6:12 |
| FedPer(K_p=5) | DLG | 0.627 | 6:47, 3:22, 8:15 | 0->6:11, 7->6:7, 4->6:6 |
| FedPer(K_p=5) | InvertingGradients | 0.660 | 6:29, 3:27, 0:26 | 4->3:9, 7->6:9, 9->0:8 |
| Per-FedAvg(FO) | DLG | 0.540 | 6:84, 7:15, 9:15 | 3->6:15, 4->6:15, 5->6:12 |
| Per-FedAvg(FO) | InvertingGradients | 1.000 | 5:15, 6:15, 7:15 | none |
| Per-FedAvg(HF) | DLG | 0.200 | 6:135, 7:6, 9:6 | 5->6:15, 8->6:15, 0->6:15 |
| Per-FedAvg(HF) | InvertingGradients | 0.993 | 6:16, 5:15, 7:15 | 8->6:1 |
| Per-FedAvg(HVP) | DLG | 0.200 | 6:135, 7:6, 9:6 | 5->6:15, 8->6:15, 0->6:15 |
| Per-FedAvg(HVP) | InvertingGradients | 0.993 | 6:16, 5:15, 7:15 | 8->6:1 |

This is useful because it shows that low label recovery is not always random failure. Some attack-algorithm pairs are biased toward predicting one specific class.

The strongest examples:

- `FedPer(K_p=2/3/4) + InvertingGradients` often predicts `6`.
- `Per-FedAvg(HF/HVP) + DLG` almost always predicts `6`.

So when label recovery is poor, the attack is often collapsing to a dominant label rather than producing uniformly random labels.

## Best And Worst Individual Reconstructions

The ten best rows by MSE are all `FedAvg + InvertingGradients`:

| algorithm | attack | run | round | client | actual | recon | mse | psnr | ssim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | InvertingGradients | 0 | 2 | 3 | 8 | 8 | 0.00095 | 30.202 | 0.901 |
| FedAvg | InvertingGradients | 1 | 8 | 3 | 8 | 8 | 0.00124 | 29.075 | 0.859 |
| FedAvg | InvertingGradients | 1 | 1 | 3 | 8 | 8 | 0.00154 | 28.135 | 0.852 |
| FedAvg | InvertingGradients | 2 | 6 | 2 | 2 | 2 | 0.00154 | 28.111 | 0.640 |
| FedAvg | InvertingGradients | 0 | 10 | 3 | 8 | 8 | 0.00155 | 28.098 | 0.840 |
| FedAvg | InvertingGradients | 1 | 4 | 3 | 8 | 8 | 0.00176 | 27.553 | 0.859 |
| FedAvg | InvertingGradients | 0 | 5 | 3 | 3 | 3 | 0.00188 | 27.247 | 0.588 |
| FedAvg | InvertingGradients | 2 | 10 | 3 | 8 | 8 | 0.00190 | 27.205 | 0.874 |
| FedAvg | InvertingGradients | 2 | 3 | 4 | 4 | 4 | 0.00196 | 27.071 | 0.609 |
| FedAvg | InvertingGradients | 1 | 5 | 3 | 3 | 3 | 0.00199 | 27.021 | 0.778 |

This confirms that the clearest image reconstructions come from the non-personalized baseline.

The ten worst rows by MSE include FedPer and Per-FedAvg cases:

| algorithm | attack | run | round | client | actual | recon | mse | psnr | ssim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedPer(K_p=3) | DLG | 0 | 8 | 0 | 5 | 5 | 0.56684 | 2.465 | 0.028 |
| FedPer(K_p=3) | DLG | 0 | 10 | 0 | 5 | 5 | 0.47270 | 3.254 | -0.020 |
| Per-FedAvg(HVP) | DLG | 0 | 2 | 2 | 7 | 7 | 0.43316 | 3.634 | 0.015 |
| Per-FedAvg(HF) | InvertingGradients | 1 | 9 | 1 | 1 | 1 | 0.42311 | 3.735 | -0.003 |
| Per-FedAvg(HVP) | InvertingGradients | 1 | 9 | 1 | 1 | 1 | 0.42278 | 3.739 | -0.004 |
| FedPer(K_p=3) | DLG | 1 | 4 | 0 | 5 | 5 | 0.41999 | 3.768 | 0.009 |
| Per-FedAvg(HF) | InvertingGradients | 0 | 2 | 2 | 7 | 7 | 0.41700 | 3.799 | 0.019 |
| Per-FedAvg(HVP) | InvertingGradients | 0 | 2 | 2 | 7 | 7 | 0.41692 | 3.799 | 0.020 |
| Per-FedAvg(FO) | InvertingGradients | 0 | 2 | 2 | 7 | 7 | 0.41156 | 3.856 | 0.022 |
| Per-FedAvg(HVP) | InvertingGradients | 1 | 6 | 0 | 0 | 0 | 0.40966 | 3.876 | 0.027 |

Notice that many of the worst image reconstructions still recover the label correctly. This again shows that image leakage and label leakage must be reported separately.

## Why DLG And InvertingGradients Behave Differently

The two attacks optimize different objectives.

`DLG`:

- Builds a dummy image and dummy label.
- Computes dummy gradients.
- Minimizes squared difference between dummy gradients and observed client gradients.
- Uses L-BFGS.

In code, this is the squared gradient loss in `attacks/DLG.py`:

```text
sum(square(dummy_gradient - real_gradient))
```

`InvertingGradients`:

- Builds a dummy image and dummy label.
- Computes dummy gradients.
- Flattens gradients into vectors.
- Minimizes cosine distance between dummy-gradient direction and real-gradient direction.
- Adds total variation regularization, but in this simulation `alpha = 6.3e-13`, which is effectively tiny.

In code, this is approximately:

```text
1 - cosine(dummy_gradient, real_gradient) + alpha * total_variation(dummy_image)
```

This distinction explains the Per-FedAvg result:

- `DLG` is stricter about matching the exact gradient values.
- `InvertingGradients` can match directional label information even when exact image reconstruction fails.

So `InvertingGradients` can recover labels from Per-FedAvg while producing bad images.

## Why FedAvg Is So Much More Vulnerable

FedAvg exposes the most complete attack surface in this simulation.

For each attacked client update:

- The model starts from the global weights.
- The client trains on one sample.
- The full client weights are sent back.
- The attack converts the weight delta into a gradient-like target.
- The attack sees gradients for all trainable layers.

Because the update is from one sample, the gradient strongly encodes both image and label.

This is why `FedAvg + InvertingGradients` is the best attack case:

```text
mean_mse = 0.0052
mean_ssim = 0.780
label_recovery = 1.000
```

That is a severe privacy leak.

## Why FedPer Reduces Image Leakage

FedPer changes the attack surface.

With `K_p > 0`, the personalized final layers are not sent to the server. The attacker only receives the shared first layers.

This reduces image leakage for two reasons:

1. Fewer gradients are available.
2. The attacker reconstructs through the global model, but the real shared-layer gradients were generated through a client-personalized head.

That second point is a model mismatch. It does not mean FedPer is mathematically impossible to attack. It means the current attacker is partly mismatched to the local personalized model state.

The practical result in this CSV is still clear:

```text
FedPer image reconstruction is much worse than FedAvg image reconstruction.
```

But the safer thesis statement is:

```text
FedPer reduces leakage against this attacker and threat model.
```

Do not overstate it as an absolute privacy guarantee.

## Why Per-FedAvg Labels Still Leak

Per-FedAvg is not hiding layers like FedPer. It sends full-model updates, but the update is a meta-update rather than a simple one-step gradient.

The current attacks still convert the weight delta into a gradient-like target:

```text
observed_target = (global_weight - client_weight) / learning_rate
```

For ordinary one-step FedAvg this is close to the actual gradient. For Per-FedAvg it is not exactly the same kind of gradient, especially for HF and HVP.

That mismatch hurts image reconstruction.

However, labels still leak because the update direction remains class-dependent. `InvertingGradients` only needs the direction to be informative enough. In the current results it is:

```text
Per-FedAvg(FO) + InvertingGradients label recovery = 1.000
Per-FedAvg(HF) + InvertingGradients label recovery = 0.993
Per-FedAvg(HVP) + InvertingGradients label recovery = 0.993
```

The important conclusion is:

```text
Per-FedAvg can hide visual information from this attacker while still leaking labels.
```

## What The Notebook Plots Mean

The notebook's attack plots should be read as follows:

### MSE Plot

Lower is better for the attacker.

Expected visual reading:

- `FedAvg + InvertingGradients` should be the lowest bar.
- `FedAvg + DLG` should also be low.
- FedPer bars should be much higher than FedAvg.
- `Per-FedAvg + InvertingGradients` should be among the highest bars.

Meaning:

```text
Pixel-level reconstruction is excellent for FedAvg and poor for most personalized methods.
```

### PSNR Plot

Higher is better for the attacker.

This should mostly invert the MSE plot:

- `FedAvg + InvertingGradients` should be highest.
- PFL methods should be much lower.

Meaning:

```text
The signal quality of reconstructed images is much higher for FedAvg.
```

### SSIM Plot

Higher is better for the attacker.

Expected visual reading:

- `FedAvg + InvertingGradients` should be highest.
- `FedAvg + DLG` should be second.
- FedPer should be low.
- Per-FedAvg should be very low, especially with `InvertingGradients`.

Meaning:

```text
FedAvg reconstructions preserve visible structure; most PFL reconstructions do not.
```

### Label Recovery Plot

Higher is better for the attacker.

Expected visual reading:

- `FedAvg` is 1.0 for both attacks.
- `FedPer` varies heavily.
- `Per-FedAvg + InvertingGradients` is almost 1.0.
- `Per-FedAvg(HF/HVP) + DLG` is very low at 0.2.

Meaning:

```text
Label privacy and image privacy diverge. Some cases protect images but still leak labels.
```

## Key Thesis Claims Supported By This CSV

The data supports these claims:

1. `FedAvg` is highly vulnerable to gradient inversion in this single-sample setup.
2. `InvertingGradients` reconstructs `FedAvg` images much better than `DLG`.
3. `FedPer` reduces image reconstruction quality relative to `FedAvg`.
4. `FedPer` does not fully prevent label leakage.
5. `Per-FedAvg` causes a strong split between image leakage and label leakage.
6. `Per-FedAvg + InvertingGradients` has very poor image reconstruction but near-perfect label recovery.
7. Attack results must be reported with both image metrics and label recovery; one does not imply the other.

The data does not safely support these stronger claims:

1. "PFL prevents gradient inversion."
2. "Higher `K_p` always improves privacy."
3. "Low MSE always means label recovery is high."
4. "Low label recovery means the attack failed randomly."
5. "Per-FedAvg is label-private."

## Important Experimental Caveats

These caveats matter when explaining the results.

### Single-Sample Updates

The attacks are run against updates produced from one client sample at a time. This is much easier to invert than larger-batch FL.

In `Client.get_sample()`, the client returns one image and one label. The algorithms call that once per local training round.

So the current results are best described as:

```text
single-sample gradient inversion results
```

not general full-batch FL privacy results.

### Non-IID Two-Class Clients

Each client has only two label classes:

```text
client 0: [0, 5]
client 1: [1, 6]
client 2: [2, 7]
client 3: [3, 8]
client 4: [4, 9]
```

This is useful for controlled PFL experiments, but it is also a simplified and very non-IID setting.

### Identical Attack Initialization Within Runs

The `_SeededAttack` wrapper reseeds before each attack call. This is good for fair comparisons because each algorithm/attack trial starts consistently, but it also means rows are not fully independent random attack attempts.

### Per-FedAvg Threat Model Mismatch

For Per-FedAvg, the update is not a simple FedAvg gradient. The current `DLG` and `InvertingGradients` implementations do not explicitly model the Per-FedAvg inner adaptation process when reconstructing dummy gradients.

This probably explains why Per-FedAvg image reconstruction is poor. It does not fully explain away the label leakage, because `InvertingGradients` still recovers labels almost perfectly.

### FedPer Personalized Head Mismatch

For FedPer, the attack uses the global model for reconstruction but the client's shared-layer gradients were generated through personalized local layers.

This makes FedPer look more private against the current attacker. It is still a valid result under this attacker model, but it should be described carefully.

## Bottom-Line Interpretation

The attack results are not saying "all PFL algorithms are safe."

They are saying:

```text
FedAvg leaks both images and labels badly.
FedPer makes images much harder to reconstruct but still leaks labels.
Per-FedAvg often hides image structure from these attacks, but InvertingGradients still extracts labels almost perfectly.
```

The most interesting result is the Per-FedAvg split:

```text
Per-FedAvg + InvertingGradients:
image reconstruction: poor
label recovery: almost perfect
```

That is the result to highlight carefully. It shows that evaluating privacy only with MSE, PSNR, or SSIM is incomplete. A model update can reveal the private label even when the reconstructed image looks useless.

## Recommended Next Checks

Before making strong thesis claims from these numbers, run these checks:

1. Add a multi-seed summary with mean and standard deviation per `(algorithm, attack)`.
2. Run a batch-size experiment to see whether label recovery drops when client updates contain multiple images.
3. Add a Per-FedAvg-aware attack that reconstructs through the inner adaptation step.
4. Add a FedPer-aware attack variant that has access to the personalized head as an upper-bound leakage test.
5. Report label recovery separately from image metrics in every result table.

The current results are meaningful, but they should be framed as attack behavior under this specific simulation and threat model.
