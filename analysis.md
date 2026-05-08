Experiment Audit Report — PFL-attack-simulation
Audit scope: scientific validity of results in results/algorithm_results/raw_results.csv and results/attack_results/raw_results.csv, produced by simulations/simulation.py, simulations/dlg_sim.py, simulations/ig_sim.py, and hyperparameter_simulation.py. Framework note: this codebase uses TensorFlow 2 / Keras + tensorflow-probability, not PyTorch as the audit prompt assumed. model.eval() / no_grad() checks below are translated to training=True/False in keras and to whether ops live inside tf.GradientTape.

A. Executive summary
Overall verdict: results are SUSPICIOUS to INVALID for thesis-grade claims. The pipeline runs and produces numerically plausible attack reconstructions, but the utility numbers are dominated by an evaluation artifact, the attack numbers cannot be compared across algorithms as advertised because of an attacker-model mismatch for Per-FedAvg, and the entire experiment is a single deterministic run with no statistical replication.

Top three highest-risk issues:

Utility evaluation is rigged towards personalized methods. Each client is tested only on test images of its own training class. FedPer/Per-FedAvg trivially score 100% (single-class memorization through personalized layers / one adaptation step), FedAvg scores 10% (chance). The algorithm_summary accuracy column does not measure FL utility and is not a fair comparison. (See finding C-2.)
Per-FedAvg attack target is mathematically inconsistent with the algorithm. The attack reconstructs (theta_global – theta_client)/beta and then minimises the discrepancy with a dummy gradient computed at theta_global. For Per-FedAvg(FO/HF/HVP) the true update is computed at the adapted model, so the gradient-matching objective has no global minimum at the true (x, y). This means the observed "Per-FedAvg leaks less" finding is partly an artefact of an under-specified threat model, not of an algorithmic privacy property. (See finding C-1.)
No statistical replication. Result CSVs are truncated on every run. num_runs = 1, the seed is fixed, and ResultHandler._initialize_csv opens the CSV in "w" mode at construction, wiping prior runs. Reported means/min/max are point estimates of a single deterministic execution, and only the most recent execution survives on disk. (See findings C-3, C-7.)
The reconstruction-quality numbers (MSE/PSNR/SSIM per algo/attack) are computed correctly given the data they receive and can stand as qualitative observations for one run. They cannot stand as quantitative or statistical claims.

B. Experiment map
Main entry points:

makefile target run → python simulation.py (stale, see C-13)
Working entry points (executed at import time, no if __name__ == "__main__" guard):
simulations/simulation.py — full grid: 9 algos × 2 attacks
simulations/dlg_sim.py — DLG only
simulations/ig_sim.py — InvertingGradients only
hyperparameter_simulation.py — DLG/IG hyperparameter grid search
Configuration: hard-coded in simulations/simulation.py (lines 65–76), plus per-algorithm settings dicts (lines 101–191). No CLI flags except for hyperparameter_simulation.py.

Datasets / partitioning:

data/data.py::CIFAR10Data.get_structured_x_y — used by main simulation and by metrics. Builds class-pure batches (each batch = batch_size images of class i mod 10).
CIFAR10Data.get_x_y — used by Client.__init__ (immediately overwritten) and by hyperparameter search.
Algorithms:

algorithms/fedAvg.py::FedAvg
algorithms/fedPer.py::FedPer
algorithms/per_fedAvg.py::PerFedAvg (FO / HF / HVP variants)
Attacks:

attacks/DLG.py::DLG — L-BFGS gradient-matching
attacks/InvertingGradients.py::InvertingGradients — Adam + cosine distance + TV
Evaluation metrics:

Reconstruction: metrics/{ComparisonMetric, MSE_metric, PSNR_metric, SSIM_metric, VisualMetric}.py
Utility: metrics/PredictionComparison.py, metrics/ModelCrossEntropy.py
Result files:

results/attack_results/raw_results.csv
results/algorithm_results/raw_results.csv
results/{attack_results,algorithm_results}/{dlg,ig}/raw_results.csv
results/true_images/, results/reconstructed_images/, results/hyperparameters/...
Logging / aggregation: results/ResultHandler.py, result_analysis/analysis_utils.py, result_analysis/result_analysis.ipynb.

C. Critical findings
C-1. Per-FedAvg attack uses an inconsistent gradient target — distorts the central thesis comparison
Severity: Critical (scientific validity)
Files: algorithms/per_fedAvg.py lines 62–183, 243; attacks/DLG.py lines 79–88, 106–117; attacks/InvertingGradients.py lines 70–101.
What is wrong. For Per-FedAvg(FO), the local update is theta_client = theta − beta · ∇L(theta_adapted; x, y) where theta_adapted = theta − alpha · ∇L(theta; x, y). The attack recovers g_obs = (theta − theta_client)/beta = ∇L(theta_adapted; x, y) (per_fedAvg.py:243, DLG.py:111, InvertingGradients.py:99), then optimises a dummy (x', y') so that ∇L(theta; x', y') ≈ g_obs. The dummy gradient is computed at theta, not theta_adapted. The recovery target therefore lives at a different parameter than the surrogate, so the global minimiser of the attacker's loss is not (x, y). For HVP/HF the situation is worse: g_obs = v − alpha·Hv where v is itself a gradient at theta_adapted. There is no (x', y') that makes ∇L(theta; x', y') = v − alpha·Hv exact.
Why it distorts results. The thesis interprets "Per-FedAvg has higher reconstruction MSE" as evidence of algorithmic privacy. The actual cause is that the attacker's optimisation problem is misspecified. A modest extension (attacker simulating one inner step of MAML) would likely recover much more. Without controlling for this, the claim "Per-FedAvg is more privacy-preserving than FedAvg" cannot be cleanly defended.
Evidence in current results. All three Per-FedAvg variants give nearly identical IG reconstruction MSE (FO 0.3009, HF 0.3028, HVP 0.3028) and SSIM near 0 / negative. This is consistent with the attacker not finding any signal beyond the regularisation prior, regardless of variant — i.e. the attack is essentially failing for structural reasons, not for privacy reasons.
How to verify. Implement the "MAML-aware" attacker: dummy gradient computed at theta − alpha · ∇L(theta; x', y'). Re-run on Per-FedAvg(FO) and compare MSE to current numbers. If the gap to FedAvg shrinks substantially, the current finding is mostly a threat-model artefact.
Suggested fix. Either (a) implement a Per-FedAvg-aware attacker and use it as an additional baseline, or (b) explicitly report this attacker as a naïve baseline and discuss the threat-model assumption in the thesis.
C-2. Utility (PredictionComparison, ModelCrossEntropy) tests each client only on its own class — accuracy comparisons are meaningless
Severity: Critical (scientific validity)
Files: metrics/PredictionComparison.py:31–34, 68–69, 95–117; metrics/ModelCrossEntropy.py:29–32, 73–89; data/data.py:92–113.
What is wrong. The test data uses get_structured_x_y(batch_size=5, number_of_batches=10*num_iterations). _test_batch_index(client, iteration) = client.id + iteration*num_classes, and get_structured_x_y puts images of class i mod 10 into batch i. With num_iterations=1, client i is tested on five images of CIFAR class i — the same class the client trained on.
Why it distorts results. A model that always predicts class i reaches accuracy 1.0 on this test. Personalised methods (FedPer with K_p ≥ 1, and Per-FedAvg with one adaptation step on a same-class sample) collapse exactly to such an "always class-i" predictor and achieve 100%. FedAvg, which has no personalisation, is forced to spread mass across all 10 classes and (with only 50 effective SGD steps total) is at chance, ~10%. The factor-10 utility gap is therefore an evaluation artefact, not a measure of personalised utility.
Evidence in current results. From results/algorithm_results/raw_results.csv:
FedAvg: accuracy 0.100, mean cross-entropy ≈ 2.31 (≈ ln 10).
All FedPer(K_p=*): accuracy 1.000, CE ≈ 0.03.
All Per-FedAvg(*): accuracy 1.000, CE ≈ 0.11. These numbers are exactly what the "predict-the-client's-only-class" degenerate model would give.
How to verify. Re-run PredictionComparison / ModelCrossEntropy against a balanced 10-class test batch per client (or a single global balanced test set) instead of the same single-class batch as the training class. Expect FedPer/Per-FedAvg accuracy to drop substantially.
Suggested fix. Replace _test_batch_index = client.id + iteration*num_classes with batches that cover all classes per client. For Per-FedAvg, also report the model's prediction before the adaptation step (the global meta-initialisation), so that adaptation gain is separated from base utility.
C-3. Single deterministic run; no replication, no error bars
Severity: Critical (statistical validity)
Files: simulations/simulation.py:67, :30–31, :197–219; identical pattern in simulations/dlg_sim.py, simulations/ig_sim.py.
What is wrong. num_runs = 1. trial_seed(run) = seed + 1000 * run, so the only trial seed used is 42. The model initialiser, client data permutation, and attack init all derive from this single seed.
Why it distorts results. All summary numbers (mean_mse, mean_psnr, mean_ssim, accuracy) are computed across the 50 reconstructions inside one run. Those 50 are not iid samples from a population: they are 5 communication rounds × 10 specific (client, class) pairs. Treating them as iid and reporting a single mean conflates between-class variance, between-round variance, and (zero) between-seed variance. No standard deviation or confidence interval reported anywhere is statistically meaningful.
How to verify. Set num_runs=5 (or run the script several times with seed re-set externally) and compute between-run variance. Alternatively pre-aggregate: per-class mean across rounds, per-round mean across clients.
Suggested fix. (i) Loop the simulation over multiple seeds; (ii) report per-(algo, attack) means with run-level standard deviation; (iii) in result_analysis/analysis_utils.py::attack_summary, replace pooled mean with mean-of-run-means and run std. Currently the analysis pools all 50 rows.
C-4. hyperparameter_simulation.py will crash before producing results — current grid-search outputs are stale
Severity: Critical (correctness / stale results)
Files: hyperparameter_simulation.py:123.
What is wrong. Line 123: train_x, train_y = client.random.sample(1). Client (in clients/client.py) does not define random as an attribute. The available methods are random_samples(n) and get_sample(). Running python hyperparameter_simulation.py will raise AttributeError: 'Client' object has no attribute 'random' inside prepare_fedavg_trials.
Why it distorts results. Any results/hyperparameters/.../raw_results.csv and results.txt predate this regression and do not reflect what the script in HEAD would do. The "best DLG hyperparameters" / "best IG hyperparameters" used in simulations/simulation.py may not be reproducible with current code.
How to verify. python hyperparameter_simulation.py --smoke --attack DLG. Expect immediate crash on line 123.
Suggested fix. Replace with client.random_samples(1) (matches the original sampling intent) or client.get_sample() and document which one is intended.
C-5. Per-FedAvg's meta-step rate beta is silently the FedAvg/FedPer step rate alpha
Severity: High (algorithmic fairness; threatens hyperparameter comparability)
Files: algorithms/per_fedAvg.py:37–38; simulations/simulation.py:148–174.
What is wrong. Settings.get("beta", 0.1) is the only source of beta, and the simulation never sets it. Therefore Per-FedAvg uses alpha = beta = 0.1. In the original Per-FedAvg formulation alpha (inner) and beta (outer) are distinct; with both equal, FO reduces to two consecutive same-rate SGD steps on the same (x, y), which is a trivial degenerate of MAML.
Why it distorts results. The privacy/utility properties claimed for Per-FedAvg cannot be cleanly attributed; the algorithm being run is closer to "two SGD steps from the same sample" than to a proper meta-learning update. Comparison with FedPer/FedAvg under "same alpha" is therefore not a comparison of algorithmic structure alone.
How to verify. Print algo.alpha, algo.beta before each Per-FedAvg trial; expect both 0.1.
Suggested fix. Set beta explicitly in the PerFedAvg settings dicts (e.g. beta = 0.01), and document the choice. Consider also surfacing alpha/beta in EXPERIMENT_SUMMARY.md's settings table — it currently lists "alpha 0.1 default / beta 0.1 default" without flagging that nothing in code overrides them.
C-6. reuse_data_batches is a documented setting that is never read
Severity: High (config silently ignored)
Files: algorithms/per_fedAvg.py (whole file — only mentioned in a docstring at line 66); simulations/simulation.py:153, 162, 171.
What is wrong. The simulation passes "reuse_data_batches": True to PerFedAvg. There is no settings.get("reuse_data_batches", ...) anywhere in per_fedAvg.py. The implementation always reuses the same (x, y) for both inner adaptation and outer meta step. So flipping this flag changes nothing.
Why it distorts results. A reader of simulations/simulation.py may believe the experiment is exploring data reuse on/off; it is not. The HF/HVP variants in particular require a separate D_2 batch to be a faithful approximation of MAML; the present code does not provide that.
Suggested fix. Either implement the flag (sample two batches when False) or remove it from the settings dict and from EXPERIMENT_SUMMARY.md to avoid confusion.
C-7. Result CSVs are truncated on every simulation start — no append, no versioning
Severity: High (reproducibility, data loss)
Files: results/ResultHandler.py:22–25 (_initialize_csv opens "w").
What is wrong. AttackResultHandler.__init__ and AlgorithmResultHandler.__init__ call _initialize_csv, which opens the file in "w" mode and writes only the header. Any previous content of results/attack_results/raw_results.csv and results/algorithm_results/raw_results.csv is lost. Running simulation.py then dlg_sim.py then ig_sim.py will leave only the last script's output for the combined file paths if you ever forget to set specific_folder.
Why it distorts results. A human running multiple seeds or multiple variants who does not manually rename / commit the CSV between runs ends up with only the last run on disk while believing they accumulated many. Confidence-interval claims based on these CSVs would be false.
How to verify. Run the simulation once, copy the CSVs aside, run it again, diff: only the second run remains.
Suggested fix. Open in "a" mode after writing the header on first creation, or include a run identifier in the filename, or fail if the file already contains rows for the same (run, algorithm, attack) triple. Add an index column already (good) — also add seed and git_sha columns.
C-8. The 50 rows per (algo, attack) are not iid — pooled mean is misleading
Severity: High (statistical interpretation)
Files: result_analysis/analysis_utils.py:122–170 (attack_summary).
What is wrong. attack_summary groups by (algorithm, attack) and reports mean(MSE) over all 50 rows. Those rows are 5 rounds × 10 (client, class) pairs. Round-1 and round-5 are not iid replications of the same condition (the global model has evolved between them). Per-class reconstruction difficulty also varies systematically (some CIFAR classes are easier).
Why it distorts results. min_mse over 50 rows can pick the easiest class at the easiest round; mean_mse averages a non-stationary process. The current report has no per-class or per-round breakdown except via attack_metric_by_round, which is not used in the bar plots.
Suggested fix. Report per-class and per-round means; treat round as a fixed effect, not a replication. For across-condition tests, average rounds within client first, then compare clients.
C-9. FedPer attacker uses global last layers in the forward pass — gradient mismatch grows with personalization
Severity: High (interpretation of FedPer privacy)
Files: attacks/DLG.py:79, attacks/InvertingGradients.py:71; clients/client.py:50–63, 96–104.
What is wrong. The attacker's surrogate forward pass keras_model(dummy_data, training=False) uses the global model's full weights. The actual client gradient on the shared first-N layers, however, was computed with the client's personalized last K layers. As personalisation diverges across rounds, the attacker's gradient at the global model on the first-N layers diverges from the client's gradient on the same layers, even ignoring (x, y). So the attack's reconstruction error is upper-bounded by this layer-mismatch, independently of any privacy property.
Why it distorts results. The reported "FedPer leaks less as K_p grows" trend is partially the attacker losing information and partially the attacker being given a model that no longer matches the client's runtime model. The two effects cannot be separated with the current design.
Suggested fix. Either (a) feed the attacker the client's full personalized state (an ablation that bounds privacy from below), or (b) explicitly note this confound in the thesis. Currently EXPERIMENT_SUMMARY.md §8 does not mention the personalized-head mismatch.
C-10. Per-FedAvg evaluation adapts on data that overlaps the training set; FedAvg/FedPer do not — comparisons are not held-out symmetric
Severity: Medium-High (fairness)
Files: metrics/PredictionComparison.py:80–93, metrics/ModelCrossEntropy.py:105–118; clients/client.py:28–33.
What is wrong. _adapt_per_fed_avg_model calls client.random_samples(1) on the client's training data (the only data attached to the client) and adapts the global model on that one sample before evaluation. With the existing setup that is one of the 5 in-distribution images of the client's only class. FedAvg and FedPer do not get this extra step.
Why it distorts results. This is approximately the canonical Per-FedAvg eval, but combined with C-2 it gives Per-FedAvg a near-deterministic shortcut to predicting class i for class-i test batches. The relative ordering "Per-FedAvg ≥ FedAvg" is partly a consequence of the extra adaptation pass.
How to verify. Disable _adapt_per_fed_avg_model (use the meta-initialisation directly) and re-run algorithm_summary for perFedAvg. Expect Per-FedAvg utility to drop towards FedAvg's.
Suggested fix. Report both adapted and non-adapted utility for Per-FedAvg so that the gain attributable to adaptation is visible. Also separate "adaptation budget = 1 sample of training data" vs. "adaptation budget = held-out client validation data" — the second is more honest.
C-11. Each Client is constructed with seed=42 (the global seed) — same shuffle pattern across clients; data sampling order is independent of run id
Severity: Medium (reproducibility, randomness diversity)
Files: simulations/simulation.py:209; clients/client.py:11–14, 28–29, 70–77.
What is wrong. The simulation passes seed=seed (42) to every Client. Client.set_data shuffles the client-local 5 images with np.random.default_rng(self.seed) — same RNG seed for every client, so the shuffled order pattern is identical across clients. random_samples uses op-level seed self.seed similarly. trial_seed(run) is unused for client-level randomness.
Why it distorts results. Reduces effective randomness diversity. Combined with num_runs=1, the only variation across clients is the data content, not the data ordering.
Suggested fix. Use seed = base_seed + client.id for client construction, and propagate trial_seed(run) into client constructors so different runs really see different orderings.
C-12. Client.__init__ performs a side-effect data fetch that is silently overwritten
Severity: Medium (lurking bug, performance)
Files: clients/client.py:11–16; simulations/simulation.py:209–211.
What is wrong. Client.__init__ does x, y = data.get_x_y(batch_size, 1) and stores the result. The simulation immediately calls client.set_data(...) which overwrites self.data_x, self.data_y. The init fetch is wasted and consumes time / random-state cycles inside tf.data. Worse, if a future caller forgets set_data, the client silently uses unstructured CIFAR images — and _get_sample_index = 0 will work fine, masking the bug.
Suggested fix. Make the fetch lazy, or make set_data mandatory and have __init__ leave data_x = None.
C-13. Stale makefile target (python simulation.py does not exist at repo root)
Severity: Medium (reproducibility)
Files: makefile:3-4.
What is wrong. make run runs python simulation.py, but the root has no such script. The actual entry is simulations/simulation.py. Anyone reproducing the experiment via the README's make run will get an immediate failure.
Suggested fix. python -m simulations.simulation (and add __init__.py if needed) or python simulations/simulation.py.
C-14. Simulation modules execute on import — no __main__ guard
Severity: Medium (code hygiene; partial-run risk)
Files: simulations/simulation.py, simulations/dlg_sim.py, simulations/ig_sim.py (none use if __name__ == "__main__":).
What is wrong. Any utility that imports these modules (e.g. an analysis notebook doing from simulations.simulation import config) will trigger the full ~50-trial run, overwriting result CSVs in the process (see C-7). Currently result_analysis/analysis_utils.py does not import them, but this is a foot-gun.
Suggested fix. Wrap top-level code in def main(): ... and if __name__ == "__main__": main().
C-15. No determinism enforcement; GPU runs may be non-bit-reproducible
Severity: Medium (reproducibility)
Files: simulations/simulation.py:25–28 and similar in other entry points.
What is wrong. reseed sets NumPy / TensorFlow / Python random seeds but does not call tf.config.experimental.enable_op_determinism() or set TF_DETERMINISTIC_OPS. On GPU some reductions and tf.image.psnr/ssim kernels may be non-deterministic.
Suggested fix. Add tf.config.experimental.enable_op_determinism() in reseed for runs intended to be reproducible.
C-16. FedAvg client_training_rounds > 1 would break the attack model, but it's silently allowed
Severity: Medium (latent bug; easy to misuse)
Files: algorithms/fedAvg.py:50–66, 110–119; attacks/{DLG,InvertingGradients}.py:run.
What is wrong. The attack's (theta_global - theta_client)/lr is the gradient only when there is exactly one local SGD step. With client_training_rounds=1 (current setting) this is fine. If a future user sets it to 2+ to "match real FL", the attack will silently inject a meaningless target gradient and report fake reconstruction qualities.
Suggested fix. Either assert client_training_rounds == 1 in the attack-enabled branch or expose a clear API for "multi-step" updates that converts the delta to a more honest "average gradient" with caveats.
C-17. algorithm_summary collapses source_attack=DLG and source_attack=InvertingGradients
Severity: Low–Medium (over-counting / mislabeling)
Files: result_analysis/analysis_utils.py:173–209.
What is wrong. The grouping key is algorithm only; the same logical training is recorded twice (once during the DLG outer loop, once during the IG outer loop) and both copies are pooled. Because training is deterministic across the two outer iterations, this just doubles every weight. If determinism breaks (see C-15), the two copies could diverge.
Suggested fix. Group by (algorithm, source_attack), then average — or at least dedupe by source_attack first.
C-18. min_mse / max_mse columns invite cherry-picking
Severity: Low (reporting hygiene)
Files: result_analysis/analysis_utils.py:147–167.
What is wrong. attack_summary reports min_mse, max_mse over the 50 rows. Reporting the minimum MSE per (algo, attack) makes good attacks look like silver bullets ("best reconstruction of the run"). It's not wrong, but it's easy to misuse in tables.
Suggested fix. Use percentiles (e.g. P25/P50/P75) instead of strict min/max, or label them as "best-case sample".
C-19. _unique_path only deduplicates within a single run
Severity: Low (logging only, but can mislead)
Files: metrics/VisualMetric.py:56–68.
What is wrong. Filenames depend on algorithm, attack, run, communication_round, client_id, sample_index, label, image_md5. The image hash is a hash of the true image; reconstructed paths share the same suffix. Across two different simulation runs (same seed), filenames collide and _unique_path appends _1, _2, etc. So the directory ends up containing both old and new reconstructions interleaved; there is no easy way to associate a _2-suffixed file with which run produced it. CSV rows reference the original (no-suffix) path that was assigned during that run, but the file on disk for that path may now be from a previous run.
Suggested fix. Include the run timestamp / git SHA in the filename; or wipe results/reconstructed_images/ at the start of each run; or append run timestamps systematically.
D. Suspicious but unconfirmed issues
D-1. Model.clone() in models/model.py:25–29 first calls Model({"layers": self.layers}) which builds a Sequential from the original layer objects, then immediately overwrites new_model.model = tf.keras.models.clone_model(self.model). The first build is wasted. More importantly, the layer objects in self.layers are shared between the original and the clone (they are the constructor's input list, not freshly built). For the current usage (no shared in-place mutation observed) this seems benign, but if anyone later relies on model.layers for introspection, they'll get the original model's Keras layers — not the cloned ones — and may modify the wrong tensors.
D-2. Client.set_data (clients/client.py:70–77) shuffles list(zip(data_x, data_y)) and then unpacks back into separate lists. After this, self.data_x is a Python list of per-image tensors of shape (32, 32, 3), while before set_data it was a single (5, 32, 32, 3) tensor (from __init__). Both representations work with tf.gather(..., [idx]), but if any future code reads self.data_x.shape it will break in one case.
D-3. tf.GradientTape() in _HVP_training_algorithm is not declared persistent=True. The current call pattern only uses each tape once, so this works, but very close to the limit. Adding a single extra outer_tape.gradient(...) call would silently fail.
D-4. attacks/DLG.py:60 and attacks/InvertingGradients.py:55, 58 use the same seed=self.seed for all random initialisations (image and label). If TF's stateful op semantics combine global and op seeds in an unexpected way under different TF versions, the two tf.random.uniform(... seed=self.seed) calls in IG could yield correlated draws. Worth verifying empirically per TF version.
D-5. aggregate() in FedAvg/FedPer/PerFedAvg is unweighted (np.mean). Standard FedAvg weights by n_k. With identical n_k=5 it's equivalent, but non-IID extensions (e.g. variable-size clients) would silently produce wrong averages.
D-6. EXPERIMENT_SUMMARY.md:732–735 quotes 901 rows including header for the combined CSVs and 451 for the split CSVs. The current files match (901 each, confirmed via wc -l). However, since _initialize_csv truncates and there is no run identifier in the rows, you cannot tell whether all rows were written by one recent execution or are a leftover from an older code state. The single index column is per-handler-instance, not globally unique.
D-7. L-BFGS settings for DLG (f_relative_tolerance=1.25e-12, tolerance=1e-15, max_iterations=300) come from hyperparameter_simulation.py outputs that — given C-4 — may not be reproducible from current code. The DLG numbers in simulations/simulation.py may be from a hyperparameter search run on an older Client API.
E. Validation tests to run
#	Test	How	What it confirms
E-1
Single-class accuracy artefact
Replace _test_batch_index so each client is tested on a 50-sample balanced test set. Re-run utility eval.
Confirms C-2; expected: FedPer/Per-FedAvg accuracy collapses substantially below 1.0.
E-2
MAML-aware attacker negative control
Add a Per-FedAvg-aware DLG variant that does one inner FO step on the dummy before computing dummy gradients, then matches g_obs. Re-run on Per-FedAvg(FO).
Confirms C-1; expected: large reconstruction-quality jump on Per-FedAvg(FO).
E-3
Determinism
python simulations/simulation.py twice; diff results/attack_results/raw_results.csv.
Confirms C-15 / reproducibility status under the current seeding scheme.
E-4
Multi-seed replication
Externally loop seed ∈ {42, 1, 2, 3, 4} (parameterise simulations/simulation.py).
Provides per-(algo,attack) std and confidence intervals.
E-5
Tiny overfit sanity
One client, one image, FedAvg, 50 communication rounds. Track per-round CE on the same image.
CE should monotonically drop near 0 → confirms training pipeline works.
E-6
Aggregation identity
Two clients with hand-set weights; manually compute mean and compare with FedAvg.aggregate.
Validates aggregate numerics.
E-7
Client isolation
After one round, assert no two clients share client.model object identity (id(...)).
Confirms model.clone() gives independent objects.
E-8
hyperparameter_simulation.py smoke
python hyperparameter_simulation.py --smoke --attack DLG.
Confirms C-4 (immediate AttributeError).
E-9
Personalisation memorisation check
Print FedPer client's last-layer bias after final round.
If the bias for the client's class dominates → confirms degenerate "predict-only-my-class" head, supports C-2.
E-10
Round-by-round MSE for Per-FedAvg
attack_metric_by_round(..., "input mse") for each Per-FedAvg variant.
Should be ~flat near random; confirms attack failure (C-1) rather than algorithmic shielding.
E-11
FedPer last-layer mismatch
For FedPer K_p=3 round 5, run a control where the attacker uses the client's full personalised state. Compare reconstruction MSE to current.
Quantifies the part of FedPer's "privacy" that is just attacker-mismatch (C-9).
E-12
_adapt_per_fed_avg_model ablation
Compare Per-FedAvg utility with and without the in-eval adaptation step.
Quantifies how much of Per-FedAvg's 100% accuracy is due to adaptation vs. base meta-init (C-10).
E-13
Truncation safety
Run sim, copy CSVs, run sim again, diff.
Confirms C-7.
E-14
Beta vs alpha
Set beta=0.01 in simulations/simulation.py's Per-FedAvg dicts and re-run.
Confirms C-5: the trio's results should change materially.
F. Result-trust assessment
Trustworthy as currently reported (with caveats):

Reconstructed image PNGs in results/{true,reconstructed}_images/. They are produced from the actual model state at attack time and saved correctly.
The qualitative observation that FedAvg with one local SGD step is highly invertible by both DLG and IG (consistent with prior literature, MSE 0.005 for IG / 0.038 for DLG).
The qualitative observation that the current attacks fail badly on the current Per-FedAvg implementation (high MSE, ~0 SSIM).
Need rerunning before being reportable:

All mean_mse, mean_psnr, mean_ssim, min_mse, min_psnr numbers in any thesis table — they need at least 5 seeds and per-class/per-round disaggregation (C-3, C-8).
The FedPer monotonicity-in-K_p claim — currently non-monotonic in DLG (K_p=4 is worst at MSE 0.167, K_p=3 is best at 0.073), which suggests run-level noise dominates the K_p effect at num_runs=1.
Invalid until fixed:

The accuracy column in algorithm_summary and the bar plot it produces in plot_algorithm_accuracy. With C-2 it is essentially "single-class memorisation rate" with FedPer/Per-FedAvg trivially at 1.0 by construction. Do not ship as a privacy-utility tradeoff figure.
Any thesis statement of the form "Per-FedAvg leaks less than FedAvg under gradient inversion" without explicit threat-model qualification (C-1) and without a Per-FedAvg-aware attacker baseline.
Any results from hyperparameter_simulation.py produced after the regression introduced by client.random.sample(1) (C-4). Older results that predate this regression may exist in results/hyperparameters/..., but they are not reproducible from current HEAD.
G. Suggested patch plan, in priority order
G.1 Correctness affecting scientific conclusions
Fix metrics/PredictionComparison.py and metrics/ModelCrossEntropy.py to test on a balanced multi-class test set per client. Remove or clearly label the "test on the same class as training" version. (addresses C-2)
Implement and report a Per-FedAvg-aware attacker (FO-aware MAML attacker) and use it as the headline number for Per-FedAvg in the thesis. Keep the current naive attacker as an ablation. (C-1)
Set explicit, distinct alpha and beta in Per-FedAvg settings. Document chosen values. Add an assert in PerFedAvg.__init__ warning if alpha == beta. (C-5)
Either implement reuse_data_batches or remove the flag from settings dicts. (C-6)
Add a control where the FedPer attacker has access to the client's personalised tail. Either drop or annotate the FedPer privacy-by-K_p curve accordingly. (C-9)
Add an "adaptation off" Per-FedAvg utility variant. (C-10)
G.2 Reproducibility
Increase num_runs to ≥ 5; loop multiple seeds; produce per-run replicates and report run-level std. (C-3)
Switch ResultHandler._initialize_csv to non-truncating mode (or insert seed / git_sha / timestamp into the filename). Add seed column to every CSV row. (C-7)
Call tf.config.experimental.enable_op_determinism() in reseed. (C-15)
Make per-client RNGs distinct (seed = base_seed + client.id). (C-11)
Fix hyperparameter_simulation.py:123 (client.random.sample(1) → client.random_samples(1)) and re-run the hyperparameter search; record the resulting best DLG/IG settings in EXPERIMENT_SUMMARY.md. (C-4)
G.3 Logging / reporting
Group algorithm_summary by (algorithm, source_attack); let plots collapse only after deduplication. (C-17)
In attack_summary, replace pooled mean/min/max with mean-of-per-run-means and per-run std. Add per-class and per-round disaggregation to the default report. (C-3, C-8)
Replace min_mse / max_mse columns with quartiles; explicitly label any "best sample" reporting. (C-18)
Encode run/timestamp in VisualMetric filenames so reconstructed PNGs do not silently mix runs. (C-19)
G.4 Cleanup / refactoring
Add if __name__ == "__main__": guards to simulations/{simulation,dlg_sim,ig_sim}.py. (C-14)
Update makefile to point run at the actual entry, e.g. python -m simulations.simulation. (C-13)
Drop or lazy-load the wasted data.get_x_y(batch_size, 1) call in Client.__init__. (C-12)
Add assert client_training_rounds == 1 or attack is None in algorithm run methods to make the multi-step / attack interaction explicit. (C-16)
Make Model.clone() skip the initial Model({"layers": self.layers}) build that is immediately discarded. (D-1)
