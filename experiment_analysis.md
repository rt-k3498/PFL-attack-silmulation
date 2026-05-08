TL;DR
Your two specific findings are substantively real, well-explained phenomena, not simulation bugs. They are consistent with established gradient-inversion literature (Zhao et al. 2020 — iDLG, Geiping et al. 2020 — InvertingGradients). However, the specific numbers in your CSV are partly shaped by a few config choices in simulation.py that bias the comparison; you should report the qualitative finding but not the exact percentages without re-running with those choices fixed.

The most important fact in your data is below — please pin this down before reading further:

algorithm            attack               label_recover  mean_mse  mean_ssim
FedAvg               DLG                  1.00 (50/50)   0.0378    0.5051
FedAvg               InvertingGradients   1.00 (50/50)   0.0050    0.7908
FedPer(K_p=1..5)     DLG                  0.34–0.48      ~0.07–.17 ~0.12–.20
FedPer(K_p=1..5)     InvertingGradients   0.58–0.78      ~0.16–.23 ~0.09–.17
Per-FedAvg(FO/HF/HVP) DLG                 0.26–0.40      ~0.09–.10 ~0.05
Per-FedAvg(FO/HF/HVP) InvertingGradients  0.98–1.00      ~0.30     ~0.00
The big jump is: Per-FedAvg + IG recovers labels almost perfectly while the image MSE is at near-random (SSIM ≈ 0). That single pattern is the most interesting result you have, and it is real.

1. Is there a bug in simulations/simulation.py that could fake these results?
I went through simulation.py line-by-line ignoring all unrelated audit items. In terms of what your attack pipeline actually receives, there is no logic bug that fakes the labels. The flow is:

algos[*] → calls algo.run(attack, …) (simulation.py:216-219).
Inside FedAvg.run / FedPer.run / PerFedAvg.run: per round, per client, training is done on model = self.model.clone(), weights are read out (client.get_weights()), and the attack is given (self.model, client_weights, {"learning_rate": …, "num_classes": 10}). The "true" (x, y) for that client/round is fetched separately as data = client.get_data_used_for_training()[-1] and only handed to the metrics, never to the attack.
The attack therefore has no path to read the true label or the true image. Label recovery is genuinely from gradients.
So your label-recovery numbers are not the result of label leakage through the API.

What the simulation does do that nudges the result is config, not code:

Config in simulation.py	Effect on label-recovery numbers
batch_size = 5, local_training_rounds = 1 (lines 66, 71)
Each client update is from one (x, y). Any single-sample gradient inversion attack has a near-trivial label-recovery channel via softmax-CE. This biases label recovery upward across the board.
get_structured_x_y(...) → each client gets one CIFAR class only (line 78, with data/data.py:92-113)
The attack is choosing 1 of 10 with a strong class-conditional signal. Label recovery is much easier here than in any realistic non-IID FL run.
Per-FedAvg dicts pass no alpha / beta (simulation.py:148-174) → both default to 0.1 (per_fedAvg.py:37-38)
Per-FedAvg(FO) reduces to two SGD steps from the same (x, y) at the same LR. The "meta-gradient" you're attacking is barely perturbed from an ordinary FedAvg gradient. This is the single biggest reason IG still recovers labels nearly perfectly on Per-FedAvg.
InvertingGradients hyperparameters (simulation.py:185-191): init_step_size=0.1, final_step_size=0.09, alpha=6.3e-13, use_signed_adam=True (default)
alpha=6.3e-13 is effectively zero TV — it's pure cosine-direction matching with sign-Adam. The cosine signal aligns the dummy label very reliably even when the image cannot be reconstructed. This is exactly the regime in which the IG label-vs-image asymmetry is biggest.
_SeededAttack reseeds with ts = 42 + 1000·run before every attack call (lines 33-48, 213)
Every one of the 50 attacks within an (algo, attack) cell starts from the same dummy init. Label recovery becomes effectively per-class deterministic: it's "did the optimizer succeed for this class yes/no", repeated across 5 rounds. The per-round breakdown confirms this — most cells are flat across rounds (r1=4/10 r2=4/10 …).
None of those is a correctness bug. They are deliberate experimental simplifications. They make the magnitude of the effect bigger and cleaner than it would be in a realistic FL setting, but they don't fabricate the effect itself.

The one thing in simulation.py that does worry me about the attack row, narrowly, is C-1 from the audit: the attack is matching a dummy gradient computed at theta against an observed quantity that equals (FO) ∇L(theta_adapted; x, y) or (HF/HVP) v − α·Hv. With α = β = 0.1 and only one inner step, theta_adapted is very close to theta, so the mismatch is small and IG label recovery still works. With a real Per-FedAvg setting (e.g. α≫β or α=0.1, β=0.001), the meta-gradient could be much further from a normal gradient and your "IG-only label leakage" rate could drop. That's worth flagging as a robustness check, not a "wrong result".

2. Why FedPer label recovery stays high (especially under IG, especially K_p ≥ 3)
This is the part I want you to understand mechanically, because the answer is not "your code is wrong".

For one sample (x, y) and softmax-CE over a model f(x; θ) = softmax(z(x; θ)):

∂L / ∂b_out_c = p_c − 1[c == y]
For the true class y: this entry is negative (since p_y − 1 < 0).
For every other class: this entry is non-negative.
So if the attacker can see the output bias gradient (or anything proportional to it), the label is recoverable in closed form (this is iDLG, Zhao et al., NeurIPS 2020). For FedAvg this gradient is sent — that's why you get 100% label recovery for FedAvg under both attacks.

For FedPer with K_p ≥ 1 the output layer is not sent, so this trivial channel is closed. Yet you still see 58–78% IG label recovery. The mechanism is more subtle, but real:

dummy_label is a tf.Variable of shape (1, num_classes) (InvertingGradients.py:57-59) that is jointly optimized with dummy_image.
The dummy gradient w.r.t. the shared first-N layers depends on the dummy label through the full forward chain. Different labels produce systematically different early-layer gradients (because different classes → different per-pixel softmax-CE weights → different backward signal at the first conv).
IG's loss is cosine distance of the direction of the shared-layer gradient (InvertingGradients.py:79-81). Cosine is invariant to magnitude, so even tiny class-conditional differences in the first-conv gradient are sufficient to discriminate among 10 candidate dummy labels.
With use_signed_adam=True (default) and alpha ≈ 0 (your config), the optimizer's update on dummy_label is sign(∂cos / ∂dummy_label) per step — a strong, label-discriminative signal.
The dummy label is therefore driven to argmax at the correct class long before the dummy image converges. That's why you see "label = 1.0, MSE = 0.30, SSIM ≈ 0" — the label channel converged, the image channel did not.
Two consequences:

Personalisation of the output head does NOT protect labels under cosine-direction gradient inversion. This is a genuine and interesting finding for your thesis.
It works under IG and not DLG because DLG uses an unnormalized L2 loss (DLG.py:83-86) whose magnitude is dominated by the first conv layer's large-norm gradient. The fine-grained label signal is washed out. Don't say "IG is better at labels in general"; say "cosine-direction matching makes the label channel separable from the image channel, even when image reconstruction fails." That's what your data shows.
The K_p ≥ 3 → high IG label recovery is the same mechanism as K_p = 1, just with progressively less first-N-layer information. That recovery only modestly degrades as K_p increases (76 → 58 → 58 → 76 → 78) is itself the finding: even the first conv layer alone (K_p=5) is enough for IG to reliably pick the right of 10 classes. That is a real privacy concern.

Two cautions before you publish that exact monotonicity:

Your numbers are not monotonic in K_p (K_p=4 is higher than K_p=3 for both DLG and IG label recovery). With num_runs = 1 and only 10 client/class trials per round, this is well within seed noise. You cannot claim "K_p has the following effect on label recovery" from a single seed. With the audit's E-4 (multi-seed) re-run, expect either (i) a clearer flat curve at ≈70-80% across K_p, or (ii) a real mild downward trend; the K_p=4 spike will most likely smooth out.
The single-class-per-client design (C-2 in the audit) means "the answer is one of 10". In a realistic FL deployment a client has many classes, and the per-update label-inference task is about a batch of labels, not a single class. Don't claim "labels leak" without specifying "for single-sample updates of a single class". Geiping et al. and others have shown that batch label inference is much harder.
3. Why Per-FedAvg leaks labels under IG even though the gradient is not what the attacker assumes
Three layered reasons, in order of importance:

α = β = 0.1: as noted, Per-FedAvg(FO) collapses to two SGD steps from the same (x, y). The "meta-gradient" the attacker observes is ∇L(θ − 0.1·∇L(θ; x, y); x, y), evaluated at a very small perturbation of θ. Cosine of this against ∇L(θ; x', y') (what IG matches) still has the right sign structure across labels. So the label channel is preserved almost untouched. Set β = 0.01 and you'd see this drop noticeably.

HF/HVP variants reduce to FO + small correction. g_obs = v − α·Hv is dominated by v (which is the gradient at the adapted point). With small α·H, this is close to v, which itself is close to ∇L(θ; x, y) for the same reason as point 1. So all three Per-FedAvg variants give nearly identical label recovery (FO=100%, HF=98%, HVP=98%) — exactly what you see. That near-identity across FO/HF/HVP is additional evidence the leak is the FO-style channel, not anything specific to second-order updates.

Image reconstruction does fail for Per-FedAvg under IG (MSE 0.30, SSIM ≈ 0). That part is genuinely the threat-model mismatch from C-1: the attacker is regressing onto a dummy gradient at θ while the truth lives at θ_adapted. So the image-reconstruction numbers for Per-FedAvg are mostly an artifact of the attacker not modeling the meta-update. The label-recovery numbers are not, because the label channel survives the perturbation.

So the headline you can responsibly write is:

"Under single-sample, one-class-per-client gradient inversion, IG-style cosine-direction attacks recover labels almost perfectly across FedAvg, FedPer (any K_p), and Per-FedAvg (FO/HF/HVP) — even when image reconstruction fails. DLG-style L2 attacks largely do not, because their loss is dominated by early-layer gradient magnitude rather than direction."

That is a genuinely defensible, interesting privacy claim, and it does not depend on the audit-flagged bugs.

4. What I would do before reporting these specific numbers
These are the smallest, most targeted re-runs that would let you publish the IG-vs-DLG label asymmetry confidently from simulation.py alone:

#	Tweak in simulation.py only	Question it answers
1
Set "beta": 0.001 (or 0.01) in the three Per-FedAvg lambdas (lines 148-174).
Is the IG label leak on Per-FedAvg an actual MAML-resistant leak, or an artifact of α = β? Expected: label recovery drops noticeably for FO, less for HF/HVP. If it stays at ~90%+, the finding is robust.
2
Set "alpha": 1e-2 and increase init_step_size/final_step_size similarly in the IG lambda (line 186).
Is the IG label leak driven by the near-zero TV weight (i.e. cosine-only)? Expected: label recovery still high but image MSE no longer at random.
3
Loop over 3-5 different seed values at the top of the file, run the whole thing, and aggregate.
Cross-validates the K_p non-monotonicity and gives you per-(algo, attack) std. Required for any monotonicity claim.
4
Set client_adaptation_rounds = 5 for one Per-FedAvg variant (e.g. FO).
Push θ_adapted further from θ and watch IG label recovery degrade (or not). If it still recovers labels, the cosine label channel is genuinely robust to MAML-style adaptation, which is a stronger version of your finding.
5
Replace _SeededAttack reseeding with a per-client, per-round trial seed.
Rules out "label recovery is binary per class because the dummy init is identical". With per-trial init you get genuine 50-trial sampling instead of 10-class × 5-replicates.
Tweak 1 is the single most important one for the Per-FedAvg story. Tweak 3 is the most important for the FedPer K_p story. Both can be done by editing only simulation.py (no changes to algorithms/attacks), so they're cheap.

5. Short answer to your question
"Is there a bug in simulation.py that could have produced wrong results, or a flaw in the attack/algorithm logic?"

No bug in simulation.py produces fake labels. The _SeededAttack reseeding makes per-(algo, attack) results less independent than they look (so 50/50 ≠ 50 independent successes), and the Per-FedAvg α = β = 0.1 makes FO collapse into a near-FedAvg update — but neither fabricates the IG label-leakage result.

"Or do I have something truly interesting?"

You have something genuinely interesting: cosine-based gradient inversion can extract labels from Per-FedAvg meta-updates and from FedPer updates with up to 5 personalised layers, even under conditions where image reconstruction is at chance. That's a real finding, well-grounded in the iDLG / cosine-attack literature, and it's specifically your contribution to extend it to PFL update structures.

"Why only IG?"

Because IG matches gradient direction (cosine), and the label channel is encoded as a sign pattern in the dummy gradient regardless of magnitude. DLG matches gradient L2 magnitude, which is dominated by early-layer norm and washes out the label sign pattern. This is mechanism, not bug.

What the thesis still needs to be defensible: re-run with multiple seeds (at minimum) and report the IG label-recovery rate with std; explicitly disclose the single-class-per-client setup and the α = β setting; and show the finding is robust (or not) to setting β ≠ α. Once you've done that, the IG-only label-leak result is publishable.