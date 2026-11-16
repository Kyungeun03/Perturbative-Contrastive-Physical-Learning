# Perturbative Contrastive Physical Learning — code

Scripts used to produce the results in the manuscript *Perturbative Contrastive Physical Learning* and its Supplementary Information. Scripts for the main text are in `main/`, those for the Supplementary Information in `supp/`. Each script is standalone: run it from inside its folder with Python 3 and it writes its figures/results next to itself.

## Requirements

- Python 3
- numpy, scipy, scikit-learn, matplotlib
- strawberryfields (Gaussian backend) for all `CV_*`, `cv_*`, and `SDR_*` scripts

Random seeds are fixed in each script (seed 42, trial seeds `42 + 13*trial`), so the reported numbers are reproducible.

## Main text

### Spring networks as contrastive learners
| Script | What it does |
|---|---|
| `main/Iris_elongation_classifier_v1.py` | Case 1: Iris classification with the triangular spring network (single run, seed 42). |
| `main/Iris_elongation_classifier_v2_plot.py` | Case 2: single run, seed 42; writes `classification2.svg`. |
| `main/Iris_elongation_run_v2_trials.py` | Runs `Iris_elongation_classifier_v2_plot.py` unchanged over 25 trial seeds and collects the accuracies (mean ± std). Both v2 files must be in the same folder. |

### Continuous-variable linear photonic circuits as contrastive learners
| Script | What it does |
|---|---|
| `main/CV_pseudoinverse_multirun.py` | Four-mode Gaussian circuit, Iris classification with the regularized pseudoinverse update, 25 trials. Writes `cv_pseudoinverse_training_curves.*`, `cv_pseudoinverse_test_accuracy.*`, `cv_pseudoinverse_measurement_dist.*`. |

## Supplementary Information

### Supplementary Note 1: Mode A and Mode B comparison
| Script | What it does |
|---|---|
| `supp/CV_modeA_contrast.py` | Mode A: free-vs-nudged contrasts, target enters the update through the error; scalar and multivariate forms. |
| `supp/CV_modeB_contrast.py` | Mode B: candidate nudges scored against the target, single-best and elite-center selection. |
| `supp/CV_modeB_ancilla.py` | Mode B with the target carried by an ancilla mode; the error is read optically on mode 0. |

### Supplementary Note 2: Gradient descent versus multivariate inversion
| Script | What it does |
|---|---|
| `supp/pseudoinverse_vs_gd_stability_4methods.py` | GD, Gauss–Newton, Tikhonov, Levenberg–Marquardt under learning-rate sweep and ill-conditioned feature scaling. Writes `pseudoinverse_vs_gd_stability.*`. |

### Supplementary Note 3: Effect of preprocessing
| Script | What it does |
|---|---|
| `supp/symplectic_component_sweep.py` | Accuracy versus number of retained symplectic components on the 8×8 digits. Prints the final table. |

### Supplementary Note 4: Partition size and input noise
| Script | What it does |
|---|---|
| `supp/CV_block_locality.py` | Pseudoinverse update with the parameters partitioned into blocks of size B; results appended to a CSV per trial. |
| `supp/CV_block_locality_withnoise.py` | Same sweep with Gaussian input noise of standard deviation σ redrawn at every circuit evaluation. |

### Supplementary Note 7: Tunable multiplier with squeezing
| Script | What it does |
|---|---|
| `supp/cv_trainable_DSR_multiplier.py` | Fits the single-mode D–S–R multiplier circuit to the classical gradient; writes `cv_multiplier_params.npy` and `cv_gradient_train.*`. |
| `supp/cv_multiplier_params.npy` | Fitted multiplier parameters used by the training loop. |
| `supp/SDR_training_loop.py` | Iris classification with the multiplier circuit supplying the gradient and the D-gate circuit applying the update, 25 trials. Writes `cv_training_curves.*`, `cv_test_accuracy_trials_ver2.*`, `cv_measurement_dist_ver2.*`. |

Run `cv_trainable_DSR_multiplier.py` first (or use the shipped `.npy`), then `SDR_training_loop.py`, from inside `supp/`.

### Supplementary Note 8: Tunable linear optical multiplier
| Script | What it does |
|---|---|
| `supp/cv_ablation_study.py` | Configurations A–D isolating the CV multiplier gradient and the CV displacement update against their classical counterparts. |

## AI-Assisted Programming

The initial implementation was developed by the authors. Multiple versions of Claude (main) and ChatGPT (sub) were used as a programming assistant for feature additions, and visualization. The underlying physical model and code design were developed by the authors. All AI-assisted code was reviewed, modified as needed, and validated by the authors.

---

## License

MIT — see `LICENSE`.
