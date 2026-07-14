
# Ambiguity Range Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-62%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey)
![Domain](https://img.shields.io/badge/topic-model%20evaluation-purple)
[![Preprint](https://img.shields.io/badge/Preprint-SSRN-darkred)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7118401)

A model-agnostic diagnostic toolkit for post-hoc evaluation of binary classifiers.

📄 **Preprint:** [Quantifying Decision-Boundary Uncertainty in Binary Classifiers: The Ambiguity Range Framework](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7118401)

Standard metrics like AUC-ROC summarise *global* discriminative performance but
say nothing about the *local* distribution of predicted probabilities near the
decision boundary. A classifier with AUC = 0.90 may still assign probabilities
clustered tightly around 0.5 for a large fraction of predictions—instances
where the model is, operationally, guessing.

The Ambiguity Range Framework makes this indecision zone explicit and measurable
through three complementary metrics.

---

## Metrics

| Metric | Symbol | What it measures |
|---|---|---|
| **Ambiguity Mass** | AA_Mass(δ) | Fraction of predictions inside the indecision interval ℐ_δ = [0.5−δ, 0.5+δ] |
| **Ambiguity Area over P·R** | AA_PR | How severely precision·recall degrades as the decision threshold varies |
| **Ambiguity Area over F₁** | AA_F1 | How severely F₁ degrades across thresholds (generalises to F_β) |

All three metrics return a value in **[0, 1]** where **lower = less ambiguous**.

---

## Installation

```bash
git clone https://github.com/drkianmaleki/ambiguity-framework.git
cd ambiguity-framework
pip install -e .
pip install -r requirements.txt
```

---

## Quick start

```python
from ambiguity_suite import compute_AAPR, compute_AAF1, compute_AAMass

# Ground-truth labels and predicted probabilities from any classifier
y_true = [0, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.4, 0.6, 0.9, 0.5, 0.8]

# Global threshold-sensitivity metrics (delta-invariant)
print(compute_AAPR(y_true, y_prob))          # → float in [0, 1]
print(compute_AAF1(y_true, y_prob))          # → float in [0, 1]

# Local indecision zone metric (varies with delta)
print(compute_AAMass(y_true, y_prob, delta=0.10))   # interval [0.40, 0.60]
print(compute_AAMass(y_true, y_prob, delta=0.20))   # interval [0.30, 0.70]
```

### Command-line usage

```bash
python main.py \
    --data path/to/predictions.csv \
    --y-true label_column \
    --y-prob score_column \
    --delta 0.05 0.10 0.20
```

---

## Project structure

```
ambiguity-framework/
├── ambiguity_suite/        ← installable package (the three metrics)
├── experiments/                  ← experiment runners and data loaders
│   ├── _runner_base.py           ← shared LR / RF / XGBoost evaluation logic
│   ├── run_all_real.py           ← run all six real-world datasets at once
│   ├── run_synthetic.py          ← Gaussian mixture sweep
│   ├── run_calibration_ablation.py ← Platt / isotonic calibration study
│   ├── run_bootstrap_ci_fast.py  ← 95% bootstrap CIs on key comparisons
│   ├── run_entropy_comparison.py ← entropy and margin uncertainty baselines
│   ├── run_paired_tests.py       ← Wilcoxon signed-rank tests (10-fold CV)
│   ├── run_aamass_tstar.py       ← AAMass at F1-optimal threshold t*
│   ├── run_calibrated_aamass.py  ← calibrated vs raw AAMass for all datasets
│   ├── plot_reliability.py       ← reliability diagrams (calibration curves)
│   ├── plot_graphical_abstract.py ← graphical abstract figure
│   └── configs/                  ← experiment_grid.yaml (all reproducible params)
├── tests/                  ← 62 pytest tests (all passing)
├── data/
│   ├── raw/                ← place downloaded CSVs here (see below)
│   └── processed/          ← auto-generated pickle cache
└── results/
    ├── tables/             ← experiment output CSVs
    └── figures/            ← generated plots
```

---

## Reproducing the experiments

### 1. Download the datasets

See [`data/raw/README.md`](data/raw/README.md) for download URLs.
Place the CSV files in `data/raw/` before running any experiments.

| Dataset | Source | Rows |
|---|---|---|
| Heart Disease | [Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction) | 918 |
| UCI Credit Card | [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 |
| Hospital Readmissions | [Kaggle](https://www.kaggle.com/datasets/dubradave/hospital-readmissions) | 25,000 |
| NSL-KDD | [Kaggle](https://www.kaggle.com/datasets/programmer3/nsl-kdd-intrusion-detection-dataset) | 4,431 |
| BRFSS 2015 | [Kaggle](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset) | 253,680 |
| Credit Card Fraud | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 |

### 2. Run all real-world experiments

```bash
python experiments/run_all_real.py --force-reload
```

Results written to `results/tables/real_summary.csv`. Estimated runtime: ~25 minutes.

### 3. Run synthetic experiments

```bash
python experiments/run_synthetic.py
```

Results written to `results/tables/synthetic_summary.csv`. Estimated runtime: ~5 minutes.

### 4. Run calibration ablation study

```bash
python experiments/run_calibration_ablation.py
```

Compares raw vs Platt vs isotonic calibration across all classifiers and datasets.
Results written to `results/tables/calibration_ablation_summary.csv`.

### 5. Run bootstrap confidence intervals

```bash
python experiments/run_bootstrap_ci_fast.py
```

Computes 95% bootstrap CIs for AAMass and AUC on the four moderate-sized datasets.
Results written to `results/tables/bootstrap_ci_fast.csv`. Estimated runtime: ~10 minutes.

### 6. Run entropy and margin baseline comparison

```bash
python experiments/run_entropy_comparison.py
```

Compares AAMass against predictive entropy and margin uncertainty.
Results written to `results/tables/entropy_comparison_summary.csv`.

### 7. Run paired statistical tests (Wilcoxon signed-rank)

```bash
python experiments/run_paired_tests.py
```

Runs 10-fold stratified cross-validation and applies two-sided Wilcoxon
signed-rank tests on fold-paired AAMass and AUC-ROC observations.
Results written to `results/tables/paired_tests_summary.csv`.
Estimated runtime: ~2 hours (all six datasets).

### 8. Run AAMass at F1-optimal threshold t*

```bash
python experiments/run_aamass_tstar.py
```

Evaluates AAMass centred at the F1-optimal operating threshold t* for each
classifier and dataset. Demonstrates metric utility on imbalanced datasets
where the default 0.5 centre is suboptimal.
Results written to `results/tables/aamass_tstar_summary.csv`.
Estimated runtime: ~1 minute.

### 9. Run calibrated AAMass comparison

```bash
python experiments/run_calibrated_aamass.py
```

Computes Platt-scaled (calibrated) AAMass alongside raw AAMass for all six
datasets. Produces the dual raw/calibrated column in the main results table.
Results written to `results/tables/calibrated_aamass_summary.csv`.
Estimated runtime: ~5 minutes.

### 10. Generate figures

```bash
python experiments/plot_reliability.py --dataset readmissions
python experiments/plot_reliability.py --all        # all six datasets
python experiments/plot_graphical_abstract.py       # graphical abstract
```

---

## Running the tests

```bash
pytest tests/ -v
```

All 62 tests should pass. The test suite covers metric correctness, edge cases,
boundary behaviour, imbalanced datasets, and input validation.

---

## API reference

### `compute_AAPR(y_true, y_prob, delta=0.1)`

Ambiguity Area over Precision and Recall. Integrates P(t)·R(t) over all
thresholds and normalises against a prevalence-dependent random-classifier
baseline. `delta` is accepted for API consistency but does not affect the
result — use `compute_AAMass` to measure score density within an interval.

### `compute_AAF1(y_true, y_prob, delta=0.1, beta=1.0)`

Ambiguity Area over F-beta. Same as AAPR but integrates the F_β score.
At `beta=1.0` this is the standard F₁; `beta>1` weights recall more heavily.

### `compute_AAMass(y_true, y_prob, delta=0.1)`

Ambiguity Mass. Returns the fraction of predictions in
ℐ_δ = [0.5−δ, 0.5+δ]. Compute across a range of δ values to trace an
ambiguity profile for your classifier.

### `get_interval(delta)` / `interval_width(delta)`

Utilities that return the bounds and full width of ℐ_δ.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{maleki2026ambiguity,
  author = {Maleki, Kian},
  title  = {The Ambiguity Range Framework: A Diagnostic Toolkit for Operational Evaluation of Binary Classifiers},
  year   = {2026},
  note   = {Manuscript under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
