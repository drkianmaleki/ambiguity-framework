"""
Centralized dataset loading with caching.

All load_dataset() calls return (X, y, feature_names) with:
  X            — np.ndarray float64, shape (n_samples, n_features)
  y            — np.ndarray int,     shape (n_samples,), values in {0, 1}
  feature_names — list[str]

Processed versions are cached in data/processed/ as .pkl files.
The REGISTRY here is the executable counterpart to experiment_grid.yaml.
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

_AGE_BRACKET_MAP = {
    "[0-10)": 5,   "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

# Executable specification — mirrors datasets section of experiment_grid.yaml.
# If you change a transformation here, update the YAML too (and vice versa).
REGISTRY = {
    "creditcard": {
        "file":        "creditcard.csv",
        "target":      "Class",
        "drop_cols":   [],
        "cat_cols":    [],
        "status":      "available",
        "note":        "All features are numeric PCA components; no preprocessing needed.",
    },
    "uci_credit": {
        "file":        "UCI_Credit_Card.csv",
        "target":      "default.payment.next.month",
        "drop_cols":   ["ID"],
        "cat_cols":    [],
        "status":      "available",
        "note":        "Drop ID; all remaining features are numeric.",
    },
    "heart": {
        "file":        "heart.csv",
        "target":      "HeartDisease",
        "drop_cols":   [],
        "cat_cols":    ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"],
        "status":      "available",
    },
    "brfss": {
        "file":        "heart_disease_health_indicators_BRFSS2015.csv",
        "target":      "HeartDiseaseorAttack",
        "drop_cols":   [],
        "cat_cols":    [],
        "status":      "available",
        "note":        "All 21 features are already numeric binary/ordinal survey responses.",
    },
    "readmissions": {
        "file":             "hospital_readmissions.csv",
        "target":           "readmitted",
        "drop_cols":        [],
        "cat_cols":         ["medical_specialty", "diag_1", "diag_2", "diag_3",
                             "glucose_test", "A1Ctest", "change", "diabetes_med"],
        "age_bracket_col":  "age",
        "target_map":       {"yes": 1, "no": 0},
        "status":           "available",
        "note":             "age bracket → midpoint int; target binarized from yes/no strings.",
    },
    "nslkdd": {
        "file":                    "nsl_kdd_dataset.csv",
        "target":                  "label",
        "drop_cols":               [],
        "cat_cols":                ["protocol_type", "service", "flag"],
        "binary_label_normal_cls": "normal",
        "status":                  "available",
        "note":                    "Binary framing: normal=0, all attack classes (DoS/Probe/U2R/R2L)=1. Dataset: simulated NSL-KDD structure, 4,431 rows (Kaggle: programmer3/nsl-kdd-intrusion-detection-dataset).",
    },
}


def load_dataset(name: str, force_reload: bool = False):
    """
    Load, clean, and cache a dataset by registry name.

    Parameters
    ----------
    name : str
        Registry key — one of: creditcard, uci_credit, heart, brfss,
        readmissions, nslkdd.
    force_reload : bool
        Bypass the cache and re-process from the raw CSV.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), int, values in {0, 1}
    feature_names : list[str]

    Raises
    ------
    ValueError       — unknown dataset name
    FileNotFoundError — raw CSV not present (with download instructions)
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'.\n"
            f"Available: {list(REGISTRY)}"
        )

    cfg = REGISTRY[name]
    raw_path = RAW_DIR / cfg["file"]

    if not raw_path.exists():
        if cfg["status"] == "pending":
            raise FileNotFoundError(
                f"Dataset '{name}' has not been downloaded yet.\n"
                f"Expected path : {raw_path}\n"
                f"See data/raw/README.md for the download URL."
            )
        raise FileNotFoundError(
            f"Raw file not found for dataset '{name}'.\n"
            f"Expected path : {raw_path}"
        )

    cache_path = PROCESSED_DIR / f"{name}.pkl"
    if cache_path.exists() and not force_reload:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    X, y, feature_names = _process(cfg, raw_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((X, y, feature_names), f)

    return X, y, feature_names


def available_datasets() -> list:
    """Return names of datasets whose raw CSV files are present on disk."""
    return [name for name, cfg in REGISTRY.items()
            if (RAW_DIR / cfg["file"]).exists()]


def dataset_info() -> pd.DataFrame:
    """Return a summary DataFrame of all registry entries and their disk status."""
    rows = []
    for name, cfg in REGISTRY.items():
        present = (RAW_DIR / cfg["file"]).exists()
        cached = (PROCESSED_DIR / f"{name}.pkl").exists()
        rows.append({
            "name":    name,
            "file":    cfg["file"],
            "target":  cfg["target"],
            "status":  cfg["status"],
            "on_disk": present,
            "cached":  cached,
        })
    return pd.DataFrame(rows)


def _process(cfg: dict, raw_path: Path):
    df = pd.read_csv(raw_path)

    df = df.drop(columns=cfg.get("drop_cols", []), errors="ignore")

    for col in cfg.get("impute_median_cols", []):
        if col in df.columns:
            median = df[col].median()
            n_missing = df[col].isna().sum()
            if n_missing:
                warnings.warn(f"Imputing {n_missing} missing values in '{col}' with median={median:.4f}")
            df[col] = df[col].fillna(median)

    age_col = cfg.get("age_bracket_col")
    if age_col and age_col in df.columns:
        df[age_col] = df[age_col].map(_AGE_BRACKET_MAP)

    for col in cfg.get("cat_cols", []):
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    target_col = cfg["target"]

    target_map = cfg.get("target_map")
    if target_map:
        df[target_col] = df[target_col].str.strip().str.lower().map(target_map)

    normal_cls = cfg.get("binary_label_normal_cls")
    if normal_cls:
        df[target_col] = (~df[target_col].str.strip().str.lower().eq(normal_cls.lower())).astype(int)

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(float)
    y = df[target_col].values.astype(int)

    return X, y, feature_names
