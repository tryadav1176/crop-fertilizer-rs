"""Train the crop and fertilizer recommendation pipelines.

Each model is ONE scikit-learn Pipeline (preprocessing + classifier), so the
scaler/encoders can never be forgotten or mismatched at serving time.

Usage:
    python -m src.train                    # train both
    python -m src.train --target crop
    python -m src.train --data-dir dataset --model-dir models
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config

logger = logging.getLogger("train")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_crop(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = set(config.CROP_FEATURES + [config.CROP_TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna().drop_duplicates()
    return df[config.CROP_FEATURES], df[config.CROP_TARGET].astype(str).str.strip()


def load_fertilizer(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=config.FERT_RENAME)
    needed = config.FERT_NUMERIC + config.FERT_CATEGORICAL + [config.FERT_TARGET]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna().drop_duplicates()
    for col in config.FERT_CATEGORICAL + [config.FERT_TARGET]:
        df[col] = df[col].astype(str).str.strip()
    return (
        df[config.FERT_NUMERIC + config.FERT_CATEGORICAL],
        df[config.FERT_TARGET],
    )


# ----------------------------------------------------------------------------
# Preprocessing + candidate models
# ----------------------------------------------------------------------------
def crop_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), config.CROP_FEATURES)])


def fertilizer_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.FERT_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.FERT_CATEGORICAL,
            ),
        ]
    )


def candidate_pipelines(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    rs = config.RANDOM_STATE
    models = {
        "decision_tree": DecisionTreeClassifier(random_state=rs),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "logistic_regression": LogisticRegression(max_iter=2000),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=rs),
    }
    return {
        name: Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
        for name, model in models.items()
    }


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def _feature_importance(pipe: Pipeline) -> Optional[Dict[str, float]]:
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    pairs = sorted(zip(names, model.feature_importances_), key=lambda t: -t[1])
    return {str(n): round(float(v), 4) for n, v in pairs}


def fit_select_evaluate(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer):
    """Compare candidates with CV, evaluate the winner on a held-out split,
    then refit the winner on ALL data for deployment."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    n_splits = max(2, min(5, int(y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    candidates = candidate_pipelines(preprocessor)
    cv_results = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}
        logger.info("CV %-24s acc=%.4f (+/- %.4f)", name, scores.mean(), scores.std())

    best_name = max(cv_results, key=lambda n: cv_results[n]["mean"])
    best = candidates[best_name].fit(X_train, y_train)

    y_pred = best.predict(X_test)
    proba = best.predict_proba(X_test)
    classes = best.classes_
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    k = config.TOP_K
    if len(classes) > k:
        test_metrics[f"top_{k}_accuracy"] = round(
            float(top_k_accuracy_score(y_test, proba, k=k, labels=classes)), 4
        )

    final = clone(candidates[best_name]).fit(X, y)  # refit on everything for serving
    meta = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_samples": int(len(X)),
        "features": list(X.columns),
        "classes": [str(c) for c in final.classes_],
        "cv_accuracy": cv_results,
        "test_metrics": test_metrics,
        "feature_importance": _feature_importance(final),
        "notes": (
            "Metrics come from a stratified 80/20 hold-out split; the deployed "
            "model is refit on the full dataset."
        ),
    }
    if len(X) < 500:
        meta["warning"] = "Small dataset: treat these metrics as indicative only."
    return final, meta


def _json_default(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def _save(model_dir: Path, key: str, pipe: Pipeline, meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / f"{key}_pipeline.joblib")
    meta_path = model_dir / "metadata.json"
    all_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    all_meta[key] = meta
    meta_path.write_text(json.dumps(all_meta, indent=2, default=_json_default))


def train_crop(csv_path: Optional[Path] = None, model_dir: Optional[Path] = None) -> dict:
    csv_path = Path(csv_path or config.DATA_DIR / config.CROP_CSV_NAME)
    X, y = load_crop(csv_path)
    pipe, meta = fit_select_evaluate(X, y, crop_preprocessor())
    _save(Path(model_dir or config.MODEL_DIR), "crop", pipe, meta)
    return meta


def train_fertilizer(csv_path: Optional[Path] = None, model_dir: Optional[Path] = None) -> dict:
    csv_path = Path(csv_path or config.DATA_DIR / config.FERT_CSV_NAME)
    X, y = load_fertilizer(csv_path)
    pipe, meta = fit_select_evaluate(X, y, fertilizer_preprocessor())
    meta["categories"] = {c: sorted(X[c].unique().tolist()) for c in config.FERT_CATEGORICAL}
    _save(Path(model_dir or config.MODEL_DIR), "fertilizer", pipe, meta)
    return meta


def _summary(title: str, meta: dict) -> None:
    tm = meta["test_metrics"]
    extra = "".join(f", {k}={v}" for k, v in tm.items() if k.startswith("top_"))
    print(f"[{title}] best={meta['model_name']}  test_acc={tm['accuracy']}  "
          f"macro_f1={tm['macro_f1']}{extra}  n={meta['n_samples']}")
    if "warning" in meta:
        print(f"    ! {meta['warning']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--target", choices=["crop", "fertilizer", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.target in ("crop", "all"):
        _summary("crop", train_crop(args.data_dir / config.CROP_CSV_NAME, args.model_dir))
    if args.target in ("fertilizer", "all"):
        _summary("fertilizer", train_fertilizer(args.data_dir / config.FERT_CSV_NAME, args.model_dir))
    print(f"Artifacts saved to {args.model_dir}")


if __name__ == "__main__":
    main()
