from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

FEATURE_COLS_DEFAULT = [
    "pcr_oi","pcr_volume","pcr_oi_chg1","pcr_volume_chg1",
    "news_count","sent_pos_share","sent_neg_share","sent_score_mean","sent_score_std",
    "news_count_chg1","sent_score_mean_chg1"
]

def add_labels(feats: pd.DataFrame) -> pd.DataFrame:
    df = feats.copy()
    if "underlying_close" in df.columns and df["underlying_close"].notna().any():
        df = df.sort_values("date")
        df["return_next"] = df["underlying_close"].pct_change().shift(-1)
        df["y"] = (df["return_next"] > 0).astype(int)
    else:
        s = df.get("sent_score_mean_chg1", pd.Series([0]*len(df)))
        p = df.get("pcr_oi_chg1", pd.Series([0]*len(df)))
        score = s - p.fillna(0)
        df["y"] = (score > 0).astype(int)
    return df

def split_xy(df: pd.DataFrame, feature_cols=None):
    if feature_cols is None: feature_cols = FEATURE_COLS_DEFAULT
    use_cols = [c for c in feature_cols if c in df.columns]
    X = df[use_cols].fillna(0.0).values
    y = df["y"].values
    return X, y, use_cols

def train_tscv(df: pd.DataFrame, feature_cols=None, n_splits: int = 5) -> Tuple[Pipeline, dict]:
    X, y, used = split_xy(df, feature_cols)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_auc = -np.inf; best_model = None
    metrics = []
    for i, (tr, te) in enumerate(tscv.split(X)):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        pipe.fit(X[tr], y[tr])
        y_prob = pipe.predict_proba(X[te])[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y[te], y_prob) if len(set(y[te])) > 1 else float("nan")
        acc = accuracy_score(y[te], y_pred)
        metrics.append({"fold": i+1, "AUC": None if np.isnan(auc) else float(auc), "ACC": float(acc)})
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc; best_model = pipe
    final = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    final.fit(X, y)
    return final, {"cv_metrics": metrics, "features_used": used}

def save_model(model: Pipeline, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

def load_model(path: Path) -> Pipeline:
    return joblib.load(path)

def predict_proba(model: Pipeline, feats_row: pd.DataFrame) -> float:
    return float(model.predict_proba(feats_row.values)[:,1][0])
