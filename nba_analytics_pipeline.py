from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FULL_DATA_FILE = "nba_2023_24_full_cleaned_dataset_RE_CLEANED_WITH_WIN.csv"
MODEL_READY_FILE = "model_ready_WITH_TEAMS_FINAL.csv"

FEATURE_COLS: List[str] = [
    "ORtg",
    "DRtg",
    "NRtg",
    "Pace",
    "eFG%",
    "TS%",
    "TOV%",
    "ORB%",
    "FT/FGA",
    "3PAr",
]


def load_data(
    full_data_path: str = FULL_DATA_FILE, model_data_path: str = MODEL_READY_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load full dataset and model-ready dataset, parsing Date if present."""
    full_df = pd.read_csv(full_data_path)
    model_df = pd.read_csv(model_data_path)

    # Convert Date column when available so downstream code can rely on datetime.
    if "Date" in model_df.columns:
        # Parse expected m/d/yy strings; if none parse, fall back to flexible parsing.
        parsed = pd.to_datetime(model_df["Date"], format="%m/%d/%y", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(model_df["Date"], errors="coerce")
        model_df["Date"] = parsed

    return full_df, model_df


def validate_model_dataset(model_df: pd.DataFrame) -> None:
    """Print quick diagnostics and verify required columns exist."""
    required_cols = FEATURE_COLS + ["Win", "Team"]

    print("\nModel dataset preview:")
    print(model_df.head())
    print(f"\nShape: {model_df.shape}")
    print("\nColumns:", list(model_df.columns))

    missing_required = [col for col in required_cols if col not in model_df.columns]
    if missing_required:
        print("\nWARNING - missing required columns:", missing_required)
    else:
        print("\nAll required columns present.")

    print("\nMissing value counts for required columns:")
    print(model_df[required_cols].isna().sum())


def _format_classification_report(report_dict: Dict) -> str:
    """Nicely format classification_report output for printing."""
    lines = []
    for label, metrics_dict in report_dict.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics_dict.get("precision", 0.0)
        recall = metrics_dict.get("recall", 0.0)
        f1 = metrics_dict.get("f1-score", 0.0)
        support = metrics_dict.get("support", 0)
        lines.append(
            f"Class {label}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, support={support}"
        )
    overall = report_dict.get("accuracy", 0.0)
    lines.append(f"Accuracy: {overall:.3f}")
    return "\n".join(lines)


def build_logistic_regression_model(
    model_df: pd.DataFrame, feature_cols: Iterable[str] = FEATURE_COLS
):
    """Train and evaluate logistic regression to predict Win."""
    feature_cols = list(feature_cols)
    df = model_df.dropna(subset=feature_cols + ["Win"])

    X = df[feature_cols].values
    y = df["Win"].astype(int).values

    # Split and scale features.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    confusion = metrics.confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    auc = metrics.roc_auc_score(y_test, y_prob)
    class_report = metrics.classification_report(y_test, y_pred, output_dict=True)

    coef_df = (
        pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_[0]})
        .sort_values(by="coefficient", ascending=False)
        .reset_index(drop=True)
    )

    # Prints for human-readable report.
    print("\n=== Logistic Regression Results ===")
    print(f"AUC: {auc:.3f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion)
    print("\nClassification Report:")
    print(_format_classification_report(class_report))
    print("\nCoefficients (higher => more win likelihood):")
    print(coef_df)

    print("\nFeature interpretations (approx around 50% baseline):")
    for _, row in coef_df.iterrows():
        feature = row["feature"]
        beta = row["coefficient"]
        delta_pct = 25 * beta  # 0.25 * beta * 100
        sign = "+" if delta_pct >= 0 else ""
        print(
            f"{feature}: +1 ≈ change win prob by {sign}{delta_pct:.2f} percentage points around a 50% baseline."
        )

    metrics_dict = {
        "confusion_matrix": confusion,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "classification_report": class_report,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    return model, scaler, metrics_dict, coef_df


def build_style_clusters(
    full_df: pd.DataFrame,
    feature_cols: Iterable[str] = FEATURE_COLS,
    k: int = 3,
):
    """Cluster team styles and return cluster assignments plus centroids."""
    feature_cols = list(feature_cols)
    df = full_df.dropna(subset=feature_cols + ["Team"]).copy()

    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # Reduce dimensionality for potential visualization downstream.
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    # Convert centroids back to original feature scale for interpretation.
    centroids_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_original_scale, columns=feature_cols)
    centroids_df["Cluster"] = centroids_df.index

    # Primary cluster per team (mode).
    team_clusters = (
        df.groupby("Team")["Cluster"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={"Cluster": "PrimaryCluster"})
    )

    print("\n=== K-Means Style Clusters ===")
    print(f"Number of clusters: {k}")
    print("\nCluster centroids (original scale):")
    print(centroids_df)
    print("\nPrimary cluster assignment per team:")
    print(team_clusters)

    return df, kmeans, pca, scaler, centroids_df, team_clusters


def main() -> None:
    # Load data
    full_df, model_df = load_data()

    # Validate modeling dataset
    validate_model_dataset(model_df)

    # Build win probability model
    build_logistic_regression_model(model_df, FEATURE_COLS)

    # Cluster team styles
    build_style_clusters(full_df, FEATURE_COLS, k=3)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
