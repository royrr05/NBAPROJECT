from __future__ import annotations

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


st.set_page_config(
    page_title="NBA 2023-24 Analytics Dashboard",
    layout="wide",
)


FEATURE_COLS = [
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


@st.cache_data
def load_data():
    """Load full and model-ready datasets, parsing Date when present."""
    base_path = Path(__file__).resolve().parent
    full_candidates = ["nba_2023_24_full_cleaned_dataset_RE_CLEANED_WITH_WIN.csv"]
    model_candidates = [
        "nba_2023_24_model_ready_WITH_TEAMS_FINAL.csv",
        "model_ready_WITH_TEAMS_FINAL.csv",  # fallback to shorter name if present
    ]

    def pick_path(candidates):
        for name in candidates:
            path = base_path / name
            if path.exists():
                return path
        return None

    full_path = pick_path(full_candidates)
    model_path = pick_path(model_candidates)

    missing = []
    if full_path is None:
        missing.extend(full_candidates)
    if model_path is None:
        missing.extend(model_candidates)
    if missing:
        raise FileNotFoundError(
            "Expected data files next to this script. Missing candidates: "
            + ", ".join(missing)
        )

    full_df = pd.read_csv(full_path)
    model_df = pd.read_csv(model_path)

    if "Date" in model_df.columns:
        parsed = pd.to_datetime(model_df["Date"], format="%m/%d/%y", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(model_df["Date"], errors="coerce")
        model_df["Date"] = parsed

    return full_df, model_df


@st.cache_resource
def train_logistic_model(model_df: pd.DataFrame):
    """Train logistic regression win model and return artifacts + diagnostics."""
    df = model_df.dropna(subset=FEATURE_COLS + ["Win"]).copy()
    X = df[FEATURE_COLS].values
    y = df["Win"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "auc": auc_val,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
    }


@st.cache_resource
def compute_team_averages(model_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate season averages per team for comparisons and metrics."""
    summary_cols = FEATURE_COLS + ["Win"]
    team_avgs = model_df.groupby("Team")[summary_cols].mean().reset_index()
    return team_avgs


@st.cache_resource
def compute_clusters(full_df: pd.DataFrame, k: int = 3):
    """Compute K-Means clusters and PCA projections for team styles."""
    df = full_df.dropna(subset=FEATURE_COLS + ["Team"]).copy()
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    return df, kmeans, FEATURE_COLS


def plot_confusion_matrix(cm: np.ndarray):
    """Return a matplotlib figure for a 2x2 confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Game Outcome Prediction)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_val: float):
    """Return a matplotlib figure for ROC curve."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})", color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Win Probability Model)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_coefficients(model: LogisticRegression, feature_cols):
    """Return a matplotlib figure of model coefficients."""
    coefs = model.coef_[0]
    order = np.argsort(coefs)
    sorted_feats = np.array(feature_cols)[order]
    sorted_coefs = coefs[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(sorted_feats, sorted_coefs, color="steelblue")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient value")
    fig.tight_layout()
    return fig


# Load data and models once at startup.
full_df, model_df = load_data()
team_avgs = compute_team_averages(model_df)
loginfo = train_logistic_model(model_df)
cluster_df, kmeans, cluster_features = compute_clusters(full_df, k=3)


st.sidebar.title("NBA 2023–24 Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Matchup Explorer", "Team Style Clusters", "Model Diagnostics"],
)


if page == "Overview":
    st.title("NBA 2023–24 Team Overview")

    teams = team_avgs["Team"].unique()
    team_choice = st.selectbox("Select a team", options=teams)
    team_row = team_avgs[team_avgs["Team"] == team_choice].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win %", f"{team_row['Win']*100:.1f}%")
    col2.metric("ORtg", f"{team_row['ORtg']:.2f}")
    col3.metric("DRtg", f"{team_row['DRtg']:.2f}")
    col4.metric("NRtg", f"{team_row['NRtg']:.2f}")

    st.subheader("Four Factors Snapshot")
    factors = [
        "eFG%",
        "TS%",
        "TOV%",
        "ORB%",
        "FT/FGA",
        "3PAr",
    ]
    factor_df = (
        team_row[factors]
        .to_frame(name=team_choice)
        .rename_axis("Metric")
        .round(3)
    )
    st.dataframe(factor_df)

    st.markdown(
        "- Higher ORtg, TS%, eFG% → more efficient scoring  \n"
        "- Lower DRtg, TOV% → better defense and ball security  \n"
        "- 3PAr shows how three-heavy the offense is"
    )

elif page == "Matchup Explorer":
    st.title("Matchup Explorer (Win Probability)")

    teams = team_avgs["Team"].unique()
    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.selectbox("Team A", options=teams, index=0)
    with col_b:
        team_b = st.selectbox("Team B", options=teams, index=1 if len(teams) > 1 else 0)

    row_a = team_avgs[team_avgs["Team"] == team_a].iloc[0]
    row_b = team_avgs[team_avgs["Team"] == team_b].iloc[0]

    metrics_list = FEATURE_COLS
    compare_df = pd.DataFrame(
        {
            team_a: row_a[metrics_list].values,
            team_b: row_b[metrics_list].values,
        },
        index=metrics_list,
    ).round(2)
    st.subheader("Team Comparison (Season Averages)")
    st.dataframe(compare_df)

    X_match = pd.DataFrame([row_a[loginfo["feature_cols"]]], columns=loginfo["feature_cols"])
    X_scaled = loginfo["scaler"].transform(X_match)
    win_prob = loginfo["model"].predict_proba(X_scaled)[0, 1]

    st.metric(
        f"Predicted chance {team_a} beats {team_b}",
        f"{win_prob*100:.1f}%",
    )

    st.markdown(
        "_Assumptions: neutral court, season-average performance, no injuries or rest adjustments._"
    )

elif page == "Team Style Clusters":
    st.title("Team Style Clusters (K-Means, k = 3)")
    st.markdown(
        "Clusters group games by stylistic profile (tempo, shooting, turnovers, rebounding). "
        "They do not reflect standings—just similar tendencies."
    )

    team_clusters = (
        cluster_df.groupby("Team")["Cluster"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    st.subheader("Primary Cluster per Team")
    st.dataframe(team_clusters)

    st.subheader("Team Games in PCA Space (Colored by Cluster)")
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster in sorted(cluster_df["Cluster"].unique()):
        subset = cluster_df[cluster_df["Cluster"] == cluster]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.6, s=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Team Games in PCA Space (Colored by Cluster)")
    ax.legend(title="Cluster")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown(
        "Clusters represent similar offensive/defensive styles. Teams heavily represented in a cluster "
        "share tendencies such as pace, shooting profile, or turnover/rebounding patterns."
    )

elif page == "Model Diagnostics":
    st.title("Win Probability Model Diagnostics")

    st.metric("ROC AUC", f"{loginfo['auc']:.3f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.pyplot(plot_confusion_matrix(loginfo["cm"]))
    with col_right:
        st.pyplot(plot_roc_curve(loginfo["fpr"], loginfo["tpr"], loginfo["auc"]))

    st.subheader("Coefficient Importance")
    st.pyplot(plot_coefficients(loginfo["model"], loginfo["feature_cols"]))

    st.markdown(
        f"- AUC ≈ {loginfo['auc']:.2f} → moderate predictive power using team-level stats.  \n"
        "- Use this as a supporting tool alongside scouting, injuries, rest, and situational factors."
    )
