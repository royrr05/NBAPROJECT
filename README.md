# NBA 2023–24 Analytics Project

Data-driven look at what drives winning in the 2023–24 NBA season. The repo includes a reproducible backend pipeline and a Streamlit dashboard for win prediction, team style profiling, and reporting.

## Repo Contents
- `nba_analytics_pipeline.py` — backend pipeline: data load/validation, logistic regression win model, K-Means team style clusters, elbow diagnostics.
- `nba_dashboard_streamlit.py` — interactive dashboard with Overview, Matchup Explorer, Team Style Clusters, and Model Diagnostics pages.
- `nba_2023_24_full_cleaned_dataset_RE_CLEANED_WITH_WIN.csv` — game-level data with advanced metrics and win flag.
- `model_ready_WITH_TEAMS_FINAL.csv` — modeling-ready game-level data (Team + Win + features).
- `data_dictionary.md` — definitions and units for core columns.
- `EXECUTIVE_SUMMARY.md` — concise narrative and insights.
- `figures/` — generated plots from the pipeline run.

## Setup
1) Python 3.10+ recommended (tested with 3.13).  
2) Install deps:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

## Run the Pipeline (text outputs)
```bash
python3 nba_analytics_pipeline.py
```
Outputs:
- Dataset validation summary
- Logistic regression AUC, confusion matrix, classification report, coefficients + basketball interpretations
- K-Means centroids and team cluster assignment (k=3)
- Elbow scores table (k vs inertia) to guide cluster selection
- EDA figures saved under `figures/` (correlation heatmap, Four Factors distributions, coefficients, ROC, confusion matrix, elbow)

## Run the Dashboard
```bash
streamlit run nba_dashboard_streamlit.py
```
Pages:
- **Overview** — single-team KPIs + Four Factors snapshot
- **Matchup Explorer** — compare two teams and model win probability (team-average inputs)
- **Team Style Clusters** — primary cluster per team, PCA scatter, elbow plot
- **Model Diagnostics** — ROC/AUC, confusion matrix, coefficient chart
- **Exploratory Analysis** — correlation heatmap, Four Factors distributions, feature–Win correlations

## Rubric Alignment (quick mapping)
- Problem framing & data provenance: see `README.md` + `EXECUTIVE_SUMMARY.md` (add source details for your pulls), `data_dictionary.md`.
- EDA quality & reproducibility: pipeline generates and saves plots to `figures/`; reproducible via `python3 nba_analytics_pipeline.py`.
- Methods & validation: logistic regression with stratified split; K-Means with elbow; diagnostics in pipeline and dashboard.
- Clarity of visuals & tables: Matplotlib/Seaborn figures + Streamlit pages; saved assets in `figures/`.
- Business insights & recommendations: `EXECUTIVE_SUMMARY.md` includes coaching/front-office/betting notes.
- Limitations & ethics: see `EXECUTIVE_SUMMARY.md` (expand with your data provenance/usage constraints as needed).

## Deliverables Coverage
- Cleaned datasets (provided CSVs)
- Predictive model + diagnostics (logistic regression)
- Team style clustering (K-Means, PCA, elbow)
- Interactive dashboard (Streamlit)
- Written context: see `EXECUTIVE_SUMMARY.md` for a concise narrative and business takeaways.
