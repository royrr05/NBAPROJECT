# Executive Summary — NBA 2023–24 Analytics

## Purpose
Quantify what drives winning in the 2023–24 NBA season, profile team styles, and surface actionable insights for coaches, front offices, and betting analysts. Outputs include a win-probability model, style clusters, and a Streamlit dashboard for exploration.

## Data & Methods
- **Data**: Team-level game logs with advanced metrics (ORtg, DRtg, NRtg, Pace, eFG%, TS%, TOV%, ORB%, FT/FGA, 3PAr) and Win flag.
- **Model**: Logistic Regression on season games, stratified split (75/25), standardized features.
- **Clustering**: K-Means on the same feature set; PCA (2D) for projection; elbow scores to guide k.
- **Tools**: pandas, scikit-learn, Streamlit, matplotlib.

## Key Findings (current run)
- **Predictive power**: AUC ≈ 0.64 (moderate; useful as supporting signal).
- **Win drivers (coefficients)**: TS% and NRtg lead the list; ORtg also positive. Higher DRtg, Pace, FT/FGA skew negative in this fit; interpretation: efficiency and net rating outweigh raw pace.
- **Baseline confusion matrix**: Model is more confident on losses than wins (higher recall on class 0), underscoring the need to combine with context (injuries, rest, travel).
- **Style clusters (k=3)**: Three broad archetypes emerge. Example tendencies:
  - Cluster 0: Balanced/efficient offenses with moderate pace.
  - Cluster 1: Faster, looser profiles (higher DRtg, more variance).
  - Cluster 2: Lower-paced, defense-leaning/drag-down profiles.
- **Elbow diagnostic**: Inertia drops sharply through k≈4–5 before flattening, suggesting k=3–5 is a reasonable band; k=3 used for simplicity and storytelling.

## Business Insights
- **Coaching**: Emphasize shooting quality (TS%/eFG%) and net efficiency; trim turnovers. Against Cluster 1 (“fast/loose”), slow pace and force half-court possessions; against Cluster 0, prioritize defensive shot quality.
- **Front Office**: Target lineups that boost TS% without inflating TOV%. Cluster placement highlights roster identity; recruit to reinforce or counter predominant styles.
- **Betting/Analytics**: Use model probabilities as a baseline; look for divergence from market when efficiency metrics shift (injuries, rest) but pace/variance stays similar. Moderate AUC means it’s a guide, not a stand-alone edge.

## Limitations & Next Steps
- No injury/availability, rest, or home/away context baked in.
- Team-average inputs miss matchup-specific dynamics and rotations.
- Logistic regression is linear; non-linear models (XGBoost, calibrated RF) could add lift.
- Consider time-aware splits (train early-season, test later), and add opponent-adjusted metrics.
- Expand dashboard with opponent filtering, recent-form windows, and schedule context.
- Data provenance: ensure source links and refresh cadence are documented alongside cleaned CSVs.
- Ethics/usage: model should not be used as sole decision support for betting; avoid overconfidence and consider responsible use guidelines.
