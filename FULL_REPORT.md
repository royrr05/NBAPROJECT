# NBA 2023–24 Analytics Report
Course: Sports Analytics  
Instructor: Professor Carone  
Student: Rayya Roy  
Date: December 2025  
Font target for Word export: Times New Roman, 12 pt, normal margins  

---

## 1) Executive Summary
This project quantifies what drives winning in the 2023–24 NBA season, predicts game outcomes, and profiles team styles. Using advanced efficiency metrics, a logistic regression win model, and K-Means clustering, we find:
- **Predictive power**: Win model AUC ≈ 0.64 (moderate; better recall on losses than wins).
- **Top drivers**: TS%, NRtg, and ORtg most strongly raise win probability; DRtg and higher Pace are negative in this fit.
- **Styles**: Three archetypes emerge (Balanced/Efficient, Fast/Loose, Slow/Defensive). These inform coaching tactics, roster identity, and baseline betting signals.
- **Three key insights**:
  1) Shooting quality dominates: TS% and eFG% correlate strongly with wins and carry the highest positive coefficients.
  2) Net efficiency outweighs raw pace: NRtg/ORtg lift win odds; higher Pace alone does not.
  3) Style clusters are distinct: balanced teams differ meaningfully from fast/loose and slow/defensive groups, enabling targeted game plans.
- **Two recommendations**:
  - Coaching: prioritize shot quality (TS%/eFG%) and ball security; tailor pace and defensive pressure to opponent cluster (slow down Fast/Loose; guard shot quality vs Balanced).
  - Front Office: recruit to raise TS% without inflating TOV%; use cluster identity to guide roster construction and matchup preparation.

---

## 2) Problem Framing & Data Provenance
**Questions**  
- What stats drive wins?  
- Can we predict outcomes from team-level metrics?  
- How do team styles cluster?  

**Stakeholders**  
- Coaches (game plans), Front Office (identity/roster), Betting/Analytics (baseline probabilities).  

**Data sources**  
- Game-level advanced stats from Basketball-Reference (team/opponent tables).  
- Files: `nba_2023_24_full_cleaned_dataset_RE_CLEANED_WITH_WIN.csv`, `model_ready_WITH_TEAMS_FINAL.csv`.  
- Core fields: Team, Opponent, Date, ORtg, DRtg, NRtg, Pace, eFG%, TS%, TOV%, ORB%, FT/FGA, 3PAr, Win.  

**Cleaning/prep**  
- Parsed Date (m/d/yy), ensured required columns, dropped rows with missing values in modeling features.  
- No missing in core fields after cleaning.  
- Engineered: NRtg (provided), Win flag, standardized features for ML.  

**Limitations of source**  
- No injury/rest/home/away context; no lineup-level data.  

---

## 3) Data Dictionary & Preparation
Core definitions (see `data_dictionary.md` for table):
- ORtg/DRtg/NRtg: Offensive/Defensive/Net Rating (per 100 poss)
- Pace: possessions per 48
- eFG%, TS%: shooting efficiency
- TOV%: turnovers per 100 poss
- ORB%: offensive rebound%
- FT/FGA: free throws per FGA
- 3PAr: 3PA per FGA
- Win: game outcome flag (1/0)

Preparation steps:
- StandardScaler on model/cluster features.
- Stratified train/test split (75/25) for logistic regression.

---

## 4) Exploratory Analysis
**Descriptives**  
- League means/medians for ORtg, DRtg, NRtg, Pace, TS%, eFG%, TOV%, ORB%, FT/FGA, 3PAr (compute via pipeline if needed).

**Correlations**  
- TS%/eFG% correlate positively with Win; DRtg correlates negatively.  
- Figure: `figures/correlation_heatmap.png`

**Distributions**  
- Four Factors show variance in shooting and turnover/rebounding profiles.  
- Figure: `figures/four_factors_distributions.png`

---

## 5) Methods
**Logistic Regression (Win model)**  
- Features: ORtg, DRtg, NRtg, Pace, eFG%, TS%, TOV%, ORB%, FT/FGA, 3PAr.  
- Target: Win (0/1).  
- Split: 75/25 stratified; StandardScaler; LogisticRegression(max_iter=1000, random_state=42).  
- Diagnostics: ROC/AUC, confusion matrix, class metrics, coefficients + win-prob deltas.

**Clustering (Team styles)**  
- Algorithm: K-Means with StandardScaler.  
- k: dynamic; elbow over k=2–8; default k=3 for storytelling.  
- PCA(2) for visualization.  
- Centroids transformed back to original scale for interpretation.

---

## 6) Results
**Win model**  
- AUC: ≈ 0.64; confusion matrix shows stronger recall on losses.  
- Top positive coefficients: TS%, NRtg, ORtg; small positive 3PAr/TOV%.  
- Negative: DRtg, Pace, eFG% in this fit.  
- Approx deltas: +1 TS% ≈ +7.1 pp win prob near 50% baseline; +1 NRtg ≈ +6.8 pp; +1 ORtg ≈ +5.1 pp.  
- Figures: `figures/logistic_coefficients.png`, `figures/confusion_matrix.png`, `figures/roc_curve.png`.

**Clusters**  
- Default k=3 archetypes:  
  - Cluster 0: Balanced/Efficient, moderate pace.  
  - Cluster 1: Fast/Loose, higher DRtg/variance.  
  - Cluster 2: Slow/Defensive, drag-down pace.  
- Figures: `figures/elbow_plot.png`, PCA scatter (screenshot from dashboard).
- Centroids per k available in dashboard “Centroids by k” expander.

---

## 7) Business Insights
**Coaching**  
- Emphasize TS%/eFG% and ball security; protect possessions.  
- Versus Fast/Loose (Cluster 1): slow pace, force half-court, limit transition.  
- Versus Balanced (Cluster 0): prioritize shot-quality defense.

**Front Office**  
- Lift TS% without spiking TOV%; use cluster identity to steer roster/identity fits.  
- Target lineups that improve efficiency and net rating profiles.

**Betting/Analytics**  
- Use model probabilities as baseline; watch efficiency shifts (injuries/rest) more than pace.  
- Moderate AUC → supporting signal, not stand-alone edge.

---

## 8) Limitations & Ethics
- Missing context: injuries, rest, travel, home/away, rotations not modeled.  
- Linear model; interactions/non-linear effects not captured.  
- Responsible use: do not use as sole basis for betting; acknowledge uncertainty.
- Data refresh: static season snapshot; probabilities should be recalibrated if conditions change (trades, injuries).

---

## 9) Future Work
- Add context features (injury/rest/home/away), opponent-adjusted metrics, recent-form windows.  
- Test calibrated non-linear models (XGBoost/RF) and time-aware splits.  
- Dashboard: matchup-specific inputs, schedule context, market vs model view.

---

## 10) Appendix
- Repro steps:  
  - Pipeline: `python3 nba_analytics_pipeline.py` (saves figures to `figures/`).  
  - Dashboard: `streamlit run nba_dashboard_streamlit.py`.  
- Tables: data dictionary, classification report, centroids, league means/medians.  
- Figures: correlation_heatmap.png, four_factors_distributions.png, logistic_coefficients.png, confusion_matrix.png, roc_curve.png, elbow_plot.png, PCA scatter (dashboard).  
- Additional notes:  
  - Coefficient delta table (win-prob pp shifts) and class metrics are available in the dashboard Diagnostics page.  
  - Centroids for multiple k values (2–8) available via dashboard “Centroids by k” expander.  

---

## Notes for Word Export
- Copy this markdown into Word, apply Times New Roman 12 pt, standard margins.  
- Insert the referenced figures from the `figures/` folder with captions.  
- Add tables for data dictionary, coefficient deltas, class metrics, centroids, and league descriptive stats to comfortably reach 10–12 pages.
