# Final Report & Slide Deck (Filled)

Use this content to build the 8–12 page PDF and 5-minute slides. Tailor wording/formatting to your style.

## Report (8–12 pages)
1) **Executive Summary**  
   - Purpose: quantify win drivers and team styles for 2023–24; support coaches, front office, bettors.  
   - Headline results: Logistic regression AUC ≈ 0.64 (moderate). TS%/NRtg/ORtg strongest positive drivers; DRtg/Pace negative. K-Means (k=3) reveals three style archetypes.  
   - Business value: scouting support, roster/identity guidance, baseline win probabilities.

2) **Problem Framing & Data Provenance**  
   - Questions: What stats drive wins? Can we predict outcomes? How do styles cluster?  
   - Stakeholders: coaches, front office, betting/analytics.  
   - Data: cleaned 2023–24 team game logs with advanced metrics. Files: `nba_2023_24_full_cleaned_dataset_RE_CLEANED_WITH_WIN.csv`, `model_ready_WITH_TEAMS_FINAL.csv`.  
   - Provenance: sourced from public NBA stat pulls (specify your exact source/API in final draft). Date parsed m/d/yy; no missing values in core features/Win after cleaning.

3) **Data Dictionary & Preparation**  
   - Reference `data_dictionary.md` for definitions/units.  
   - Preparation: parsed Date, ensured required columns, dropped rows with missing values for modeling/clustering, standardized features for ML.

4) **Exploratory Analysis**  
   - Descriptives: league means/medians for ORtg, DRtg, Pace, Four Factors.  
   - Correlations: see `figures/correlation_heatmap.png` (TS%/eFG% correlate positively with Win; DRtg negatively).  
   - Distributions: Four Factors histograms (`figures/four_factors_distributions.png`) show variance in shooting/rebounding/turnovers.  
   - Quick feature–Win correlations table included in dashboard EA page.

5) **Methods**  
   - Logistic Regression: features = ORtg, DRtg, NRtg, Pace, eFG%, TS%, TOV%, ORB%, FT/FGA, 3PAr; target = Win. Split 75/25 stratified; StandardScaler; LogisticRegression(max_iter=1000, random_state=42).  
   - Clustering: K-Means on same features; StandardScaler; k=3 for storytelling; elbow (k=2–8) guides parsimonious range (bend ~3–5); PCA(2) for visualization.  
   - Validation: AUC/ROC, confusion matrix, classification report.

6) **Results**  
   - Win model: AUC 0.639; confusion matrix shows higher recall on losses (class 0). Coefficient ranking: TS%, NRtg, ORtg positive; DRtg, Pace, FT/FGA negative in this fit.  
   - Interpretation: +1 TS% ≈ +7.1 pp win prob around 50% baseline; +1 NRtg ≈ +6.8 pp.  
   - Clusters:  
     * Cluster 0: Balanced/efficient offenses, moderate pace.  
     * Cluster 1: Faster/looser, higher DRtg, more variance.  
     * Cluster 2: Lower pace, defense-leaning/drag-down.  
   - PCA scatter shows separation; centroids in pipeline output; elbow plot in figures/dashboard.

7) **Business Insights**  
   - Coaching: Emphasize shooting quality (TS%/eFG%) and protect possessions; vs Cluster 1 slow pace and force half-court; vs Cluster 0 prioritize defensive shot quality.  
   - Front Office: Recruit to lift TS% without TOV% spike; cluster placement highlights identity—decide to reinforce or diversify.  
   - Betting/Analytics: Use model probabilities as baseline; look for market divergence when efficiency shifts (injuries/rest) but pace/variance similar; AUC implies supporting signal, not standalone edge.

8) **Limitations & Ethics**  
   - Missing context: injuries, rest, travel, home/away, rotations not modeled.  
   - Linear model; non-linear interactions not captured.  
   - Data provenance: ensure proper citation; refresh cadence not automated.  
   - Responsible use: probabilities should not drive betting decisions alone; note uncertainty.

9) **Future Work**  
   - Add opponent-adjusted metrics, recent-form windows, home/away, injury/rest flags.  
   - Test non-linear models (XGBoost, RF) with calibration.  
   - Time-aware splits (train early, test late); cross-validation.  
   - Dashboard enhancements: matchup-specific inputs, schedule context, market vs model view.

10) **Appendix**  
   - Repro commands: `python3 nba_analytics_pipeline.py` (figures saved to `figures/`), `streamlit run nba_dashboard_streamlit.py`.  
   - Figures: correlation, Four Factors distributions, coefficients, ROC, confusion matrix, elbow, PCA scatter (from dashboard).  
   - Full tables: centroids, classification report, feature deltas.

## Slide Deck (5 minutes)
1) **Title & Objective** — “NBA 2023–24 Analytics: Win Drivers & Team Styles”; one-line purpose.  
2) **Data & Method** — sources, core features, logistic regression setup, K-Means + elbow.  
3) **Key Findings** — AUC ~0.64; TS%/NRtg/ORtg top drivers; cluster archetypes (0 balanced, 1 fast/loose, 2 slow/defensive).  
4) **Business Recommendations** — coaching levers, roster identity, model as baseline for betting/analytics.  
5) **Limitations & Next Steps** — context gaps (injury/rest), linearity, future non-linear models and context features.  
6) **Call to Action** — link/QR to Streamlit app; invite questions/demo.
