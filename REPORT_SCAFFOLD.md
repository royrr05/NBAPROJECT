# Report & Slide Deck Scaffold

Use this template to draft the 8–12 page report and 5-minute deck. Replace bracketed prompts with your content.

## Report (8–12 pages)
1) **Executive Summary**  
   - Purpose: [what you set out to answer]  
   - Headline results: [top metrics/findings]  
   - Business value: [who benefits and how]

2) **Problem Framing & Data Provenance**  
   - Questions: [list the key questions]  
   - Stakeholders: [coaches/front office/analysts/etc.]  
   - Sources: [list datasets/APIs with links]; collection dates; refresh cadence  
   - Cleaning steps: [brief description of merge, parsing, missing handling]

3) **Data Dictionary & Preparation**  
   - Schema: [columns/features used]  
   - Definitions/units: see `data_dictionary.md`; add any extra engineered fields  
   - Missingness: [how handled]; transformations: [scaling, encoding]

4) **Exploratory Analysis**  
   - Descriptive stats: [means/medians for key metrics]  
   - Correlations: [summary + pointer to heatmap]  
   - Distributions: [which factors explored; note any skew/outliers]  
   - Key observations: [bullet 2–4 takeaways]

5) **Methods**  
   - Model 1: Logistic Regression — features, target, split, scaling, hyperparams  
   - Model 2: Clustering — algorithm, features, k selection (elbow), scaling, PCA viz  
   - Validation: [metrics and rationale]

6) **Results**  
   - Win model: AUC/ROC, confusion matrix, classification report; top coefficients with interpretations  
   - Clusters: centroids (original scale), PCA scatter, cluster descriptions  
   - Supporting visuals: [list included figures]

7) **Business Insights**  
   - Coaching: [recommendations tied to metrics/clusters]  
   - Front Office: [roster/identity implications]  
   - Betting/Analytics: [how to use probabilities; caveats]

8) **Limitations & Ethics**  
   - Data gaps: [injuries, rest, home/away, etc.]  
   - Method limits: [linearity, feature scope]  
   - Responsible use: [betting caution, uncertainty]

9) **Future Work**  
   - [Context features], [non-linear models], [time-aware splits], [dashboard enhancements]

10) **Appendix**  
   - Repro steps: commands to run pipeline/dashboard  
   - Tables/figures: [list any extras]  
   - Source links and licenses

## Slide Deck (5 minutes)
1) Title & Objective — [project title; one-line purpose]  
2) Data & Method — [sources, features, model + clustering in 1–2 bullets each]  
3) Key Findings — [top 3 win drivers; 1–2 cluster archetypes; AUC headline]  
4) Business Recommendations — [3 bullets: coaching/front office/analytics]  
5) Limitations & Next Steps — [context gaps; future improvements]  
6) Call to Action — [link/QR to dashboard; invite discussion/demo]
