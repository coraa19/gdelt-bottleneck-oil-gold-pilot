Markdown# GDELT Supply Bottleneck Sentiment & Gold Quantile Risk (Pilot)

An end-to-end reproducible research pipeline evaluating whether a news-derived **Supply Bottleneck Sentiment Index** improves out-of-sample predictability of extreme gold price shocks across conditional quantiles.

---

## 1. Research Question & Hypothesis
During geopolitical crises, supply bottlenecks trigger abrupt energy disruptions. While central tendency models (OLS) often average out such tail events, safe-haven assets (gold) typically respond asymmetrically to supply shocks. 
* **Core Question**: Does adding high-frequency news-derived bottleneck sentiment reduce out-of-sample forecast errors in conditional quantiles ($\tau \in \{0.05, 0.50, 0.95\}$) of gold returns?

---

## 2. Pipeline Architecture
```text
GDELT GKG 2.0 (2022 H1)
       │
       ▼
 [Ingestion & Cleaning] ──> Drop nulls, deduplicate same-day titles (1,507 clean records)
       │
       ├─► [Measurement Validity] ──> 200 Hand-labeled Ground Truth titles (Macro-F1: 0.90)
       │
       ▼
 [FinBERT Sentiment]   ──> Extract negative sentiment probability (ProsusAI/finbert)
       │
       ▼
 [Index Aggregation]   ──> Daily Bottleneck Share * Mean Negative Sentiment (Weekend-pooled)
       │
       ▼
 [Market Integration]  ──> Align Brent Crude (BZ=F), Gold (GC=F), and VIX (^VIX) via yfinance
       │
       ▼
 [Quantile Modeling]   ──> QuantReg (tau=0.05, 0.50, 0.95) with Expanding Window OOS Evaluation
       │
       ▼
 [Hypothesis Testing]  ──> Out-of-sample Pinball Loss & Diebold-Mariano test
3. Measurement Validity & Hand-Labeling EvidenceTo avoid circular reasoning and ungrounded claims:A transparent heuristic rule (src/lexicon.py) was formulated to separate physical supply disruption from diplomatic/procedural events.Validation: 200 stratified titles were manually inspected and hand-labeled as Ground Truth (data/sample_200_for_labeling.csv).Performance:Bottleneck Precision: 0.94 | Recall: 1.00 | F1: 0.97Diplomatic Precision: 0.70 | Recall: 0.78 | F1: 0.74Macro-F1: 0.90 (Accuracy: 0.97)Identified Boundary Failure: Polysemous terms like university union strikes (e.g., ASUU strike) and talk shows were flagged and analyzed.4. Empirical Evaluation & Key FindingsModels were evaluated using an expanding window out-of-sample procedure (initial training window = 40 trading days; strictly 1-step-ahead forecast; no look-ahead bias).Out-of-Sample Pinball Loss & Diebold-Mariano (DM) TestQuantile (τ)Base Model LossAugmented Model LossDM StatisticOne-sided p-valueEmpirical Verdict0.05 (Downside Tail)0.0010200.001102-0.3310.630No improvement0.50 (Median)0.0031680.0030890.7040.241Null result0.95 (Upside Tail)0.0009560.0008255.320< 0.001***Statistically Significant Improvement5. What This Does NOT Show (Critical Limitations)No Evidence for Median Directionality: News sentiment provides negligible predictive edge over median day-to-day fluctuations ($p = 0.241$).GDELT Source Redundancy & Bias: Automated ingestion aggregates English-origin feeds which disproportionately over-represent Western institutional wire coverage.Futures Roll Adjustment: Front-month futures (BZ=F, GC=F) were sourced without unrolling contract spreads, introducing potential roll noise around expiry dates.Sample Span: The evaluation covers an acute conflict shock window in early 2022; generalization to structural regimes requires broader multi-year validation.n src/model_quantreg.py
