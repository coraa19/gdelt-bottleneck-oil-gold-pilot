# GDELT Supply Bottleneck Sentiment & Gold Quantile Risk (Pilot)

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

```
### Data Cleaning & Attrition Funnel
| Stage | Raw Record Count | Retained Records | Filtering / Cleaning Rationale |
| :--- | :---: | :---: | :--- |
| **1. Raw GDELT GKG Ingestion** | 5,420 | 5,420 | Scraped hourly GDELT 2.0 translingual feeds across 2022 H1. |
| **2. Null & Integrity Filter** | 5,420 | 4,110 | Dropped records with missing titles, invalid URLs, or broken tone strings. |
| **3. Syndicate De-duplication** | 4,110 | 1,507 | Collapsed multi-outlet wire syndication into unique same-day narrative events. |
| **4. Feature Alignment** | 1,507 | 68 trading days | Aggregated to trading-day frequency; mapped weekend events via Friday/Monday roll. |
Due to bandwidth and compute bounds in this pilot stage, we sample two 15-minute GKG snapshots daily (06:00 and 18:00 UTC, covering ~2% of daily GKG volume) as an exploratory proxy rather than the full 96-snapshot streaming corpus.
---

## 3. Measurement Validity & Hand-Labeling Evidence

To avoid circular reasoning and ungrounded claims:

* A transparent heuristic rule (`src/lexicon.py`) was formulated to separate physical supply disruption from diplomatic/procedural events.
* **Validation**: 200 stratified titles were manually inspected and hand-labeled as Ground Truth (`data/sample_200_for_labeling.csv`).
* **Performance**:
* **Bottleneck Precision**: 0.94 | **Recall**: 1.00 | **F1**: 0.97
* **Diplomatic Precision**: 0.70 | **Recall**: 0.78 | **F1**: 0.74
* **Macro-F1**: **0.90** (Accuracy: 0.97)
* *Identified Boundary Failure*: Polysemous terms like university union strikes (e.g., ASUU strike) and talk shows were flagged and analyzed.



---

## 4. Empirical Evaluation & Key Findings

Models were evaluated using an **expanding window** out-of-sample procedure (initial training window = 40 trading days; strictly 1-step-ahead forecast; no look-ahead bias).

### Out-of-Sample Pinball Loss & Diebold-Mariano (DM) Test

| Quantile ($\tau$) | Base Model Loss | Augmented Model Loss | DM Statistic | One-sided $p$-value | Empirical Verdict |
| --- | --- | --- | --- | --- | --- |
| **0.05** (Downside Tail) | 0.001020 | 0.001102 | -0.331 | 0.630 | No improvement |
| **0.50** (Median) | 0.003168 | 0.003089 | 0.704 | 0.241 | Null result |
| **0.95** (Upside Tail) | **0.000956** | **0.000825** | **5.320** | **< 0.001***** | **Statistically Significant Improvement** |

---

## 5. What This Does NOT Show (Critical Limitations)

1. **No Evidence for Median Directionality**: News sentiment provides negligible predictive edge over median day-to-day fluctuations ($p = 0.241$).
2. **GDELT Source Redundancy & Bias**: Automated ingestion aggregates English-origin feeds which disproportionately over-represent Western institutional wire coverage.
3. **Futures Roll Adjustment**: Front-month futures (`BZ=F`, `GC=F`) were sourced without unrolling contract spreads, introducing potential roll noise around expiry dates.
4. **Sample Span**: The evaluation covers an acute conflict shock window in early 2022; generalization to structural regimes requires broader multi-year validation.

---

## 6. Reproducibility

```powershell
# 1. Environment setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Run measurement validation
python src/evaluate_lexicon.py

# 3. FinBERT scoring & feature aggregation
python src/score_sentiment.py
python src/build_features.py

# 4. Out-of-sample quantile regression evaluation
python src/model_quantreg.py

```

---

## 7. Future Work

* Implementing **Markov-Switching Quantile Regression (MS-QR)** to account for endogenous regime shifts between crisis and calm periods.
* Fine-tuning a domain-adapted sequence classifier on active maritime/chokepoint datasets.
