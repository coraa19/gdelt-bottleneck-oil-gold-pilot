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
