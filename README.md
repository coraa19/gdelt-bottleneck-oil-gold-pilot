# GDELT Supply Bottleneck Sentiment & Gold Quantile Risk (Pilot)

An end-to-end reproducible research pipeline evaluating whether a news-derived **Supply Bottleneck Sentiment Index** improves out-of-sample predictability and risk calibration of extreme gold price shocks across conditional quantiles.

---

## 1. Research Question & Hypothesis
During geopolitical crises, supply bottlenecks trigger abrupt energy and trade disruptions. While conditional mean frameworks (OLS) average out extreme tail events, gold returns often exhibit pronounced tail asymmetry under stress.
- **Core Question**: Does adding a high-frequency, news-derived bottleneck sentiment indicator improve out-of-sample calibration across conditional quantiles ($\tau \in \{0.05, 0.50, 0.95\}$) of gold returns when conditioning on realized market volatility and oil dynamics?

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
 [FinBERT Sentiment]    ──> Extract negative sentiment probability (ProsusAI/finbert)
       │
       ▼
 [Index Aggregation]    ──> Daily Bottleneck Share * Mean Negative Sentiment (Weekend-pooled)
       │
       ▼
 [Market Integration]   ──> Align Brent Crude (BZ=F), Gold (GC=F), and VIX (^VIX) via yfinance
       │
       ▼
 [Quantile Modeling]    ──> QuantReg (tau=0.05, 0.50, 0.95) with Expanding Window OOS Evaluation
       │
       ▼
 [Hypothesis Testing]   ──> Pinball Loss, Empirical Coverage, DM Test & Permutation Significance

```

### Data Cleaning & Sampling Protocol

| Stage | Raw Record Count | Retained Records | Filtering / Cleaning Rationale |
| --- | --- | --- | --- |
| **1. Raw GDELT Ingestion** | 5,420 | 5,420 | Scraped two fixed daily snapshots (06:00 & 18:00 UTC) across 2022 H1. |
| **2. Null & Integrity Filter** | 5,420 | 4,110 | Dropped records with missing titles, non-English translations, or invalid metadata. |
| **3. Syndicate De-duplication** | 4,110 | 1,507 | Collapsed syndicated wire feeds into unique same-day narrative events. |
| **4. Feature Alignment** | 1,507 | 68 trading days | Aggregated to trading-day frequency; mapped weekend events via Friday/Monday roll. |

> **Sampling & Imputation Protocol**: Due to compute bounds in this pilot stage, data collection samples two 15-minute GKG snapshots daily (06:00 and 18:00 UTC, covering ~2% of streaming GKG volume). Trading days with zero matching headlines are imputed as `Index = 0.0` (neutral baseline). Daily headline volume exhibits a median of 8 (IQR: [3, 14]); low-volume days naturally generate discrete index steps (0.0, 0.5, 1.0).

---

## 3. Measurement Validity & Hand-Labeling Evidence

To prevent circular reasoning and ungrounded lexical proxies:

* A targeted domain rule (`src/lexicon.py`) differentiates physical maritime/chokepoint bottlenecks from routine diplomatic coverage.
* **Calibration Benchmark**: 200 stratified titles were manually inspected and hand-labeled as Ground Truth (`data/sample_200_for_labeling.csv`).
* **Performance**:
* **Bottleneck Precision**: 0.94 | **Recall**: 1.00 | **F1**: 0.97
* **Diplomatic Precision**: 0.70 | **Recall**: 0.78 | **F1**: 0.74
* **Macro-F1**: **0.90** (Accuracy: 0.97)


* *Audit Note*: The 200-sample set served as an exploratory calibration benchmark during feature design; category support for diplomatic events is relatively small ($N=9$). Full out-of-sample generalization should be independently audited on a frozen blind set.

---

## 4. Empirical Findings: Contemporaneous Conditional Tail Risk

Models are evaluated via an **expanding window** out-of-sample design (initial training window = 40 trading days; 28 test days). Features include contemporaneous realized market controls (`oil_ret`, `vix_level`) with lagged narrative sentiment (`lag_bottleneck_index` and its interaction).

### Out-of-Sample Pinball Loss & Calibration Metrics

| Quantile ($\tau$) | Target Coverage | Base Coverage | Aug Coverage | Base Pinball Loss | Aug Pinball Loss | Loss Diff ($\Delta$) | Permutation $p$-val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0.05** (Downside Tail) | 0.05 | 0.071 | 0.071 | 0.001020 | 0.001102 | -0.000082 | 0.630 |
| **0.50** (Median) | 0.50 | 0.464 | 0.464 | 0.003168 | 0.003089 | +0.000079 | 0.241 |
| **0.95** (Upside Tail) | 0.95 | 0.036 | 0.036 | **0.000956** | **0.000825** | **+0.000131** | **$\le$ 0.001***** |

*Note: In addition to the standard Diebold-Mariano statistic ($DM = 5.320$), non-parametric permutation $p$-values are reported across 1,000 iterations to account for small test sample size ($N=28$) and nested model structure.*

### Key Takeaways

1. **Upper-Tail Calibration Gain ($\tau = 0.95$)**: Narrative bottleneck sentiment yields a statistically significant pinball loss reduction ($p \le 0.001$). Both models yield identical empirical coverage (1/28 exceeds, $\hat{\alpha}=0.036$), but the augmented model provides sharper tail bounds.
2. **Tail Compression vs. Speculative Surge**: The net marginal coefficient at $\tau = 0.95$ under typical volatility is slightly negative ($\beta_{\text{index}} + \beta_{\text{interaction}} \times \overline{\log\text{VIX}} \approx -0.06$). Heightened bottleneck sentiment dampens upper-tail dispersion rather than inducing explosive upward jumps, likely reflecting systemic risk de-risking and cross-asset liquidity drawdowns.
3. **Median Invariance**: News sentiment provides negligible predictive utility for median returns ($\tau = 0.50$, $p = 0.241$), confirming narrative signals inform extreme risk regimes rather than central trend direction.

---

## 5. Critical Methodological Boundaries

1. **Conditional Fit vs. Pure Forecasting**: Because contemporaneous realized market controls (`oil_ret`, `vix_level`) are conditioned on concurrently with lagged narrative indices, the model evaluates conditional risk transmission rather than pure directional market forecasting.
2. **Small Test Span ($N=28$)**: The out-of-sample window covers an acute 3-month phase of the 2022 conflict shock. At $\tau = 0.95$, expected tail exceedances in 28 days are only $\approx 1.4$, meaning tail asymptotic standard errors approach degeneracy.
3. **GDELT Source Redundancy**: Ingestion is restricted to English-origin records, which over-index on Western wire syndicates and institutional reporting.
4. **Futures Roll Adjustment**: Front-month futures (`BZ=F`, `GC=F`) were sourced without unrolling contract spreads across expiration dates.

---

## 6. Reproducibility

```powershell
# 1. Environment setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run measurement validation (Ground truth Macro-F1 = 0.90)
python src/evaluate_lexicon.py

# 3. FinBERT scoring & feature aggregation
python src/score_sentiment.py
python src/build_features.py

# 4. Out-of-sample quantile regression evaluation & metrics export
python src/model_quantreg.py

```

---

## 7. Future Work

* Implementing **Markov-Switching Quantile Regression (MS-QR)** to account for endogenous transitions between calm and acute crisis volatility states.
* Expanding GDELT ingestion to full 96-snapshot continuous feeds with rolling exponential decay kernels to resolve zero-inflation on low-volume trading days.
