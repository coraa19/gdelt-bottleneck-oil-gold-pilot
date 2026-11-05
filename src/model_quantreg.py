from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = Path("data/model_dataset.csv")
REPORT_PATH = Path("results/model_evaluation_report.txt")
METRICS_CSV_PATH = Path("results/quantreg_metrics.csv")


def pinball_loss(y_true, y_pred, q):
    err = y_true - y_pred
    return np.maximum(q * err, (q - 1) * err)


def diebold_mariano(loss_base, loss_aug):
    d = np.array(loss_base) - np.array(loss_aug)
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0 or np.isnan(var_d):
        return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 1.0 - stats.norm.cdf(dm_stat)
    return float(dm_stat), float(p_value)


def run_permutation_test(train_df, test_df, q, formula_aug, mean_loss_base, actual_diff, n_permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    perm_diffs = []
    
    perm_train = train_df.copy()
    y_test = test_df["gold_ret"].values
    
    for _ in range(n_permutations):
        shuffled_idx = rng.permutation(len(perm_train))
        perm_train["lag_bottleneck_index"] = train_df["lag_bottleneck_index"].iloc[shuffled_idx].values
        perm_train["interaction_vix_index"] = train_df["interaction_vix_index"].iloc[shuffled_idx].values
        
        try:
            fit_perm = smf.quantreg(formula_aug, data=perm_train).fit(q=q, max_iter=2500)
            pred_perm = fit_perm.predict(test_df).values
            loss_perm = np.mean(pinball_loss(y_test, pred_perm, q))
            perm_diffs.append(mean_loss_base - loss_perm)
        except Exception:
            continue
            
    if not perm_diffs:
        return 1.0
        
    perm_p_val = np.mean(np.array(perm_diffs) >= actual_diff)
    return float(perm_p_val)


def main():
    df = pd.read_csv(DATA_PATH)
    quantiles = [0.05, 0.50, 0.95]
    
    init_train = 40
    n = len(df)
    n_test = n - init_train
    
    formula_base = "gold_ret ~ oil_ret + vix_level"
    formula_aug = "gold_ret ~ oil_ret + vix_level + lag_bottleneck_index + interaction_vix_index"
    
    # 1. Full-sample quantile regression
    full_models_summary = ["=== FULL SAMPLE QUANTILE REGRESSION COEFFICIENTS ==="]
    for q in quantiles:
        res_aug = smf.quantreg(formula_aug, data=df).fit(q=q, max_iter=2500)
        full_models_summary.append(f"\n--- Quantile: {q} ---")
        full_models_summary.append(res_aug.summary().as_text())
        
    # 2. Out-of-sample expanding window estimation
    oos_data = {
        q: {
            "y_true": [],
            "pred_base": [],
            "pred_aug": [],
            "loss_base": [],
            "loss_aug": []
        } for q in quantiles
    }
    
    print(f"Running expanding window out-of-sample evaluation ({n_test} test days)...")
    for t in range(init_train, n):
        train_df = df.iloc[:t]
        test_row = df.iloc[t:t+1]
        y_true = test_row["gold_ret"].values[0]
        
        for q in quantiles:
            fit_b = smf.quantreg(formula_base, data=train_df).fit(q=q, max_iter=2500)
            fit_a = smf.quantreg(formula_aug, data=train_df).fit(q=q, max_iter=2500)
            
            p_b = fit_b.predict(test_row).values[0]
            p_a = fit_a.predict(test_row).values[0]
            
            oos_data[q]["y_true"].append(y_true)
            oos_data[q]["pred_base"].append(p_b)
            oos_data[q]["pred_aug"].append(p_a)
            oos_data[q]["loss_base"].append(pinball_loss(y_true, p_b, q))
            oos_data[q]["loss_aug"].append(pinball_loss(y_true, p_a, q))
            
    # 3. Compute calibration, coverage, and permutation significance
    metrics_records = []
    metrics_text_lines = [
        "\n=== OUT-OF-SAMPLE EVALUATION: PINBALL LOSS, COVERAGE & SIGNIFICANCE ===",
        f"{'Quantile':<10} | {'Target':<8} | {'Base Cov':<10} | {'Aug Cov':<10} | {'Base Loss':<12} | {'Aug Loss':<12} | {'DM Stat':<9} | {'DM p-val':<10} | {'Perm p-val':<10}",
        "-" * 105
    ]
    
    # Pre-slice fixed sets for the permutation check
    train_fixed = df.iloc[:init_train].copy()
    test_fixed = df.iloc[init_train:].copy()
    
    for q in quantiles:
        y_arr = np.array(oos_data[q]["y_true"])
        pred_b_arr = np.array(oos_data[q]["pred_base"])
        pred_a_arr = np.array(oos_data[q]["pred_aug"])
        
        loss_b = oos_data[q]["loss_base"]
        loss_a = oos_data[q]["loss_aug"]
        
        mean_lb = float(np.mean(loss_b))
        mean_la = float(np
