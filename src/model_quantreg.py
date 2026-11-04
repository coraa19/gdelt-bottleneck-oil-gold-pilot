from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = Path("data/model_dataset.csv")
REPORT_PATH = Path("results/model_evaluation_report.txt")

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
    # 单尾检验：检查 Augmented 模型 loss 是否显著更低 (d > 0)
    p_value = 1.0 - stats.norm.cdf(dm_stat)
    return float(dm_stat), float(p_value)

def main():
    df = pd.read_csv(DATA_PATH)
    quantiles = [0.05, 0.50, 0.95]
    
    init_train = 40
    n = len(df)
    
    formula_base = "gold_ret ~ oil_ret + vix_level"
    formula_aug = "gold_ret ~ oil_ret + vix_level + lag_bottleneck_index + interaction_vix_index"
    
    # 1. 全样本参数拟合估计
    full_models_summary = []
    full_models_summary.append("=== FULL SAMPLE QUANTILE REGRESSION COEFFICIENTS ===")
    for q in quantiles:
        res_aug = smf.quantreg(formula_aug, data=df).fit(q=q, max_iter=2500)
        full_models_summary.append(f"\n--- Quantile: {q} ---")
        full_models_summary.append(res_aug.summary().as_text())
        
    # 2. 样本外 Expanding Window 预测
    oos_results = {q: {"base_loss": [], "aug_loss": []} for q in quantiles}
    
    print(f"Running expanding window out-of-sample evaluation ({n - init_train} test days)...")
    for t in range(init_train, n):
        train_df = df.iloc[:t]
        test_row = df.iloc[t:t+1]
        y_true = test_row["gold_ret"].values[0]
        
        for q in quantiles:
            fit_b = smf.quantreg(formula_base, data=train_df).fit(q=q, max_iter=2500)
            fit_a = smf.quantreg(formula_aug, data=train_df).fit(q=q, max_iter=2500)
            
            pred_b = fit_b.predict(test_row).values[0]
            pred_a = fit_a.predict(test_row).values[0]
            
            oos_results[q]["base_loss"].append(pinball_loss(y_true, pred_b, q))
            oos_results[q]["aug_loss"].append(pinball_loss(y_true, pred_a, q))
            
    # 3. 统计 Pinball Loss 与 Diebold-Mariano 检验
    metrics_summary = []
    metrics_summary.append("\n=== OUT-OF-SAMPLE PINBALL LOSS & DIEBOLD-MARIANO TEST ===")
    metrics_summary.append(f"{'Quantile':<10} | {'Base Loss':<12} | {'Aug Loss':<12} | {'DM Stat':<10} | {'p-value':<10}")
    metrics_summary.append("-" * 65)
    
    for q in quantiles:
        base_l = oos_results[q]["base_loss"]
        aug_l = oos_results[q]["aug_loss"]
        
        m_base = np.mean(base_l)
        m_aug = np.mean(aug_l)
        dm_stat, p_val = diebold_mariano(base_l, aug_l)
        
        row_str = f"{q:<10.2f} | {m_base:<12.6f} | {m_aug:<12.6f} | {dm_stat:<10.3f} | {p_val:<10.3f}"
        metrics_summary.append(row_str)
        print(row_str)
        
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(full_models_summary))
        f.write("\n\n")
        f.write("\n".join(metrics_summary))
        f.write("\n")
        
    print(f"\nEvaluation report successfully written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
