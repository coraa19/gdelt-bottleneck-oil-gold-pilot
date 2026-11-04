from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

SENTIMENT_CSV = Path("data/titles_with_sentiment.csv")
OUTPUT_CSV = Path("data/model_dataset.csv")

def build_daily_index(df_titles: pd.DataFrame) -> pd.DataFrame:
    df = df_titles.copy()
    df["day"] = pd.to_datetime(df["day"])
    
    # 周末舆情合并到周一 (Saturday -> Monday (+2 days), Sunday -> Monday (+1 day))
    weekday = df["day"].dt.weekday
    df.loc[weekday == 5, "day"] += pd.Timedelta(days=2)
    df.loc[weekday == 6, "day"] += pd.Timedelta(days=1)
    
    # 按天聚合：计算瓶颈比例与平均负面概率
    grouped = df.groupby("day").agg(
        total_count=("title", "count"),
        bottleneck_count=("is_bottleneck", "sum"),
        mean_neg=("prob_negative", "mean")
    ).reset_index()
    
    grouped["bottleneck_share"] = grouped["bottleneck_count"] / grouped["total_count"]
    grouped["bottleneck_index"] = grouped["bottleneck_share"] * grouped["mean_neg"]
    return grouped[["day", "bottleneck_index"]]

def fetch_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching market data from {start_date} to {end_date} via yfinance...")
    tickers = {"BZ=F": "oil", "GC=F": "gold", "^VIX": "vix"}
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)["Close"]
    data = data.rename(columns=tickers)
    data = data.dropna(subset=["oil", "gold"])
    
    # 计算日对数收益率
    data["oil_ret"] = np.log(data["oil"] / data["oil"].shift(1))
    data["gold_ret"] = np.log(data["gold"] / data["gold"].shift(1))
    # VIX 取对数水平值作为波动率代理
    data["vix_level"] = np.log(data["vix"])
    
    data = data.reset_index().rename(columns={"Date": "day"})
    data["day"] = pd.to_datetime(data["day"]).dt.tz_localize(None)
    return data[["day", "gold_ret", "oil_ret", "vix_level"]].dropna()

def main():
    df_titles = pd.read_csv(SENTIMENT_CSV)
    idx_df = build_daily_index(df_titles)
    
    min_date = idx_df["day"].min().strftime("%Y-%m-%d")
    # 扩展拉取窗口以保证滞后阶对齐
    start_market = (idx_df["day"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_market = (idx_df["day"].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    
    mkt_df = fetch_market_data(start_market, end_market)
    
    # 交易日日历合并
    merged = pd.merge(mkt_df, idx_df, on="day", how="left")
    merged["bottleneck_index"] = merged["bottleneck_index"].fillna(0.0)
    
    # 核心步骤：情绪指数严格滞后 1 交易日，消除前视偏差
    merged["lag_bottleneck_index"] = merged["bottleneck_index"].shift(1)
    
    # 构造交互项: 瓶颈情绪与 VIX 市场恐慌的乘积
    merged["interaction_vix_index"] = merged["lag_bottleneck_index"] * merged["vix_level"]
    
    # 剔除首行 lag 生成的 NaN
    final_df = merged.dropna().copy()
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\nDataset ready for quantile regression!")
    print(f"Total trading days: {len(final_df)}")
    print(f"Saved to: {OUTPUT_CSV}")
    print(final_df.head(3).to_string())

if __name__ == "__main__":
    main()
