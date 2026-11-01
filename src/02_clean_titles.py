import pandas as pd
from pathlib import Path

RAW = Path("data/gdelt_titles.csv")
CLEAN = Path("data/titles_clean.csv")
SUMMARY = Path("results/cleaning_summary.csv")


def norm(title):
    t = title.lower()
    return "".join(ch for ch in t if ch.isalnum())


def main():
    df = pd.read_csv(RAW, dtype=str)
    n0 = len(df)
    df = df.dropna(subset=["title"]).copy()
    df["key"] = df["title"].map(norm)
    df = df[df["key"].str.len() > 15].copy()
    n1 = len(df)
    df = df.drop_duplicates(subset=["day", "key"]).drop(columns="key")
    n2 = len(df)
    df.to_csv(CLEAN, index=False)

    per_day = df.groupby("day").size()
    per_day.index = pd.to_datetime(per_day.index)
    all_days = pd.date_range(per_day.index.min(), per_day.index.max())
    per_day = per_day.reindex(all_days, fill_value=0)

    print("raw rows:", n0)
    print("after dropping empty or very short titles:", n1)
    print("after same-day de-duplication:", n2)
    print("days covered:", len(all_days))
    print("days with zero headlines:", int((per_day == 0).sum()))
    print(per_day.describe().round(1).to_string())

    out = pd.DataFrame({"step": ["raw", "non_empty", "deduplicated"],
                        "rows": [n0, n1, n2]})
    out.to_csv(SUMMARY, index=False)


if __name__ == "__main__":
    main()