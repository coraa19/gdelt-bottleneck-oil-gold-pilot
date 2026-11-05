import argparse
import csv
from datetime import datetime, timedelta
import io
from pathlib import Path
import re
import time
import zipfile

import pandas as pd
import requests

BASE = "http://data.gdeltproject.org/gdeltv2/{}.gkg.csv.zip"
OUT = Path("data/gdelt_titles.csv")
DONE = Path("data/processed_files.txt")

ENERGY = (
    r"\b(?:oil|crude|gas|lng|fuel|energy|opec|tankers?|shipping|pipelines?|"
    r"refiner(?:y|ies)|hormuz|suez|red sea|strait|chokepoint)\b"
)
CONFLICT = (
    r"\b(?:war|attacks?|strikes?|missiles?|drones?|sanctions?|blockade|embargo|"
    r"ceasefire|talks|conflict|military|invasion|houthis?|threat|seiz(?:e|ed|ure))\b"
)


def fetch(stamp):
    url = BASE.format(stamp)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"  retry {attempt + 1} for {stamp}: {e}")
            time.sleep(3)
    return None


def parse(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                usecols=[1, 3, 25, 26],
                dtype=str,
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip",
                encoding="utf-8",
                encoding_errors="replace",
            )
    df.columns = ["date", "source", "translation", "extras"]
    n_raw = len(df)
    df = df[df["translation"].isna()].copy()
    df["title"] = df["extras"].str.extract(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>")
    df = df.dropna(subset=["title"]).copy()
    t = df["title"].str.lower()
    df = df[
        t.str.contains(ENERGY, regex=True) & t.str.contains(CONFLICT, regex=True)
    ].copy()
    df["raw_day"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S").dt.date
    df["day"] = df["raw_day"]
    return n_raw, df[["day", "source", "title"]]


def stamps(start, end, hours):
    d = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")
    while d <= last:
        for h in hours:
            yield d.replace(hour=h).strftime("%Y%m%d%H%M%S")
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--hours", type=int, nargs="+", default=[6, 18])
    ap.add_argument("--test", help="one timestamp, e.g. 20220301120000")
    args = ap.parse_args()

    Path("data").mkdir(exist_ok=True)
    todo = [args.test] if args.test else list(stamps(args.start, args.end, args.hours))
    done = set(DONE.read_text().split()) if DONE.exists() else set()

    for s in todo:
        if s in done and not args.test:
            continue
        content = fetch(s)
        if content is None:
            print(f"{s}: missing or failed")
            continue
        n_raw, df = parse(content)
        msg = f"{s}: zip {len(content)/1e6:.1f} MB, raw {n_raw}, kept {len(df)}"
        print(msg)

        if len(df) > 0:
            header = not OUT.exists()
            df.to_csv(OUT, mode="a", index=False, header=header)

        with open(DONE, "a", encoding="utf-8") as f_done:
            f_done.write(f"{s}\n")


if __name__ == "__main__":
    main()