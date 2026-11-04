import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent))
from lexicon import classify_title

LABELED_PATH = Path("data/sample_200_for_labeling.csv")
REPORT_PATH = Path("results/lexicon_validation_report.txt")

def main():
    df = pd.read_csv(LABELED_PATH)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].str.strip().str.lower()
    
    df["pred"] = df["title"].apply(classify_title)
    classes = ["bottleneck", "diplomatic", "unrelated"]
    
    report = classification_report(df["label"], df["pred"], labels=classes, zero_division=0)
    cm = confusion_matrix(df["label"], df["pred"], labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== Lexicon Validation Against Hand-Labeled Ground Truth ===\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n")
        
    print(report)
    print("\nConfusion Matrix:\n", cm_df)
    print(f"\nReport written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
