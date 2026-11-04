import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lexicon import classify_title

INPUT_PATH = Path("data/titles_clean.csv")
OUTPUT_PATH = Path("data/titles_with_sentiment.csv")
MODEL_NAME = "ProsusAI/finbert"

def main():
    print(f"Loading model: {MODEL_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} cleaned headlines.")
    
    # 规则分类标注
    print("Classifying headlines with lexicon...")
    df["bottleneck_pred"] = df["title"].apply(classify_title)
    df["is_bottleneck"] = (df["bottleneck_pred"] == "bottleneck").astype(int)
    
    # FinBERT 批量打分
    titles = df["title"].tolist()
    batch_size = 32
    neg_probs = []
    
    print("Scoring sentiment with FinBERT...")
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i : i + batch_size]
        inputs = tokenizer(batch_titles, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # FinBERT 默认输出顺序: [positive, negative, neutral]
            probs = torch.softmax(outputs.logits, dim=-1)
            # 提取 negative 概率 (索引 1)
            batch_neg = probs[:, 1].cpu().tolist()
            neg_probs.extend(batch_neg)
            
        if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(titles):
            print(f"Processed {min(i + batch_size, len(titles))}/{len(titles)} headlines...")

    df["prob_negative"] = neg_probs
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Saved scored headlines to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
