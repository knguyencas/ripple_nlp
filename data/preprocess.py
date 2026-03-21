"""
Preprocessing pipeline cho Vietnamese Emotion Dataset
1. Filter samples không phù hợp
2. Clean text
3. Validate translation quality
4. Merge nhiều sources
"""

import pandas as pd
import re
import unicodedata
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

SENSITIVE_KEYWORDS = [
    "tự tử", "tự sát", "muốn chết", "không muốn sống",
    "suicide", "kill myself", "want to die",
]

PROFANITY_PATTERNS = [
    r'\bfuck\b', r'\bshit\b', r'\bass\b', r'\bdick\b',
    r'\bwank\b', r'\bcunt\b', r'\bbitch\b',
]

MIN_WORDS = 3
MAX_WORDS = 100

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize('NFC', text)

    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    text = re.sub(r'@\w+|#\w+', '', text)

    text = re.sub(
        r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ,.!?]',
        ' ', text
    )

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_words(text: str) -> int:
    return len(text.split())

def is_sensitive(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in SENSITIVE_KEYWORDS)

def has_profanity(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in PROFANITY_PATTERNS)

def is_valid_length(text: str) -> bool:
    wc = count_words(text)
    return MIN_WORDS <= wc <= MAX_WORDS

def is_valid_translation(text_en: str, text_vi: str) -> bool:
    if not text_vi or not isinstance(text_vi, str):
        return False

    if text_en.lower().strip() == text_vi.lower().strip():
        return False

    ratio = len(text_vi) / (len(text_en) + 1)
    if ratio < 0.3 or ratio > 3.0:
        return False

    return True

def process_go_emotions(filepath: str) -> pd.DataFrame:
    print(f"\nProcessing GoEmotions: {filepath}")
    df = pd.read_csv(filepath)
    original_count = len(df)

    filtered_sensitive = 0
    filtered_profanity = 0
    filtered_length = 0
    filtered_translation = 0

    valid_rows = []

    for _, row in df.iterrows():
        text_en = str(row.get('text_en', ''))
        text_vi = str(row.get('text_vi', ''))

        if is_sensitive(text_en) or is_sensitive(text_vi):
            filtered_sensitive += 1
            continue

        if has_profanity(text_en):
            filtered_profanity += 1
            continue

        cleaned_vi = clean_text(text_vi)
        cleaned_en = clean_text(text_en)

        if not is_valid_length(cleaned_vi):
            filtered_length += 1
            continue

        if text_vi and text_vi != 'None':
            if not is_valid_translation(text_en, text_vi):
                filtered_translation += 1
                continue

        valid_rows.append({
            **row.to_dict(),
            'text_vi_clean': cleaned_vi,
            'text_en_clean': cleaned_en,
            'word_count': count_words(cleaned_vi),
        })

    result_df = pd.DataFrame(valid_rows)

    print(f"Original:              {original_count}")
    print(f"Filtered sensitive:    {filtered_sensitive}")
    print(f"Filtered profanity:    {filtered_profanity}")
    print(f"Filtered length:       {filtered_length}")
    print(f"Filtered translation:  {filtered_translation}")
    print(f"Valid samples:      {len(result_df)}")

    return result_df

def process_uit_vsfc(filepath: str) -> pd.DataFrame:
    print(f"\nProcessing UIT-VSFC: {filepath}")
    df = pd.read_csv(filepath)
    original_count = len(df)

    sentiment_map = {
        0: {"primary_emotion": "sadness",  "primary_emotion_vi": "buồn bã",    "valence": -0.7, "arousal": 0.3},
        1: {"primary_emotion": "neutral",  "primary_emotion_vi": "bình thường", "valence": 0.0,  "arousal": 0.0},
        2: {"primary_emotion": "joy",      "primary_emotion_vi": "vui vẻ",      "valence": 0.7,  "arousal": 0.6},
    }

    valid_rows = []
    filtered = 0

    for _, row in df.iterrows():
        text = str(row.get('text', '') or row.get('sentence', ''))
        sentiment = row.get('sentiment', row.get('label', 1))

        if is_sensitive(text):
            filtered += 1
            continue

        cleaned = clean_text(text)
        if not is_valid_length(cleaned):
            filtered += 1
            continue

        emotion_info = sentiment_map.get(int(sentiment), sentiment_map[1])

        valid_rows.append({
            'text_en': None,
            'text_vi': text,
            'text_vi_clean': cleaned,
            'text_en_clean': None,
            'word_count': count_words(cleaned),
            'source': 'uit_vsfc',
            **emotion_info,
        })

    result_df = pd.DataFrame(valid_rows)
    print(f"  Original: {original_count}, Valid: {len(result_df)}, Filtered: {filtered}")
    return result_df

def merge_and_export(dfs: list, output_name: str) -> pd.DataFrame:
    merged = pd.concat(dfs, ignore_index=True)

    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    training_cols = [
        'text_vi_clean', 'text_en_clean',
        'primary_emotion', 'primary_emotion_vi',
        'secondary_emotions', 'valence', 'arousal',
        'word_count', 'source',
    ]

    cols = [c for c in training_cols if c in merged.columns]
    result = merged[cols].copy()
    result = result.rename(columns={'text_vi_clean': 'text'})

    output_path = PROCESSED_DIR / output_name
    result.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nExported {len(result)} samples → {output_path}")

    print("\nEmotion distribution:")
    print(result['primary_emotion'].value_counts().to_string())

    print("\nSource distribution:")
    print(result['source'].value_counts().to_string())

    return result

def main():
    dfs = []

    go_vi_path = RAW_DIR / "go_emotions_vi_sample.csv"
    if go_vi_path.exists():
        df_go = process_go_emotions(str(go_vi_path))
        if len(df_go) > 0:
            df_go['source'] = 'go_emotions'
            dfs.append(df_go)
    else:
        print(f"{go_vi_path} not found, skipping GoEmotions")

    # Process UIT-VSFC
    uit_path = RAW_DIR / "uit_vsfc_train.csv"
    if uit_path.exists():
        df_uit = process_uit_vsfc(str(uit_path))
        if len(df_uit) > 0:
            dfs.append(df_uit)
    else:
        print(f"{uit_path} not found, skipping UIT-VSFC")

    if not dfs:
        print("No data to process")
        return

    final_df = merge_and_export(dfs, "dataset_v1.csv")

    print("\nSample rows:")
    print(final_df.head(5).to_string())


if __name__ == "__main__":
    main()