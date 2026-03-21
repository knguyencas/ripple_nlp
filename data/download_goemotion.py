from datasets import load_dataset
from deep_translator import GoogleTranslator
from tqdm import tqdm
import pandas as pd
import json
import time
import os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

GO_EMOTION_MAP = {
    # Joy family
    "joy":          {"primary": "joy",          "vi": "hạnh phúc",      "valence": 0.9,  "arousal": 0.7},
    "amusement":    {"primary": "joy",          "vi": "vui vẻ",         "valence": 0.8,  "arousal": 0.6},
    "excitement":   {"primary": "joy",          "vi": "hứng khởi",      "valence": 0.8,  "arousal": 0.9},
    "gratitude":    {"primary": "joy",          "vi": "biết ơn",        "valence": 0.85, "arousal": 0.4},
    "pride":        {"primary": "joy",          "vi": "tự hào",         "valence": 0.75, "arousal": 0.6},
    "relief":       {"primary": "joy",          "vi": "nhẹ nhõm",       "valence": 0.7,  "arousal": 0.3},
    "love":         {"primary": "love",         "vi": "yêu thương",     "valence": 0.95, "arousal": 0.5},
    "caring":       {"primary": "trust",        "vi": "quan tâm",       "valence": 0.7,  "arousal": 0.4},
    "optimism":     {"primary": "optimism",     "vi": "lạc quan",       "valence": 0.8,  "arousal": 0.6},
    "admiration":   {"primary": "trust",        "vi": "ngưỡng mộ",      "valence": 0.75, "arousal": 0.5},
    "desire":       {"primary": "anticipation", "vi": "khao khát",      "valence": 0.6,  "arousal": 0.7},
    "curiosity":    {"primary": "anticipation", "vi": "tò mò",          "valence": 0.5,  "arousal": 0.6},

    # Sadness family
    "sadness":      {"primary": "sadness",      "vi": "buồn bã",        "valence": -0.8, "arousal": 0.3},
    "grief":        {"primary": "sadness",      "vi": "đau buồn",       "valence": -0.9, "arousal": 0.2},
    "remorse":      {"primary": "remorse",      "vi": "hối hận",        "valence": -0.7, "arousal": 0.3},
    "disappointment":{"primary": "sadness",     "vi": "thất vọng",      "valence": -0.7, "arousal": 0.3},
    "embarrassment":{"primary": "sadness",      "vi": "xấu hổ",         "valence": -0.6, "arousal": 0.5},

    # Anger family
    "anger":        {"primary": "anger",        "vi": "tức giận",       "valence": -0.8, "arousal": 0.9},
    "annoyance":    {"primary": "anger",        "vi": "khó chịu",       "valence": -0.6, "arousal": 0.6},
    "disapproval":  {"primary": "disapproval",  "vi": "phản đối",       "valence": -0.5, "arousal": 0.5},

    # Fear family
    "fear":         {"primary": "fear",         "vi": "sợ hãi",         "valence": -0.8, "arousal": 0.8},
    "nervousness":  {"primary": "fear",         "vi": "lo lắng",        "valence": -0.6, "arousal": 0.7},

    # Disgust
    "disgust":      {"primary": "disgust",      "vi": "ghê tởm",        "valence": -0.85,"arousal": 0.6},

    # Surprise
    "surprise":     {"primary": "surprise",     "vi": "ngạc nhiên",     "valence": 0.1,  "arousal": 0.8},
    "confusion":    {"primary": "surprise",     "vi": "bối rối",        "valence": -0.3, "arousal": 0.5},
    "realization":  {"primary": "surprise",     "vi": "nhận ra",        "valence": 0.2,  "arousal": 0.5},

    # Neutral
    "neutral":      {"primary": "neutral",      "vi": "bình thường",    "valence": 0.0,  "arousal": 0.0},
}

def get_primary_label(labels: list, id2label: dict) -> dict:
    """Lấy emotion chính từ list label IDs"""
    if not labels:
        return GO_EMOTION_MAP["neutral"]
    
    label_name = id2label[labels[0]]
    return GO_EMOTION_MAP.get(label_name, GO_EMOTION_MAP["neutral"])

def translate_batch(texts: list, batch_size: int = 10) -> list:
    translator = GoogleTranslator(source='en', target='vi')
    translated = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        batch_translated = []
        
        for text in batch:
            try:
                vi_text = translator.translate(text)
                batch_translated.append(vi_text)
                time.sleep(0.1)
            except Exception as e:
                print(f"Translation error: {e}")
                batch_translated.append(text)
        
        translated.extend(batch_translated)
        time.sleep(0.5) 
    
    return translated

def process_dataset(split: str, ds, id2label: dict, max_samples: int = None) -> list:
    """Xử lý 1 split của dataset"""
    data = ds[split]
    samples = []
    
    limit = max_samples or len(data)
    
    for i in range(min(limit, len(data))):
        item = data[i]
        labels = item['labels']
        
        if not labels:
            continue
        
        emotion_info = get_primary_label(labels, id2label)
        
        all_emotions = [id2label[l] for l in labels]
        secondary = [GO_EMOTION_MAP.get(e, {}).get("vi", e) 
                    for e in all_emotions[1:]]
        
        samples.append({
            "text_en": item['text'],
            "text_vi": None,
            "primary_emotion": emotion_info["primary"],
            "primary_emotion_vi": emotion_info["vi"],
            "secondary_emotions": secondary,
            "all_go_emotions": all_emotions,
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "source": "go_emotions",
            "split": split,
        })
    
    return samples


def main():
    print("Downloading GoEmotions.")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    
    label_names = ds['train'].features['labels'].feature.names
    id2label = {i: name for i, name in enumerate(label_names)}
    
    print(f"Dataset loaded!")
    print(f"Train: {len(ds['train'])} samples")
    print(f"Val:   {len(ds['validation'])} samples")
    print(f"Test:  {len(ds['test'])} samples")
    
    print("\nLabel distribution (train, top 10):")
    all_labels = []
    for item in ds['train']:
        all_labels.extend([id2label[l] for l in item['labels']])
    
    from collections import Counter
    top_labels = Counter(all_labels).most_common(10)
    for label, count in top_labels:
        print(f"  {label}: {count}")
    
    print("\nProcessing dataset")
    
    TEST_SIZE = 500
    print(f"Processing {TEST_SIZE} samples for testing")
    
    samples = process_dataset('train', ds, id2label, max_samples=TEST_SIZE)
    
    df = pd.DataFrame(samples)
    df.to_csv("data/raw/go_emotions_en.csv", index=False, encoding='utf-8')
    print(f"\nSaved {len(df)} English samples to data/raw/go_emotions_en.csv")
        
    texts_en = df['text_en'].tolist()[:100]
    texts_vi = translate_batch(texts_en, batch_size=5)
    
    df_vi = df.head(100).copy()
    df_vi['text_vi'] = texts_vi
    df_vi.to_csv("data/raw/go_emotions_vi_sample.csv", index=False, encoding='utf-8')
    
    print(f"\nSaved translated sample to data/raw/go_emotions_vi_sample.csv")
    
    print("\nPreview:")
    for _, row in df_vi.head(5).iterrows():
        print(f"\n  EN: {row['text_en']}")
        print(f"  VI: {row['text_vi']}")
        print(f"  Emotion: {row['primary_emotion']} ({row['primary_emotion_vi']})")
        print(f"  Valence: {row['valence']}, Arousal: {row['arousal']}")


if __name__ == "__main__":
    main()