from transformers import pipeline
from typing import Optional
import torch
from .cause_extractor import extract_causes, extract_keywords

# Map sentiment
SENTIMENT_SCORE_MAP = {
    "LABEL_0": 1.5,  # negative
    "LABEL_1": 3.0,  # neutral  
    "LABEL_2": 4.5,  # positive
}

# Emotion keywords map
EMOTION_KEYWORDS = {
    0: ["tê liệt", "vô cảm", "trống rỗng", "không cảm"],
    1: ["mơ hồ", "không biết", "lẫn lộn", "confused"],
    2: ["buồn", "khóc", "thất vọng", "đau lòng", "tổn thương"],
    3: ["mệt", "kiệt sức", "uể oải", "không muốn"],
    4: ["tức", "bực", "cáu", "giận", "khó chịu"],
    5: ["bình thường", "không sao", "thờ ơ", "nhàm"],
    6: ["bình tĩnh", "ổn định", "nhẹ nhàng", "thư giãn"],
    7: ["ổn", "tốt", "được", "khá"],
    8: ["vui", "tích cực", "hứng", "năng lượng"],
    9: ["rất vui", "hạnh phúc", "phấn khởi", "tuyệt"],
    10: ["xuất sắc", "tuyệt vời", "hứng khởi", "phấn chấn"],
}

EMOTION_NAMES = [
    "Tê liệt", "Mơ hồ", "Buồn bã", "Mệt mỏi", "Cáu kỉnh",
    "Thờ ơ", "Bình tĩnh", "Ổn", "Tích cực", "Vui vẻ", "Phấn khởi"
]

RISK_KEYWORDS = {
    "high": [
        "tự tử", "không muốn sống", "chết", "tuyệt vọng",
        "không còn hy vọng", "bỏ cuộc tất cả"
    ],
    "moderate": [
        "lo lắng nhiều", "stress nặng", "không chịu được",
        "quá áp lực", "muốn bỏ trốn", "khóc mãi"
    ],
}

class NLPAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=0 if torch.cuda.is_available() else -1,
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model load failed: {e}, using rule-based fallback")
            self.sentiment_pipeline = None

    def analyze(self, text: str) -> dict:
        # Sentiment score
        sentiment_score, sentiment_label, confidence = self._get_sentiment(text)

        # Emotion detection
        emotion_index = self._detect_emotion(text, sentiment_score)
        emotion_name = EMOTION_NAMES[emotion_index]

        # Cause extraction
        causes = extract_causes(text)
        keywords = extract_keywords(text)

        # Risk level
        risk_level = self._detect_risk(text, sentiment_score)

        return {
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_label": sentiment_label,
            "emotion": emotion_name,
            "emotion_index": emotion_index,
            "causes": causes,
            "keywords": keywords,
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
        }

    def _get_sentiment(self, text: str):
        if self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text[:512])[0]
                label = result["label"]
                score = SENTIMENT_SCORE_MAP.get(label, 3.0)
                confidence = result["score"]
                sentiment_label = (
                    "positive" if score >= 4 else
                    "negative" if score <= 2 else
                    "neutral"
                )
                return score, sentiment_label, confidence
            except:
                pass

        # Fallback
        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str):
        positive_words = [
            "vui", "hạnh phúc", "tốt", "ổn", "tích cực",
            "yêu", "thích", "tuyệt", "hài lòng", "phấn khởi"
        ]
        negative_words = [
            "buồn", "tức", "mệt", "lo", "sợ", "ghét",
            "tệ", "xấu", "khó", "stress", "áp lực"
        ]
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)

        if pos > neg:
            score = min(3.5 + pos * 0.3, 5.0)
            return score, "positive", 0.6
        elif neg > pos:
            score = max(2.5 - neg * 0.3, 1.0)
            return score, "negative", 0.6
        return 3.0, "neutral", 0.5

    def _detect_emotion(self, text: str, sentiment_score: float) -> int:
        text_lower = text.lower()
        scores = [0] * 11

        for idx, keywords in EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[idx] += 1

        if max(scores) == 0:
            if sentiment_score >= 4.5: return 10
            if sentiment_score >= 4.0: return 9
            if sentiment_score >= 3.5: return 8
            if sentiment_score >= 3.0: return 7
            if sentiment_score >= 2.5: return 6
            if sentiment_score >= 2.0: return 5
            if sentiment_score >= 1.5: return 3
            return 2

        return scores.index(max(scores))

    def _detect_risk(self, text: str, score: float) -> str:
        text_lower = text.lower()

        for kw in RISK_KEYWORDS["high"]:
            if kw in text_lower:
                return "high"

        for kw in RISK_KEYWORDS["moderate"]:
            if kw in text_lower:
                return "moderate"

        if score <= 1.5:
            return "moderate"

        return "gentle"


# Singleton
analyzer = NLPAnalyzer()