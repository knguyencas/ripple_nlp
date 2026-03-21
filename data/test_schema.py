from schema import *
import json

# Sample test
sample = EmotionSample(
    text=TextInfo(
        original="mệt mỏi đau khổ quá, deadline dồn dập mà sếp cứ thêm việc",
        normalized="mệt mỏi đau khổ quá deadline dồn dập mà sếp cứ thêm việc",
        language=Language.VI,
        style=TextStyle.INFORMAL,
        word_count=12,
        has_emoji=False,
    ),
    emotion=EmotionLabel(
        primary=BasicEmotion.SADNESS,
        secondary=[BasicEmotion.ANGER, BasicEmotion.FEAR],
        vietnamese_specific=VietnameseEmotion.CHAN_NAN,
        intensity=0.8,
        valence=-0.75,
        arousal=0.4,
        sentiment_score=1.8,
    ),
    context=ContextInfo(
        causes=[CauseCategory.WORK],
        keywords=["deadline", "sếp", "mệt mỏi"],
        time_of_day="evening",
        day_of_week="thursday",
        risk_level=RiskLevel.LOW,
    ),
    label_info=LabelInfo(
        human_verified=False,
        confidence=0.85,
    ),
    meta=MetaInfo(
        source=DataSource.SELF_CREATED,
        age_group=AgeGroup.YOUNG_ADULT,
        user_consent=True,
        anonymized=True,
        tags=["work_stress", "fatigue"],
    )
)

print(json.dumps(sample.to_dict(), ensure_ascii=False, indent=2))

print(json.dumps(sample.to_training_row(), ensure_ascii=False, indent=2))