"""
Schema định nghĩa cấu trúc dataset cảm xúc tiếng Việt
Vietnamese Emotion Dataset (ViED) - Ripple Project
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import uuid
from datetime import datetime


# ─── Enums ────────────────────────────────────────────────

class DataSource(str, Enum):
    RIPPLE_APP    = "ripple_app"        # User journal từ app
    GO_EMOTIONS   = "go_emotions"       # Google GoEmotions (dịch)
    UIT_VSFC      = "uit_vsfc"          # UIT dataset
    SEMEVAL       = "semeval"           # SemEval competition
    SELF_CREATED  = "self_created"      # Tự tạo
    SYNTHETIC     = "synthetic"         # Claude generate

class Language(str, Enum):
    VI    = "vi"        # Tiếng Việt thuần
    EN    = "en"        # Tiếng Anh
    VI_EN = "vi_en"     # Mixed Việt-Anh (Viet + English code-switching)

class TextStyle(str, Enum):
    FORMAL    = "formal"        # Văn viết, trang trọng
    INFORMAL  = "informal"      # Thông thường
    TEEN      = "teen"          # Teen slang
    DIALECT   = "dialect"       # Phương ngữ

class AgeGroup(str, Enum):
    TEEN        = "13-17"
    YOUNG_ADULT = "18-25"
    ADULT       = "26-35"
    MIDDLE      = "36-50"
    SENIOR      = "50+"
    UNKNOWN     = "unknown"

class RiskLevel(str, Enum):
    NONE        = "none"        # Bình thường
    LOW         = "low"         # Hơi lo
    MODERATE    = "moderate"    # Cần chú ý
    HIGH        = "high"        # Cần can thiệp


# ─── Basic Emotions (Ekman's 6 + Extended) ────────────────

class BasicEmotion(str, Enum):
    # Ekman's 6 basic
    JOY       = "joy"           # Vui mừng
    SADNESS   = "sadness"       # Buồn bã
    ANGER     = "anger"         # Tức giận
    FEAR      = "fear"          # Sợ hãi
    DISGUST   = "disgust"       # Ghê tởm
    SURPRISE  = "surprise"      # Ngạc nhiên

    # Extended (Plutchik)
    TRUST         = "trust"         # Tin tưởng
    ANTICIPATION  = "anticipation"  # Háo hức, mong đợi

    # Complex combinations
    LOVE        = "love"        # Joy + Trust
    SUBMISSION  = "submission"  # Trust + Fear
    AWE         = "awe"         # Fear + Surprise
    DISAPPROVAL = "disapproval" # Surprise + Sadness
    REMORSE     = "remorse"     # Sadness + Disgust
    CONTEMPT    = "contempt"    # Disgust + Anger
    AGGRESSION  = "aggression"  # Anger + Anticipation
    OPTIMISM    = "optimism"    # Anticipation + Joy


class VietnameseEmotion(str, Enum):
    """
    Cảm xúc đặc trưng tiếng Việt
    khó map trực tiếp sang tiếng Anh
    """
    BUO_MAN_MAC   = "buồn man mác"     # Melancholy, bittersweet sadness
    TUI_THAN      = "tủi thân"         # Self-pity, feeling wronged
    NHAT_NHAN     = "nhẫn nhịn"        # Suppressed emotion, endurance
    CHAN_NAN      = "chán nản"         # Despair + boredom
    BON_CHON      = "bồn chồn"         # Restlessness, anxiety mix
    NHO_NHUNG     = "nhớ nhung"        # Longing, nostalgia
    XAU_HO        = "xấu hổ"           # Shame
    NGAI_NGUNG    = "ngại ngùng"       # Social awkwardness
    THUONG_CAM    = "thương cảm"       # Compassion, empathy
    HANH_PHUC     = "hạnh phúc"        # Deep happiness (deeper than joy)
    AN_NHIEN      = "an nhiên"         # Inner peace, contentment
    VO_VONG       = "vô vọng"          # Hopelessness


# ─── Cause Categories ─────────────────────────────────────

class CauseCategory(str, Enum):
    WORK          = "công việc"
    SLEEP         = "giấc ngủ"
    RELATIONSHIP  = "mối quan hệ"
    HEALTH        = "sức khỏe"
    FINANCE       = "tài chính"
    STUDY         = "học tập"
    FAMILY        = "gia đình"
    WEATHER       = "thời tiết"
    FOOD          = "ăn uống"
    SOCIAL        = "xã hội"
    SELF          = "bản thân"
    HOBBY         = "sở thích"
    UNKNOWN       = "không rõ"


# ─── Core Schema ──────────────────────────────────────────

@dataclass
class EmotionLabel:
    """Label cảm xúc chi tiết"""

    # Primary emotion (bắt buộc)
    primary: BasicEmotion

    # Secondary emotions (tùy chọn, có thể nhiều)
    secondary: List[BasicEmotion] = field(default_factory=list)

    # Vietnamese-specific emotion (nếu có)
    vietnamese_specific: Optional[VietnameseEmotion] = None

    # Intensity: 0.0 (rất nhẹ) → 1.0 (rất mạnh)
    intensity: float = 0.5

    # Valence: -1.0 (rất tiêu cực) → +1.0 (rất tích cực)
    valence: float = 0.0

    # Arousal: 0.0 (thụ động) → 1.0 (kích động)
    arousal: float = 0.5

    # Sentiment score tổng: 1.0 → 5.0
    sentiment_score: float = 3.0


@dataclass
class ContextInfo:
    """Thông tin ngữ cảnh"""

    # Nguyên nhân
    causes: List[CauseCategory] = field(default_factory=list)

    # Keywords trích xuất
    keywords: List[str] = field(default_factory=list)

    # Thời điểm trong ngày
    time_of_day: Optional[str] = None  # morning/afternoon/evening/night

    # Ngày trong tuần
    day_of_week: Optional[str] = None

    # Risk level
    risk_level: RiskLevel = RiskLevel.NONE

    # Trigger events (sự kiện kích hoạt)
    trigger_events: List[str] = field(default_factory=list)


@dataclass
class LabelInfo:
    """Thông tin về quá trình label"""

    # Auto label bằng Claude
    claude_label: Optional[dict] = None
    claude_model: Optional[str] = None  # claude-haiku/sonnet/opus

    # Auto label bằng ML model
    model_label: Optional[dict] = None
    model_name: Optional[str] = None    # phobert/xlm-roberta

    # Human verified
    human_verified: bool = False
    verified_by: Optional[str] = None   # annotator ID (anonymous)
    verified_at: Optional[str] = None

    # Confidence score
    confidence: float = 0.0

    # Agreement score (nếu nhiều annotator)
    inter_annotator_agreement: Optional[float] = None


@dataclass
class TextInfo:
    """Thông tin về text"""

    # Text gốc
    original: str

    # Text đã normalize
    normalized: Optional[str] = None

    # Ngôn ngữ
    language: Language = Language.VI

    # Style viết
    style: TextStyle = TextStyle.INFORMAL

    # Độ dài (số từ)
    word_count: int = 0

    # Có emoji không
    has_emoji: bool = False

    # Text gốc (nếu dịch từ ngôn ngữ khác)
    source_text: Optional[str] = None
    source_language: Optional[Language] = None
    translation_method: Optional[str] = None  # googletrans/helsinki/deepl


@dataclass
class MetaInfo:
    """Metadata"""

    # Source
    source: DataSource = DataSource.SELF_CREATED

    # Demographic
    age_group: AgeGroup = AgeGroup.UNKNOWN
    region: Optional[str] = None  # north/central/south Vietnam

    # Consent
    user_consent: bool = False
    anonymized: bool = True

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None

    # Dataset version
    version: str = "1.0.0"

    # Tags (tùy ý)
    tags: List[str] = field(default_factory=list)


@dataclass
class EmotionSample:
    """
    Schema chính — 1 sample trong dataset
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: TextInfo = field(default_factory=TextInfo)
    emotion: EmotionLabel = field(default_factory=EmotionLabel)
    context: ContextInfo = field(default_factory=ContextInfo)
    label_info: LabelInfo = field(default_factory=LabelInfo)
    meta: MetaInfo = field(default_factory=MetaInfo)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def to_training_row(self) -> dict:
        return {
            "id": self.id,
            "text": self.text.normalized or self.text.original,
            "primary_emotion": self.emotion.primary.value,
            "secondary_emotions": [e.value for e in self.emotion.secondary],
            "vietnamese_emotion": self.emotion.vietnamese_specific.value if self.emotion.vietnamese_specific else None,
            "intensity": self.emotion.intensity,
            "valence": self.emotion.valence,
            "arousal": self.emotion.arousal,
            "sentiment_score": self.emotion.sentiment_score,
            "causes": [c.value for c in self.context.causes],
            "risk_level": self.context.risk_level.value,
            "language": self.text.language.value,
            "source": self.meta.source.value,
            "confidence": self.label_info.confidence,
            "human_verified": self.label_info.human_verified,
        }