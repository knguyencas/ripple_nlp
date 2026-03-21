from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    text: str
    language: Optional[str] = "vi"

class AnalyzeResponse(BaseModel):
    sentiment_score: float
    sentiment_label: str
    emotion: str 
    emotion_index: int 
    causes: List[str]
    keywords: List[str]
    risk_level: str 
    confidence: float

class WeeklyRequest(BaseModel):
    logs: List[dict]

class WeeklyResponse(BaseModel):
    dominant_emotion: str
    avg_score: float
    trend: str 
    pattern: str
    recommendation: str