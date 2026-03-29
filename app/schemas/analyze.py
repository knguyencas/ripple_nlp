from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    text: str
    user_id: str | None = None

class AnalyzeResponse(BaseModel):
    severity:    str
    severity_id: int
    confidence:  float
    phq_score:   float
    dsm:         dict
    risk_flag:   bool
    c9_ideation: float