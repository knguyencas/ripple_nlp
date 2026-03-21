from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas.analyze import AnalyzeRequest, AnalyzeResponse
from .services.analyzer import analyzer

app = FastAPI(
    title="Ripple NLP Service",
    description="NLP microservice for Ripple app",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ripple-nlp"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text quá ngắn")

    try:
        result = analyzer.analyze(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
def analyze_batch(texts: list[str]):
    results = []
    for text in texts[:20]:
        try:
            result = analyzer.analyze(text)
            results.append(result)
        except:
            results.append(None)
    return results