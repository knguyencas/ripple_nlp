from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.analyzer import load_model, predict
from app.services.nli import check_context          # ← thêm import này
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
import uvicorn

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Ripple NLP model...")
    ml['model'], ml['tokenizer'] = load_model()
    print("Model ready.")
    yield
    ml.clear()

app = FastAPI(title="Ripple NLP Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text quá ngắn")
    if len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Text quá dài")

    result = predict(req.text, ml['model'], ml['tokenizer'])

    nli_result = None
    if result['severity_id'] >= 3 or result['c9_ideation'] > 0.7:
        nli_result = check_context(req.text)

    return {**result, 'nli': nli_result}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)