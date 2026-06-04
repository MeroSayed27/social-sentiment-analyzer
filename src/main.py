from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import os
from .config import settings


# Initialize FastAPI App using settings configuration
app = FastAPI(
    title=settings.APP_TITLE,
    version="1.0.0",
    description="Production-grade Arabic-English text sentiment classifier."
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── PYDANTIC DATA MODEL VALIDATION ───
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    probabilities: dict

# Global holders for model artifacts
model = None
vectorizer = None

# Lifespan startup event validation
@app.on_event("startup")
def load_artifacts():
    global model, vectorizer
    if not os.path.exists(settings.MODEL_PATH) or not os.path.exists(settings.VECTORIZER_PATH):
        raise RuntimeError("Machine Learning artifacts missing. Run the training script first!")
    
    model = joblib.load(settings.MODEL_PATH)
    vectorizer = joblib.load(settings.VECTORIZER_PATH)
    print("✨ Global model and vectorizer loaded into memory smoothly!")

@app.get("/")
async def root():
    return {"status": "healthy", "project": settings.APP_TITLE}

# Async prediction route feeding confidence percentages to our UI
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        transformed_text = vectorizer.transform([request.text])
        prediction = model.predict(transformed_text)[0]
        probs = model.predict_proba(transformed_text)[0]
        
        classes = model.classes_
        prob_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
        
        return SentimentResponse(
            text=request.text,
            sentiment=prediction,
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

# Interactive User Interface Dashboard Route
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multilingual Sentiment Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
            .card { border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
            .badge-positive { background-color: #d1e7dd; color: #0f5132; }
            .badge-negative { background-color: #f8d7da; color: #842029; }
        </style>
    </head>
    <body>
        <div class="container py-5">
            <div class="row justify-content-center">
                <div class="col-md-7">
                    <div class="text-center mb-4">
                        <h2 class="fw-bold text-dark">Arabic-English Sentiment Analyzer</h2>
                        <p class="text-muted">Enter social text to test the active inference pipeline model.</p>
                    </div>
                    <div class="card p-4 mb-4">
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Input Text / النص المدخل</label>
                            <textarea id="textInput" class="form-control" rows="3" placeholder="Type something here... / اكتب شيئاً هنا..."></textarea>
                        </div>
                        <button onclick="analyzeSentiment()" class="btn btn-primary w-100 fw-bold py-2">Analyze Sentiment</button>
                    </div>
                    <div id="resultCard" class="card p-4 d-none">
                        <h5 class="fw-bold mb-3">Analysis Report</h5>
                        <div class="d-flex align-items-center mb-3">
                            <span class="me-2 fw-semibold">Predicted Classification:</span>
                            <span id="sentimentLabel" class="badge px-3 py-2 fs-6"></span>
                        </div>
                        <h6 class="fw-semibold text-muted mb-2">Confidence Matrix Distributions:</h6>
                        <div id="probList" class="small"></div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            async function analyzeSentiment() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) return alert('Please enter valid text.');
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    const data = await response.json();
                    
                    document.getElementById('resultCard').classList.remove('d-none');
                    const label = document.getElementById('sentimentLabel');
                    label.innerText = data.sentiment.toUpperCase();
                    
                    if(data.sentiment === 'positive') {
                        label.className = 'badge px-3 py-2 fs-6 badge-positive';
                    } else {
                        label.className = 'badge px-3 py-2 fs-6 badge-negative';
                    }
                    
                    let probHtml = '';
                    for (const [key, val] of Object.entries(data.probabilities)) {
                        probHtml += `<div class='mb-1'><strong>${key}:</strong> ${(val * 100).toFixed(2)}%</div>`;
                    }
                    document.getElementById('probList').innerHTML = probHtml;
                } catch (err) {
                    alert('API Gateway communications failure.');
                }
            }
        </script>
    </body>
    </html>
    """