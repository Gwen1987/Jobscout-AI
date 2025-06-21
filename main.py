from fastapi import FastAPI
from app.predict import predict_fraud
from app.schemas import JobInput, PredictionOutput

app = FastAPI(title="JobScout AI", description="Detect fraudulent job listings")

@app.post("/predict", response_model=PredictionOutput)
def predict(job: JobInput):
    score = predict_fraud(job.title, job.description)
    return PredictionOutput(
        fraudulent=score > 0.5,
        confidence=round(score, 4)
    )