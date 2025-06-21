from pydantic import BaseModel

class JobInput(BaseModel):
    title: str
    description: str

class PredictionOutput(BaseModel):
    fraudulent: bool
    confidence: float
