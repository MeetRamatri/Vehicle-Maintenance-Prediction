from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Vehicle Maintenance AI API",
    description="API for vehicle health predictions and maintenance recommendations",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---
class ChatRequest(BaseModel):
    query: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3

class PredictRequest(BaseModel):
    vehicle_model: str
    mileage: float
    vehicle_age: float
    reported_issues: int

# --- Endpoints ---

@app.get("/health")
def read_health():
    """Health check endpoint."""
    return {"status": "ok"}
