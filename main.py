from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the project root to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.vehicle_ai import VehicleAI
from rag_pipeline.retriever import RAGRetriever

app = FastAPI(
    title="Vehicle Maintenance AI API",
    description="API for vehicle health predictions and maintenance recommendations",
    version="1.0.0"
)

# Initialize AI & RAG components
try:
    vehicle_ai = VehicleAI()
    rag_retriever = RAGRetriever()
except Exception as e:
    print(f"Warning: Failed to initialize AI components: {e}")
    vehicle_ai = None
    rag_retriever = None


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

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Endpoint for conversing with the Vehicle AI Agent."""
    if not vehicle_ai:
        raise HTTPException(status_code=503, detail="Vehicle AI is not initialized")
    
    try:
        response = vehicle_ai.ask(request.query)
        return {"response": response, "memory_length": len(vehicle_ai.memory)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    """Endpoint for retrieving maintenance guidelines via RAG."""
    if not rag_retriever:
        raise HTTPException(status_code=503, detail="RAG Retriever is not initialized")
        
    try:
        # Note: retriever.retrieve might print instead of returning cleanly,
        # but based on the code it returns a list of strings
        chunks = rag_retriever.retrieve(request.query, k=request.top_k)
        return {"query": request.query, "retrieved_chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def predict_endpoint(request: PredictRequest):
    """Placeholder endpoint for ML risk prediction."""
    # TODO: Load actual ML model and return real predictions
    
    # Dummy logic based on inputs
    risk_score = min(100.0, request.vehicle_age * 5 + request.mileage / 1000 + request.reported_issues * 10)
    recommendation = "Immediate Service Needed" if risk_score > 70 else "Routine Check"
    
    return {
        "vehicle_model": request.vehicle_model,
        "risk_score": round(risk_score, 2),
        "recommendation": recommendation,
        "note": "This is a placeholder prediction."
    }

