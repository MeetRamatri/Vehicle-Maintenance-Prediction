from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
import math
import numpy as np
import pandas as pd
import joblib

# Add the project root to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.vehicle_ai import VehicleAI
from rag_pipeline.retriever import RAGRetriever

app = FastAPI(
    title="Vehicle Maintenance AI API",
    description="API for vehicle health predictions and maintenance recommendations",
    version="1.0.0"
)

# Initialize AI & RAG components (lazy - heavy loading deferred to first request)
vehicle_ai = VehicleAI()
rag_retriever = None  # Will be shared with VehicleAI on first RAG request

# Load ML model and feature columns
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
try:
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    print(f"ML model loaded successfully ({len(feature_columns)} features)")
except Exception as e:
    print(f"Warning: Failed to load ML model: {e}")
    xgb_model = None
    feature_columns = None


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
    vehicle_model: str = Field(..., description="One of: Car, Motorcycle, Suv, Truck, Van, Bus")
    mileage: float = Field(..., description="Total mileage in km")
    vehicle_age: int = Field(..., description="Age of vehicle in years")
    reported_issues: int = Field(..., description="Number of reported issues")
    engine_size: int = Field(default=2000, description="Engine size in cc")
    odometer_reading: float = Field(default=50000, description="Current odometer reading")
    insurance_premium: float = Field(default=15000, description="Annual insurance premium")
    service_history: int = Field(default=5, description="Number of past services")
    accident_history: int = Field(default=0, description="Number of past accidents")
    fuel_efficiency: float = Field(default=15.0, description="Fuel efficiency (km/l)")
    fuel_type: str = Field(default="Petrol", description="One of: Petrol, Diesel, Electric")
    transmission_type: str = Field(default="Automatic", description="One of: Automatic, Manual")
    owner_type: str = Field(default="First", description="One of: First, Second, Third")
    tire_condition: str = Field(default="Good", description="One of: New, Good, Worn Out")
    brake_condition: str = Field(default="Good", description="One of: New, Good, Worn Out")
    battery_status: str = Field(default="Good", description="One of: New, Good, Weak")
    maintenance_history: str = Field(default="Average", description="One of: Good, Average, Poor")
    days_since_last_service: int = Field(default=180, description="Days since last service")
    warranty_remaining_days: int = Field(default=365, description="Warranty remaining days")

# --- Endpoints ---

@app.get("/health")
def read_health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Endpoint for conversing with the Vehicle AI Agent."""
    try:
        response = vehicle_ai.ask(request.query)
        return {"response": response, "memory_length": len(vehicle_ai.memory)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    """Endpoint for retrieving maintenance guidelines via RAG."""
    global rag_retriever
    # Lazy-init: share the retriever from VehicleAI or create one
    if rag_retriever is None:
        if vehicle_ai and vehicle_ai.retriever:
            rag_retriever = vehicle_ai.retriever
        else:
            try:
                rag_retriever = RAGRetriever()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"RAG Retriever failed to init: {e}")

    try:
        chunks = rag_retriever.retrieve(request.query, k=request.top_k)
        return {"query": request.query, "retrieved_chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def predict_endpoint(request: PredictRequest):
    """ML-powered vehicle maintenance risk prediction using XGBoost."""
    if not xgb_model or not feature_columns:
        raise HTTPException(status_code=503, detail="ML model is not loaded")

    try:
        # Build engineered features matching training pipeline
        aging_factor = math.sqrt(request.mileage * request.vehicle_age) if request.mileage * request.vehicle_age >= 0 else 0
        mileage_age_interaction = request.mileage * request.vehicle_age
        issues_mileage_interaction = request.reported_issues * request.mileage

        # Build raw feature dict with numeric columns
        raw = {
            'Mileage': request.mileage,
            'Reported_Issues': request.reported_issues,
            'Vehicle_Age': request.vehicle_age,
            'Engine_Size': request.engine_size,
            'Odometer_Reading': request.odometer_reading,
            'Insurance_Premium': request.insurance_premium,
            'Service_History': request.service_history,
            'Accident_History': request.accident_history,
            'Fuel_Efficiency': request.fuel_efficiency,
            'aging_factor': aging_factor,
            'mileage_age_interaction': mileage_age_interaction,
            'issues_mileage_interaction': issues_mileage_interaction,
            'Days_Since_Last_Service': request.days_since_last_service,
            'Warranty_Remaining_Days': request.warranty_remaining_days,
        }

        # One-hot encode categorical features (match training pipeline's pd.get_dummies with drop_first=True)
        # Vehicle_Model: base = Bus → dummies for Car, Motorcycle, Suv, Truck, Van
        raw['Vehicle_Model_Car'] = 1 if request.vehicle_model == 'Car' else 0
        raw['Vehicle_Model_Motorcycle'] = 1 if request.vehicle_model == 'Motorcycle' else 0
        raw['Vehicle_Model_Suv'] = 1 if request.vehicle_model == 'Suv' else 0
        raw['Vehicle_Model_Truck'] = 1 if request.vehicle_model == 'Truck' else 0
        raw['Vehicle_Model_Van'] = 1 if request.vehicle_model == 'Van' else 0

        # Maintenance_History: base = Average → dummies for Good, Poor
        raw['Maintenance_History_Good'] = 1 if request.maintenance_history == 'Good' else 0
        raw['Maintenance_History_Poor'] = 1 if request.maintenance_history == 'Poor' else 0

        # Fuel_Type: base = Diesel → dummies for Electric, Petrol
        raw['Fuel_Type_Electric'] = 1 if request.fuel_type == 'Electric' else 0
        raw['Fuel_Type_Petrol'] = 1 if request.fuel_type == 'Petrol' else 0

        # Transmission_Type: base = Automatic → dummy for Manual
        raw['Transmission_Type_Manual'] = 1 if request.transmission_type == 'Manual' else 0

        # Owner_Type: base = First → dummies for Second, Third
        raw['Owner_Type_Second'] = 1 if request.owner_type == 'Second' else 0
        raw['Owner_Type_Third'] = 1 if request.owner_type == 'Third' else 0

        # Tire_Condition: base = Good → dummies for New, Worn Out
        raw['Tire_Condition_New'] = 1 if request.tire_condition == 'New' else 0
        raw['Tire_Condition_Worn Out'] = 1 if request.tire_condition == 'Worn Out' else 0

        # Brake_Condition: base = Good → dummies for New, Worn Out
        raw['Brake_Condition_New'] = 1 if request.brake_condition == 'New' else 0
        raw['Brake_Condition_Worn Out'] = 1 if request.brake_condition == 'Worn Out' else 0

        # Battery_Status: base = Good → dummies for New, Weak
        raw['Battery_Status_New'] = 1 if request.battery_status == 'New' else 0
        raw['Battery_Status_Weak'] = 1 if request.battery_status == 'Weak' else 0

        # Build DataFrame in the exact column order the model expects
        df = pd.DataFrame([raw])[feature_columns]

        # Predict
        prediction = int(xgb_model.predict(df)[0])
        probability = float(xgb_model.predict_proba(df)[0][1])
        risk_score = round(probability * 100, 2)

        if risk_score >= 80:
            recommendation = "Critical - Immediate Service Required"
        elif risk_score >= 60:
            recommendation = "High Risk - Schedule Service Soon"
        elif risk_score >= 40:
            recommendation = "Moderate Risk - Monitor Closely"
        elif risk_score >= 20:
            recommendation = "Low Risk - Routine Check Recommended"
        else:
            recommendation = "Healthy - No Immediate Action Needed"

        return {
            "vehicle_model": request.vehicle_model,
            "needs_maintenance": bool(prediction),
            "risk_score": risk_score,
            "probability": round(probability, 4),
            "recommendation": recommendation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

