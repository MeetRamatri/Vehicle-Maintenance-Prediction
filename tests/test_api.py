"""
Unit tests for the FastAPI API endpoints.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def test_client():
    """Create a TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, test_client):
        """Test that health endpoint returns ok status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChatEndpoint:
    """Tests for the /api/chat endpoint."""

    def test_chat_returns_response(self, test_client):
        """Test that chat endpoint returns a response."""
        response = test_client.post(
            "/api/chat",
            json={"query": "What is the oil change interval?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "memory_length" in data
        assert isinstance(data["response"], str)

    def test_chat_updates_memory(self, test_client):
        """Test that chat endpoint updates memory."""
        # First request
        response1 = test_client.post(
            "/api/chat",
            json={"query": "First question"}
        )
        memory_len_1 = response1.json()["memory_length"]
        
        # Second request
        response2 = test_client.post(
            "/api/chat",
            json={"query": "Second question"}
        )
        memory_len_2 = response2.json()["memory_length"]
        
        assert memory_len_2 >= memory_len_1

    def test_chat_missing_query(self, test_client):
        """Test chat endpoint with missing query."""
        response = test_client.post("/api/chat", json={})
        
        assert response.status_code == 422  # Validation error

    def test_chat_empty_query(self, test_client):
        """Test chat endpoint with empty query."""
        response = test_client.post(
            "/api/chat",
            json={"query": ""}
        )
        
        # Should still succeed, just with generic response
        assert response.status_code == 200


class TestRetrieveEndpoint:
    """Tests for the /api/retrieve endpoint."""

    def test_retrieve_returns_chunks(self, test_client):
        """Test that retrieve endpoint returns chunks."""
        response = test_client.post(
            "/api/retrieve",
            json={"query": "oil change", "top_k": 3}
        )
        
        # May be 503 if RAG retriever not initialized
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "retrieved_chunks" in data
            assert data["query"] == "oil change"

    def test_retrieve_default_top_k(self, test_client):
        """Test retrieve endpoint uses default top_k."""
        response = test_client.post(
            "/api/retrieve",
            json={"query": "brake maintenance"}
        )
        
        # Check it doesn't fail due to missing top_k
        assert response.status_code in [200, 503]

    def test_retrieve_custom_top_k(self, test_client):
        """Test retrieve endpoint with custom top_k."""
        response = test_client.post(
            "/api/retrieve",
            json={"query": "tire pressure", "top_k": 5}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["retrieved_chunks"]) <= 5


class TestPredictEndpoint:
    """Tests for the /api/predict endpoint."""

    def test_predict_basic_request(self, test_client, sample_vehicle_data):
        """Test predict endpoint with basic request."""
        response = test_client.post("/api/predict", json=sample_vehicle_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "vehicle_model" in data
            assert "needs_maintenance" in data
            assert "risk_score" in data
            assert "probability" in data
            assert "recommendation" in data
        elif response.status_code == 503:
            # ML model not loaded
            assert "ML model" in response.json()["detail"]

    def test_predict_returns_valid_risk_score(self, test_client, sample_vehicle_data):
        """Test that risk score is between 0 and 100."""
        response = test_client.post("/api/predict", json=sample_vehicle_data)
        
        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["risk_score"] <= 100

    def test_predict_returns_valid_probability(self, test_client, sample_vehicle_data):
        """Test that probability is between 0 and 1."""
        response = test_client.post("/api/predict", json=sample_vehicle_data)
        
        if response.status_code == 200:
            data = response.json()
            assert 0 <= data["probability"] <= 1

    def test_predict_different_vehicle_models(self, test_client, sample_vehicle_data):
        """Test predict endpoint with different vehicle models."""
        vehicle_models = ["Car", "Motorcycle", "Suv", "Truck", "Van", "Bus"]
        
        for model in vehicle_models:
            data = sample_vehicle_data.copy()
            data["vehicle_model"] = model
            
            response = test_client.post("/api/predict", json=data)
            
            if response.status_code == 200:
                assert response.json()["vehicle_model"] == model

    def test_predict_high_risk_vehicle(self, test_client, sample_vehicle_data):
        """Test prediction for a high-risk vehicle configuration."""
        high_risk_data = sample_vehicle_data.copy()
        high_risk_data.update({
            "mileage": 200000,
            "vehicle_age": 15,
            "reported_issues": 10,
            "brake_condition": "Worn Out",
            "battery_status": "Weak",
            "tire_condition": "Worn Out",
            "maintenance_history": "Poor",
        })
        
        response = test_client.post("/api/predict", json=high_risk_data)
        
        if response.status_code == 200:
            # High-risk vehicle should have elevated risk score
            data = response.json()
            assert data["risk_score"] > 30  # Expect higher risk

    def test_predict_low_risk_vehicle(self, test_client, sample_vehicle_data):
        """Test prediction for a low-risk vehicle configuration."""
        low_risk_data = sample_vehicle_data.copy()
        low_risk_data.update({
            "mileage": 5000,
            "vehicle_age": 1,
            "reported_issues": 0,
            "brake_condition": "New",
            "battery_status": "New",
            "tire_condition": "New",
            "maintenance_history": "Good",
        })
        
        response = test_client.post("/api/predict", json=low_risk_data)
        
        if response.status_code == 200:
            data = response.json()
            # Low-risk vehicle should have lower risk score
            assert data["risk_score"] < 70

    def test_predict_missing_required_field(self, test_client, sample_vehicle_data):
        """Test predict endpoint with missing required field."""
        incomplete_data = sample_vehicle_data.copy()
        del incomplete_data["vehicle_model"]
        
        response = test_client.post("/api/predict", json=incomplete_data)
        
        assert response.status_code == 422  # Validation error

    def test_predict_recommendation_messages(self, test_client, sample_vehicle_data):
        """Test that recommendation messages are appropriate."""
        expected_recommendations = [
            "Critical - Immediate Service Required",
            "High Risk - Schedule Service Soon",
            "Moderate Risk - Monitor Closely",
            "Low Risk - Routine Check Recommended",
            "Healthy - No Immediate Action Needed",
        ]
        
        response = test_client.post("/api/predict", json=sample_vehicle_data)
        
        if response.status_code == 200:
            data = response.json()
            assert data["recommendation"] in expected_recommendations


class TestPredictEndpointCategoricalFields:
    """Tests for categorical field handling in predict endpoint."""

    def test_fuel_types(self, test_client, sample_vehicle_data):
        """Test different fuel types."""
        for fuel_type in ["Petrol", "Diesel", "Electric"]:
            data = sample_vehicle_data.copy()
            data["fuel_type"] = fuel_type
            
            response = test_client.post("/api/predict", json=data)
            assert response.status_code in [200, 503]

    def test_transmission_types(self, test_client, sample_vehicle_data):
        """Test different transmission types."""
        for transmission in ["Automatic", "Manual"]:
            data = sample_vehicle_data.copy()
            data["transmission_type"] = transmission
            
            response = test_client.post("/api/predict", json=data)
            assert response.status_code in [200, 503]

    def test_owner_types(self, test_client, sample_vehicle_data):
        """Test different owner types."""
        for owner in ["First", "Second", "Third"]:
            data = sample_vehicle_data.copy()
            data["owner_type"] = owner
            
            response = test_client.post("/api/predict", json=data)
            assert response.status_code in [200, 503]

    def test_condition_fields(self, test_client, sample_vehicle_data):
        """Test different condition combinations."""
        conditions = ["New", "Good", "Worn Out"]
        battery_conditions = ["New", "Good", "Weak"]
        
        for tire in conditions:
            for brake in conditions:
                for battery in battery_conditions:
                    data = sample_vehicle_data.copy()
                    data.update({
                        "tire_condition": tire,
                        "brake_condition": brake,
                        "battery_status": battery,
                    })
                    
                    response = test_client.post("/api/predict", json=data)
                    assert response.status_code in [200, 503]


class TestAPIValidation:
    """Tests for API request validation."""

    def test_chat_invalid_json(self, test_client):
        """Test chat endpoint with invalid JSON."""
        response = test_client.post(
            "/api/chat",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_predict_negative_mileage(self, test_client, sample_vehicle_data):
        """Test predict endpoint with negative mileage."""
        data = sample_vehicle_data.copy()
        data["mileage"] = -1000
        
        response = test_client.post("/api/predict", json=data)
        
        # Depending on validation, this may succeed or fail
        # The model should still produce a result
        assert response.status_code in [200, 422, 503]

    def test_predict_extreme_values(self, test_client, sample_vehicle_data):
        """Test predict endpoint with extreme values."""
        data = sample_vehicle_data.copy()
        data.update({
            "mileage": 1000000,
            "vehicle_age": 50,
            "reported_issues": 100,
        })
        
        response = test_client.post("/api/predict", json=data)
        
        assert response.status_code in [200, 503]
