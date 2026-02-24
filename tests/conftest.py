"""
Pytest configuration and shared fixtures for the Vehicle Maintenance Prediction test suite.
"""
import os
import sys
import pytest
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for memory tests."""
    return [
        {"role": "user", "content": "What is the recommended oil change interval?"},
        {"role": "assistant", "content": "Regular oil changes should be performed every 5,000 to 7,500 km or every 6 months."},
        {"role": "user", "content": "What about for synthetic oil?"},
        {"role": "assistant", "content": "Synthetic oil can extend the interval to 10,000 km."},
    ]


@pytest.fixture
def sample_vehicle_data():
    """Sample vehicle data for prediction tests."""
    return {
        "vehicle_model": "Car",
        "mileage": 75000,
        "vehicle_age": 5,
        "reported_issues": 3,
        "engine_size": 2000,
        "odometer_reading": 75000,
        "insurance_premium": 15000,
        "service_history": 5,
        "accident_history": 0,
        "fuel_efficiency": 15.0,
        "fuel_type": "Petrol",
        "transmission_type": "Automatic",
        "owner_type": "First",
        "tire_condition": "Good",
        "brake_condition": "Good",
        "battery_status": "Good",
        "maintenance_history": "Average",
        "days_since_last_service": 180,
        "warranty_remaining_days": 365,
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_chunks():
    """Sample text chunks for RAG testing."""
    return [
        "Regular oil changes should be performed every 5,000 to 7,500 km.",
        "Tire pressure should be checked monthly. Recommended pressure is 30-35 PSI.",
        "Brake pads typically need replacement every 40,000 to 70,000 km.",
        "Battery life averages 3-5 years. Test battery voltage regularly.",
        "Vehicle: Car, Age: 5 years, Mileage: 75000 km, Issues: 2, Risk: Medium.",
    ]


@pytest.fixture
def sample_instruction_data():
    """Sample instruction data for chunking tests."""
    return [
        {
            "instruction": "What maintenance is needed for high mileage vehicles?",
            "input": "",
            "output": "High mileage vehicles require more frequent oil changes and suspension checks."
        },
        {
            "instruction": "How often should I replace brake pads?",
            "input": "",
            "output": "Brake pads should be replaced every 40,000 to 70,000 km depending on driving conditions."
        },
    ]
