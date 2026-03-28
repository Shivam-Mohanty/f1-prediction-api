from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test to ensure the API is running and responsive."""
    response = client.get("/")
    assert response.status_code == 200
    # Adjust the assertion below based on what your root endpoint actually returns
    assert "status" in response.json() or response.json() != {}

def test_predict_endpoint():
    """Test to ensure the prediction endpoint returns a result for a valid race."""
    # Use a known past race to ensure data is available from FastF1
    response = client.get("/predict/2023/Bahrain Grand Prix")
    
    assert response.status_code == 200
    response_data = response.json()
    
    # Verify that the API returns a prediction list
    assert "prediction" in response_data
    assert len(response_data["prediction"]) > 0
    assert "driver" in response_data["prediction"][0]
    assert "win_probability" in response_data["prediction"][0]