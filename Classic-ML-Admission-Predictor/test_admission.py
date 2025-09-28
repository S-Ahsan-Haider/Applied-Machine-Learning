import pytest 
from admission import app

@pytest.fixture
def client():
    return app.test_client()

def test_home(client):
    r = client.get('/')
    assert r.status_code == 200
    assert r.text == "<p>Predictor is Online!</p>"

def test_predict(client):
    test_data = {
        "gre_score": 137,
        "toefl_score": 118,
        "univ_rating": 5,
        "sop": 4.0,
        "lor": 4.5,
        "cgpa": 9.65,
        "research": "Yes"
    }

    r = client.post('/predict', json = test_data)
    assert r.status_code == 200