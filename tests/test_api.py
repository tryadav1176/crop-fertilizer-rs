import pytest
from fastapi.testclient import TestClient

from src.api import create_app

CROP_OK = {"N": 20, "P": 20, "K": 20, "temperature": 18, "humidity": 50, "ph": 5, "rainfall": 60}
FERT_OK = {"temperature": 26, "humidity": 52, "moisture": 30, "soil_type": "Loamy", "crop_type": "Maize",
           "nitrogen": 10, "potassium": 0, "phosphorous": 0}


@pytest.fixture()
def client(model_dir):
    with TestClient(create_app(model_dir)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] == {"crop": True, "fertilizer": True}


def test_options(client):
    body = client.get("/api/v1/options").json()
    assert "Loamy" in body["soil_type"] and "Maize" in body["crop_type"]


def test_predict_crop(client):
    r = client.post("/api/v1/predict/crop", json=CROP_OK)
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"] == body["top_k"][0]["name"]
    assert len(body["top_k"]) == 3
    assert "model_version" in body


def test_predict_fertilizer(client):
    r = client.post("/api/v1/predict/fertilizer", json=FERT_OK)
    assert r.status_code == 200
    assert r.json()["recommendation"] == "Urea"


def test_out_of_range_input_rejected(client):
    r = client.post("/api/v1/predict/crop", json={**CROP_OK, "ph": 20})
    assert r.status_code == 422


def test_missing_field_rejected(client):
    bad = dict(CROP_OK)
    bad.pop("rainfall")
    assert client.post("/api/v1/predict/crop", json=bad).status_code == 422


def test_unknown_soil_rejected(client):
    r = client.post("/api/v1/predict/fertilizer", json={**FERT_OK, "soil_type": "Moon dust"})
    assert r.status_code == 422
    assert "Allowed values" in r.json()["detail"]


def test_no_models_returns_503(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/api/v1/predict/crop", json=CROP_OK).status_code == 503
