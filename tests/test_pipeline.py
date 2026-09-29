import json

from src.predict import Predictor


def test_artifacts_written(model_dir):
    assert (model_dir / "crop_pipeline.joblib").exists()
    assert (model_dir / "fertilizer_pipeline.joblib").exists()
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert {"crop", "fertilizer"} <= set(meta)
    assert meta["crop"]["test_metrics"]["accuracy"] > 0.9
    assert set(meta["fertilizer"]["categories"]) == {"soil_type", "crop_type"}


def test_crop_prediction_is_top_k_sorted(model_dir):
    p = Predictor(model_dir)
    top = p.predict_crop({"N": 20, "P": 20, "K": 20, "temperature": 18, "humidity": 50, "ph": 5, "rainfall": 60})
    assert len(top) == 3
    probs = [t["probability"] for t in top]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0001
    assert top[0]["name"] == "rice"


def test_fertilizer_prediction_and_case_insensitive_categories(model_dir):
    p = Predictor(model_dir)
    row = dict(temperature=26, humidity=52, moisture=30, soil_type="loamy", crop_type="MAIZE",
               nitrogen=10, potassium=0, phosphorous=0)
    top = p.predict_fertilizer(row)
    assert top[0]["name"] == "Urea"


def test_unknown_category_rejected(model_dir):
    p = Predictor(model_dir)
    row = dict(temperature=26, humidity=52, moisture=30, soil_type="Moon dust", crop_type="Maize",
               nitrogen=10, potassium=0, phosphorous=0)
    try:
        p.predict_fertilizer(row)
    except ValueError as exc:
        assert "Allowed values" in str(exc)
    else:
        raise AssertionError("expected ValueError")
