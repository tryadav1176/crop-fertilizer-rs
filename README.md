# Crop & Fertilizer Recommendation

Top-3 crop and fertilizer recommendations (with confidence) from soil and weather inputs.
Scikit-learn `Pipeline` models served through a FastAPI API and a small web page.
The Crop and Fertilizer Recommendation System is a machine learning-based web application designed to help farmers and agricultural enthusiasts make informed decisions. The system suggests:
The best crop to cultivate based on soil and environmental conditions.
The most suitable fertilizer to maximize yield and maintain soil health.
This solution bridges the gap between modern data-driven insights and traditional farming, promoting sustainable agriculture.

```
src/
  config.py    paths, column names, constants
  train.py     CV model comparison -> hold-out evaluation -> saves pipeline + metadata.json
  predict.py   loads pipelines, returns top-k, validates categories
  schemas.py   request/response models with range validation
  api.py       FastAPI app (create_app factory)
static/        index.html front end (calls the API)
tests/         pytest suite (uses synthetic data, no CSVs needed)
dataset/       put your two CSVs here
models/        trained artifacts are written here
```

## 🛠️ Installation and Setup
### Clone the Repository:
Open a terminal or command prompt on your system.
Run the following command to clone the repository:
```bash
git clone https://github.com/tryadav1176/crop-fertilizer-rs.git
```
### Navigate to the cloned repository directory:
```bash
cd crop-fertilizer-rs
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Copy your CSVs into `dataset/` using these names:

- `Crop_recommendation.csv`
- `Fertilizer Prediction.csv`

## Train

```powershell
python -m src.train                 # both models
python -m src.train --target crop   # just one
```

This compares Decision Tree, Random Forest, Logistic Regression and Gradient Boosting with
cross-validation, evaluates the winner on a held-out 20% split (accuracy, macro-F1, top-3
accuracy, per-class report), then saves `models/*_pipeline.joblib` and `models/metadata.json`
(model name, metrics, feature importances, class list, scikit-learn version).

Because the scaler and encoders live inside the pipeline, the mismatch bugs from the old
version (missing scaler, hand-typed soil/crop codes) cannot happen. Soil and crop types are
sent as plain text (`"Loamy"`, `"Maize"`); the allowed values come from `GET /api/v1/options`.

## Run

```powershell
uvicorn src.api:app --reload
```

- Web page: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/predict/crop -H "Content-Type: application/json" `
  -d '{"N":90,"P":42,"K":43,"temperature":21,"humidity":82,"ph":6.5,"rainfall":203}'
```

## Test and lint

```powershell
pytest -q
ruff check src tests
```

## Docker

```powershell
python -m src.train
docker build -t crop-rs .
docker run -p 8000:8000 crop-rs
```

## Current project status

The Crop & Fertilizer Recommendation project is implemented and currently functioning as expected.
The repository includes a complete machine learning workflow for crop and fertilizer recommendation,
with a FastAPI-based service layer, a lightweight frontend, automated validation tests, and CI
configuration.

### Validation

I ran the repository test suite successfully:

- 12 passed
- 0 failed
- 11 warnings, all non-blocking dependency deprecations

### What is already in place

- Crop and fertilizer model training pipeline
- Saved model artifacts and metadata
- Prediction API endpoints
- Input validation and categorical restrictions
- Static web interface
- Automated pytest suite
- GitHub Actions CI configuration

### Overall assessment

- Functional status: Good
- Implementation status: Complete for its intended scope
- Production readiness: Strong for internal/demo use, with minor maintenance work recommended
  before broader production deployment

### Key considerations

- The repo contains a few non-blocking dependency deprecation warnings from FastAPI/Starlette and
  SciPy/sklearn.
- The fertilizer model dataset is relatively small, so its performance should be treated as indicative
  rather than definitive.
- Dependency pinning and upgrade management would improve reproducibility and long-term stability.

### Conclusion

This repository is a solid, working ML application with clear architecture and validated behavior.
It is ready for local use, demonstration, and internal evaluation. With minor dependency and
data-quality improvements, it would be well positioned for more robust deployment scenarios.

## Notes

- Load `.joblib` files only if you trained them yourself; they are Python pickles.
- Retrain whenever you change scikit-learn versions. The API logs a warning on a mismatch.
- The fertilizer dataset is small (about 99 rows). Its metrics are indicative, not proof.
- Recommendations are decision support. Confirm with a local agronomist.
