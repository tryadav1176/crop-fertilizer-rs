# Crop & Fertilizer Recommendation

Top-3 crop and fertilizer recommendations (with confidence) from soil and weather inputs.
Scikit-learn `Pipeline` models served through a FastAPI API and a small web page.

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

## Notes

- Load `.joblib` files only if you trained them yourself; they are Python pickles.
- Retrain whenever you change scikit-learn versions. The API logs a warning on a mismatch.
- The fertilizer dataset is small (about 99 rows). Its metrics are indicative, not proof.
- Recommendations are decision support. Confirm with a local agronomist.
