import pytest

from src.train import train_crop, train_fertilizer
from tests.synthetic import make_crop_csv, make_fertilizer_csv


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    crop_csv, fert_csv = root / "crop.csv", root / "fert.csv"
    make_crop_csv(crop_csv)
    make_fertilizer_csv(fert_csv)
    out = root / "models"
    train_crop(crop_csv, out)
    train_fertilizer(fert_csv, out)
    return out
