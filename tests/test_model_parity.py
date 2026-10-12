"""The deployed model (deploy_data/model.txt, loaded as a LightGBM Booster) must reproduce the
predictions saved from the trained pickle, on the log1p -> expm1 scale the dashboard uses."""
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy_data"
PICKLE = ROOT / "models" / "lightgbm_model.pkl"

pytestmark = pytest.mark.skipif(not (DEPLOY / "model.txt").exists(), reason="deploy_data/ not present")


@pytest.fixture(scope="module")
def sample():
    preds = pd.read_parquet(DEPLOY / "predictions.parquet")
    return preds.sample(5000, random_state=0)


@pytest.fixture(scope="module")
def booster():
    return lgb.Booster(model_file=str(DEPLOY / "model.txt"))


def test_booster_has_the_expected_feature_names(booster):
    assert booster.feature_name() == FEATURE_COLS


def test_booster_reproduces_saved_predictions(booster, sample):
    pred = np.expm1(booster.predict(sample[FEATURE_COLS], num_threads=1)).clip(0)
    np.testing.assert_allclose(pred, sample["Predicted"].to_numpy(), rtol=1e-6)


@pytest.mark.skipif(not PICKLE.exists(), reason="models/lightgbm_model.pkl is gitignored; run day2_model.py")
def test_booster_matches_the_pickled_model(booster, sample):
    with open(PICKLE, "rb") as f:
        model = pickle.load(f)
    np.testing.assert_allclose(
        booster.predict(sample[FEATURE_COLS], num_threads=1), model.predict(sample[FEATURE_COLS]), rtol=1e-9
    )
