import json

import numpy as np
import pytest

from anomalias import model_store
from anomalias.data import FEATURES, STDS, generate_dataset
from anomalias.train import main, predict_labels, score_and_label, train

MIN_F1 = 0.80
MIN_AUC = 0.90


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quality_above_minimum(seed):
    _, meta = train(n_samples=3000, contamination=0.05, seed=seed)
    assert meta["metrics"]["f1"] >= MIN_F1, meta["metrics"]
    assert meta["metrics"]["roc_auc"] >= MIN_AUC, meta["metrics"]


def test_metadata_has_reference_stats():
    _, meta = train(n_samples=1000, seed=0, n_estimators=50)
    stats = meta["reference_stats"]
    assert list(stats) == FEATURES
    assert all(s["std"] > 0 for s in stats.values())
    assert meta["data"]["synthetic"] is True
    assert meta["reference_criterion"] == "predicted_normal"
    assert 0 < meta["reference_n_samples"] < meta["data"]["n_train"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_std_close_to_generator(seed):
    # Sem as anomalias, o desvio de referência deve ficar perto do gerador
    _, meta = train(n_samples=5000, contamination=0.05, seed=seed)
    ref_std = np.array([meta["reference_stats"][f]["std"] for f in FEATURES])
    np.testing.assert_allclose(ref_std, STDS, rtol=0.15)


def test_cli_saves_versioned_model(tmp_path, capsys):
    args = ["--n-samples", "1000", "--n-estimators", "50", "--models-dir", str(tmp_path)]
    assert main(args) == 1
    assert main(args) == 2

    out = capsys.readouterr().out
    assert '"f1"' in out
    assert '"roc_auc"' in out
    _, meta = model_store.load_current(tmp_path)
    assert meta["version"] == 2
    assert json.loads((tmp_path / "model_v1.json").read_text())["hyperparams"]["n_estimators"] == 50


def test_score_and_label_matches_predict_and_score_samples():
    model, _ = train(n_samples=2000, seed=42)
    X, _ = generate_dataset(1000, 0.05, seed=7)
    scores, labels = score_and_label(model, X)
    np.testing.assert_array_equal(labels, predict_labels(model, X))
    np.testing.assert_array_equal(scores, -model.score_samples(X))
    assert 0 < labels.sum() < len(X)
