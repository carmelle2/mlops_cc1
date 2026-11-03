# tests/test_p04_training.py
"""Tests de l'étape 3 : seuil, stratégies d'entraînement, modèle pyfunc (MLflow SQLite local)."""
import os
import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from pandas.testing import assert_frame_equal

from mlops_common.registry import promote_if_better
from projects.p04_fraud import training
from projects.p04_fraud.data import FEATURES, TARGET, temporal_split
from projects.p04_fraud.model import ThresholdedModel
from projects.p04_fraud.training import (STRATEGIES, log_final, pick_threshold, train, xy)

MODEL = "p04-fraud-clf"
FAST = {"n_estimators": 40}  # tests rapides


# ---------- données et environnement ----------
def make_fraud_data(n=8000, seed=0):
    """Jeu synthétique (même schéma que creditcard.csv) où la fraude dépend de V1 et V2 (~5 %)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(rng.integers(0, 172_800, size=n)).astype(float)
    df["Amount"] = rng.exponential(50, size=n)
    score = df["V1"] + df["V2"] + rng.normal(scale=0.3, size=n)
    df[TARGET] = (score >= np.quantile(score, 0.95)).astype("int8")
    return df[FEATURES + [TARGET]]


@pytest.fixture(scope="module")
def splits():
    tr, va, te = temporal_split(make_fraud_data())
    return xy(tr) + xy(va) + xy(te)  # X_tr, y_tr, X_va, y_va, X_te, y_te


@pytest.fixture()
def mlflow_env(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    exp_id = mlflow.create_experiment("test-p04", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    return MlflowClient(), uri


# ---------- pick_threshold ----------
def _recall_precision(y, proba, t):
    pred = proba >= t
    tp = int((pred & (y == 1)).sum())
    return tp / (y == 1).sum(), (tp / pred.sum() if pred.any() else 0.0)


def test_pick_threshold_maximizes_recall_under_precision_constraint():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    proba = np.clip(0.55 * y + rng.normal(0.25, 0.15, size=2000), 0, 1)
    thr, met = pick_threshold(y, proba, min_precision=0.90)
    assert met is True
    recall, precision = _recall_precision(y, proba, thr)
    assert precision >= 0.90
    # Recherche exhaustive : aucun autre seuil respectant la contrainte n'a un meilleur rappel.
    best = max(r for r, p in (_recall_precision(y, proba, t) for t in np.unique(proba)) if p >= 0.90)
    assert recall == pytest.approx(best)


def test_pick_threshold_warns_and_flags_when_constraint_unreachable():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.9, 0.8, 0.7, 0.2, 0.1])  # les fraudes ont les scores les plus bas
    with pytest.warns(UserWarning, match="Aucun seuil"):
        thr, met = pick_threshold(y, proba, min_precision=0.90)
    assert (thr, met) == (0.5, False)


def test_pick_threshold_rejects_validation_without_fraud():
    with pytest.raises(ValueError, match="aucune fraude"):
        pick_threshold(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


# ---------- xy ----------
def test_xy_keeps_time_and_excludes_target():
    X, y = xy(make_fraud_data(n=50))
    assert list(X.columns) == FEATURES and "Time" in X.columns and TARGET not in X.columns
    assert y.name == TARGET


# ---------- train ----------
def test_train_rejects_unknown_strategy(splits, mlflow_env):
    X_tr, y_tr, X_va, y_va, *_ = splits
    with mlflow.start_run():
        with pytest.raises(ValueError, match="Stratégie inconnue"):
            train("undersampling", X_tr, y_tr, X_va, y_va)


def test_train_requires_active_run(splits, mlflow_env):
    X_tr, y_tr, X_va, y_va, *_ = splits
    with pytest.raises(RuntimeError, match="run MLflow actif"):
        train("class_weight", X_tr, y_tr, X_va, y_va, params_override=FAST)


def test_class_weight_uses_scale_pos_weight_and_never_smote(splits, mlflow_env, monkeypatch):
    X_tr, y_tr, X_va, y_va, *_ = splits

    class ForbiddenSMOTE:
        def __init__(self, *a, **k):
            raise AssertionError("SMOTE ne doit pas être appelé avec class_weight")

    monkeypatch.setattr(training, "SMOTE", ForbiddenSMOTE)
    client, _ = mlflow_env
    with mlflow.start_run() as run:
        res = train("class_weight", X_tr, y_tr, X_va, y_va, params_override=FAST)
    expected = (y_tr == 0).sum() / (y_tr == 1).sum()
    assert res.model.get_params()["scale_pos_weight"] == pytest.approx(expected)
    params = client.get_run(run.info.run_id).data.params
    assert int(params["fit_rows"]) == len(y_tr)  # aucune ligne ajoutée


def test_smote_resamples_the_training_set_only(splits, mlflow_env, monkeypatch):
    X_tr, y_tr, X_va, y_va, *_ = splits
    X_va_before, y_va_before = X_va.copy(), y_va.copy()
    X_tr_before = X_tr.copy()
    seen = {"smote": [], "fit": []}

    class SpySMOTE(training.SMOTE):
        def fit_resample(self, X, y):
            seen["smote"].append(len(y))
            return super().fit_resample(X, y)

    class SpyLGBM(training.LGBMClassifier):
        def fit(self, X, y, *a, **k):
            seen["fit"].append((len(y), int((y == 1).sum()), int((y == 0).sum())))
            return super().fit(X, y, *a, **k)

    monkeypatch.setattr(training, "SMOTE", SpySMOTE)
    monkeypatch.setattr(training, "LGBMClassifier", SpyLGBM)
    client, _ = mlflow_env
    with mlflow.start_run() as run:
        train("smote", X_tr, y_tr, X_va, y_va, params_override=FAST)

    assert seen["smote"] == [len(y_tr)]  # appelé UNE fois, sur le train uniquement
    rows, pos, neg = seen["fit"][0]
    assert rows > len(y_tr) and pos / neg == pytest.approx(0.1, abs=0.01)
    assert_frame_equal(X_va, X_va_before)  # validation intacte
    assert_frame_equal(X_tr, X_tr_before)  # le train d'origine n'est pas modifié non plus
    assert y_va.equals(y_va_before)
    assert int(client.get_run(run.info.run_id).data.params["fit_rows"]) == rows


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_train_logs_params_metrics_and_figure(splits, mlflow_env, strategy):
    X_tr, y_tr, X_va, y_va, *_ = splits
    client, _ = mlflow_env
    with mlflow.start_run() as run:
        res = train(strategy, X_tr, y_tr, X_va, y_va, min_precision=0.90, params_override=FAST)
    data = client.get_run(run.info.run_id).data
    assert data.params["strategy"] == strategy
    assert data.params["subsample_freq"] == "1"  # sinon subsample est ignoré par LightGBM
    assert data.params["threshold_constraint_met"] == "True"
    assert float(data.params["threshold"]) == pytest.approx(res.threshold)
    assert data.metrics["val_pr_auc"] == pytest.approx(res.val_pr_auc)
    assert res.val_pr_auc > 0.5  # le signal V1 + V2 est appris
    assert data.metrics["val_precision"] >= 0.90  # la contrainte tient sur la validation
    assert any(a.path == "val_pr_curve.png" for a in client.list_artifacts(run.info.run_id))


# ---------- ThresholdedModel ----------
def test_thresholded_model_rejects_invalid_threshold():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="seuil"):
            ThresholdedModel(bad)


def test_to_frame_applies_threshold_inclusively_with_stable_dtypes():
    out = ThresholdedModel.to_frame([0.1, 0.5, 0.9], 0.5)
    assert out["is_fraud"].tolist() == [0, 1, 1]  # proba == seuil -> fraude (>=)
    assert out["fraud_proba"].dtype == "float64" and out["is_fraud"].dtype == "int64"


# ---------- log_final : aller-retour MLflow ----------
@pytest.fixture()
def logged_model(splits, mlflow_env):
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits
    with mlflow.start_run() as run:
        res = train("class_weight", X_tr, y_tr, X_va, y_va, params_override=FAST)
        test_pr_auc = log_final(res.model, res.threshold, X_te, y_te, MODEL)
    return run.info.run_id, res, test_pr_auc


def test_log_final_roundtrip_predictions_match(splits, mlflow_env, logged_model):
    *_, X_te, y_te = splits
    run_id, res, _ = logged_model
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    out = loaded.predict(X_te)
    assert list(out.columns) == ["fraud_proba", "is_fraud"]
    np.testing.assert_allclose(out["fraud_proba"], res.model.predict_proba(X_te)[:, 1])
    assert (out["is_fraud"] == (out["fraud_proba"] >= res.threshold).astype(int)).all()
    assert loaded.predict(X_te[list(reversed(FEATURES))]).equals(out)  # ordre des colonnes sans effet


def test_log_final_saves_signature_and_input_example(mlflow_env, logged_model):
    run_id, *_ = logged_model
    info = mlflow.models.Model.load(f"runs:/{run_id}/model")
    assert info.signature is not None
    assert [c.name for c in info.signature.inputs.inputs] == FEATURES
    assert [c.name for c in info.signature.outputs.inputs] == ["fraud_proba", "is_fraud"]
    assert info.saved_input_example_info is not None


def test_log_final_metrics_figure_and_registry_link(mlflow_env, logged_model):
    client, _ = mlflow_env
    run_id, _, test_pr_auc = logged_model
    data = client.get_run(run_id).data
    assert data.metrics["test_pr_auc"] == pytest.approx(test_pr_auc)
    assert {"test_precision", "test_recall", "val_pr_auc"} <= set(data.metrics)
    assert any(a.path == "test_pr_curve.png" for a in client.list_artifacts(run_id))
    # La version enregistrée est liée au run qui porte test_pr_auc : le gate peut donc la juger.
    assert [v.run_id for v in client.search_model_versions(f"name='{MODEL}'")] == [run_id]
    assert promote_if_better(MODEL, run_id, "test_pr_auc", min_delta=0.005, floor=0.0) == "champion"


def test_log_final_requires_active_run(splits, mlflow_env):
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits
    with mlflow.start_run():
        res = train("class_weight", X_tr, y_tr, X_va, y_va, params_override=FAST)
    with pytest.raises(RuntimeError, match="run MLflow actif"):
        log_final(res.model, res.threshold, X_te, y_te, MODEL)


def test_pinned_requirements_are_exact_versions():
    reqs = training._pinned_requirements()
    assert all("==" in r for r in reqs) and any(r.startswith("lightgbm==") for r in reqs)


def test_model_reloads_without_the_repository(splits, mlflow_env, logged_model, tmp_path):
    """Le modèle embarque son code (code_paths) : il se recharge dans un processus où le
    dépôt n'est PAS importable, comme sur un serveur de modèles."""
    *_, X_te, _ = splits
    _, uri = mlflow_env
    run_id, res, _ = logged_model
    X_te.head(5).to_csv(tmp_path / "x.csv", index=False)
    script = (
        "import sys, mlflow, pandas as pd\n"
        f"mlflow.set_tracking_uri({uri!r})\n"
        f"m = mlflow.pyfunc.load_model('runs:/{run_id}/model')\n"
        f"print('COLS', list(m.predict(pd.read_csv({str(tmp_path / 'x.csv')!r})).columns))\n"
        "print('MODULE', sys.modules['projects.p04_fraud.model'].__file__)\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc = subprocess.run([sys.executable, "-c", script], cwd=elsewhere, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "COLS ['fraud_proba', 'is_fraud']" in proc.stdout
    module_line = [l for l in proc.stdout.splitlines() if l.startswith("MODULE")][0]
    # Chargé depuis le code EMBARQUÉ dans le modèle (dossier `code/`), pas depuis le dépôt.
    # Selon l'OS, MLflow lit les artefacts sur place ou les télécharge dans un dossier temporaire :
    # on ne teste donc que la fin du chemin (normalisée, `\\` -> `/`).
    assert "/code/projects/p04_fraud/model.py" in module_line.replace("\\", "/")