# tests/test_registry.py
"""Teste la promotion par alias contre un vrai serveur MLflow local (SQLite)."""
import mlflow
import pytest
from mlflow import MlflowClient

from mlops_common.registry import promote_if_better, mark_rejected

MODEL = "p04-fraud-clf"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test")
    c = MlflowClient()
    c.create_registered_model(MODEL)
    return c


def _register(client, metric_value):
    """Crée un run avec test_pr_auc et une version de modèle liée à ce run."""
    with mlflow.start_run() as run:
        mlflow.log_metric("test_pr_auc", metric_value)
    client.create_model_version(MODEL, source=f"runs:/{run.info.run_id}/model",
                                run_id=run.info.run_id)
    return run.info.run_id


def _alias_version(client, alias):
    return str(client.get_model_version_by_alias(MODEL, alias).version)


def test_first_model_above_floor_becomes_champion(client):
    r1 = _register(client, 0.80)
    assert promote_if_better(MODEL, r1, "test_pr_auc", min_delta=0.005, floor=0.70) == "champion"
    assert _alias_version(client, "champion") == "1"


def test_below_floor_is_rejected_and_never_champion(client):
    r1 = _register(client, 0.60)
    assert promote_if_better(MODEL, r1, "test_pr_auc", min_delta=0.005, floor=0.70) == "rejected"
    assert _alias_version(client, "rejected") == "1"
    with pytest.raises(Exception):
        client.get_model_version_by_alias(MODEL, "champion")


def test_better_by_delta_replaces_champion(client):
    r1 = _register(client, 0.80)
    promote_if_better(MODEL, r1, "test_pr_auc", min_delta=0.005, floor=0.70)
    r2 = _register(client, 0.82)
    assert promote_if_better(MODEL, r2, "test_pr_auc", min_delta=0.005, floor=0.70) == "champion"
    assert _alias_version(client, "champion") == "2"


def test_not_better_enough_becomes_challenger(client):
    r1 = _register(client, 0.80)
    promote_if_better(MODEL, r1, "test_pr_auc", min_delta=0.005, floor=0.70)
    r2 = _register(client, 0.802)  # +0.002 < min_delta 0.005
    assert promote_if_better(MODEL, r2, "test_pr_auc", min_delta=0.005, floor=0.70) == "challenger"
    assert _alias_version(client, "champion") == "1"
    assert _alias_version(client, "challenger") == "2"


def test_rerun_on_current_champion_is_idempotent(client):
    r1 = _register(client, 0.80)
    promote_if_better(MODEL, r1, "test_pr_auc", floor=0.70)
    assert promote_if_better(MODEL, r1, "test_pr_auc", min_delta=0.005, floor=0.70) == "champion"


def test_unknown_run_raises_clear_error(client):
    with pytest.raises(ValueError, match="Aucune version"):
        mark_rejected(MODEL, "inexistant")
