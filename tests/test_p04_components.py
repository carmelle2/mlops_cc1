# tests/test_p04_components.py
"""Tests de l'étape 4 : on exécute le code réel de chaque composant (`.python_func`, sans cluster,
sur MLflow SQLite local), puis on vérifie que chacun se compile en spécification KFP."""
import inspect
import json
from types import SimpleNamespace

import joblib
import mlflow
import pandas as pd
import pytest
from kfp import compiler, dsl
from mlflow import MlflowClient

from mlops_common import tracking
from projects.p04_fraud import components as C
from projects.p04_fraud.data import TARGET, TIME
from tests.test_p04_training import make_fraud_data

MODEL = "p04-fraud-clf"
EXPERIMENT = "test-p04-components"
PROJECT = "p04-fraud"
COMPONENTS = ["ingest_validate", "temporal_split", "train_strategy",
              "pick_and_register", "gate", "mark_rejected"]


def _ds(tmp, name):
    return dsl.Dataset(name=name, uri=str(tmp / f"{name}.parquet"))


def _model(tmp, name):
    return dsl.Model(name=name, uri=str(tmp / f"{name}.joblib"))


def _train_both(tmp, tr, va, summary, dataset_version="v-test", tag=""):
    """Exécute `train_strategy` pour les deux stratégies ; renvoie (result_a, result_b, model_a, model_b)."""
    out = []
    for strategy in ("class_weight", "smote"):
        m = _model(tmp, f"model_{strategy}{tag}")
        r = C.train_strategy.python_func(
            strategy=strategy, train=tr, val=va, model_out=m, experiment=EXPERIMENT,
            project=PROJECT, splits_summary=summary, kfp_run_id="kfp-run-123",
            dataset_name="synthetic-fraud", dataset_version=dataset_version, n_jobs=2)
        out.append((r.result, m))
    return out[0][0], out[1][0], out[0][1], out[1][1]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """MLflow SQLite local + expérience dont les artefacts vont dans le dossier temporaire."""
    tmp = tmp_path_factory.mktemp("p04")
    uri = f"sqlite:///{tmp.as_posix()}/mlflow.db"
    mp = pytest.MonkeyPatch()
    mp.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp / "art").as_uri())
    yield SimpleNamespace(tmp=tmp, client=MlflowClient())
    mp.undo()


@pytest.fixture(scope="module")
def chain(env):
    """Enchaîne ingest -> split -> train x2 -> pick_and_register, comme le fera le pipeline."""
    tmp = env.tmp
    df = make_fraud_data(n=6000)
    csv = tmp / "creditcard.csv"
    df.to_csv(csv, index=False)
    frauds = int(df[TARGET].sum())

    dataset, tr, va, te = (_ds(tmp, n) for n in ("dataset", "train", "val", "test"))
    ing = C.ingest_validate.python_func(csv_path=str(csv), expected_rows=len(df),
                                        expected_frauds=frauds, dataset=dataset)
    split = C.temporal_split.python_func(dataset=dataset, train=tr, val=va, test=te)
    ra, rb, ma, mb = _train_both(tmp, tr, va, split.splits_summary, dataset_version=ing.dataset_sha256)
    best = C.pick_and_register.python_func(
        result_a=ra, result_b=rb, model_a=ma, model_b=mb, test=te, experiment=EXPERIMENT,
        project=PROJECT, model_name=MODEL, floor=0.0)
    return SimpleNamespace(df=df, csv=csv, frauds=frauds, dataset=dataset, tr=tr, va=va, te=te,
                           ing=ing, split=split, ra=json.loads(ra), rb=json.loads(rb),
                           ma=ma, mb=mb, best=best)


# ---------- ingest_validate ----------
def test_ingest_validate_outputs(chain):
    assert chain.ing.rows == len(chain.df) and chain.ing.frauds == chain.frauds
    assert chain.ing.dataset_sha256 == tracking.sha256_file(chain.csv)
    out = pd.read_parquet(chain.dataset.path)
    assert len(out) == len(chain.df) and out[TIME].is_monotonic_increasing


def test_ingest_validate_rejects_wrong_dataset(tmp_path):
    df = make_fraud_data(n=300)
    csv = tmp_path / "cc.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="fraudes"):
        C.ingest_validate.python_func(csv_path=str(csv), expected_rows=len(df),
                                      expected_frauds=int(df[TARGET].sum()) + 1,
                                      dataset=_ds(tmp_path, "dataset"))


# ---------- temporal_split ----------
def test_temporal_split_outputs(chain):
    tr, va, te = (pd.read_parquet(a.path) for a in (chain.tr, chain.va, chain.te))
    assert len(tr) + len(va) + len(te) == len(chain.df)
    assert tr[TIME].max() < va[TIME].min() and va[TIME].max() < te[TIME].min()
    summary = json.loads(chain.split.splits_summary)
    assert set(summary) == {"train", "val", "test"}
    assert sum(s["frauds"] for s in summary.values()) == chain.frauds


# ---------- train_strategy ----------
def test_train_strategy_never_receives_the_test_set():
    assert "test" not in inspect.signature(C.train_strategy.python_func).parameters


@pytest.mark.parametrize("which,strategy", [("ra", "class_weight"), ("rb", "smote")])
def test_train_strategy_run_is_fully_traced(env, chain, which, strategy):
    res = getattr(chain, which)
    run = env.client.get_run(res["run_id"])
    assert res["strategy"] == strategy and run.info.status == "FINISHED"
    tags = run.data.tags
    assert tags["project"] == PROJECT and tags["strategy"] == strategy
    assert tags["kfp_run_id"] == "kfp-run-123"
    assert tags["dataset_version"] == chain.ing.dataset_sha256 and tags["dataset_name"] == "synthetic-fraud"
    assert run.data.metrics["val_pr_auc"] == pytest.approx(res["val_pr_auc"])
    assert float(run.data.params["threshold"]) == pytest.approx(res["threshold"])
    assert any(a.path == "splits_summary.json" for a in env.client.list_artifacts(res["run_id"]))
    model_artifact = getattr(chain, "ma" if which == "ra" else "mb")
    assert hasattr(joblib.load(model_artifact.path), "predict_proba")


# ---------- pick_and_register ----------
def test_pick_and_register_retains_best_validation_strategy(env, chain):
    better = max((chain.ra, chain.rb), key=lambda r: r["val_pr_auc"])
    worse = chain.rb if better is chain.ra else chain.ra
    assert chain.best.run_id == better["run_id"] and chain.best.strategy == better["strategy"]
    assert chain.best.above_floor == "true"
    tags = lambda r: env.client.get_run(r["run_id"]).data.tags  # noqa: E731
    assert tags(better)["selected"] == "true" and tags(worse)["selected"] == "false"


def test_pick_and_register_evaluates_test_once_on_the_winner_and_registers_it(env, chain):
    winner = env.client.get_run(chain.best.run_id)
    loser_id = chain.rb["run_id"] if chain.best.run_id == chain.ra["run_id"] else chain.ra["run_id"]
    assert winner.data.metrics["test_pr_auc"] == pytest.approx(chain.best.test_pr_auc)
    assert "test_pr_auc" not in env.client.get_run(loser_id).data.metrics  # perdant : jamais évalué
    versions = env.client.search_model_versions(f"name='{MODEL}'")
    assert [v.run_id for v in versions] == [chain.best.run_id]


def test_above_floor_is_false_below_the_floor(env, chain):
    tmp = env.tmp / "below"
    tmp.mkdir()
    ra, rb, ma, mb = _train_both(tmp, chain.tr, chain.va, chain.split.splits_summary, tag="2")
    best = C.pick_and_register.python_func(
        result_a=ra, result_b=rb, model_a=ma, model_b=mb, test=chain.te, experiment=EXPERIMENT,
        project=PROJECT, model_name=MODEL, floor=1.01)  # plancher impossible
    assert best.above_floor == "false"
    # Branche « plancher non atteint » du pipeline : le modèle est alors marqué rejected.
    assert C.mark_rejected.python_func(run_id=best.run_id, model_name=MODEL,
                                       experiment=EXPERIMENT, project=PROJECT) == "rejected"
    rejected = env.client.get_model_version_by_alias(MODEL, "rejected")
    assert rejected.run_id == best.run_id


# ---------- gate ----------
def test_gate_promotes_to_champion_and_is_idempotent(env, chain):
    kwargs = dict(run_id=chain.best.run_id, model_name=MODEL, experiment=EXPERIMENT,
                  project=PROJECT, min_delta=0.005, floor=0.0)
    assert C.gate.python_func(**kwargs) == "champion"
    assert env.client.get_model_version_by_alias(MODEL, "champion").run_id == chain.best.run_id
    assert C.gate.python_func(**kwargs) == "champion"  # relance KFP : rien ne change


@pytest.mark.parametrize("name", ["gate", "mark_rejected"])
def test_registry_components_refuse_to_run_without_tracking_uri(monkeypatch, name):
    """Sans MLFLOW_TRACKING_URI, MLflow écrirait en silence dans ./mlruns : on exige l'erreur."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fn = getattr(C, name).python_func
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        fn(run_id="x", model_name=MODEL, experiment=EXPERIMENT, project=PROJECT)


# ---------- compilation KFP ----------
@pytest.mark.parametrize("name", COMPONENTS)
def test_component_compiles_with_the_ml_base_image(tmp_path, name):
    out = tmp_path / f"{name}.yaml"
    compiler.Compiler().compile(getattr(C, name), str(out))
    text = out.read_bytes().decode("utf-8", errors="replace")
    assert f"comp-{name.replace('_', '-')}" in text and "ml-base:" in text
