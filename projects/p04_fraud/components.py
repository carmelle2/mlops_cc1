# projects/p04_fraud/components.py
"""Composants KFP v2 du projet 4 (détection de fraude).

Chaque composant est une fonction Python légère : KFP en extrait le SOURCE, donc tous les
imports (`projects`, `mlops_common`, `mlflow`...) sont faits DANS le corps de la fonction, et
le code métier est dans `data.py` / `training.py` (testés sans KFP). L'image `ml-base` doit
contenir `mlops_common` et `projects` (PYTHONPATH=/opt/mlops).

Les ressources (CPU/mémoire), `with_mlflow` et les conditions `dsl.If` sont posées dans le
pipeline (étape 5), pas ici. Pour tester une fonction sans cluster : `composant.python_func(...)`.
"""
from typing import NamedTuple

from kfp import dsl

from mlops_common.kfp_utils import cpu_component


@cpu_component
def ingest_validate(
    csv_path: str,
    expected_rows: int,
    expected_frauds: int,
    dataset: dsl.Output[dsl.Dataset],
) -> NamedTuple("Outputs", [("dataset_sha256", str), ("rows", int), ("frauds", int)]):
    """Charge le CSV (chemin LOCAL : volume monté ou fichier de l'image), vérifie le contenu,
    trie par `Time` et écrit le jeu au format Parquet. Renvoie l'empreinte SHA-256 du CSV
    (= `dataset_version` dans MLflow)."""
    from collections import namedtuple

    from mlops_common.tracking import sha256_file
    from projects.p04_fraud import data

    df = data.load_csv(csv_path)
    info = data.validate(df, expected_rows=expected_rows, expected_frauds=expected_frauds)
    data.sort_by_time(df).to_parquet(dataset.path, index=False)
    dataset.metadata.update(info)

    Outputs = namedtuple("Outputs", ["dataset_sha256", "rows", "frauds"])
    return Outputs(sha256_file(csv_path), info["rows"], info["frauds"])


@cpu_component
def temporal_split(
    dataset: dsl.Input[dsl.Dataset],
    train: dsl.Output[dsl.Dataset],
    val: dsl.Output[dsl.Dataset],
    test: dsl.Output[dsl.Dataset],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> NamedTuple("Outputs", [("splits_summary", str)]):
    """Découpage chronologique : le passé entraîne, puis valide, puis teste sur le futur.
    Renvoie un résumé JSON (tailles, fraudes, bornes de `Time`) journalisé ensuite dans MLflow."""
    import json
    from collections import namedtuple

    import pandas as pd

    from projects.p04_fraud import data

    parts = data.temporal_split(pd.read_parquet(dataset.path),
                                train_frac=train_frac, val_frac=val_frac)
    for part, out in zip(parts, (train, val, test)):
        part.to_parquet(out.path, index=False)
        out.metadata["rows"] = len(part)
    summary = data.summarize_splits(*parts)

    Outputs = namedtuple("Outputs", ["splits_summary"])
    return Outputs(json.dumps(summary))


@cpu_component
def train_strategy(
    strategy: str,
    train: dsl.Input[dsl.Dataset],
    val: dsl.Input[dsl.Dataset],
    model_out: dsl.Output[dsl.Model],
    experiment: str,
    project: str,
    splits_summary: str,
    kfp_run_id: str = "",
    dataset_name: str = "",
    dataset_version: str = "",
    min_precision: float = 0.90,
    n_jobs: int = 4,
) -> NamedTuple("Outputs", [("result", str)]):
    """Entraîne LightGBM avec UNE stratégie (`class_weight` ou `smote`) dans son propre run MLflow.

    Ce composant ne reçoit PAS le jeu de test : il ne peut pas l'utiliser, ni pour entraîner
    ni pour choisir le seuil (aucune fuite par construction). Renvoie un JSON
    {strategy, run_id, val_pr_auc, threshold, threshold_met}."""
    import json
    from collections import namedtuple

    import joblib
    import mlflow
    import pandas as pd

    from mlops_common import tracking
    from projects.p04_fraud import training

    tags = tracking.init(experiment, project, kfp_run_id=kfp_run_id, dataset_name=dataset_name,
                         dataset_version=dataset_version, extra_tags={"strategy": strategy})
    X_tr, y_tr = training.xy(pd.read_parquet(train.path))
    X_va, y_va = training.xy(pd.read_parquet(val.path))

    with mlflow.start_run(run_name=strategy, tags=tags) as run:
        mlflow.log_dict(json.loads(splits_summary), "splits_summary.json")
        res = training.train(strategy, X_tr, y_tr, X_va, y_va,
                             min_precision=min_precision, n_jobs=n_jobs)
        joblib.dump(res.model, model_out.path)
    model_out.metadata.update({"framework": "lightgbm", "strategy": strategy})

    Outputs = namedtuple("Outputs", ["result"])
    return Outputs(json.dumps({
        "strategy": strategy, "run_id": run.info.run_id, "val_pr_auc": res.val_pr_auc,
        "threshold": res.threshold, "threshold_met": res.threshold_met}))


@cpu_component
def pick_and_register(
    result_a: str,
    result_b: str,
    model_a: dsl.Input[dsl.Model],
    model_b: dsl.Input[dsl.Model],
    test: dsl.Input[dsl.Dataset],
    experiment: str,
    project: str,
    model_name: str,
    floor: float = 0.70,
) -> NamedTuple("Outputs", [("run_id", str), ("test_pr_auc", float),
                            ("above_floor", str), ("strategy", str)]):
    """Retient la stratégie de meilleure PR-AUC de VALIDATION (à égalité : la première), l'évalue
    UNE fois sur le test et enregistre le modèle pyfunc (seuil embarqué) dans le run retenu.
    `above_floor` vaut "true" / "false" (chaîne, pour les `dsl.If` du pipeline)."""
    import json
    from collections import namedtuple

    import joblib
    import mlflow
    import pandas as pd
    from mlflow import MlflowClient

    from mlops_common import tracking
    from projects.p04_fraud import training

    tracking.init(experiment, project)
    candidates = [(json.loads(result_a), model_a), (json.loads(result_b), model_b)]
    best, best_model = max(candidates, key=lambda c: c[0]["val_pr_auc"])
    other = next(c[0] for c in candidates if c[0]["run_id"] != best["run_id"])

    model = joblib.load(best_model.path)
    X_te, y_te = training.xy(pd.read_parquet(test.path))
    # On rouvre le run retenu : la version enregistrée et `test_pr_auc` sont ainsi sur le même
    # run, que registry.promote_if_better retrouve par son run_id.
    with mlflow.start_run(run_id=best["run_id"]):
        test_pr_auc = training.log_final(model, best["threshold"], X_te, y_te, model_name)
    client = MlflowClient()
    client.set_tag(best["run_id"], "selected", "true")
    client.set_tag(other["run_id"], "selected", "false")

    Outputs = namedtuple("Outputs", ["run_id", "test_pr_auc", "above_floor", "strategy"])
    return Outputs(best["run_id"], float(test_pr_auc),
                   "true" if test_pr_auc >= floor else "false", best["strategy"])


@cpu_component
def gate(
    run_id: str,
    model_name: str,
    experiment: str,
    project: str,
    min_delta: float = 0.005,
    floor: float = 0.70,
) -> str:
    """Porte de validation : alias `champion` si le modèle bat le champion d'au moins `min_delta`
    et dépasse `floor`, sinon `challenger` (ou `rejected` sous le plancher)."""
    from mlops_common import tracking
    from mlops_common.registry import promote_if_better

    tracking.init(experiment, project)  # vérifie MLFLOW_TRACKING_URI (sinon : RuntimeError)
    return promote_if_better(model_name, run_id, "test_pr_auc", higher_is_better=True,
                             min_delta=min_delta, floor=floor)


@cpu_component
def mark_rejected(run_id: str, model_name: str, experiment: str, project: str) -> str:
    """Pose l'alias `rejected` sur la version issue du run (modèle sous le plancher)."""
    from mlops_common import tracking
    from mlops_common.registry import mark_rejected as _mark_rejected

    tracking.init(experiment, project)
    return _mark_rejected(model_name, run_id)
