# projects/p04_fraud/pipeline.py
"""Pipeline KFP v2 du projet 4 (détection de fraude bancaire).

    ingest_validate -> temporal_split -> train_strategy x2 (class_weight, smote)
                    -> pick_and_register -> dsl.If(above_floor == "true")  : gate
                                            dsl.If(above_floor == "false") : mark_rejected

Le choix de livraison du CSV (décision S1) se fait à la COMPILATION, par l'environnement du
poste (comme `MLOPS_REGISTRY` dans kfp_utils) :
  - `MLOPS_DATA_PVC=<nom du PVC>` : le volume est monté dans `ingest_validate` uniquement
    (option A, défaut recommandé) ; `csv_path` pointe alors dans `DATA_MOUNT`.
  - variable absente : rien n'est monté (option B : le CSV est dans l'image) ; il faut alors
    passer `csv_path` au lancement.
"""
import os

from kfp import dsl, kubernetes

from mlops_common.kfp_utils import with_mlflow, with_resources
from projects.p04_fraud.components import (gate, ingest_validate, mark_rejected,
                                           pick_and_register, temporal_split, train_strategy)

PIPELINE_NAME = "p04-fraud"
PROJECT = "p04-fraud"
DATASET_NAME = "creditcard-fraud"
STRATEGIES = ("class_weight", "smote")  # même ordre que training.STRATEGIES
DATA_MOUNT = "/data"


def make_pipeline(data_pvc: str = ""):
    """Construit le pipeline. `data_pvc` : nom du PVC contenant creditcard.csv (vide = rien à monter)."""

    @dsl.pipeline(
        name=PIPELINE_NAME,
        description="Fraude bancaire : class_weight vs SMOTE, seuil embarqué, porte de validation.")
    def pipeline_func(
        experiment: str = "equipe-ml/p04-fraud",
        model_name: str = "p04-fraud-clf",
        csv_path: str = f"{DATA_MOUNT}/creditcard.csv",
        expected_rows: int = 284807,
        expected_frauds: int = 492,
        floor: float = 0.70,
        min_delta: float = 0.005,
        min_precision: float = 0.90,
    ):
        # 1. Chargement et vérification du CSV (2 CPU, 4 Go)
        ing = with_mlflow(with_resources(
            ingest_validate(csv_path=csv_path, expected_rows=expected_rows,
                            expected_frauds=expected_frauds), "2", "4G"))
        if data_pvc:
            kubernetes.mount_pvc(ing, pvc_name=data_pvc, mount_path=DATA_MOUNT)

        # 2. Découpage chronologique 70 / 15 / 15 (1 CPU, 2 Go)
        split = with_mlflow(with_resources(
            temporal_split(dataset=ing.outputs["dataset"]), "1", "2G"))

        # 3. Une tâche par stratégie (4 CPU, 8 Go). Le jeu de test n'est PAS transmis.
        trained = []
        for strategy in STRATEGIES:
            t = train_strategy(
                strategy=strategy,
                train=split.outputs["train"],
                val=split.outputs["val"],
                experiment=experiment,
                project=PROJECT,
                splits_summary=split.outputs["splits_summary"],
                kfp_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
                dataset_name=DATASET_NAME,
                dataset_version=ing.outputs["dataset_sha256"],
                min_precision=min_precision)
            t.set_display_name(f"train-{strategy}")
            trained.append(with_mlflow(with_resources(t, "4", "8G")))
        a, b = trained

        # 4. Choix de la stratégie, évaluation unique sur le test, enregistrement (1 CPU, 2 Go)
        best = with_mlflow(with_resources(
            pick_and_register(
                result_a=a.outputs["result"], result_b=b.outputs["result"],
                model_a=a.outputs["model_out"], model_b=b.outputs["model_out"],
                test=split.outputs["test"],
                experiment=experiment, project=PROJECT, model_name=model_name,
                floor=floor), "1", "2G"))

        # 5. Porte de validation (1 CPU)
        with dsl.If(best.outputs["above_floor"] == "true", name="plancher-atteint"):
            with_mlflow(with_resources(
                gate(run_id=best.outputs["run_id"], model_name=model_name,
                     experiment=experiment, project=PROJECT,
                     min_delta=min_delta, floor=floor), "1", "1G"))

        with dsl.If(best.outputs["above_floor"] == "false", name="plancher-non-atteint"):
            with_mlflow(with_resources(
                mark_rejected(run_id=best.outputs["run_id"], model_name=model_name,
                              experiment=experiment, project=PROJECT), "1", "1G"))

    return pipeline_func


# Pipeline par défaut : le PVC est lu dans l'environnement du poste qui compile.
pipeline_func = make_pipeline(os.environ.get("MLOPS_DATA_PVC", ""))
