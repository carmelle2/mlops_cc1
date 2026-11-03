# projects/p04_fraud/training.py
"""Entraînement LightGBM par stratégie (class_weight / smote), choix du seuil sous contrainte
de précision, et journalisation finale du modèle pyfunc dans MLflow.

Pas de dépendance à KFP : les composants de l'étape 4 appellent ces fonctions telles quelles.
`train` et `log_final` doivent être appelées DANS un run MLflow actif (mlflow.start_run).
"""
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import joblib
import mlflow
import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score)

from projects.p04_fraud.data import FEATURES, TARGET
from projects.p04_fraud.model import ThresholdedModel

STRATEGIES = ("class_weight", "smote")
DEFAULT_THRESHOLD = 0.5  # valeur de repli si la contrainte de précision est inatteignable
PROJECTS_DIR = Path(__file__).resolve().parents[1]  # dossier `projects/`, embarqué dans le modèle


class TrainResult(NamedTuple):
    model: LGBMClassifier
    threshold: float
    val_pr_auc: float
    threshold_met: bool  # la contrainte de précision a-t-elle pu être respectée ?


def xy(df, features=FEATURES):
    """Sépare variables et cible. `Time` reste dans FEATURES (décision de l'utilisateur)."""
    return df[list(features)], df[TARGET]


def pick_threshold(y_val, proba, min_precision=0.90):
    """Seuil maximisant le rappel sous contrainte de précision, sur la validation.

    Renvoie (seuil, contrainte_respectée). Si aucun seuil n'atteint `min_precision`, renvoie
    (0.5, False) ET émet un UserWarning : le cahier des charges renvoyait 0.5 en silence.
    """
    y_val = np.asarray(y_val)
    if not (y_val == 1).any():
        raise ValueError("Impossible de choisir un seuil : aucune fraude dans la validation")
    prec, rec, thr = precision_recall_curve(y_val, proba)
    # prec et rec ont un élément de plus que thr (point final precision=1, rappel=0) : on l'écarte.
    ok = np.where(prec[:-1] >= min_precision)[0]
    if len(ok) == 0:
        warnings.warn(
            f"Aucun seuil n'atteint la précision {min_precision} sur la validation : "
            f"seuil de repli {DEFAULT_THRESHOLD}", UserWarning, stacklevel=2)
        return DEFAULT_THRESHOLD, False
    return float(thr[ok[np.argmax(rec[ok])]]), True


def pr_curve_figure(y_true, proba, threshold, title):
    """Courbe précision-rappel avec le point correspondant au seuil retenu."""
    prec, rec, _ = precision_recall_curve(y_true, proba)
    pred = np.asarray(proba) >= threshold
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, proba):.3f}")
    ax.scatter([recall_score(y_true, pred)], [precision_score(y_true, pred, zero_division=0)],
               color="red", zorder=3, label=f"seuil = {threshold:.3f}")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def train(strategy, X_tr, y_tr, X_va, y_va, min_precision=0.90, n_jobs=4,
          smote_ratio=0.1, params_override=None):
    """Entraîne LightGBM selon la stratégie, choisit le seuil sur la validation et journalise.

    - class_weight : `scale_pos_weight` = négatifs / positifs du jeu d'entraînement ;
    - smote        : suréchantillonnage des fraudes UNIQUEMENT sur le jeu d'entraînement
                     (validation et test ne sont jamais rééchantillonnés : pas de fuite).
    `params_override` (ex. {"n_estimators": 50}) sert aux tests rapides ; il est journalisé.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy!r} (attendu : {STRATEGIES})")
    if mlflow.active_run() is None:
        raise RuntimeError("train() doit être appelée dans un run MLflow actif (mlflow.start_run)")

    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, subsample_freq=1,  # sans subsample_freq, subsample est IGNORÉ
                  colsample_bytree=0.8, random_state=42, n_jobs=n_jobs, verbose=-1)
    params.update(params_override or {})

    X_fit, y_fit = X_tr, y_tr
    extra = {}
    if strategy == "class_weight":
        params["scale_pos_weight"] = float((y_tr == 0).sum() / (y_tr == 1).sum())
    else:  # smote
        X_fit, y_fit = SMOTE(sampling_strategy=smote_ratio, random_state=42).fit_resample(X_tr, y_tr)
        extra["smote_sampling_strategy"] = smote_ratio

    model = LGBMClassifier(**params).fit(X_fit, y_fit)
    proba = model.predict_proba(X_va)[:, 1]
    thr, met = pick_threshold(y_va, proba, min_precision)
    pred = proba >= thr
    val_pr_auc = float(average_precision_score(y_va, proba))

    mlflow.log_params({"strategy": strategy, "min_precision": min_precision, "threshold": thr,
                       "threshold_constraint_met": met, "n_features": X_tr.shape[1],
                       "fit_rows": len(y_fit), "fit_frauds": int((y_fit == 1).sum()),
                       **extra, **params})
    mlflow.log_metrics({"val_pr_auc": val_pr_auc,
                        "val_precision": float(precision_score(y_va, pred, zero_division=0)),
                        "val_recall": float(recall_score(y_va, pred))})
    mlflow.log_figure(pr_curve_figure(y_va, proba, thr, f"Validation - {strategy}"),
                      "val_pr_curve.png")
    return TrainResult(model, thr, val_pr_auc, met)


def _pinned_requirements():
    """Versions exactes des bibliothèques utilisées : le modèle reste rechargeable à l'identique."""
    return [f"{p}=={version(p)}" for p in
            ("mlflow", "lightgbm", "scikit-learn", "pandas", "numpy", "joblib")]


def log_final(model, thr, X_te, y_te, model_name):
    """Évalue sur le test, journalise la métrique et la figure, enregistre le modèle pyfunc.

    À appeler dans le run de la stratégie retenue : la version du modèle et `test_pr_auc` sont
    alors rattachées au même run, que `registry.promote_if_better` retrouve par son run_id.
    Renvoie test_pr_auc (float).
    """
    if mlflow.active_run() is None:
        raise RuntimeError("log_final() doit être appelée dans un run MLflow actif")
    proba = model.predict_proba(X_te)[:, 1]
    pred = proba >= thr
    test_pr_auc = float(average_precision_score(y_te, proba))
    mlflow.log_metrics({"test_pr_auc": test_pr_auc,
                        "test_precision": float(precision_score(y_te, pred, zero_division=0)),
                        "test_recall": float(recall_score(y_te, pred))})
    mlflow.log_figure(pr_curve_figure(y_te, proba, thr, "Test - modèle retenu"), "test_pr_curve.png")

    head = X_te.head(5)
    signature = infer_signature(
        head, ThresholdedModel.to_frame(model.predict_proba(head)[:, 1], thr))
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "model.joblib")
        joblib.dump(model, path)
        mlflow.pyfunc.log_model(
            "model", python_model=ThresholdedModel(thr), artifacts={"model": path},
            signature=signature, input_example=X_te.head(3),
            registered_model_name=model_name,
            # La classe ThresholdedModel est sérialisée PAR RÉFÉRENCE (module importable) :
            # sans code_paths, recharger le modèle ailleurs échouerait (ModuleNotFoundError).
            code_paths=[str(PROJECTS_DIR)],
            pip_requirements=_pinned_requirements())
    return test_pr_auc
