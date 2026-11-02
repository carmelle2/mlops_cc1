# projects/p04_fraud/data.py
"""Chargement, validation et découpage temporel du jeu ULB Credit Card Fraud.

Fonctions pures (pas de dépendance à KFP ni à MLflow) : elles sont testables
localement et réutilisées telles quelles dans les composants KFP (étape 4).
"""
import argparse
import json

import numpy as np
import pandas as pd

TARGET = "Class"
TIME = "Time"
FEATURES = [TIME] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def load_csv(path, storage_options=None):
    """Lit le CSV (chemin local ou s3://...). La cible est convertie en entier :
    selon la source (Kaggle / OpenML), `Class` peut arriver sous forme de texte."""
    df = pd.read_csv(path, storage_options=storage_options)
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype("int8")
    return df


def validate(df, expected_rows=EXPECTED_ROWS, expected_frauds=EXPECTED_FRAUDS):
    """Vérifie la conformité du jeu de données ; lève ValueError listant TOUS les problèmes."""
    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    problems = []
    if len(df) != expected_rows:
        problems.append(f"{len(df)} lignes au lieu de {expected_rows}")
    bad_labels = sorted(set(df[TARGET].unique()) - {0, 1})
    if bad_labels:
        problems.append(f"valeurs de {TARGET} invalides : {bad_labels}")
    n_null = int(df[needed].isna().sum().sum())
    if n_null:
        problems.append(f"{n_null} valeurs manquantes")
    elif not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        problems.append("valeurs infinies dans les variables")
    frauds = int((df[TARGET] == 1).sum())
    if frauds != expected_frauds:
        problems.append(f"{frauds} fraudes au lieu de {expected_frauds}")
    if problems:
        raise ValueError("Validation échouée : " + " ; ".join(problems))
    return {"rows": len(df), "frauds": frauds, "fraud_rate": frauds / len(df)}


def sort_by_time(df):
    """Tri chronologique STABLE (l'ordre initial est conservé à temps égal)."""
    return df.sort_values(TIME, kind="stable").reset_index(drop=True)


def _tie_safe_boundary(times, idx):
    """Recule la frontière au début du groupe de transactions de même `Time`,
    pour qu'aucune seconde ne soit partagée entre deux jeux (pas de fuite)."""
    if idx <= 0 or idx >= len(times):
        return idx
    return int(np.searchsorted(times, times[idx], side="left"))


def temporal_split(df, train_frac=0.70, val_frac=0.15, min_positives=1):
    """Découpe chronologique : le passé entraîne, puis valide, puis teste sur le futur.

    Aucun mélange. Les frontières sont ajustées pour ne pas couper un groupe de
    transactions simultanées. Lève ValueError si un jeu est vide ou contient moins
    de `min_positives` fraudes (la PR-AUC n'y serait pas définie).
    """
    if not (train_frac > 0 and val_frac > 0 and train_frac + val_frac < 1):
        raise ValueError("Il faut train_frac > 0, val_frac > 0 et train_frac + val_frac < 1")
    df = sort_by_time(df)
    times = df[TIME].to_numpy()
    n = len(df)
    i1 = _tie_safe_boundary(times, int(n * train_frac))
    i2 = _tie_safe_boundary(times, int(n * (train_frac + val_frac)))
    parts = {"train": df.iloc[:i1], "val": df.iloc[i1:i2], "test": df.iloc[i2:]}
    for name, part in parts.items():
        if len(part) == 0:
            raise ValueError(f"Le jeu '{name}' est vide")
        pos = int((part[TARGET] == 1).sum())
        if pos < min_positives:
            raise ValueError(f"Le jeu '{name}' contient {pos} fraude(s) (minimum {min_positives})")
    return tuple(p.reset_index(drop=True) for p in parts.values())


def summarize_splits(train, val, test):
    """Résumé (tailles, fraudes, bornes de Time) à journaliser dans MLflow."""
    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        out[name] = {
            "rows": len(part),
            "frauds": int((part[TARGET] == 1).sum()),
            "fraud_rate": float((part[TARGET] == 1).mean()),
            "time_min": float(part[TIME].min()),
            "time_max": float(part[TIME].max()),
        }
    return out


if __name__ == "__main__":
    # Contrôle du VRAI fichier : python -m projects.p04_fraud.data creditcard.csv
    ap = argparse.ArgumentParser(description="Valide creditcard.csv et affiche le découpage temporel")
    ap.add_argument("csv_path")
    args = ap.parse_args()
    data = load_csv(args.csv_path)
    print("Validation :", json.dumps(validate(data)))
    print("Découpage  :", json.dumps(summarize_splits(*temporal_split(data)), indent=2))