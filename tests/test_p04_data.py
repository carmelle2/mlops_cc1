# tests/test_p04_data.py
import numpy as np
import pandas as pd
import pytest

from projects.p04_fraud.data import (
    FEATURES, TARGET, TIME, load_csv, sort_by_time, summarize_splits,
    temporal_split, validate)

N, FRAUDS = 1000, 60


def make_df(n=N, frauds=FRAUDS, seed=0, max_time=172_800):
    """Jeu synthétique au même schéma que creditcard.csv (Time avec des égalités possibles)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df[TIME] = np.sort(rng.integers(0, max_time, size=n)).astype(float)
    df["Amount"] = rng.exponential(50, size=n)
    y = np.zeros(n, dtype="int8")
    y[rng.choice(n, frauds, replace=False)] = 1
    df[TARGET] = y
    return df[FEATURES + [TARGET]]


# ---------- validate ----------
def test_validate_ok():
    info = validate(make_df(), expected_rows=N, expected_frauds=FRAUDS)
    assert info == {"rows": N, "frauds": FRAUDS, "fraud_rate": FRAUDS / N}


def test_validate_reports_all_problems_at_once():
    df = make_df()
    with pytest.raises(ValueError) as e:
        validate(df, expected_rows=N + 1, expected_frauds=FRAUDS + 1)
    assert "lignes" in str(e.value) and "fraudes" in str(e.value)


def test_validate_missing_column():
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate(make_df().drop(columns=["V7"]), expected_rows=N, expected_frauds=FRAUDS)


def test_validate_bad_label():
    df = make_df()
    df.loc[0, TARGET] = 2
    with pytest.raises(ValueError, match="invalides"):
        validate(df, expected_rows=N, expected_frauds=FRAUDS)


def test_validate_nan_and_inf():
    df = make_df()
    df.loc[3, "V1"] = np.nan
    with pytest.raises(ValueError, match="manquantes"):
        validate(df, expected_rows=N, expected_frauds=FRAUDS)
    df = make_df()
    df.loc[3, "Amount"] = np.inf
    with pytest.raises(ValueError, match="infinies"):
        validate(df, expected_rows=N, expected_frauds=FRAUDS)


# ---------- tri ----------
def test_sort_by_time_is_chronological_and_stable():
    df = make_df(max_time=50)  # beaucoup d'égalités
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    out = sort_by_time(shuffled)
    assert out[TIME].is_monotonic_increasing
    assert sort_by_time(out).equals(out)  # idempotent


# ---------- découpage ----------
def test_split_proportions_and_partition():
    tr, va, te = temporal_split(make_df())
    assert len(tr) + len(va) + len(te) == N
    assert abs(len(tr) / N - 0.70) < 0.02
    assert abs(len(va) / N - 0.15) < 0.02
    assert abs(len(te) / N - 0.15) < 0.02


def test_split_no_temporal_leakage_even_with_many_ties():
    tr, va, te = temporal_split(make_df(max_time=40))  # ~25 transactions par seconde
    assert tr[TIME].max() < va[TIME].min()
    assert va[TIME].max() < te[TIME].min()


def test_split_ignores_input_order():
    df = make_df()
    shuffled = df.sample(frac=1, random_state=3).reset_index(drop=True)
    a = temporal_split(df)
    b = temporal_split(shuffled)
    for x, y in zip(a, b):
        assert x[TIME].tolist() == y[TIME].tolist()


def test_split_raises_when_test_has_no_fraud():
    df = make_df()
    df[TARGET] = 0
    df.loc[0:840:20, TARGET] = 1  # fraudes dans train et val seulement (lignes < 850)
    with pytest.raises(ValueError, match="'test'"):
        temporal_split(df)


@pytest.mark.parametrize("tr,va", [(0.9, 0.2), (0, 0.15), (0.7, 0)])
def test_split_rejects_bad_fractions(tr, va):
    with pytest.raises(ValueError, match="train_frac"):
        temporal_split(make_df(), train_frac=tr, val_frac=va)


def test_split_raises_when_all_times_equal():
    df = make_df()
    df[TIME] = 5.0
    with pytest.raises(ValueError, match="vide"):
        temporal_split(df)


# ---------- chargement / résumé ----------
def test_load_csv_casts_string_labels(tmp_path):
    df = make_df(n=50, frauds=5)
    df[TARGET] = df[TARGET].astype(str)
    p = tmp_path / "cc.csv"
    df.to_csv(p, index=False, quoting=1)  # tout entre guillemets, comme certaines sources
    out = load_csv(str(p))
    assert out[TARGET].dtype == np.dtype("int8")
    assert int(out[TARGET].sum()) == 5


def test_summarize_splits_counts():
    tr, va, te = temporal_split(make_df())
    s = summarize_splits(tr, va, te)
    assert sum(s[k]["rows"] for k in s) == N
    assert sum(s[k]["frauds"] for k in s) == FRAUDS
    assert s["train"]["time_max"] < s["val"]["time_min"]