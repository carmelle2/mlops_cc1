# tests/test_tracking.py
import pytest
from mlops_common import tracking


def test_init_requires_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        tracking.init("equipe-ml/p04-fraud", "p04-fraud")


def test_init_returns_standard_tags_without_empty_values(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    monkeypatch.setenv("GIT_SHA", "abc123")
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    tags = tracking.init("equipe-ml/p04-fraud", "p04-fraud",
                         dataset_name="creditcard-fraud", dataset_version="2026-10")
    assert tags["project"] == "p04-fraud"
    assert tags["git_sha"] == "abc123"
    assert tags["dataset_version"] == "2026-10"
    assert "kfp_run_id" not in tags  # vide -> retiré


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert tracking.sha256_file(str(f)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
