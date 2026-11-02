# tests/test_kfp_utils.py
"""Vérifie que les helpers produisent un pipeline KFP compilable (sans cluster)."""
from kfp import dsl, compiler

from mlops_common.kfp_utils import cpu_component, with_mlflow, with_resources


@cpu_component
def hello() -> str:
    return "ok"


def test_helpers_compile(tmp_path):
    @dsl.pipeline(name="p04-test")
    def pipe():
        t = hello()
        with_resources(t, cpu="2", memory="4G")
        with_mlflow(t)

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(pipe, str(out))
    text = out.read_text()
    assert "MLFLOW_TRACKING_URI" in text
    assert "datasets-s3" not in text  # plus de Secret S3 (decision_01)
    assert "ml-base:" in text