# tests/test_p04_pipeline.py
"""Tests de l'étape 5 : on compile le pipeline et on vérifie la SPÉCIFICATION produite
(tâches, branches `dsl.If`, ressources, absence de fuite du test), puis `run.py`.

Le runner local de KFP ignore les `dsl.If` : les branches ne peuvent se vérifier que dans la
spécification compilée ; le code de `gate` / `mark_rejected` est testé à l'étape 4."""
import argparse
from types import SimpleNamespace

import pytest
import yaml

from projects.p04_fraud import run
from projects.p04_fraud.pipeline import DATA_MOUNT, STRATEGIES, make_pipeline

# executor -> (CPU, mémoire) imposés par le cahier des charges
RESOURCES = {
    "exec-ingest-validate": ("2", "4G"),
    "exec-temporal-split": ("1", "2G"),
    "exec-train-strategy": ("4", "8G"),
    "exec-train-strategy-2": ("4", "8G"),
    "exec-pick-and-register": ("1", "2G"),
    "exec-gate": ("1", "1G"),
    "exec-mark-rejected": ("1", "1G"),
}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return list(yaml.safe_load_all(f))


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    path = run.compile_pipeline(str(tmp_path_factory.mktemp("p05") / "pipeline.yaml"))
    docs = _load(path)
    return SimpleNamespace(path=path, spec=docs[0], docs=docs,
                           text=open(path, encoding="utf-8").read())


def _root_tasks(c):
    return c.spec["root"]["dag"]["tasks"]


def _branch_tasks(c, condition_value):
    """Tâches de la branche dont la condition vaut 'true' ou 'false'."""
    for name, task in _root_tasks(c).items():
        cond = task.get("triggerPolicy", {}).get("condition", "")
        if name.startswith("condition") and cond.endswith(f"== '{condition_value}'"):
            return set(c.spec["components"][task["componentRef"]["name"]]["dag"]["tasks"])
    raise AssertionError(f"aucune branche pour {condition_value}")


# ---------- structure ----------
def test_pipeline_metadata_and_default_parameters(compiled):
    assert compiled.spec["pipelineInfo"]["name"] == "p04-fraud"
    params = compiled.spec["root"]["inputDefinitions"]["parameters"]
    defaults = {k: v.get("defaultValue") for k, v in params.items()}
    assert defaults == {
        "experiment": "equipe-ml/p04-fraud", "model_name": "p04-fraud-clf",
        "csv_path": f"{DATA_MOUNT}/creditcard.csv", "expected_rows": 284807,
        "expected_frauds": 492, "floor": 0.70, "min_delta": 0.005, "min_precision": 0.90}


def test_tasks_are_chained_in_the_expected_order(compiled):
    tasks = _root_tasks(compiled)
    assert {"ingest-validate", "temporal-split", "train-strategy", "train-strategy-2",
            "pick-and-register"} <= set(tasks)
    assert tasks["temporal-split"]["dependentTasks"] == ["ingest-validate"]
    assert set(tasks["pick-and-register"]["dependentTasks"]) >= {"train-strategy", "train-strategy-2"}


def test_one_train_task_per_strategy(compiled):
    tasks = _root_tasks(compiled)
    got = {tasks[t]["inputs"]["parameters"]["strategy"]["runtimeValue"]["constant"]
           for t in ("train-strategy", "train-strategy-2")}
    assert got == set(STRATEGIES) == {"class_weight", "smote"}


def test_train_tasks_never_receive_the_test_set(compiled):
    """Aucune fuite par construction : seuls train et val arrivent aux tâches d'entraînement."""
    tasks = _root_tasks(compiled)
    for name in ("train-strategy", "train-strategy-2"):
        arts = tasks[name]["inputs"]["artifacts"]
        assert set(arts) == {"train", "val"}, name
        # chaque entrée reçoit la sortie de même nom (jamais `test` à la place de `train` ou `val`)
        for key, art in arts.items():
            assert art["taskOutputArtifact"] == {
                "producerTask": "temporal-split", "outputArtifactKey": key}, (name, key)
    assert tasks["pick-and-register"]["inputs"]["artifacts"]["test"][
        "taskOutputArtifact"]["outputArtifactKey"] == "test"


def test_train_tasks_get_run_id_and_dataset_version(compiled):
    tasks = _root_tasks(compiled)
    for name in ("train-strategy", "train-strategy-2"):
        p = tasks[name]["inputs"]["parameters"]
        assert p["kfp_run_id"]["runtimeValue"]["constant"] == "{{$.pipeline_job_uuid}}"
        assert p["dataset_version"]["taskOutputParameter"] == {
            "producerTask": "ingest-validate", "outputParameterKey": "dataset_sha256"}
        assert p["project"]["runtimeValue"]["constant"] == "p04-fraud"


# ---------- porte de validation ----------
def test_two_complementary_conditions_on_above_floor(compiled):
    conds = sorted(t["triggerPolicy"]["condition"] for n, t in _root_tasks(compiled).items()
                   if n.startswith("condition"))
    assert len(conds) == 2
    assert all("pick-and-register-above_floor" in c for c in conds)
    assert [c.rsplit("== ", 1)[1] for c in conds] == ["'false'", "'true'"]


def test_gate_runs_only_above_the_floor_and_mark_rejected_only_below(compiled):
    assert _branch_tasks(compiled, "true") == {"gate"}
    assert _branch_tasks(compiled, "false") == {"mark-rejected"}


def test_branches_receive_the_run_chosen_by_pick_and_register(compiled):
    for comp in ("comp-condition-1", "comp-condition-2"):
        for task in compiled.spec["components"][comp]["dag"]["tasks"].values():
            ref = task["inputs"]["parameters"]["run_id"]["componentInputParameter"]
            assert ref == "pipelinechannel--pick-and-register-run_id"


# ---------- configuration Kubeflow (critère 2.8) ----------
def _executors(c):
    return c.spec["deploymentSpec"]["executors"]


def test_seven_executors_with_the_expected_resources(compiled):
    got = {n: (e["container"]["resources"]["resourceCpuLimit"],
               e["container"]["resources"]["resourceMemoryLimit"])
           for n, e in _executors(compiled).items()}
    assert got == RESOURCES


def test_requests_equal_limits_everywhere(compiled):
    for name, e in _executors(compiled).items():
        r = e["container"]["resources"]
        assert r["resourceCpuRequest"] == r["resourceCpuLimit"], name
        assert r["resourceMemoryRequest"] == r["resourceMemoryLimit"], name


def test_every_executor_gets_mlflow_uri_and_the_ml_base_image(compiled):
    for name, e in _executors(compiled).items():
        env = {x["name"]: x["value"] for x in e["container"]["env"]}
        assert env.get("MLFLOW_TRACKING_URI", "").startswith("http"), name
        assert "ml-base:" in e["container"]["image"], name


def test_cache_is_disabled_on_every_task(compiled):
    tasks = dict(_root_tasks(compiled))
    for comp in ("comp-condition-1", "comp-condition-2"):
        tasks.update(compiled.spec["components"][comp]["dag"]["tasks"])
    real = {n: t for n, t in tasks.items() if not n.startswith("condition")}
    assert len(real) == 7
    assert all(not t["cachingOptions"].get("enableCache") for t in real.values())


def test_no_secret_and_no_s3_reference(compiled):
    """Décision S1 : aucun Secret AWS ; tout transite par le proxy MLflow et le stockage interne."""
    for word in ("datasets-s3", "secretAsEnv", "AWS_", "S3_ENDPOINT"):
        assert word not in compiled.text


# ---------- livraison du CSV (PVC) ----------
def test_pvc_is_mounted_only_in_ingest_validate(tmp_path):
    path = run.compile_pipeline(str(tmp_path / "p.yaml"), pipeline=make_pipeline("fraud-data"))
    docs = _load(path)
    assert len(docs) == 2
    execs = docs[1]["platforms"]["kubernetes"]["deploymentSpec"]["executors"]
    assert list(execs) == ["exec-ingest-validate"]
    mount = execs["exec-ingest-validate"]["pvcMount"][0]
    assert mount["constant"] == "fraud-data" and mount["mountPath"] == DATA_MOUNT


def test_nothing_is_mounted_without_a_pvc(compiled):
    assert len(compiled.docs) == 1  # pas de section « platforms » : rien à monter


# ---------- run.py ----------
def test_ensure_utf8_repairs_a_cp1252_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run.locale, "getpreferredencoding", lambda do_setlocale=True: "cp1252")
    f = tmp_path / "x.yaml"
    f.write_bytes("seuil embarqué\n".encode("cp1252"))
    with pytest.raises(UnicodeDecodeError):
        f.read_bytes().decode("utf-8")
    run._ensure_utf8(f)
    assert f.read_text(encoding="utf-8") == "seuil embarqué\n"


def test_ensure_utf8_leaves_a_valid_file_untouched(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_bytes("seuil embarqué\n".encode("utf-8"))
    before = f.read_bytes()
    run._ensure_utf8(f)
    assert f.read_bytes() == before


def test_parse_param_types_values():
    assert run.parse_param("floor=0.75") == ("floor", 0.75)
    assert run.parse_param("expected_rows=100") == ("expected_rows", 100)
    assert run.parse_param("csv_path=/data/x.csv") == ("csv_path", "/data/x.csv")
    with pytest.raises(argparse.ArgumentTypeError):
        run.parse_param("sans-egal")


def test_main_compile_only_writes_the_file(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("KFP_HOST", raising=False)
    out = tmp_path / "out.yaml"
    assert run.main(["--compile-only", "--output", str(out)]) == 0
    assert out.exists() and "comp-gate" in out.read_text(encoding="utf-8")
    assert "Pipeline compilé" in capsys.readouterr().out


def test_main_refuses_to_submit_without_kfp_host(tmp_path, monkeypatch):
    monkeypatch.delenv("KFP_HOST", raising=False)
    with pytest.raises(SystemExit) as e:
        run.main(["--output", str(tmp_path / "o.yaml")])
    assert e.value.code == 2
    assert not (tmp_path / "o.yaml").exists()  # on échoue avant de compiler


def test_main_submits_with_typed_arguments(tmp_path, monkeypatch):
    calls = {}

    class FakeClient:
        def create_run_from_pipeline_package(self, package, **kw):
            calls.update(package=package, **kw)
            return SimpleNamespace(run_id="run-42")

    monkeypatch.setenv("KFP_HOST", "http://kfp.example")
    monkeypatch.setattr(run, "_make_client", lambda: FakeClient())
    out = tmp_path / "p.yaml"
    code = run.main(["--output", str(out), "--run-name", "r1", "--param", "floor=0.75",
                     "--param", "csv_path=/data/c.csv"])
    assert code == 0
    assert calls["package"] == str(out) and calls["run_name"] == "r1"
    assert calls["arguments"] == {"floor": 0.75, "csv_path": "/data/c.csv"}
    assert calls["experiment_name"] == run.DEFAULT_EXPERIMENT


def test_make_client_passes_only_the_auth_settings_that_are_set(monkeypatch):
    import kfp

    seen = {}
    monkeypatch.setattr(kfp, "Client", lambda **kw: seen.update(kw))
    monkeypatch.setenv("KFP_HOST", "http://kfp.example")
    monkeypatch.delenv("KFP_TOKEN", raising=False)
    monkeypatch.setenv("KFP_COOKIES", "authservice_session=abc")
    monkeypatch.delenv("KFP_NAMESPACE", raising=False)
    run._make_client()
    assert seen == {"host": "http://kfp.example", "cookies": "authservice_session=abc"}
