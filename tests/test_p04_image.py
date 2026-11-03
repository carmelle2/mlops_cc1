# tests/test_p04_image.py
"""Tests de l'étape 6 : l'image `ml-base` (sans Docker : on vérifie les fichiers qui la définissent).

Le vrai contrôle reste `docker build` + l'import dans l'image ; ces tests attrapent les oublis
classiques (dépendance non listée, version non figée, code non copié) avant de construire."""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "images" / "ml-base" / "requirements.txt"
DOCKERFILE = ROOT / "images" / "ml-base" / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"

SERVER_MLFLOW = "3.14.0"  # version du serveur MLflow (relevée avec /version)
FIRST_PARTY = {"mlops_common", "projects"}
IMPORT_TO_PIP = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn"}


def _read(path):
    return path.read_text(encoding="utf-8")  # les fichiers contiennent des accents


def _lines(path):
    return [l.strip() for l in _read(path).splitlines() if l.strip() and not l.strip().startswith("#")]


def _pins():
    out = {}
    for line in _lines(REQUIREMENTS):
        name, _, version = line.partition("==")
        out[name.strip().lower()] = version.strip()
    return out


def _third_party_imports():
    """Modules tiers importés (au niveau module OU dans le corps des fonctions) par le code de l'image."""
    found = {}
    for folder in ("mlops_common", "projects"):
        for py in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(_read(py))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for name in names:
                    top = name.split(".")[0]
                    if top not in sys.stdlib_module_names and top not in FIRST_PARTY:
                        found.setdefault(top, set()).add(py.name)
    return found


# ---------- requirements.txt ----------
def test_every_requirement_is_pinned_exactly():
    lines = _lines(REQUIREMENTS)
    assert lines
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9.\-+]*", l) for l in lines), lines


def test_mlflow_matches_the_server_version():
    assert _pins()["mlflow"] == SERVER_MLFLOW


def test_kfp_is_in_the_image_because_components_do_not_install_it():
    from mlops_common.kfp_utils import cpu_component  # noqa: F401 (install_kfp_package=False)
    assert "kfp" in _pins()
    assert "kfp-kubernetes" not in _pins()  # sert à compiler (poste), pas à exécuter


def test_parquet_engine_is_listed():
    """pandas lit/écrit le Parquet via pyarrow, jamais importé directement : le scan d'imports ne le voit pas."""
    assert "pyarrow" in _pins()


def test_every_third_party_import_of_the_code_is_in_requirements():
    pins = _pins()
    missing = {mod: sorted(files) for mod, files in _third_party_imports().items()
               if IMPORT_TO_PIP.get(mod, mod).lower() not in pins}
    assert not missing, f"à ajouter à requirements.txt : {missing}"


# ---------- Dockerfile ----------
@pytest.fixture(scope="module")
def dockerfile():
    return _read(DOCKERFILE)


def test_dockerfile_exposes_the_code_on_pythonpath(dockerfile):
    assert re.search(r"PYTHONPATH=/opt/mlops\b", dockerfile)
    assert re.search(r"COPY\s+mlops_common\s+/opt/mlops/mlops_common", dockerfile)
    assert re.search(r"COPY\s+projects\s+/opt/mlops/projects", dockerfile)


def test_dockerfile_installs_the_pinned_requirements_and_openmp(dockerfile):
    assert "COPY images/ml-base/requirements.txt" in dockerfile
    assert re.search(r"pip install\s+-r\s+/tmp/requirements\.txt", dockerfile)
    assert "libgomp1" in dockerfile  # LightGBM
    assert dockerfile.index("requirements.txt") < dockerfile.index("COPY mlops_common")  # cache de couches


def test_dockerfile_passes_build_info_read_by_tracking(dockerfile):
    from mlops_common import tracking
    source = _read(Path(tracking.__file__))
    for var in ("GIT_SHA", "IMAGE_TAG"):
        assert f'"{var}"' in source  # tracking.init lit bien ces variables
        assert f"ARG {var}" in dockerfile and f"{var}=${{{var}}}" in dockerfile


def test_dockerfile_has_no_leftovers_data_or_secrets(dockerfile):
    """D8 : plus de pré-téléchargement de jeux ; option A : le CSV n'est pas dans l'image ; décision S1 : pas d'AWS."""
    for word in ("california_housing", "adult", "creditcard", "AWS_", "SECRET", "PASSWORD"):
        assert word not in dockerfile, word


def test_dockerignore_keeps_data_and_environments_out_of_the_build_context():
    ignored = set(_lines(DOCKERIGNORE))
    assert {"creditcard.csv", ".git", "mlops-env", "mlflow.db"} <= ignored