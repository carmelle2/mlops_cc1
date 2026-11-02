# mlops_common/tracking.py
"""Initialisation MLflow et tags standard (convention 2.2 du cahier des charges)."""
import hashlib
import os

import mlflow


def init(experiment, project, kfp_run_id="", dataset_name="",
         dataset_version="", extra_tags=None):
    """Configure MLflow et renvoie les tags standard à passer à start_run(tags=...).

    Les tags dont la valeur est vide sont retirés (ex. kfp_run_id en local).
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI n'est pas défini. Dans un pipeline KFP, "
            "appliquer with_mlflow(task) ; en local, exporter la variable.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    tags = {
        "project": project,
        "kfp_run_id": kfp_run_id,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "git_sha": os.environ.get("GIT_SHA", "unknown"),
        "image": os.environ.get("IMAGE_TAG", "unknown"),
    }
    tags.update(extra_tags or {})
    return {k: v for k, v in tags.items() if v}


def sha256_file(path, chunk=1 << 20):
    """Empreinte SHA-256 d'un fichier (lecture par blocs, mémoire constante)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()