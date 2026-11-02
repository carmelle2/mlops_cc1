# mlops_common/kfp_utils.py
"""Décorateurs de composants KFP et helpers de configuration de tâches."""
import os
from functools import partial

from kfp import dsl

# Valeurs lues dans l'environnement du poste qui COMPILE le pipeline
# (pas de valeur en dur : critère 2.8). Les valeurs par défaut sont celles du cahier des charges.
REGISTRY = os.environ.get("MLOPS_REGISTRY", "registry.example.com/mlops")
IMAGE_VERSION = os.environ.get("MLOPS_IMAGE_VERSION", "1.0")
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000")

cpu_component = partial(
    dsl.component, base_image=f"{REGISTRY}/ml-base:{IMAGE_VERSION}",
    install_kfp_package=False)
gpu_component = partial(
    dsl.component, base_image=f"{REGISTRY}/ml-torch:{IMAGE_VERSION}",
    install_kfp_package=False)


def with_mlflow(task):
    """Injecte l'adresse MLflow dans une tâche et désactive son cache."""
    task.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
    task.set_caching_options(False)
    return task


def with_resources(task, cpu, memory):
    """Fixe requests = limits (CPU et mémoire) : chaque composant doit les déclarer (2.8).

    cpu : "2" ou "500m" ; memory : "4G", "512M", "2Gi"...
    """
    task.set_cpu_request(cpu).set_cpu_limit(cpu)
    task.set_memory_request(memory).set_memory_limit(memory)
    return task