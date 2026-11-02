# mlops_common/registry.py
"""Promotion de modèles par alias (champion / challenger / rejected)."""
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def _version_of_run(client, model_name, run_id):
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.run_id == run_id:
            return v
    raise ValueError(
        f"Aucune version du modèle '{model_name}' n'est issue du run {run_id}.")


def _current_champion(client, model_name):
    """Renvoie la version champion, ou None s'il n'y en a pas encore."""
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as e:
        msg = str(e).lower()
        # MLflow 2.22 renvoie INVALID_PARAMETER_VALUE (et non RESOURCE_DOES_NOT_EXIST)
        # pour un alias absent : on se fie donc au message, pas au code d'erreur.
        if e.error_code == "RESOURCE_DOES_NOT_EXIST" or ("alias" in msg and "not found" in msg):
            return None  # l'alias n'existe pas encore
        raise  # vraie erreur (réseau, droits...) : ne pas la masquer


def mark_rejected(model_name, run_id):
    """Pose l'alias 'rejected' sur la version issue de run_id."""
    c = MlflowClient()
    mv = _version_of_run(c, model_name, run_id)
    c.set_registered_model_alias(model_name, "rejected", mv.version)
    c.set_model_version_tag(model_name, mv.version, "validated_by", "kfp-pipeline")
    return "rejected"


def promote_if_better(model_name, run_id, metric, higher_is_better=True,
                      min_delta=0.0, floor=None):
    """Attribue l'alias champion/challenger/rejected à la version issue de run_id.

    - sous le plancher (floor)            -> 'rejected'
    - aucun champion, ou meilleur d'au
      moins min_delta que le champion     -> 'champion'
    - sinon                               -> 'challenger'
    """
    c = MlflowClient()
    mv = _version_of_run(c, model_name, run_id)
    new = c.get_run(run_id).data.metrics[metric]

    if floor is not None and ((new < floor) if higher_is_better else (new > floor)):
        return mark_rejected(model_name, run_id)

    champ = _current_champion(c, model_name)
    if champ is None:
        better = True  # aucun champion pour l'instant
    elif str(champ.version) == str(mv.version):
        return "champion"  # relance du gate sur le champion actuel : rien à faire
    else:
        old = c.get_run(champ.run_id).data.metrics[metric]
        better = (new - old >= min_delta) if higher_is_better else (old - new >= min_delta)

    alias = "champion" if better else "challenger"
    c.set_registered_model_alias(model_name, alias, mv.version)
    c.set_model_version_tag(model_name, mv.version, "validated_by", "kfp-pipeline")
    return alias