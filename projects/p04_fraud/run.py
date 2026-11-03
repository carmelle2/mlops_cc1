# projects/p04_fraud/run.py
"""Compile le pipeline du projet 4 puis le soumet à Kubeflow Pipelines.

Depuis la racine du dépôt :
    python -m projects.p04_fraud.run --compile-only
    python -m projects.p04_fraud.run --run-name manuel-test-01
    python -m projects.p04_fraud.run --param floor=0.75 --param csv_path=/data/creditcard.csv

Variables d'environnement lues au moment de COMPILER (aucune valeur en dur, critère 2.8) :
    MLOPS_REGISTRY, MLOPS_IMAGE_VERSION, MLFLOW_TRACKING_URI, MLOPS_DATA_PVC
et, pour SOUMETTRE :
    KFP_HOST (obligatoire), KFP_TOKEN, KFP_COOKIES, KFP_NAMESPACE (facultatifs : selon l'authentification).
"""
import argparse
import json
import locale
import os
import sys
from pathlib import Path

from kfp import compiler

from projects.p04_fraud.pipeline import pipeline_func

DEFAULT_OUTPUT = "pipeline.yaml"
DEFAULT_EXPERIMENT = "equipe-ml"  # expérience KFP (distincte de l'expérience MLflow)


def _ensure_utf8(path) -> None:
    """Garantit que le fichier est en UTF-8.

    Sous Windows, le compilateur KFP écrit avec l'encodage local (cp1252) : les accents du YAML
    deviennent des octets invalides en UTF-8, que Kubeflow refuserait. On réécrit en UTF-8.
    """
    p = Path(path)
    raw = p.read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        p.write_bytes(raw.decode(locale.getpreferredencoding(False)).encode("utf-8"))


def compile_pipeline(output: str = DEFAULT_OUTPUT, pipeline=pipeline_func) -> str:
    """Compile `pipeline` en spécification KFP (YAML UTF-8) et renvoie le chemin du fichier."""
    compiler.Compiler().compile(pipeline, str(output))
    _ensure_utf8(output)
    return str(output)


def parse_param(text: str):
    """`clé=valeur` -> (clé, valeur typée). `floor=0.75` donne un float, `csv_path=/x` une chaîne.

    Le typage compte : KFP refuse une chaîne pour un paramètre entier ou flottant.
    """
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"attendu clé=valeur, reçu : {text!r}")
    key, raw = text.split("=", 1)
    try:
        return key.strip(), json.loads(raw)
    except json.JSONDecodeError:
        return key.strip(), raw


def _make_client():
    import kfp

    kwargs = {"host": os.environ["KFP_HOST"]}
    for env, arg in (("KFP_TOKEN", "existing_token"), ("KFP_COOKIES", "cookies"),
                     ("KFP_NAMESPACE", "namespace")):
        if os.environ.get(env):
            kwargs[arg] = os.environ[env]
    return kfp.Client(**kwargs)


def submit(package: str, arguments: dict, run_name: str, experiment_name: str, client=None) -> str:
    """Soumet le pipeline compilé ; renvoie l'identifiant du run KFP."""
    client = client or _make_client()
    run = client.create_run_from_pipeline_package(
        package, arguments=arguments, run_name=run_name, experiment_name=experiment_name)
    return run.run_id


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compile-only", action="store_true", help="compiler sans soumettre")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="fichier YAML produit")
    ap.add_argument("--run-name", default="manuel-test-01")
    ap.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT, help="expérience KFP")
    ap.add_argument("--param", action="append", default=[], type=parse_param, metavar="CLE=VALEUR",
                    help="surcharge d'un paramètre du pipeline (répétable)")
    args = ap.parse_args(argv)

    if not args.compile_only and not os.environ.get("KFP_HOST"):
        ap.error("KFP_HOST n'est pas défini (adresse de Kubeflow Pipelines) ; "
                 "ou utiliser --compile-only")

    package = compile_pipeline(args.output)
    print(f"Pipeline compilé : {package}")
    if args.compile_only:
        return 0

    run_id = submit(package, dict(args.param), args.run_name, args.experiment_name)
    print(f"Run soumis : {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
