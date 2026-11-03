# images/ml-base/Dockerfile
# Image de base des composants CPU. Construire depuis la RACINE du dépôt :
#   docker build -f images/ml-base/Dockerfile -t <registre>/ml-base:<version> \
#       --build-arg GIT_SHA=$(git rev-parse --short HEAD) --build-arg IMAGE_TAG=<version> .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/mlops

# LightGBM a besoin d'OpenMP (absent des images « slim »).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY images/ml-base/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Code partagé et projets : toute modification impose de reconstruire et re-tagger l'image.
COPY mlops_common /opt/mlops/mlops_common
COPY projects /opt/mlops/projects

# Lus par mlops_common.tracking.init (tags git_sha et image des runs MLflow).
ARG GIT_SHA=unknown
ARG IMAGE_TAG=unknown
ENV GIT_SHA=${GIT_SHA} \
    IMAGE_TAG=${IMAGE_TAG}

# Le jeu de données n'est PAS dans l'image (option A : volume monté, décision S1).