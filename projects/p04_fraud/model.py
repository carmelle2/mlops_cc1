# projects/p04_fraud/model.py
"""Modèle pyfunc MLflow qui embarque le seuil de décision (propriété versionnée du modèle)."""
import mlflow.pyfunc
import numpy as np
import pandas as pd


class ThresholdedModel(mlflow.pyfunc.PythonModel):
    """Enveloppe un classifieur (predict_proba) et applique un seuil fixé à l'entraînement.

    Sortie : DataFrame à deux colonnes
      - fraud_proba : probabilité de fraude ;
      - is_fraud    : 1 si fraud_proba >= seuil, sinon 0.
    """

    def __init__(self, threshold: float):
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Le seuil doit être dans [0, 1] (reçu : {threshold})")
        self.threshold = threshold

    def load_context(self, context):
        import joblib  # import tardif : seulement nécessaire au chargement
        self.model = joblib.load(context.artifacts["model"])

    @staticmethod
    def to_frame(proba, threshold):
        """Construit la sortie du modèle. Partagé avec `training.log_final`, qui s'en sert
        pour déduire la signature : une seule définition du format de sortie."""
        proba = np.asarray(proba, dtype="float64")
        return pd.DataFrame({
            "fraud_proba": proba,
            "is_fraud": (proba >= threshold).astype("int64"),  # int64 explicite (Windows : int32 sinon)
        })

    def predict(self, context, model_input, params=None):
        # Colonnes réordonnées comme à l'entraînement (LightGBM s'appuie sur leur nom).
        cols = list(self.model.feature_name_)
        proba = self.model.predict_proba(model_input[cols])[:, 1]
        return self.to_frame(proba, self.threshold)
