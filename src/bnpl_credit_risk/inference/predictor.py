"""Predictor — the reusable scoring core.

Deliberately IO-agnostic: it takes a dataframe in and returns predictions
out. `inference/batch.py` wraps it with file reading/writing and validation
for scheduled batch runs; the FastAPI service wraps this same
class for realtime single-record scoring (see api/service.py) instead
of reimplementing scoring logic.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bnpl_credit_risk.evaluation.thresholding import assign_risk_band
from bnpl_credit_risk.models.persistence import ArtifactBundle
from bnpl_credit_risk.models.registry import resolve_model_dir
from bnpl_credit_risk.settings import RiskBandConfig


class Predictor:
    def __init__(self, bundle: ArtifactBundle) -> None:
        """Conserver l'artefact déjà chargé pour les prochains appels de scoring.

        bundle contient le pipeline entraîné, son seuil, ses bandes et sa version.
        Cette classe ne nettoie pas les entrées et ne vérifie pas l'approbation :
        ces étapes sont assurées par BatchPredictor ou RealtimeInferenceService."""
        self.bundle = bundle

    @classmethod
    def load(cls, models_dir: Path, model_version: str = "latest") -> Predictor:
        """Résoudre une version du registre et charger son ArtifactBundle.

        Paramètres :
            models_dir : dossier du registre contenant les versions et latest.json.
            model_version : nom exact de version ou latest, valeur par défaut.

        Retour :
            Predictor prêt à utiliser le pipeline sauvegardé, sans réentraînement.

        Les erreurs du registre, des fichiers ou de joblib sont propagées. Le contrôle
        d'approbation est du ressort de l'appelant, notamment du service HTTP."""
        version_dir = resolve_model_dir(models_dir, model_version)
        return cls(ArtifactBundle.load(version_dir))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Extraire la probabilité de défaut pour chaque ligne du DataFrame.

        Paramètre :
            df : lignes déjà préparées selon le contrat d'entrée du pipeline sauvegardé.

        Retour :
            Tableau NumPy de probabilités, une valeur par ligne.

        Lève ValueError si predict_proba ne fournit pas une matrice d'au moins deux
        colonnes. La colonne d'indice 1 est interprétée comme la classe défaut ; l'API
        vérifie l'ordre [0, 1] au démarrage et borne les probabilités dans sa réponse."""
        probabilities = np.asarray(self.bundle.pipeline.predict_proba(df))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                "Published model predict_proba must return two class-probability columns"
            )
        return probabilities[:, 1]

    def score(self, df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        """Calculer les scores et appliquer la politique publiée à toutes les lignes.

        Paramètres :
            df : DataFrame d'entrée compatible avec le modèle, contenant aussi l'identifiant.
            id_column : nom de la colonne à recopier pour associer entrée et prédiction.

        Retour :
            DataFrame avec identifiant, probabilité, classe, seuil, bande de risque,
            version et horodatage. L'ordre des lignes d'entrée est conservé.

        La classe vaut 1 si probabilité >= seuil. Le seuil et les bandes proviennent de
        l'artefact, jamais d'un recalcul sur les nouvelles données. L'horodatage local
        historique est commun à cet appel ; l'API le remplace par un datetime UTC explicite.
        Les erreurs du pipeline, des métadonnées ou de colonne manquante remontent."""
        threshold = self.bundle.threshold["threshold"]
        risk_bands = [RiskBandConfig.model_validate(rb) for rb in self.bundle.threshold.get("risk_bands", [])]

        probabilities = self.predict_proba(df)
        predicted = (probabilities >= threshold).astype(int)
        scoring_timestamp = datetime.now().isoformat()

        return pd.DataFrame(
            {
                id_column: df[id_column].to_numpy(),
                "default_probability": probabilities,
                "predicted_default": predicted,
                "decision_threshold": threshold,
                "risk_band": [assign_risk_band(p, risk_bands) for p in probabilities],
                "model_version": self.bundle.version,
                "scoring_timestamp": scoring_timestamp,
            }
        )
