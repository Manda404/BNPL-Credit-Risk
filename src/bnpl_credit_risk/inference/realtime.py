"""Scoring unitaire bas niveau pour les appels Python internes.

Le service HTTP complet réside dans bnpl_credit_risk.api. Cette fonction
conserve la compatibilité des appels existants ; elle attend des entrées déjà
validées. Les clients HTTP passent par RealtimeInferenceService.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bnpl_credit_risk.inference.predictor import Predictor


def score_single(predictor: Predictor, record: dict[str, Any], id_column: str) -> dict[str, Any]:
    """Adapter une seule demande Python au moteur de scoring en DataFrame.

    Paramètres :
        predictor : moteur déjà chargé par l'appelant.
        record : dictionnaire contenant les features et l'identifiant de la demande.
        id_column : nom du champ identifiant à recopier dans la sortie.

    Retour :
        Dictionnaire correspondant à la première et unique ligne produite par score().

    Cette fonction conserve l'ancien point d'entrée Python. Elle n'applique ni
    validation métier, ni authentification, ni contrôle de capacité. Les appels HTTP
    passent par RealtimeInferenceService ; ses garanties ne s'appliquent pas ici."""
    df = pd.DataFrame([record])
    result = predictor.score(df, id_column)
    return {str(k): v for k, v in result.iloc[0].to_dict().items()}
