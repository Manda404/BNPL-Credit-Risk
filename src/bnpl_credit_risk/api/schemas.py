"""Contrats HTTP versionnés ; les bornes métier restent dans configs/data.yaml."""

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

# Les nombres JSON sont acceptés ; chaînes numériques, booléens et NaN sont refusés.
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Category = Annotated[str, Field(strict=True, min_length=1, max_length=64)]


class ApplicationRequest(BaseModel):
    """Données disponibles au checkout, avant tout événement de remboursement.

    Toute colonne supplémentaire est rejetée, notamment la cible, les variables
    comportementales et les features calculées que le pipeline doit produire.
    """

    # extra="forbid" empêche notamment de glisser default_flag dans la requête.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    # Identifiant de traçabilité de la réponse ; ce n’est pas une variable explicative.
    user_id: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]
    age: Number
    employment_type: Category
    monthly_income: Number
    credit_score: Number
    purchase_amount: Number
    product_category: Category
    bnpl_installments: Number
    app_usage_frequency: Number
    location: Category
    transaction_date: date
    debt_to_income_ratio: Number

    @field_validator("transaction_date", mode="before")
    @classmethod
    def validate_date(cls, value: Any) -> date:
        """Accepter une date Python ou une chaîne JSON strictement YYYY-MM-DD.

        Paramètre :
            value : valeur brute reçue avant la conversion automatique de Pydantic.

        Retour :
            Objet datetime.date représentant une date civile valide.

        Lève ValueError pour les timestamps numériques, datetime, formats alternatifs
        ou jours inexistants. Le test isinstance(datetime) est nécessaire parce que
        datetime hérite de date. Pydantic transforme cette erreur en erreur de validation."""
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError("transaction_date must use YYYY-MM-DD")
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("transaction_date must use YYYY-MM-DD")
        return parsed


class PredictionResponse(BaseModel):
    """Même décision que Predictor, enrichie d'un identifiant de corrélation."""

    user_id: int
    # La validation de sortie protège aussi le consommateur contre un modèle défaillant.
    default_probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    default_risk_class: Literal[0, 1]
    decision_threshold: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    risk_band: str
    model_version: str
    scoring_timestamp: datetime
    request_id: str


class ModelResponse(BaseModel):
    """Expose seulement le contrat utile au client, jamais les chemins internes."""

    model_version: str
    algorithm: str
    risk_scope: Literal["application_risk"]
    approved_for_inference: bool
    decision_threshold: float
    input_features: list[str]
    required_request_fields: list[str]
    bounds: dict[str, dict[str, float]]
    allowed_values: dict[str, list[str]]


class HealthResponse(BaseModel):
    """Sonde volontairement minimale et publique pour l'orchestrateur."""

    status: Literal["alive", "ready"]


class ErrorDetail(BaseModel):
    """Erreur exploitable sans copie de valeurs personnelles envoyées par le client."""

    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    """Enveloppe identique pour validation, authentification et pannes internes."""

    error: ErrorDetail
