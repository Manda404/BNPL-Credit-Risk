"""Configuration du service par variables BNPL_API_* ou fichier .env."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Isole les paramètres HTTP des configurations d'entraînement existantes."""

    model_config = SettingsConfigDict(env_prefix="BNPL_API_", env_file=".env", extra="ignore")

    # development rend Swagger visible ; production exige une version immuable explicite.
    environment: Literal["development", "production"] = "development"
    # Une clé est obligatoire, même en local ; SecretStr évite son affichage accidentel.
    api_key: SecretStr = Field(min_length=32)
    # None délègue le choix à configs/inference.yaml ; config_dir remplace le dossier YAML.
    model_version: str | None = None
    config_dir: Path | None = None
    # Le serveur reste local par défaut ; Docker utilise 0.0.0.0 dans son conteneur.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    # 16 Kio par défaut pour une demande unique ; cette limite porte sur les octets reçus.
    max_body_bytes: int = Field(default=16384, ge=1024, le=1048576)
    # Limite de calcul par processus, distincte de la limite de connexions Uvicorn.
    max_concurrent_predictions: int = Field(default=1, ge=1, le=32)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def validate_deployment(self) -> "APISettings":
        """Contrôler la cohérence des paramètres après validation Pydantic.

        Retour :
            La même instance si la version respecte les contraintes du déploiement.

        Lève ValueError lorsque production utilise une version absente/latest, ou
        lorsqu'un nom de version ressemble à un chemin. Pydantic encapsule cette erreur
        dans ValidationError. La présence des fichiers et l'approbation sont vérifiées
        plus tard par le service ; cette méthode ne lit pas le registre."""
        if self.environment == "production" and self.model_version in (None, "latest"):
            raise ValueError("Production requires an explicit BNPL_API_MODEL_VERSION")
        if self.model_version and (
            self.model_version in (".", "..")
            or "/" in self.model_version
            or "\\" in self.model_version
        ):
            raise ValueError("model_version must be a version name, not a path")
        return self
