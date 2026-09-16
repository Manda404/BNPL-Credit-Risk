"""Service métier : charge une version approuvée et réutilise Predictor."""

from datetime import UTC, datetime
from threading import BoundedSemaphore

import numpy as np
import pandas as pd
from loguru import logger

from bnpl_credit_risk.api.schemas import ApplicationRequest, ModelResponse, PredictionResponse
from bnpl_credit_risk.api.settings import APISettings
from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.schemas import inference_input_schema
from bnpl_credit_risk.data.validation import BNPLDataValidator
from bnpl_credit_risk.inference.predictor import Predictor
from bnpl_credit_risk.models.boosting import FittedBoostingModel
from bnpl_credit_risk.models.registry import resolve_model_dir
from bnpl_credit_risk.settings import ProjectConfig, Settings


class ServiceBusyError(Exception):
    """Capacité atteinte : le client peut réessayer après un court délai."""


class RealtimeInferenceService:
    """Un modèle immuable par processus et une capacité de calcul bornée.

    La limite par défaut sérialise les appels au pipeline natif. Les requêtes
    concurrentes excédentaires sont refusées sans constituer une file illimitée.
    """

    def __init__(self, predictor: Predictor, config: ProjectConfig, api: APISettings) -> None:
        """Assembler le moteur, le nettoyage, la validation et la limite de concurrence.

        Paramètres :
            predictor : moteur déjà chargé avec son ArtifactBundle.
            config : configurations YAML du projet, notamment les règles de données.
            api : paramètres HTTP, dont le nombre de prédictions simultanées admises.

        Le constructeur vérifie la compatibilité de l'artefact mais ne le charge pas
        et n'effectue pas de préchauffage. Utiliser load() pour le démarrage normal.
        Une incohérence détectée par _validate_artifact() interrompt la construction."""
        self.predictor = predictor
        self.config = config
        # Un créneau représente un calcul autorisé. Le compteur est protégé entre threads.
        self._slots = BoundedSemaphore(api.max_concurrent_predictions)
        self._cleaner = BNPLDataCleaner(config.data)
        schema = inference_input_schema(config.data, config.features, "application_risk")
        # Le temps réel exige strict : aucune ligne invalide ne doit être scorée.
        self._validator = BNPLDataValidator(schema, "strict")
        self._validate_artifact()

    @classmethod
    def load(
        cls, config: ProjectConfig, settings: Settings, api: APISettings
    ) -> "RealtimeInferenceService":
        """Charger une version du registre et rendre un service prêt à prédire.

        Paramètres :
            config : configurations du projet ; inference.model_version sert de repli.
            settings : résout le chemin artifacts/models depuis les chemins du projet.
            api : fournit une éventuelle version explicite et les paramètres du service.

        Retour :
            Instance construite, contrôlée et préchauffée de RealtimeInferenceService.

        La version explicite de l'API est prioritaire. latest est résolu une seule fois.
        Le dossier doit rester directement dans le registre avant de charger joblib.
        Les erreurs de chemin, de fichiers, de désérialisation ou de préchauffage remontent
        au lifespan, qui refusera alors de démarrer le serveur."""
        models_dir = (settings.resolve(settings.artifacts_dir) / "models").resolve()
        version = api.model_version or config.inference.model_version
        version_dir = resolve_model_dir(models_dir, version).resolve()
        # Vérifie aussi latest.json et les liens symboliques avant de désérialiser joblib.
        if version_dir.parent != models_dir:
            raise ValueError("Model artifact must be directly inside the model registry")
        predictor = Predictor.load(models_dir, version_dir.name)
        service = cls(predictor, config, api)
        service.warmup()
        logger.bind(pipeline="api", model_version=predictor.bundle.version).info("model_ready")
        return service

    def _validate_artifact(self) -> None:
        """Vérifier que l'artefact correspond au contrat de scoring v1.

        Contrôles : approbation, scope application_risk, features disponibles, colonnes
        requises, seuil fini entre 0 et 1, bandes ordonnées couvrant 1 et classes [0, 1].
        Les classes viennent de estimator pour FittedBoostingModel, du pipeline sinon.

        Effet :
            Mémorise input_features, utilisées ensuite par describe(). Aucun fit effectué.

        Lève ValueError pour les incompatibilités explicitement détectées. Un artefact
        mal formé peut aussi produire KeyError ou TypeError ; le démarrage sera bloqué.
        Ce contrôle est une vérification de compatibilité, pas une évaluation statistique."""
        bundle = self.predictor.bundle
        if bundle.metadata.get("approved_for_inference") is not True:
            raise ValueError("Publish an approved model with notebook 05 before serving")
        if bundle.metadata.get("risk_scope") != "application_risk":
            raise ValueError("Checkout API requires an application_risk model")
        if self.config.model.risk_scope != "application_risk":
            raise ValueError("API configuration must use application_risk")
        if bundle.feature_schema.get("risk_scope", "application_risk") != "application_risk":
            raise ValueError("Artifact scope differs from the API scope")
        # Accepter les deux représentations de features déjà utilisées dans le registre.
        self.input_features = bundle.feature_schema.get("input_features") or [
            *bundle.feature_schema.get("input_numeric_features", []),
            *bundle.feature_schema.get("input_categorical_features", []),
        ]
        # Le client ne peut fournir que les champs déclarés par le contrat HTTP v1.
        fields = set(ApplicationRequest.model_fields)
        if not self.input_features or not set(self.input_features) <= fields:
            raise ValueError("Published input features do not match API v1")
        schema = inference_input_schema(self.config.data, self.config.features, "application_risk")
        if set(schema.required_columns) != fields:
            raise ValueError("Project data contract does not match API v1")
        threshold = float(bundle.threshold["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid published decision threshold")
        bands = bundle.threshold.get("risk_bands", [])
        # assign_risk_band utilise une borne supérieure stricte : dépasser 1 couvre p=1.
        limits = [float(band["max_probability"]) for band in bands]
        if (
            not limits
            or not all(np.isfinite(limits))
            or limits[0] <= 0
            or limits[-1] <= 1
            or any(a >= b for a, b in zip(limits, limits[1:], strict=False))
        ):
            raise ValueError("Risk bands must be ordered and cover probabilities through 1")
        if any(not isinstance(band.get("name"), str) or not band["name"] for band in bands):
            raise ValueError("Every published risk band needs a name")
        # Predictor interprète la deuxième colonne comme la probabilité de défaut.
        classifier = (
            bundle.pipeline.estimator
            if isinstance(bundle.pipeline, FittedBoostingModel)
            else bundle.pipeline
        )
        classes = getattr(classifier, "classes_", None)
        if classes is None or list(classes) != [0, 1]:
            raise ValueError("Published classifier classes must be ordered [0, 1]")

    def warmup(self) -> None:
        """Faire une première prédiction synthétique avant d'accepter le trafic.

        Les nombres et la date proviennent d'un exemple fixe ; les catégories sont
        remplacées par les premières valeurs admises dans la configuration du projet.
        L'identifiant de corrélation vaut startup, sans référence à un client réel.

        Retour :
            None si la prédiction et la validation de sortie réussissent.

        Toute exception remonte au chargement. Ce test exerce réellement le nettoyage,
        le pipeline natif et la sortie ; il ne mesure pas les performances du modèle."""
        sample = dict(
            user_id=0,
            age=35,
            employment_type="Salaried",
            monthly_income=4000,
            credit_score=700,
            purchase_amount=300,
            product_category="Electronics",
            bnpl_installments=3,
            app_usage_frequency=10,
            location="USA",
            transaction_date="2026-01-01",
            debt_to_income_ratio=0.2,
        )
        for field, values in self.config.data.allowed_values.items():
            if field in sample:
                sample[field] = values[0]
        self.predict(ApplicationRequest.model_validate(sample), "startup")

    def predict(self, application: ApplicationRequest, request_id: str) -> PredictionResponse:
        """Produire une prédiction pour une demande unique, sans réentraîner le modèle.

        Paramètres :
            application : données déjà validées structurellement par ApplicationRequest.
            request_id : identifiant de corrélation généré par le middleware.

        Retour :
            PredictionResponse validée, avec horodatage UTC et version du modèle.

        Lève ServiceBusyError si aucun créneau n'est libre ; DataValidationError si les
        règles YAML sont violées ; les erreurs du modèle et ValidationError de sortie
        remontent au middleware. Le créneau est toujours libéré par finally.

        Le DataFrame ne contient qu'une ligne. Predictor calcule la probabilité et
        applique le seuil et les bandes sauvegardés, identiquement au chemin batch."""
        # blocking=False refuse immédiatement la surcharge au lieu de créer une attente.
        if not self._slots.acquire(blocking=False):
            raise ServiceBusyError()
        try:
            # mode="json" transforme notamment date en chaîne, compatible avec le cleaner.
            frame = pd.DataFrame([application.model_dump(mode="json")])
            frame = self._validator.validate(self._cleaner.clean(frame)).valid
            # Réutiliser le moteur batch garantit le même seuil et la même règle de décision.
            scored = self.predictor.score(frame, self.config.data.id_column)
            row = scored.iloc[0].to_dict()
            # Adapter le nom public HTTP sans modifier le contrat batch existant.
            row["default_risk_class"] = row.pop("predicted_default")
            # Horodatage UTC explicite pour les clients distribués ; le score reste inchangé.
            row["scoring_timestamp"] = datetime.now(UTC)
            # Une probabilité NaN, infinie ou hors [0, 1] échoue ici : pas de faux HTTP 200.
            response = PredictionResponse.model_validate({**row, "request_id": request_id})
            logger.bind(
                pipeline="api",
                request_id=request_id,
                model_version=response.model_version,
            ).info("prediction_completed")
            return response
        finally:
            # Même une exception native ne doit pas immobiliser définitivement un créneau.
            self._slots.release()

    def describe(self) -> ModelResponse:
        """Décrire le modèle réellement chargé et les règles acceptées par l'API.

        Retour :
            ModelResponse avec version, algorithme, seuil, features du modèle et champs
            de la requête. Les bornes et catégories proviennent de configs/data.yaml.

        Les champs requis par HTTP peuvent être plus nombreux que les features du
        modèle : user_id sert par exemple à identifier la réponse. Les règles sont
        filtrées sur ApplicationRequest pour ne pas exposer les colonnes comportementales.
        Aucun fichier n'est relu ; la réponse reste liée à la version en mémoire."""
        bundle = self.predictor.bundle
        fields = set(ApplicationRequest.model_fields)
        return ModelResponse(
            model_version=bundle.version,
            algorithm=bundle.metadata.get("algorithm", "unknown"),
            risk_scope="application_risk",
            approved_for_inference=True,
            decision_threshold=bundle.threshold["threshold"],
            input_features=self.input_features,
            required_request_fields=list(ApplicationRequest.model_fields),
            bounds={k: v.model_dump() for k, v in self.config.data.bounds.items() if k in fields},
            allowed_values={
                k: v for k, v in self.config.data.allowed_values.items() if k in fields
            },
        )
