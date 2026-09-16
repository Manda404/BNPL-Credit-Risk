"""Factory FastAPI : cycle de vie, dépendances, routes et erreurs publiques."""

import secrets
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException

from bnpl_credit_risk.api.middleware import APIMetrics, RequestMiddleware, error_response
from bnpl_credit_risk.api.schemas import (
    ApplicationRequest,
    ErrorResponse,
    HealthResponse,
    ModelResponse,
    PredictionResponse,
)
from bnpl_credit_risk.api.service import RealtimeInferenceService, ServiceBusyError
from bnpl_credit_risk.api.settings import APISettings
from bnpl_credit_risk.exceptions import DataValidationError
from bnpl_credit_risk.settings import Settings, load_config


def create_app(
    api_settings: APISettings | None = None,
    project_settings: Settings | None = None,
    service_factory: Callable[[], RealtimeInferenceService] | None = None,
) -> FastAPI:
    """Construire l'application HTTP et enregistrer ses routes.

    Paramètres :
        api_settings : paramètres HTTP explicites ; sinon, lecture de BNPL_API_*.
        project_settings : chemins du projet ; sinon, utilisation de Settings().
        service_factory : fonction sans argument fournissant un service de test.
            Lorsqu'elle est fournie, elle remplace le chargement/préchauffage normal.

    Retour :
        Instance FastAPI configurée, dont le modèle sera chargé au démarrage.

    La factory évite de charger joblib lors d'un simple import Python. Ses fonctions
    internes partagent les paramètres de cette instance et sont appelées par FastAPI.
    Une configuration invalide peut lever une ValidationError dès la construction."""
    api = api_settings or APISettings()  # type: ignore[call-arg]  # BaseSettings lit la clé dans env.
    settings = project_settings or Settings()
    # Le registre appartient à cette application, pas à une variable globale.
    metrics = APIMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Gérer les ressources pendant toute la vie de l'application.

        Paramètre :
            app : instance FastAPI dont app.state.service contiendra le service partagé.

        Avant yield : configurer Loguru puis charger, contrôler et préchauffer le modèle.
        Pendant yield : laisser FastAPI traiter les requêtes avec cette même instance.
        Après yield, même en cas d'erreur : retirer la référence au service et logger l'arrêt.

        Une panne de chargement devient une RuntimeError générique et empêche le démarrage.
        Le type de l'erreur est journalisé, sans son contenu potentiellement sensible."""
        # Centraliser les logs du processus ; serialize=True produit du JSON exploitable.
        # diagnose=False empêche Loguru de joindre les valeurs des variables aux erreurs.
        logger.remove()
        logger.configure(extra={"pipeline": "api"})
        logger.add(sys.stderr, level=api.log_level, serialize=True, backtrace=False, diagnose=False)
        app.state.service = None
        try:
            app.state.service = (
                service_factory()
                if service_factory
                else RealtimeInferenceService.load(
                    load_config(api.config_dir),
                    settings,
                    api,
                )
            )
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error("startup_failed")
            raise RuntimeError(
                "API startup failed; check configuration and approved artifact"
            ) from None
        try:
            yield
        finally:
            app.state.service = None
            logger.info("service_stopped")

    docs_enabled = api.environment == "development"
    app = FastAPI(
        title="BNPL Credit Risk — Realtime Inference",
        version="1.0.0",
        description="Scoring application_risk avec le modèle approuvé du notebook 05.",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_enabled else None,
        responses={code: {"model": ErrorResponse} for code in (401, 413, 415, 422, 500, 503)},
    )
    # app.state partage le service entre les routes ; request.state est propre à un appel.
    app.state.service = None
    app.add_middleware(RequestMiddleware, max_body_bytes=api.max_body_bytes, metrics=metrics)
    # auto_error=False laisse notre handler construire la réponse 401 uniforme.
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def require_api_key(key: Annotated[str | None, Security(key_header)]) -> None:
        """Vérifier la clé extraite du header X-API-Key par FastAPI.

        Paramètre :
            key : secret fourni par le client, ou None si le header est absent.

        Retour :
            None en cas de succès ; la route protégée peut ensuite continuer.

        Lève HTTPException(401) si la clé est absente ou incorrecte. compare_digest
        évite une comparaison qui s'arrêterait au premier caractère différent.
        Les octets permettent aussi de comparer des chaînes Unicode sans les logger."""
        expected = api.api_key.get_secret_value().encode()
        if key is None or not secrets.compare_digest(key.encode(), expected):
            raise HTTPException(
                401, "Authentication required", headers={"WWW-Authenticate": "APIKey"}
            )

    def get_service(request: Request) -> RealtimeInferenceService:
        """Fournir aux routes le service initialisé pendant le lifespan.

        Paramètre :
            request : requête utilisée pour retrouver l'état de son application.

        Retour :
            RealtimeInferenceService partagé ; aucun rechargement du modèle ici.

        Lève HTTPException(503) lorsque le service est absent, par exemple après
        l'arrêt. Depends(get_service) demande à FastAPI d'injecter cet objet à la route."""
        service = request.app.state.service
        if service is None:
            raise HTTPException(503, "Model is not ready")
        return service

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        """Transformer une erreur de structure JSON/Pydantic en réponse 422.

        Paramètres :
            request : porte le request_id ajouté par le middleware.
            exc : erreur interceptée par FastAPI ; son contenu est volontairement ignoré.

        Retour :
            JSONResponse avec le code validation_error et l'identifiant de corrélation.

        Ne pas renvoyer exc.errors() brut : il peut contenir les valeurs du client."""
        return error_response(
            422,
            "validation_error",
            "Request does not match API v1 schema",
            request.state.request_id,
        )

    @app.exception_handler(DataValidationError)
    async def invalid_business_data(request: Request, exc: DataValidationError):
        """Transformer une violation de configs/data.yaml en réponse 422.

        Paramètres :
            request : fournit l'identifiant de la requête.
            exc : DataValidationError issue du validateur métier, non exposée au client.

        Retour :
            JSONResponse avec le code data_contract_error. Le client peut consulter
            GET /v1/model pour connaître les bornes et catégories admises."""
        return error_response(
            422,
            "data_contract_error",
            "Values violate the model data contract",
            request.state.request_id,
        )

    @app.exception_handler(ServiceBusyError)
    async def busy(request: Request, exc: ServiceBusyError):
        """Indiquer que tous les créneaux de prédiction sont occupés.

        Paramètres :
            request : fournit le request_id.
            exc : ServiceBusyError levée par le sémaphore du service.

        Retour :
            Réponse 503 service_busy avec Retry-After: 1. Ce header propose au client
            d'attendre une seconde ; le serveur ne programme aucun retry à sa place."""
        return error_response(
            503,
            "service_busy",
            "Prediction capacity reached; retry later",
            request.state.request_id,
            {"Retry-After": "1"},
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        """Uniformiser les erreurs HTTP connues, dont 401, 404, 405 et 503.

        Paramètres :
            request : fournit le request_id pour relier réponse et logs.
            exc : fournit le statut et les headers utiles, comme WWW-Authenticate.

        Retour :
            JSONResponse utilisant l'enveloppe d'erreur commune. Le détail original de
            l'exception est remplacé par un message public choisi dans le dictionnaire."""
        messages = {
            401: "Authentication required",
            404: "Route not found",
            405: "Method not allowed",
            503: "Model is not ready",
        }
        return error_response(
            exc.status_code,
            f"http_{exc.status_code}",
            messages.get(exc.status_code, "HTTP request failed"),
            request.state.request_id,
            exc.headers,
        )

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live():
        """Répondre à GET /health/live sans authentification.

        Retour :
            HealthResponse(status="alive") lorsque le processus peut répondre.

        Cette sonde n'exécute pas de prédiction et ne garantit pas la santé du modèle.
        Elle permet à l'orchestrateur de distinguer un processus arrêté d'un processus actif."""
        return HealthResponse(status="alive")

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def ready(service: Annotated[RealtimeInferenceService, Depends(get_service)]):
        """Répondre à GET /health/ready après résolution du service.

        Paramètre :
            service : injecté par get_service ; sa présence est le contrôle effectué ici.

        Retour :
            HealthResponse(status="ready"). Si le service est absent, la dépendance
            renvoie déjà 503 et cette fonction n'est pas exécutée.

        Le préchauffage a lieu au démarrage, pas à chaque appel de cette sonde."""
        return HealthResponse(status="ready")

    @app.get(
        "/v1/model",
        response_model=ModelResponse,
        tags=["model"],
        dependencies=[Depends(require_api_key)],
    )
    def model(service: Annotated[RealtimeInferenceService, Depends(get_service)]):
        """Exposer le contrat de la version en mémoire via GET /v1/model.

        Paramètre :
            service : instance injectée après contrôle de disponibilité.

        Retour :
            ModelResponse : version, seuil, colonnes requises et règles métier.

        L'authentification est déclarée sur le décorateur de route. La méthode describe
        lit les objets déjà chargés et ne consulte pas à nouveau latest.json."""
        return service.describe()

    @app.post(
        "/v1/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
    )
    def predict(
        application: ApplicationRequest,
        request: Request,
        service: Annotated[RealtimeInferenceService, Depends(get_service)],
    ):
        """Traiter une demande validée reçue sur POST /v1/predict.

        Paramètres :
            application : données converties en ApplicationRequest par FastAPI/Pydantic.
            request : fournit le request_id produit par le middleware.
            service : moteur métier injecté avec Depends(get_service).

        Retour :
            PredictionResponse contenant score, décision, version et corrélation.

        Cette route utilise def : FastAPI exécute le calcul synchrone dans son pool de
        threads. Cela évite de bloquer directement la boucle async avec pandas/le modèle.
        Les erreurs métier remontent vers les handlers enregistrés plus haut."""
        return service.predict(application, request.state.request_id)

    @app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_api_key)])
    def prometheus_metrics():
        """Exporter les compteurs de cette application au format Prometheus.

        Retour :
            Response contenant du texte et le Content-Type attendu par Prometheus.

        La route est authentifiée mais masquée dans OpenAPI. Elle expose les requêtes
        et durées enregistrées par le middleware, sans recalculer de prédiction.
        Chaque processus dispose de son registre ; Prometheus agrège les réplicas."""
        return Response(
            generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    return app
