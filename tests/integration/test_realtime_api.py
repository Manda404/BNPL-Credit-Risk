"""Tests HTTP et métier de l'API ; artefacts synthétiques isolés du registre réel."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from pydantic import ValidationError

from bnpl_credit_risk.api.app import create_app
from bnpl_credit_risk.api.client import BNPLAPIClient
from bnpl_credit_risk.api.schemas import ApplicationRequest
from bnpl_credit_risk.api.service import RealtimeInferenceService
from bnpl_credit_risk.api.settings import APISettings
from bnpl_credit_risk.inference.predictor import Predictor
from bnpl_credit_risk.models.persistence import ArtifactBundle

KEY = "integration-test-secret-32-characters"
HEADERS = {"X-API-Key": KEY}


class ConstantClassifier:
    """Petit modèle sérialisable ; la parité avec un modèle entraîné est testée plus bas."""

    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        """Renvoyer [0.6, 0.4] pour chaque ligne afin de tester le seuil sans entraînement."""
        return np.tile([0.6, 0.4], (len(frame), 1))


@pytest.fixture
def payload():
    """Fournir une demande synthétique valide, recréée pour chaque test afin de rester isolée."""
    return dict(
        user_id=123,
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


@pytest.fixture
def api_settings():
    """Construire des paramètres de test avec une clé fictive et sans lecture du fichier .env."""
    return APISettings(api_key=KEY, _env_file=None)


@pytest.fixture
def bundle(payload):
    """Assembler un artefact fictif approuvé avec un seuil de 0.4 et des bandes connues."""
    return ArtifactBundle(
        pipeline=ConstantClassifier(),
        metadata={
            "approved_for_inference": True,
            "risk_scope": "application_risk",
            "algorithm": "test",
        },
        metrics={},
        threshold={
            "threshold": 0.4,
            "risk_bands": [
                {"name": "Low Risk", "max_probability": 0.25},
                {"name": "High Risk", "max_probability": 1.01},
            ],
        },
        feature_schema={"input_features": [k for k in payload if k != "user_id"]},
        version="test-v1",
    )


@pytest.fixture
def service(bundle, project_config, api_settings):
    """Injecter le Predictor fictif dans le vrai service, avec les règles YAML du projet."""
    return RealtimeInferenceService(Predictor(bundle), project_config, api_settings)


@pytest.fixture
def client(service, api_settings):
    """Démarrer la vraie application via TestClient et la fermer après le test, même en cas d’erreur."""
    with TestClient(create_app(api_settings, service_factory=lambda: service)) as client:
        yield client


def test_prediction_contract_and_threshold_boundary(client, payload):
    """Vérifier la réponse HTTP, la corrélation, UTC et le cas limite probabilité égale au seuil."""
    response = client.post("/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["user_id"] == 123
    assert result["default_probability"] == 0.4
    assert "predicted_default" not in result
    assert result["default_risk_class"] == 1  # p == threshold is positive in Predictor.
    assert result["risk_band"] == "High Risk"
    assert result["model_version"] == "test-v1"
    assert result["scoring_timestamp"].endswith("Z")
    assert result["request_id"] == response.headers["x-request-id"]
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "patch",
    [
        {"age": 17},
        {"age": "35"},
        {"age": True},
        {"monthly_income": -1},
        {"credit_score": 851},
        {"bnpl_installments": 0},
        {"employment_type": "invalid"},
        {"transaction_date": "not-a-date"},
        {"transaction_date": "2026-02-30"},
        {"transaction_date": 1767225600},
        {"transaction_date": "2026-01-01T00:00:00"},
        {"user_id": "123"},
        {"user_id": True},
        {"monthly_income": None},
        {"default_flag": 1},
        {"missed_payments": 1},
        {"risk_score": 300},
        {"installment_amount": 10},
        {"unknown": "private-value"},
    ],
)
def test_invalid_input_is_rejected_without_echo(client, payload, patch):
    """Envoyer chaque patch invalide et exiger 422 sans copie des valeurs personnelles dans la réponse."""
    response = client.post("/v1/predict", json={**payload, **patch}, headers=HEADERS)
    assert response.status_code == 422
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert "private-value" not in response.text
    assert "input" not in response.json()["error"]


def test_missing_field_and_nonfinite_json(client, payload):
    """Vérifier qu’un champ absent et un NaN JSON produisent une erreur de validation contrôlée."""
    payload.pop("age")
    assert client.post("/v1/predict", json=payload, headers=HEADERS).status_code == 422
    payload["age"] = 35
    import json

    raw = json.dumps({**payload, "age": float("nan")})
    response = client.post(
        "/v1/predict", content=raw, headers={**HEADERS, "Content-Type": "application/json"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/v1/model", "/metrics", "/v1/predict"])
def test_authentication_required(client, payload, path):
    """Exiger 401 sur chaque route protégée lorsque la clé manque ou est incorrecte."""
    method = client.post if path.endswith("predict") else client.get
    kwargs = {"json": payload} if path.endswith("predict") else {}
    for headers in ({}, {"X-API-Key": "wrong"}):
        response = method(path, headers=headers, **kwargs)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "APIKey"


def test_health_metadata_openapi_and_metrics(client):
    """Contrôler les sondes publiques, le contrat exposé, la sécurité OpenAPI et les métriques protégées."""
    assert client.get("/health/live").json() == {"status": "alive"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    metadata = client.get("/v1/model", headers=HEADERS).json()
    assert metadata["risk_scope"] == "application_risk"
    assert metadata["bounds"]["age"] == {"min": 18, "max": 100}
    spec = client.get("/openapi.json").json()
    assert spec["paths"]["/v1/predict"]["post"]["security"]
    response = client.get("/metrics", headers=HEADERS)
    assert response.status_code == 200
    assert "bnpl_http_requests_total" in response.text
    assert "bnpl_http_request_duration_seconds" in response.text
    assert KEY not in response.text


def test_http_errors_and_body_limits(client):
    """Exercer 404, 405, 415, JSON invalide et limite de taille avec ou sans Content-Length."""
    assert client.get("/missing").status_code == 404
    assert client.get("/v1/predict").status_code == 405
    assert client.post("/v1/predict", content="x", headers=HEADERS).status_code == 415
    headers = {**HEADERS, "Content-Type": "application/json"}
    assert client.post("/v1/predict", content="{broken", headers=headers).status_code == 422
    assert client.post("/v1/predict", content="x" * 17000, headers=headers).status_code == 413
    # Un générateur enlève Content-Length : la limite doit rester effective.
    response = client.post("/v1/predict", content=iter([b"x" * 9000, b"y" * 9000]), headers=headers)
    assert response.status_code == 413


def test_internal_errors_are_private_and_slot_recovers(client, service, payload, monkeypatch):
    """Simuler une panne contenant un secret et vérifier son masquage puis la récupération du créneau."""
    original = service.predictor.score

    def fail(*args):
        """Lever volontairement une exception sensible pour exercer le traitement sécurisé des erreurs."""
        raise RuntimeError("private-customer-secret")

    monkeypatch.setattr(service.predictor, "score", fail)
    logs = []
    sink = logger.add(lambda message: logs.append(str(message)), serialize=True)
    try:
        response = client.post("/v1/predict", json=payload, headers=HEADERS)
    finally:
        logger.remove(sink)
    assert response.status_code == 500
    assert "private-customer-secret" not in response.text + "".join(logs)
    assert KEY not in "".join(logs)
    monkeypatch.setattr(service.predictor, "score", original)
    assert client.post("/v1/predict", json=payload, headers=HEADERS).status_code == 200


def test_busy_rejects_concurrent_prediction(client, service, payload, monkeypatch):
    """Maintenir une prédiction occupée et vérifier que la suivante reçoit 503 tandis que la santé répond."""
    entered, release = Event(), Event()
    original = service.predictor.score

    def slow(*args):
        """Signaler l’entrée dans le modèle puis attendre le signal de libération, avec une limite de 5 secondes."""
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(service.predictor, "score", slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/v1/predict", json=payload, headers=HEADERS)
        try:
            assert entered.wait(5)
            assert client.get("/health/live").status_code == 200
            rejected = client.post("/v1/predict", json=payload, headers=HEADERS)
            assert rejected.status_code == 503
            assert rejected.headers["retry-after"] == "1"
        finally:
            release.set()
        assert pending.result().status_code == 200


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.metadata.update(approved_for_inference=False),
        lambda b: b.metadata.update(risk_scope="behavioral_risk"),
        lambda b: b.feature_schema.update(input_features=["missed_payments"]),
        lambda b: b.threshold.update(threshold=float("nan")),
        lambda b: b.threshold.update(threshold=1.1),
        lambda b: b.threshold.update(risk_bands=[]),
        lambda b: b.threshold.update(risk_bands=[{"name": "Low", "max_probability": 0.5}]),
        lambda b: setattr(b.pipeline, "classes_", np.array([1, 0])),
    ],
)
def test_bad_artifacts_fail_before_serving(bundle, project_config, api_settings, mutate):
    """Appliquer une corruption ciblée à l’artefact et vérifier son rejet avant le démarrage HTTP."""
    mutate(bundle)
    with pytest.raises(ValueError):
        RealtimeInferenceService(Predictor(bundle), project_config, api_settings)


def test_startup_loading_pinning_and_shutdown(
    bundle, tmp_path, project_config, api_settings, test_settings, payload
):
    """Vérifier un chargement par démarrage, la stabilité malgré une promotion et le nettoyage à l’arrêt."""
    models = test_settings.resolve(test_settings.artifacts_dir) / "models"
    bundle.save(models)
    loads = []

    def factory():
        """Compter les chargements et charger le modèle depuis le registre temporaire propre au test."""
        loads.append(1)
        return RealtimeInferenceService.load(project_config, test_settings, api_settings)

    app = create_app(api_settings, service_factory=factory)
    with TestClient(app) as client:
        # Promouvoir une autre version ne change pas le modèle déjà chargé.
        other = deepcopy(bundle)
        other.version = "test-v2"
        other.save(models)
        for _ in range(2):
            result = client.post("/v1/predict", json=payload, headers=HEADERS).json()
            assert result["model_version"] == "test-v1"
    assert loads == [1]
    assert app.state.service is None
    with TestClient(create_app(api_settings, service_factory=factory)) as client:
        assert client.get("/v1/model", headers=HEADERS).json()["model_version"] == "test-v2"


def test_missing_artifact_fails_startup(api_settings, test_settings):
    """Exiger l’échec du lifespan lorsque le registre temporaire ne contient aucun modèle publié."""
    with (
        pytest.raises(RuntimeError, match="API startup failed"),
        TestClient(create_app(api_settings, test_settings)),
    ):
        pass


def test_registry_escape_rejected(tmp_path, bundle, project_config, api_settings, test_settings):
    """Écrire un latest.json pointant hors du registre temporaire et vérifier le refus avant joblib."""
    import json

    models = test_settings.resolve(test_settings.artifacts_dir) / "models"
    models.mkdir(parents=True)
    (models.parent / "escape").mkdir()
    (models / "latest.json").write_text(json.dumps({"version": "../escape"}))
    with pytest.raises(ValueError, match="inside the model registry"):
        RealtimeInferenceService.load(project_config, test_settings, api_settings)


def test_production_configuration_and_docs(api_settings, service):
    """Vérifier clé minimale, version figée, refus des chemins et absence de Swagger en production."""
    with pytest.raises(ValidationError):
        APISettings(api_key=KEY, environment="production", _env_file=None)
    with pytest.raises(ValidationError):
        APISettings(api_key="short", _env_file=None)
    with pytest.raises(ValidationError):
        APISettings(api_key=KEY, model_version="../escape", _env_file=None)
    production = APISettings(
        api_key=KEY, environment="production", model_version="v1", _env_file=None
    )
    with TestClient(create_app(production, service_factory=lambda: service)) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_unready_returns_503(client):
    """Retirer le service de l’état applicatif et distinguer liveness 200 de readiness 503."""
    client.app.state.service = None
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503


def test_trained_pipeline_http_parity(sample_df, project_config, api_settings, bundle, payload):
    """Entraîner un petit pipeline réel et comparer sa prédiction directe à la réponse HTTP de l’API."""
    from bnpl_credit_risk.models.trainer import ModelTrainer

    model_config = project_config.model.model_copy(
        update={
            "xgboost_params": {
                **project_config.model.xgboost_params,
                "n_estimators": 5,
                "max_depth": 2,
            },
        }
    )
    bundle.pipeline = ModelTrainer(model_config, project_config.features).train(
        sample_df.drop(columns=["default_flag"]),
        sample_df["default_flag"],
        date_column="transaction_date",
    )
    service = RealtimeInferenceService(Predictor(bundle), project_config, api_settings)
    expected = service.predictor.score(pd.DataFrame([payload]), "user_id").iloc[0]
    with TestClient(create_app(api_settings, service_factory=lambda: service)) as client:
        response = client.post("/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["default_probability"] == pytest.approx(expected.default_probability)
    assert response.json()["default_risk_class"] == expected.predicted_default


def test_python_client(payload):
    """Vérifier l’envoi de la clé et la lecture d’une réponse typée avec un transport HTTP simulé."""
    import httpx

    def handler(request):
        """Vérifier le header secret du client puis retourner une réponse HTTP synthétique cohérente."""
        assert request.headers["X-API-Key"] == KEY
        return httpx.Response(
            200,
            json={
                "user_id": payload["user_id"],
                "default_probability": 0.4,
                "default_risk_class": 1,
                "decision_threshold": 0.4,
                "risk_band": "High Risk",
                "model_version": "v1",
                "scoring_timestamp": "2026-01-01T00:00:00Z",
                "request_id": "test",
            },
        )

    with BNPLAPIClient("http://test", KEY, transport=httpx.MockTransport(handler)) as client:
        assert client.predict(ApplicationRequest(**payload)).model_version == "v1"


def test_published_boosting_wrapper_is_supported(bundle, payload, project_config, api_settings):
    """Vérifier le format FittedBoostingModel publié par 05, dont les classes résident dans estimator."""
    from bnpl_credit_risk.models.boosting import CategoricalFrameAdapter, FittedBoostingModel

    features = ["age", "employment_type"]
    frame = pd.DataFrame([payload])
    bundle.pipeline = FittedBoostingModel(
        model_name="CatBoost",
        estimator=ConstantClassifier(),
        adapter=CategoricalFrameAdapter.fit(frame, ["employment_type"]),
        feature_names=features,
    )
    bundle.feature_schema = {"input_features": features}
    service = RealtimeInferenceService(Predictor(bundle), project_config, api_settings)
    service.warmup()
    assert service.predict(ApplicationRequest(**payload), "wrapper-test").default_probability == 0.4


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_model_probability_returns_500(client, service, payload, probability, monkeypatch):
    """Injecter une probabilité non finie ou hors intervalle et vérifier un HTTP 500 générique."""
    original = service.predictor.score

    def corrupt(*args):
        """Conserver une sortie normale sauf sa probabilité, remplacée par la valeur invalide à tester."""
        result = original(*args)
        result["default_probability"] = probability
        return result

    monkeypatch.setattr(service.predictor, "score", corrupt)
    response = client.post("/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
