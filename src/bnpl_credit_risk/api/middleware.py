"""Protection ASGI des corps HTTP, corrélation et observabilité sans données clients."""

from collections.abc import Mapping
from time import perf_counter
from uuid import uuid4

from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class APIMetrics:
    """Registre indépendant par application, adapté à un processus par conteneur."""

    def __init__(self) -> None:
        """Créer un registre Prometheus isolé et ses deux instruments.

        requests compte les réponses par route/statut. latency répartit les durées,
        en secondes, dans des tranches de 5 ms à 5 s, plus la tranche infinie ajoutée
        par Prometheus. Les instruments commencent sans observation.

        Un registre par instance évite les doublons lors des tests créant plusieurs
        applications. Aucun identifiant de client ou de requête n'est utilisé comme label."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "bnpl_http_requests_total",
            "Requêtes HTTP terminées",
            ["route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "bnpl_http_request_duration_seconds",
            "Latence HTTP incluant validation et scoring",
            ["route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )


def error_response(
    status: int, code: str, message: str, request_id: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """Construire l'enveloppe JSON commune aux erreurs publiques.

    Paramètres :
        status : code HTTP, par exemple 422 ou 503.
        code : code applicatif stable, par exemple validation_error.
        message : texte public déjà choisi par l'appelant, sans donnée sensible.
        request_id : identifiant permettant de retrouver la requête dans les logs.
        headers : headers facultatifs, par exemple Retry-After ou WWW-Authenticate.

    Retour :
        JSONResponse contenant error.code, error.message et error.request_id.

    Cette fonction n'assainit pas un texte arbitraire : l'appelant doit fournir un
    message sûr, jamais str(exception) ou le corps reçu."""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers=headers,
    )


class RequestMiddleware:
    """Borne aussi les corps chunked, avant le décodage JSON par FastAPI.

    L'identifiant est généré par le serveur : aucun header client arbitraire ne
    peut injecter du texte dans les logs. Les métriques n'utilisent que les routes
    connues, jamais les URL, identifiants clients ou clés d'authentification.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int, metrics: APIMetrics):
        """Conserver les dépendances nécessaires au traitement HTTP.

        Paramètres :
            app : application ASGI suivante dans la chaîne de middleware.
            max_body_bytes : taille maximale du corps reçu, exprimée en octets.
            metrics : registre partagé avec la route /metrics de cette application.

        Aucune requête n'est traitée et aucun modèle n'est chargé à la construction."""
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Encadrer une requête ASGI et transmettre sa réponse au serveur.

        Paramètres :
            scope : contexte ASGI contenant type, méthode, headers et état de la requête.
            receive : coroutine permettant de lire les fragments du corps ou la déconnexion.
            send : coroutine permettant d'envoyer statut, headers et fragments de réponse.

        Les événements non HTTP sont transmis directement. Pour HTTP : créer un UUID,
        borner le corps, le relire pour FastAPI, ajouter les headers de réponse puis
        compter le statut et la durée, même en cas d'erreur. Retourne None.

        Une exception avant l'envoi des headers devient une réponse 500 générique.
        Après cet envoi, elle est relancée : on ne peut plus remplacer le statut HTTP."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        # Compteur monotone : une correction de l’horloge système ne fausse pas la durée.
        start = perf_counter()
        status = 500
        # Après les headers, le protocole HTTP ne permet plus de changer le statut.
        started = False

        async def traced_send(message: Message) -> None:
            """Intercepter un message de réponse avant son envoi au client.

            message est un événement ASGI. Au premier http.response.start, mémoriser le
            statut et ajouter corrélation, interdiction de cache et protection nosniff.
            Les fragments de corps sont transmis tels quels à send().
            nonlocal modifie l'état de la requête englobante, pas un état partagé global."""
            nonlocal status, started
            if message["type"] == "http.response.start":
                status = message["status"]
                started = True
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        async def reject(status_code: int, code: str, message: str) -> None:
            """Envoyer une erreur dès le middleware sans appeler la route.

            status_code, code et message décrivent l'erreur publique. request_id et les
            canaux ASGI viennent de la requête englobante. traced_send ajoute les mêmes
            headers que pour une réponse normale ; le bloc finally comptera aussi ce refus."""
            await error_response(status_code, code, message, request_id)(
                scope, receive, traced_send
            )

        try:
            headers = dict(scope.get("headers", []))
            # Content-Length permet un refus rapide ; le comptage réel reste indispensable.
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    size = int(length)
                    if size < 0:
                        raise ValueError()
                except ValueError:
                    await reject(400, "invalid_content_length", "Invalid Content-Length")
                    return
                if size > self.max_body_bytes:
                    await reject(413, "payload_too_large", "Request body exceeds the limit")
                    return
            if scope["method"] == "POST":
                content_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
                if content_type != b"application/json":
                    await reject(415, "unsupported_media_type", "Use application/json")
                    return
            # La taille est comptée sur les octets reçus, même sans Content-Length.
            # Le corps est conservé uniquement en mémoire et ne doit jamais entrer dans les logs.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    # 499 est une convention de suivi interne : rien n’est envoyé au client parti.
                    status = 499
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.max_body_bytes:
                    await reject(413, "payload_too_large", "Request body exceeds the limit")
                    return
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay() -> Message:
                """Fournir à FastAPI le corps déjà consommé pour contrôler sa taille.

                Le premier appel renvoie tous les octets conservés avec more_body=False.
                Les appels suivants délèguent à receive(), notamment pour une déconnexion.
                Sans cette fonction, FastAPI essaierait de relire un flux déjà consommé.
                Retourne un Message ASGI ; delivered appartient uniquement à cette requête."""
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            # Le contexte Loguru suit aussi le travail transféré au pool de threads.
            with logger.contextualize(request_id=request_id):
                await self.app(scope, replay, traced_send)
        except Exception as exc:
            # Ne pas logger str(exc), traceback ou variables locales : un modèle peut
            # inclure les données d'entrée dans son message d'erreur.
            logger.bind(pipeline="api", request_id=request_id, error_type=type(exc).__name__).error(
                "request_failed"
            )
            if started:
                raise
            await reject(500, "internal_error", "Internal server error")
        finally:
            elapsed = perf_counter() - start
            # Une route connue, ou unmatched, borne le nombre de séries dans Prometheus.
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.requests.labels(route, str(status)).inc()
            self.metrics.latency.labels(route).observe(elapsed)
            logger.bind(
                pipeline="api",
                request_id=request_id,
                route=route,
                status=status,
                duration_ms=round(elapsed * 1000, 3),
            ).info("http_request")
