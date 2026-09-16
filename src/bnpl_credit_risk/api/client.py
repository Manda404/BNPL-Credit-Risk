"""Client Python réutilisable par le notebook 07 et les consommateurs de l'API."""

from typing import Any

import httpx

from bnpl_credit_risk.api.schemas import ApplicationRequest, ModelResponse, PredictionResponse


class BNPLAPIClient:
    """Réutilise les connexions HTTP et impose des délais bornés.

    Aucun retry implicite : le notebook et l'appelant distinguent une panne
    réseau d'un refus 422/503 et décident explicitement de leur politique.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ):
        """Configurer un client HTTP réutilisable sans envoyer de requête.

        Paramètres :
            base_url : URL racine du service, par exemple http://127.0.0.1:8000.
            api_key : secret à transmettre dans X-API-Key à chaque appel.
            timeout : délais de lecture, écriture et acquisition de connexion en secondes.
                Le délai de connexion est fixé séparément à 3 secondes.
            transport : transport HTTPX facultatif ; les tests peuvent fournir un MockTransport.

        Le pool conserve jusqu'à 5 connexions réutilisables et autorise 10 connexions.
        Les redirections ne sont pas suivies automatiquement. Ces délais ne constituent
        pas un timeout total et n'interrompent pas un calcul déjà commencé sur le serveur."""
        self._client = httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(timeout, connect=3.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            transport=transport,
            follow_redirects=False,
        )

    def __enter__(self) -> "BNPLAPIClient":
        """Ouvrir un bloc with et retourner ce client.

        Permet l'écriture with BNPLAPIClient(url, key) as client. Aucune requête n'est
        envoyée ici ; __exit__ fermera le pool même si le contenu du bloc échoue."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Fermer le client à la sortie d'un bloc with, avec ou sans exception.

        args reçoit le type, la valeur et le traceback de l'éventuelle exception,
        selon le protocole Python des gestionnaires de contexte. Ils ne sont pas loggés.
        Le retour None laisse une éventuelle exception remonter à l'appelant."""
        self.close()

    def close(self) -> None:
        """Libérer les connexions et ressources du client HTTPX.

        À appeler explicitement si le client n'est pas utilisé avec with. Retourne None.
        Le client fermé ne doit plus servir à envoyer des requêtes."""
        self._client.close()

    def model(self) -> ModelResponse:
        """Lire GET /v1/model et convertir le JSON en ModelResponse.

        Retour :
            Contrat typé de la version actuellement servie.

        Lève une exception HTTPX en cas de réseau/timeout/statut non réussi, ou une
        ValidationError si le JSON ne respecte pas le schéma attendu. Aucun retry implicite."""
        response = self._client.get("/v1/model")
        # Séparer une erreur HTTP du serveur d’un JSON de succès mal formé.
        response.raise_for_status()
        return ModelResponse.model_validate(response.json())

    def predict(self, application: ApplicationRequest) -> PredictionResponse:
        """Envoyer une demande à POST /v1/predict et valider la réponse.

        Paramètre :
            application : ApplicationRequest, dont la date sera sérialisée en chaîne JSON.

        Retour :
            PredictionResponse contenant score, seuil, classe, version et request_id.

        raise_for_status() fait remonter les statuts non réussis via HTTPX ; les erreurs
        réseau et de validation ne sont pas masquées. Le client ne calcule aucun score."""
        response = self._client.post("/v1/predict", json=application.model_dump(mode="json"))
        # Séparer une erreur HTTP du serveur d’un JSON de succès mal formé.
        response.raise_for_status()
        return PredictionResponse.model_validate(response.json())
