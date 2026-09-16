"""Commande bnpl-api : un processus par instance, arrêt gracieux par Uvicorn."""

import uvicorn

from bnpl_credit_risk.api.settings import APISettings


def main() -> None:
    """Démarrer Uvicorn avec la configuration BNPL_API_*.

    Aucun argument Python : la commande bnpl-api est déclarée dans pyproject.toml.
    Valide d'abord les paramètres, puis demande à Uvicorn d'appeler create_app grâce
    à factory=True. Un seul worker garde un modèle et un registre de métriques.

    L'appel bloque pendant la vie du serveur. Les connexions inactives peuvent être
    fermées après 5 secondes ; à l'arrêt, les requêtes disposent de 30 secondes.
    La limite Uvicorn de 64 connexions/tâches est distincte de la limite de calcul
    du service. Aucun reloader ni confiance implicite dans un proxy n'est activé."""
    settings = APISettings()  # type: ignore[call-arg]  # BaseSettings lit la clé dans env.
    uvicorn.run(
        "bnpl_credit_risk.api.app:create_app",
        # Uvicorn appelle la factory ; le lifespan charge ensuite le modèle.
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
        # Les logs ASGI structurés remplacent les access logs contenant les URL brutes.
        access_log=False,
        server_header=False,
        proxy_headers=False,
        # Ces délais gèrent les connexions et l’arrêt, pas la durée maximale d’un calcul.
        timeout_keep_alive=5,
        timeout_graceful_shutdown=30,
        limit_concurrency=64,
        backlog=128,
    )


if __name__ == "__main__":
    main()
