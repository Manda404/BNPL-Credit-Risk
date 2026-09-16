# API d'inférence en temps réel — notebook 07

Pour découvrir chaque classe et le parcours d’une requête, commencer par le
[README pédagogique de l’API](../README_API.md). Ce guide décrit son exploitation.

## Architecture et intégration

Le notebook [07](../notebooks/07_realtime_inference.ipynb) teste le service ; le code
exécuté en déploiement réside dans `src/bnpl_credit_risk/api/`.

```text
Requête JSON → middleware (taille, request_id, logs, métriques)
             → authentification X-API-Key + schéma Pydantic
             → RealtimeInferenceService
             → BNPLDataCleaner + BNPLDataValidator (règles configs/data.yaml)
             → Predictor → artefact approuvé du notebook 05
             → probabilité + seuil publié + classe + risk_band + version + UTC
```

| Module | Responsabilité |
|---|---|
| `settings.py` | Configuration typée `BNPL_API_*`, secret masqué et version figée en production |
| `schemas.py` | Contrats v1, nombres finis, champs supplémentaires interdits |
| `service.py` | Chargement unique, approbation, compatibilité, préchauffage, capacité bornée |
| `app.py` | Factory FastAPI, lifespan, injection, authentification, routes et erreurs |
| `middleware.py` | Limite du corps, logs JSON Loguru, corrélation, métriques Prometheus |
| `client.py` | Client HTTP Python avec connexions réutilisées et délais bornés |
| `cli.py` | Démarrage Uvicorn et arrêt gracieux |

Le scope v1 est **application_risk**, pour les informations disponibles lors de
la demande. Les modèles behavioral_risk sont refusés. L'API refuse aussi la cible,
les variables de remboursement et les features calculées envoyées par un client.
Le pipeline sauvegardé garde la responsabilité de ses transformations ; aucune
opération de fit ou recalibration n'est exécutée par le service.

Le modèle actuellement publié par 05 est un `FittedBoostingModel` : ses colonnes
numériques et son vocabulaire catégoriel sont conservés dans l'artefact. Les
pipelines sklearn complets sont également acceptés quand leur contrat d'entrée
correspond à v1. Le préchauffage détecte les incompatibilités avant la readiness.

## Démarrage local

Depuis la racine, après publication du modèle à la fin de 05 :

```bash
poetry install
# Garder ce secret dans l'environnement local, jamais dans Git ou le notebook.
export BNPL_API_API_KEY="$(poetry run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
poetry run bnpl-api
```

L'API écoute sur `http://127.0.0.1:8000`. Swagger est disponible sur `/docs` en
mode development ; utiliser **Authorize** avec la clé créée. Les variables de
[.env.api.example](../.env.api.example) peuvent être ajoutées à `.env`, déjà ignoré
par Git. Ne pas laisser une variable `BNPL_API_MODEL_VERSION` vide : omettre la
variable pour utiliser la configuration d'inférence, ou renseigner une version.

Le notebook 07 lance Uvicorn en arrière-plan juste après les imports, avec une clé
aléatoire transmise au processus. La cellule suivante envoie un vrai POST de test
sur `127.0.0.1:8000` et vérifie la réponse. Les tests complémentaires utilisent
TestClient en mémoire. La cellule « Arrêter l’API » ferme le serveur créé par le
notebook ; l’exécuter après les essais. Le port local doit être disponible.

## Contrat HTTP

| Méthode et route | Authentification | Usage |
|---|---|---|
| `GET /health/live` | Publique | Le processus répond |
| `GET /health/ready` | Publique | Le modèle est chargé et a passé le préchauffage |
| `GET /v1/model` | X-API-Key | Version, seuil, features, champs requis, bornes et catégories |
| `POST /v1/predict` | X-API-Key | Une demande de crédit par requête |
| `GET /metrics` | X-API-Key | Compteurs et histogrammes Prometheus |
| `GET /docs`, `/openapi.json` | Development uniquement | Documentation interactive |

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/predict \
  -H "X-API-Key: $BNPL_API_API_KEY" -H 'Content-Type: application/json' \
  -d '{"user_id":123,"age":35,"employment_type":"Salaried",
       "monthly_income":4000,"credit_score":700,"purchase_amount":300,
       "product_category":"Electronics","bnpl_installments":3,
       "app_usage_frequency":10,"location":"USA",
       "transaction_date":"2026-01-01","debt_to_income_ratio":0.2}'
```

Réponse : `user_id`, `default_probability`, `default_risk_class`,
`decision_threshold`, `risk_band`, `model_version`, `scoring_timestamp` en UTC et
`request_id`. La classe vaut 1 lorsque `probabilité >= seuil`. Il s'agit d'un score
de défaut, pas d'une décision automatique d'octroi complète.
Le champ HTTP `default_risk_class` correspond à la colonne `predicted_default`
du batch, dont le contrat reste inchangé.

Les données numériques doivent être des nombres JSON finis ; les chaînes de
chiffres et les booléens sont rejetés. `transaction_date` utilise `YYYY-MM-DD`.
Le contrat métier suit `configs/data.yaml` ; `/v1/model` expose les valeurs
admises. Les identifiants sont des entiers positifs ou nuls sur 64 bits.

```json
{"error":{"code":"validation_error","message":"Request does not match API v1 schema","request_id":"..."}}
```

| Code HTTP | Sens | Action du client |
|---|---|---|
| 401 | Clé absente ou incorrecte | Vérifier le secret |
| 413 | Corps supérieur à 16 Kio par défaut, y compris chunked | Réduire le corps |
| 415 | Content-Type incorrect | Envoyer application/json |
| 422 | JSON, schéma ou règles métier invalides | Corriger les entrées selon le contrat |
| 503 + Retry-After | Capacité de scoring atteinte | Réessayer avec attente et jitter bornés |
| 503 | Modèle indisponible | Vérifier la readiness et le déploiement |
| 500 | Erreur de calcul ou sortie invalide | Rechercher request_id dans les logs |

Les erreurs n'incluent pas les valeurs d'entrée. Les logs contiennent route,
statut, durée, request_id et type d'erreur, sans corps, clé, identifiant client ni
URL arbitraire. Les sorties HTTP utilisent `Cache-Control: no-store`.

## Déploiement conteneurisé

```bash
export BNPL_API_MODEL_VERSION=2026-08-28_131115  # Remplacer par la version approuvée à servir.
# BNPL_API_API_KEY doit aussi être défini dans le shell ou .env.
docker compose -f docker-compose.api.yml up --build -d
curl --fail http://127.0.0.1:8000/health/ready
docker compose -f docker-compose.api.yml logs --tail=50 api
```

Le Dockerfile dédié utilise Poetry et son lock, un utilisateur non-root et
OpenMP. Compose monte les artefacts en lecture seule, limite CPU/mémoire, retire
les capabilities et rend le filesystem en lecture seule avec un `/tmp` dédié.
L'image n'embarque ni données, ni modèles, ni secrets. Le service batch historique
et son Dockerfile restent utilisables séparément.

Les fichiers joblib sont exécutables à la désérialisation : le registre monté
doit provenir de la chaîne de publication de confiance. La clé est injectée par
l'environnement, idéalement depuis le gestionnaire de secrets de la plateforme.
L'approbation du modèle est toujours exigée pour l'API, même si le batch a été
configuré pour l'ignorer. `latest` est accepté uniquement en développement ;
en production la version doit être exacte et la publication doit être immuable.

## Exploitation

- **Mise en ligne** : terminer 05, fournir le secret et la version, démarrer une
  instance, attendre `/health/ready`, vérifier un scoring connu, puis router le trafic.
- **Mise à jour / rollback** : déployer une nouvelle instance avec une autre version
  approuvée. Le service ne recharge pas `latest` pendant une requête. Redéployer
  la version précédente constitue le rollback ; ne pas réécrire un artefact actif.
- **Concurrence** : une prédiction simultanée par défaut et par instance ; les autres
  reçoivent 503. Les routes de calcul sont exécutées dans le pool de threads FastAPI.
  Augmenter les réplicas après mesure. Une valeur supérieure à 1 exige des tests
  de sûreté et de charge sur le pipeline natif choisi.
- **Mesure** : `bnpl_http_requests_total{route,status}` et
  `bnpl_http_request_duration_seconds{route}`. Les labels ne contiennent aucune
  donnée client. Scraper `/metrics` avec `X-API-Key` via un secret Prometheus.
  Le registre est par processus : garder un worker Uvicorn par conteneur et
  agréger les réplicas dans Prometheus. La limite Uvicorn de 64 connexions peut
  produire ses propres 503 avant le middleware, hors métriques applicatives.
- **Arrêt** : SIGTERM laisse Uvicorn terminer les requêtes pendant 30 secondes ;
  Compose accorde 40 secondes avant de tuer le processus.
- **Panne** : une erreur de chargement arrête le démarrage ; vérifier version,
  approbation, scope, contrat, bibliothèques et droits de lecture. Une exception
  de prédiction rend 500 sans contenu personnel et libère le créneau de calcul.

### Éléments à fournir sur la plateforme cible

L'application et le conteneur ne constituent pas à eux seuls un déploiement
validé sous charge. Fournir un ingress/reverse proxy pour TLS, quotas par client,
limites et délais de lecture HTTP, ainsi que la collecte des logs, alertes et
secrets. Aucun proxy n'est implicitement considéré comme fiable. Le binding
Compose est local pour permettre cette mise en place explicite.

Les délais HTTP du client ne tuent pas une prédiction native déjà en cours.
Un blocage natif prolongé nécessite le remplacement du processus par la plateforme
et des alertes sur les latences/503 ; il n'y a pas de timeout de calcul dur dans
ce service. Le notebook mesure une latence locale indicative, sans revendiquer
un SLA ni simuler une charge de production.

## Validation reproductible

```bash
poetry run python -m pytest tests/integration/test_realtime_api.py --no-cov
poetry run python -m ruff check src/bnpl_credit_risk/api tests/integration/test_realtime_api.py
poetry run jupyter execute notebooks/07_realtime_inference.ipynb --output /tmp/07_realtime_inference.executed.ipynb
```

La suite couvre contrats, authentification, non-fuite des erreurs/logs, santé,
métriques, saturation, taille chunked, démarrage/arrêt, version figée, refus des
modèles incompatibles et parité avec un pipeline sklearn réellement entraîné.
Le notebook vérifie en plus la parité avec le modèle publié et le batch du projet.

Références FastAPI utilisées : [lifespan](https://fastapi.tiangolo.com/advanced/events/),
[tests du cycle de vie](https://fastapi.tiangolo.com/advanced/testing-events/),
[concurrence](https://fastapi.tiangolo.com/async/),
[déploiement Docker](https://fastapi.tiangolo.com/deployment/docker/).
