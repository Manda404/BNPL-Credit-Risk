# Comprendre l’API FastAPI et le notebook 07

Ce document explique ce qui a été ajouté au projet, pourquoi chaque partie existe
et comment les morceaux fonctionnent ensemble. Commence par les sections 1 à 4,
puis ouvre les fichiers Python dans l’ordre indiqué à la section 5.

Le [guide d’exploitation](docs/realtime_api.md) complète ce README avec les commandes
de déploiement, la supervision et le retour à une version précédente.

## 1. Ce que l’on peut faire maintenant

Une application peut envoyer **une demande de crédit en JSON** à notre service et
recevoir immédiatement le score calculé par le modèle déjà publié :

```text
Un client Python, un site ou un autre service
    → POST /v1/predict avec les informations de la demande
    → API FastAPI
    → modèle approuvé et chargé en mémoire
    → réponse JSON avec probabilité, classe, bande de risque et version
```

**Inférence** signifie utiliser un modèle entraîné pour produire une prédiction.
Le service ne réentraîne pas le modèle et ne choisit pas un nouveau seuil.
Le score indique un risque de défaut ; il ne représente pas à lui seul une
politique complète d’acceptation ou de refus du crédit.

### Relation avec les notebooks

| Étape | Responsabilité |
|---|---|
| Notebook 04 | Comparer et entraîner les modèles du workflow de benchmark |
| Notebook 05 | Évaluer, choisir le seuil et publier un modèle approuvé |
| Notebook 06 | Scorer un fichier de demandes en batch |
| **Notebook 07** | Tester le même modèle à travers l’API HTTP |

Le **batch** traite un fichier de plusieurs lignes. Le **temps réel** traite une
demande par appel HTTP. Les deux utilisent `Predictor`, ce qui conserve la même
règle de décision et les mêmes bandes de risque pour une version donnée.

L’API v1 traite le scope `application_risk` : les informations disponibles au
moment de la demande. Elle refuse les modèles `behavioral_risk`, ainsi que les
entrées comme `missed_payments`, `repayment_delay_days` ou `default_flag`.
Ces valeurs décrivent des événements futurs ou la cible à prédire.

## 2. Où se trouve chaque élément

Les liens suivants ouvrent directement les fichiers concernés.

| Fichier | Ce qu’il apporte |
|---|---|
| [api/app.py](src/bnpl_credit_risk/api/app.py) | Assemble FastAPI, les routes, l’authentification, le démarrage et les erreurs |
| [api/service.py](src/bnpl_credit_risk/api/service.py) | Charge le modèle, vérifie sa compatibilité et orchestre la prédiction |
| [api/schemas.py](src/bnpl_credit_risk/api/schemas.py) | Définit la forme et les types des données entrantes et sortantes |
| [api/settings.py](src/bnpl_credit_risk/api/settings.py) | Lit et valide les paramètres `BNPL_API_*` |
| [api/middleware.py](src/bnpl_credit_risk/api/middleware.py) | Encadre les appels HTTP : taille, corrélation, logs et métriques |
| [api/client.py](src/bnpl_credit_risk/api/client.py) | Fournit une classe Python pour appeler l’API |
| [api/cli.py](src/bnpl_credit_risk/api/cli.py) | Lance le serveur via la commande `bnpl-api` |
| [api/__init__.py](src/bnpl_credit_risk/api/__init__.py) | Déclare le package ; ne charge aucun modèle à l’import |
| [Notebook 07](notebooks/07_realtime_inference.ipynb) | Exécute les tests et explique leurs résultats |
| [Tests de l’API](tests/integration/test_realtime_api.py) | Automatise les cas valides, invalides et les pannes simulées |
| [Dockerfile.api](Dockerfile.api) | Décrit la construction de l’image du service |
| [docker-compose.api.yml](docker-compose.api.yml) | Configure le conteneur, son port, ses ressources et son registre de modèles |
| [.env.api.example](.env.api.example) | Liste les paramètres à fournir, sans contenir de vraie clé |
| [.dockerignore](.dockerignore) | Exclut notamment secrets, données et modèles du contexte de construction |
| [Workflow CI](.github/workflows/api-tests.yml) | Prévoit lint, typage, tests et construction Docker dans GitHub Actions |

### Éléments existants réutilisés

- `BNPLDataCleaner` normalise les types et les catégories du DataFrame.
- `BNPLDataValidator` vérifie les règles définies dans `configs/data.yaml`.
- `Predictor` applique le modèle, son seuil et ses bandes.
- `ArtifactBundle` rassemble pipeline, métadonnées, seuil et contrat des features.
- Le registre `artifacts/models/` conserve les versions et le pointeur `latest.json`.

Le modèle publié par le workflow 04–05 utilise `FittedBoostingModel`, qui conserve
l’estimateur et son adaptation des catégories. Le service prend aussi en charge
les pipelines sklearn compatibles. Il **ne recalcule pas arbitrairement toutes
les features du projet** : les transformations exécutées sont celles de l’artefact.

### Autres adaptations réalisées

`pyproject.toml` déclare les dépendances FastAPI, Uvicorn, HTTPX et Prometheus,
ainsi que la commande `bnpl-api`. `poetry.lock` verrouille les dépendances.
`Makefile` propose `serve-api` et `test-api`. `.gitignore` protège les fichiers
`.env` et leurs variantes tout en permettant le modèle public `.env.api.example`.

La fonction historique `inference/realtime.py::score_single` reste disponible
pour les appels Python internes ; elle n’applique pas les contrôles HTTP.
Les docstrings du moteur `Predictor` expliquent cette distinction.

La fixture des tests historiques de parité a aussi été isolée de MLflow : tester
les scores ne doit pas écrire dans la base personnelle configurée dans le projet.
Ce réglage concerne les tests, pas la configuration de suivi des entraînements.

## 3. Ce qui se passe au démarrage

1. `bnpl-api` appelle `cli.main()`.
2. `APISettings()` lit les paramètres et valide notamment la clé et la version.
3. Uvicorn démarre et appelle `create_app()` : cette fonction construit l’application.
4. FastAPI entre dans `lifespan()` avant de servir les requêtes.
5. `RealtimeInferenceService.load()` résout la version puis charge son artefact.
6. `_validate_artifact()` contrôle approbation, scope, features, seuil, bandes et classes.
7. `warmup()` fait une vraie prédiction sur un exemple synthétique.
8. Le service est conservé dans `app.state.service` et l’application devient disponible.

**Un modèle est chargé par processus**, pas à chaque requête. En développement,
`latest` est résolu au démarrage. Modifier `latest.json` ensuite ne remplace pas
le modèle en mémoire : il faut redémarrer une instance pour changer sa version.
En production, une version exacte est obligatoire.

Si le modèle manque, n’est pas approuvé ou échoue au préchauffage, le démarrage
échoue. Le serveur ne doit pas annoncer qu’il est prêt à calculer des scores.

### Pourquoi `lifespan` contient `yield`

```text
avant yield : préparation des ressources
pendant yield : fonctionnement normal du serveur
après yield : nettoyage à l’arrêt
```

Le `finally` retire le service de l’état de l’application, même si une erreur
intervient. Le modèle reste utilisé par les requêtes en cours selon la gestion
d’arrêt d’Uvicorn ; il n’est pas rechargé ou republié par ce bloc.

## 4. Parcours concret d’une prédiction

Prenons une demande synthétique :

```json
{
  "user_id": 123,
  "age": 35,
  "employment_type": "Salaried",
  "monthly_income": 4000,
  "credit_score": 700,
  "purchase_amount": 300,
  "product_category": "Electronics",
  "bnpl_installments": 3,
  "app_usage_frequency": 10,
  "location": "USA",
  "transaction_date": "2026-01-01",
  "debt_to_income_ratio": 0.2
}
```

### A. Le middleware encadre la requête

Un **middleware** est une couche exécutée autour des routes HTTP. Ici, il :

- crée un `request_id`, identifiant aléatoire de cet appel ;
- contrôle le type de contenu et la limite de taille ;
- compte aussi les octets réellement reçus, même sans `Content-Length` ;
- conserve le corps pour le transmettre ensuite à FastAPI ;
- ajoute les headers de corrélation et de cache à la réponse ;
- enregistre le statut et la durée.

Pourquoi `replay()` ? Le flux du corps HTTP a déjà été lu pour contrôler sa taille.
FastAPI doit pouvoir lire les mêmes octets pour décoder le JSON. Cette fonction
lui fournit une fois le corps conservé en mémoire.

### B. FastAPI contrôle l’accès et le schéma

`require_api_key()` vérifie le header `X-API-Key`. `ApplicationRequest` impose les
types et les champs : par exemple, `monthly_income` doit être un nombre JSON,
pas la chaîne `"4000"`. La date doit être civile et au format `YYYY-MM-DD`.
Un champ supplémentaire, comme `default_flag`, est refusé.

`Depends(get_service)` signifie : « FastAPI, appelle cette fonction pour me fournir
le service dont cette route a besoin ». C’est l’**injection de dépendance**.
La route ne recharge donc pas le modèle elle-même.

### C. Le service contrôle les règles métier

`RealtimeInferenceService.predict()` réserve un créneau de calcul, transforme la
demande en DataFrame d’une ligne, nettoie ses valeurs et les valide.

Il y a deux niveaux de validation :

| Niveau | Exemple | Emplacement |
|---|---|---|
| Structure et types | `age: "35"` est une chaîne, donc invalide | `schemas.py` / Pydantic |
| Règles métier | `age: 12` est numérique, mais inférieur à la borne autorisée | `configs/data.yaml` / validateur |

Les deux peuvent produire HTTP 422, avec des codes d’erreur applicatifs distincts.

### D. Predictor applique le modèle et la politique publiée

Le pipeline fournit une probabilité de défaut. `Predictor.score()` applique :

```python
default_risk_class = int(default_probability >= decision_threshold)
```

Exemple **purement illustratif**, sans prédire le résultat du JSON ci-dessus :
si la probabilité vaut `0.40` et le seuil publié vaut `0.30`, la classe vaut `1`.
La bande de risque est choisie selon les bandes enregistrées dans l’artefact.
Le client ne peut modifier ni le seuil ni ces bandes via sa requête.

### E. La réponse est contrôlée et renvoyée

`PredictionResponse` vérifie notamment que la probabilité est finie et entre 0 et 1.
Le service ajoute un horodatage UTC et le `request_id`.

| Champ | Signification |
|---|---|
| `user_id` | Identifiant de la demande recopié depuis l’entrée |
| `default_probability` | Probabilité de défaut renvoyée par le modèle |
| `default_risk_class` | Classe 0 ou 1 après application du seuil |
| `decision_threshold` | Seuil issu de l’artefact |
| `risk_band` | Bande associée à la probabilité |
| `model_version` | Version exacte ayant servi au calcul |
| `scoring_timestamp` | Date et heure UTC de la sortie API |
| `request_id` | Identifiant technique de cet appel, également dans `X-Request-ID` |

Enfin, le créneau de calcul est libéré dans `finally`. Une exception ne doit pas
laisser le service croire indéfiniment qu’une prédiction est encore en cours.

## 5. Guide de lecture des classes et fonctions

Toutes les fonctions de l’API, leurs helpers internes et les fonctions de scoring
partagées disposent de docstrings en français. Elles décrivent leurs entrées,
leurs sorties et leurs effets ou erreurs. Les tests ont aussi leurs docstrings.

### Commencer par `schemas.py` et `settings.py`

Les classes héritant de `BaseModel` décrivent des **contrats de données** ; elles
ne calculent pas les prédictions. Pydantic fournit automatiquement leur constructeur
et des méthodes comme `model_validate()` et `model_dump()`.

- `ApplicationRequest.validate_date()` refuse les conversions ambiguës de date.
- `PredictionResponse` contrôle la sortie de scoring.
- `ModelResponse` décrit le modèle et les règles acceptées.
- `HealthResponse` décrit les sondes.
- `ErrorDetail` et `ErrorResponse` décrivent l’enveloppe commune des erreurs.
- `APISettings.validate_deployment()` impose la cohérence du mode de déploiement.

### Lire ensuite `service.py`

| Méthode | Question à laquelle elle répond |
|---|---|
| `__init__()` | Comment assembler un service à partir d’un Predictor déjà chargé ? |
| `load()` | Où charger le modèle et comment obtenir un service prêt ? |
| `_validate_artifact()` | Cet artefact est-il compatible avec l’API v1 ? |
| `warmup()` | Une première prédiction complète fonctionne-t-elle ? |
| `predict()` | Comment passer de la demande validée à la réponse ? |
| `describe()` | Quel modèle et quelles règles le service utilise-t-il ? |

`ServiceBusyError` est une exception dédiée à la saturation. Le préfixe `_` de
`_validate_artifact` indique une méthode interne, appelée par le constructeur.
`@classmethod` permet d’appeler `RealtimeInferenceService.load(...)` avant d’avoir
une instance : cette méthode se charge précisément de la construire.

### Lire `app.py` pour comprendre la couche HTTP

| Fonction | Rôle |
|---|---|
| `create_app()` | Construire une application indépendante et enregistrer ses composants |
| `lifespan()` | Initialiser puis nettoyer le service |
| `require_api_key()` | Contrôler l’accès aux routes protégées |
| `get_service()` | Récupérer le service en mémoire ou signaler son absence |
| `invalid_request()` | Convertir les erreurs de structure en 422 |
| `invalid_business_data()` | Convertir les erreurs métier en 422 |
| `busy()` | Convertir la saturation en 503 avec `Retry-After` |
| `http_error()` | Uniformiser les erreurs HTTP connues |
| `live()` / `ready()` | Répondre aux sondes de santé |
| `model()` / `predict()` | Exposer les métadonnées et les prédictions |
| `prometheus_metrics()` | Exporter les mesures au format Prometheus |

Ces fonctions sont définies dans `create_app()` pour partager ses paramètres.
Elles sont ensuite appelées par FastAPI grâce aux décorateurs `@app.get`,
`@app.post` et `@app.exception_handler`.

### Terminer par `middleware.py`, `client.py` et `cli.py`

`RequestMiddleware.__call__()` reçoit les trois éléments du protocole **ASGI** :
`scope` décrit la requête, `receive` lit les messages entrants et `send` transmet
les messages sortants. `traced_send()` enrichit la réponse, `reject()` envoie un
refus anticipé et `replay()` restitue le corps lu. `error_response()` construit
le JSON commun. `APIMetrics.__init__()` prépare les compteurs et histogrammes.

`BNPLAPIClient.__init__()` prépare les connexions HTTP. `model()` et `predict()`
envoient les requêtes et contrôlent les réponses. `__enter__()` et `__exit__()`
permettent `with`; `close()` libère les connexions. Un statut en erreur est propagé
avec HTTPX, sans retry automatique.

`cli.main()` est le point d’entrée de `bnpl-api`. Uvicorn est le serveur réseau ;
FastAPI est l’application qu’il fait fonctionner.

## 6. Concurrence, logs et erreurs : pourquoi ces mécanismes ?

### Le sémaphore limite les calculs simultanés

Un **sémaphore** est un compteur de places disponibles. Avec une place :

```text
Demande A → prend la place → calcule → rend la place
Demande B pendant ce calcul → aucune place → HTTP 503 + Retry-After: 1
Demande C après le calcul → prend la place → calcule normalement
```

Le service n’accumule pas une file de calculs en attente. Ce réglage limite aussi
les appels simultanés au même pipeline natif. La limite de connexions Uvicorn est
un mécanisme supplémentaire, distinct du nombre de places de calcul.

La route de prédiction utilise `def`. FastAPI la fait travailler dans un pool de
threads, tandis que ses fonctions `async def` peuvent gérer des attentes sans
bloquer directement la boucle asynchrone. Cela ne garantit pas une capacité
illimitée : les ressources CPU, mémoire et threads restent bornées.

### Les logs expliquent un événement ; les métriques résument le trafic

Loguru écrit du JSON sur la sortie d’erreur du processus. Exemples d’événements :
`model_ready`, `prediction_completed`, `request_failed`, `http_request` et
`service_stopped`. Les informations liées à la requête se trouvent notamment dans
`record.extra` : request_id, route, statut, durée et, pour le scoring, version.

Les corps, clés d’accès et identifiants clients ne sont pas loggés par cette couche.
Les exceptions publiques sont génériques pour ne pas renvoyer de données d’entrée.
Le `request_id` permet de retrouver l’événement correspondant à une réponse.

Prometheus expose deux instruments :

- `bnpl_http_requests_total` : nombre de réponses par route et statut ;
- `bnpl_http_request_duration_seconds` : répartition des durées par route.

Un **p95** de 30 ms signifie que 95 % des observations sont inférieures ou égales
à 30 ms. Les mesures du notebook sont locales, séquentielles et hors réseau.

### Interpréter les erreurs

| Statut | Ce qui s’est passé | Quoi vérifier |
|---|---|---|
| 401 | Clé manquante ou incorrecte | Header X-API-Key |
| 413 | Corps trop volumineux | Une seule demande, taille maximale configurée |
| 415 | Mauvais type de contenu | Content-Type: application/json |
| 422 | Données invalides | Types, champs et règles exposés par `/v1/model` |
| 503 | Service indisponible ou saturé | Readiness et éventuel Retry-After |
| 500 | Calcul ou sortie modèle invalide | Logs associés au request_id |

`/health/live` indique que le processus répond. `/health/ready` indique qu’un
service a été chargé et préchauffé. La readiness ne relance pas une prédiction à
chaque appel et ne constitue pas une surveillance continue de la qualité du modèle.

## 7. Lancer puis essayer l’API

Depuis la racine du dépôt, avec un artefact approuvé disponible :

```bash
poetry install
export BNPL_API_API_KEY="$(poetry run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
poetry run bnpl-api
```

Dans un autre terminal :

```bash
curl --fail http://127.0.0.1:8000/health/ready
```

Les deux terminaux ne partagent pas automatiquement une variable exportée : pour
appeler les routes protégées, le client doit utiliser **la même clé** que le serveur.
Un fichier `.env` local peut fournir cette configuration aux processus qui le lisent.

Ouvre `http://127.0.0.1:8000/docs` en développement. Swagger décrit les endpoints ;
le bouton **Authorize** permet de fournir la clé. Sur `POST /v1/predict`, utilise
**Try it out**, colle le JSON de la section 4 et exécute la requête.

### Paramètres à connaître

| Variable | Valeur par défaut / obligation | Utilité |
|---|---|---|
| `BNPL_API_API_KEY` | Obligatoire, 32 caractères minimum | Secret partagé avec les clients |
| `BNPL_API_ENVIRONMENT` | `development` | `production` masque Swagger et exige une version exacte |
| `BNPL_API_MODEL_VERSION` | Repli sur `configs/inference.yaml` | Choix de la version à servir |
| `BNPL_API_CONFIG_DIR` | Dossier `configs` du projet | Dossier alternatif de configurations YAML |
| `BNPL_API_HOST` / `BNPL_API_PORT` | `127.0.0.1` / `8000` | Adresse d’écoute |
| `BNPL_API_MAX_BODY_BYTES` | `16384` | Taille du corps HTTP en octets |
| `BNPL_API_MAX_CONCURRENT_PREDICTIONS` | `1` | Nombre de calculs simultanés autorisés |
| `BNPL_API_LOG_LEVEL` | `INFO` | Verbosité des logs |
| `BNPL_ARTIFACTS_DIR` | `artifacts` | Paramètre existant du projet, contenant le sous-dossier `models` |

Les arguments Python explicites de `APISettings` prennent priorité sur les valeurs
chargées depuis l’environnement et `.env`. Ne laisse pas une version vide : omets
la variable ou donne une version exacte. `SecretStr` masque la représentation de
la clé en Python ; ce n’est pas un mécanisme de chiffrement du fichier `.env`.

## 8. Ce que teste le notebook 07

| Partie | Preuve recherchée |
|---|---|
| Configuration isolée | Clé temporaire non affichée et version fixée |
| Santé et contrat | Démarrage réussi, bon modèle, nettoyage à l’arrêt |
| Demande synthétique | Réponse conforme, règle de seuil et horodatage UTC |
| Erreurs | Rejets prévus pour clé, données et corps trop volumineux |
| Parité batch/API | Mêmes probabilités, classes, bandes et version sur 25 demandes |
| Latence et métriques | Mesures locales sur 30 requêtes et export Prometheus |
| HTTP réel après les imports | Démarrage Uvicorn puis POST synthétique avec vérification du score |

Le notebook démarre maintenant **un vrai serveur Uvicorn après les imports**,
puis envoie une requête de prédiction sur le port local 8000. La clé aléatoire est
partagée automatiquement avec le processus ; aucun copier-coller n’est nécessaire.
La cellule « Arrêter l’API » termine le serveur à la fin des essais.

Les tests complémentaires utilisent **TestClient sans port réseau** : ils appellent
l’application ASGI en mémoire tout en exerçant routes, validation, middleware et
lifespan. Le bloc `with TestClient(app)` démarre puis arrête cette application de test.

Les tests pytest vont plus loin : saturation concurrente, sortie modèle invalide,
confidentialité des erreurs, fichiers manquants, versions incompatibles, chargement
unique et changement du pointeur latest. Leurs fixtures créent des artefacts
isolés ; certains tests utilisent aussi un petit pipeline réellement entraîné.

```bash
poetry run python -m pytest tests/integration/test_realtime_api.py --no-cov
poetry run python -m ruff check src/bnpl_credit_risk/api tests/integration/test_realtime_api.py
poetry run python -m mypy src/bnpl_credit_risk/api --follow-imports=silent
```

## 9. Docker, CI et limites de ce qui est livré

`Dockerfile.api` construit l’environnement avec Poetry, puis prépare une image
d’exécution avec un utilisateur non-root et la bibliothèque native OpenMP.
`docker-compose.api.yml` injecte la clé et la version et monte le registre en
lecture seule. Le modèle doit donc déjà exister sur la machine hôte.

Le workflow CI décrit les vérifications qui seront lancées dans GitHub Actions.
Sa présence dans le dépôt ne signifie pas qu’une exécution distante a déjà eu lieu.
Lors de l’implémentation, l’API et le notebook ont été testés localement ; Docker
étant arrêté, le fonctionnement du conteneur n’a pas encore été validé localement.

Pour une mise en service exposée, il reste à fournir sur la plateforme cible TLS,
gestion des secrets, quotas, collecte des logs/métriques, alertes et tests de charge.
La clé d’API est un secret partagé : il n’y a pas de gestion de comptes, de rôles ou
de quotas par client implémentée ici. Aucun timeout dur n’interrompt un calcul
natif déjà lancé. Ces limites et le rollback sont détaillés dans le
[guide d’exploitation](docs/realtime_api.md).

## 10. Les cinq idées à retenir

1. **Le modèle est déjà entraîné** : l’API charge et utilise l’artefact approuvé.
2. **Predictor est partagé** : batch et HTTP appliquent la même politique publiée.
3. **Le service gère le scoring ; FastAPI gère son exposition HTTP.**
4. **Le notebook est un consommateur de l’API** et une démonstration de ses tests.
5. **Les commentaires expliquent pourquoi** les contrôles existent ; les docstrings
   permettent de comprendre chaque fonction depuis l’éditeur ou avec `help()`.
