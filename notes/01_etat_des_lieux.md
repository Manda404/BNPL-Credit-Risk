# État des lieux au 25 août 2026

## Périmètre de l'audit

L'audit couvre `README.md`, `docs/`, `configs/`, `src/`, `scripts/`, `tests/`, `notebooks/`, les métadonnées d'artefacts présentes localement, le workflow GitHub Actions et les fichiers de packaging. Le contenu de `Dev/` n'a pas été analysé et aucun fichier de ce répertoire n'a été modifié.

## Vue d'ensemble

Le projet traite un problème de classification binaire : estimer le risque de défaut d'une demande BNPL. Le jeu de données versionné contient 10 345 lignes, 17 colonnes, une cible `default_flag` positive à environ 39,05 %, six marchés et des transactions datées de 2023 à 2024.

Deux cas d'usage sont séparés :

- `application_risk` : décision avant octroi, sans signaux de remboursement futurs ;
- `behavioral_risk` : suivi après octroi, incluant les signaux comportementaux.

Cette séparation est la décision méthodologique la plus solide du projet. Elle retire du modèle de décision les colonnes `repayment_delay_days`, `missed_payments`, `risk_score` et `customer_segment`, identifiées comme indisponibles au moment de l'octroi.

## Ce qui est déjà réalisé

### Données et qualité

- Chargement CSV configurable.
- Nettoyage déterministe : doublons, chaînes, numériques et dates.
- Contrat de données typé pour entraînement et inférence.
- Trois stratégies de validation : `strict`, `warn`, `quarantine`.
- Contrôles de colonnes, valeurs manquantes, doublons d'identifiant, bornes, catégories et cible binaire.
- Rapport de qualité des données.
- Trois stratégies de découpage : aléatoire stratifié, groupé par utilisateur et temporel.

### Features et modèle

- Feature engineering encapsulé dans un transformer scikit-learn.
- Prétraitement et modèle réunis dans une seule `Pipeline` sérialisable.
- Imputation numérique et catégorielle.
- Encodage `OneHotEncoder(handle_unknown="ignore")`.
- XGBoost comme algorithme implémenté et fabrique extensible.
- Calcul de `scale_pos_weight` sur le train uniquement.
- Validation croisée, probabilités out-of-fold et sélection configurable du seuil.
- Calibration optionnelle.

### Évaluation

- ROC-AUC, PR-AUC, Brier, log-loss, KS, précision, rappel, F1, spécificité et balanced accuracy.
- Matrice de confusion et métriques selon plusieurs seuils.
- Coût métier simple des faux positifs et faux négatifs.
- Graphiques d'évaluation et de calibration.
- Artefact local de référence daté du 13 juillet 2026.

Résultats de l'artefact `application_risk` présent localement :

| Indicateur | Valeur |
|---|---:|
| ROC-AUC test | 0,7008 |
| PR-AUC test | 0,5373 |
| Brier score | 0,2223 |
| F1 au seuil choisi | 0,6376 |
| Seuil | 0,2328 |
| Rappel | 0,9572 |
| Spécificité | 0,3490 |
| Taux prédit positif | 76,85 % |

Le seuil actuel favorise très fortement le rappel : 830 faux positifs contre 34 faux négatifs sur 2 069 observations. Ce n'est pas intrinsèquement mauvais, mais ce choix doit être validé avec de vrais coûts métier.

### Industrialisation présente

- CLI Typer : validation, entraînement, réévaluation et batch scoring.
- Scripts fins pour cron/CI.
- Persistance d'une pipeline, des métriques, du seuil, du schéma et des métadonnées.
- Inférence batch avec écriture atomique.
- Squelette d'inférence temps réel.
- Dockerfile et docker-compose.
- Configuration YAML et variables d'environnement.
- Intégration MLflow optionnelle.
- Workflow GitHub Actions nocturne.

### Documentation et notebooks

- Documentation d'architecture, contrat de données, modélisation, inférence et model card.
- Sept notebooks numérotés de `00` à `06`.
- Les notebooks appellent majoritairement le package et ne redéfinissent pas les fonctions métier.
- Tous les notebooks contiennent des sorties exécutées et aucune erreur enregistrée.

## Vérifications effectuées

| Vérification | Résultat |
|---|---|
| Ruff | Réussi |
| mypy sur `src` | Réussi, aucune erreur dans 44 fichiers |
| pytest | 51 tests réussis |
| Couverture globale | 68 % |
| Erreurs enregistrées dans les notebooks | 0 |

Commande pytest nécessaire dans l'environnement d'audit :

```bash
MPLBACKEND=Agg JOBLIB_MULTIPROCESSING=0 .venv/bin/python -m pytest -q
```

L'environnement virtuel a été déplacé : les scripts `.venv/bin/pytest` et `.venv/bin/mypy` conservent un ancien shebang. L'appel via `.venv/bin/python -m ...` fonctionne. Le backend macOS interactif de Matplotlib et le multiprocessing joblib échouent dans un environnement sans interface graphique ou avec des restrictions sur les sémaphores ; le projet ne fixe pas encore des valeurs sûres pour CI.

## Niveau de maturité estimé

| Domaine | Niveau | Commentaire |
|---|---|---|
| Structure logicielle | Bon | Modules clairs et responsabilités bien séparées. |
| Reproductibilité locale | Moyen à bon | Lockfile et configs présents, mais environnement déplacé et artefacts non portables. |
| Qualité des tests | Moyen | Bon cœur unitaire, couverture globale 68 %, plusieurs chemins opérationnels non testés. |
| Qualité ML | Moyen | Fuite principale identifiée, mais CV temporelle, calibration et validation métier à renforcer. |
| Inférence | Moyen-faible | Le chemin existe, mais son contrat exige encore des données post-octroi. |
| CI/CD | Faible | Le workflow nocturne ne dispose ni d'un modèle versionné ni d'un vrai batch d'entrée. |
| Monitoring | Faible | Pas de suivi de drift, de qualité des prédictions ou de performance différée. |
| Gouvernance crédit | Faible | Pas encore d'explicabilité individuelle, de fairness audit ni de mécanisme d'adverse action. |
| Prêt pour production | Non | Prototype local solide, pas encore un service de crédit exploitable. |
