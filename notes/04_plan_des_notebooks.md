# Plan des notebooks

## Principe

Les notebooks doivent rester des documents pédagogiques et analytiques. Toute fonction réutilisable doit d'abord être implémentée et testée dans `src/bnpl_credit_risk/`, puis appelée depuis un notebook. Un notebook ne doit pas devenir une seconde pipeline indépendante.

## État actuel et évolution proposée

| Notebook | État actuel | Amélioration proposée |
|---|---|---|
| `00_data_discovery` | Charge les données, affiche forme, tête, types et qualité. | Ajouter provenance, période couverte, cardinalités, unicité de l'identifiant et définition métier de chaque ligne. |
| `01_data_validation` | Exécute le contrat d'entraînement et affiche le début du contrat d'inférence. | Montrer des cas invalides contrôlés, les trois politiques et surtout une demande applicative sans colonnes post-octroi. |
| `02_exploratory_data_analysis` | Reproduit les principaux graphiques. | Séparer EDA descriptive et analyse de fuite ; ajouter évolution temporelle, dérive mensuelle et segmentation par marché. |
| `03_feature_engineering` | Crée les features métier d'accessibilité, de durée et de saisonnalité pour le scope applicatif sûr. | Ajouter ultérieurement tests de sensibilité et comparaison de performance par famille de features. |
| `04_model_training` | Charge les partitions de features produites par `03` et entraîne uniquement le modèle `application_risk`. | Ajouter ultérieurement baseline logistique et manifeste de comparaison. |
| `05_model_evaluation` | Recharge `latest`, reconstruit le test et trace les résultats. | Appeler directement la pipeline de réévaluation fidèle ; ajouter PR, calibration, stabilité temporelle, coûts et comparaison baseline. |
| `06_batch_inference` | Fabrique 30 entrées depuis le dataset complet et score `latest`. | Construire une entrée réaliste avant octroi, vérifier le schéma d'artefact, démontrer quarantaine et idempotence. |

## Notebooks à ajouter plus tard

Ajouter uniquement après stabilisation du socle :

- `07_business_thresholding.ipynb` : politique coût/profit et zones accept/review/reject ;
- `08_explainability.ipynb` : explications globales et individuelles ;
- `09_fairness_audit.ipynb` : métriques par groupe et analyse des disparités ;
- `10_drift_monitoring.ipynb` : référence d'entraînement contre batch récent.

## Critères de qualité d'un notebook

- Exécutable depuis la racine du projet sans chemin `../` fragile.
- Kernel et versions documentés.
- Graine fixée lorsque nécessaire.
- Aucune logique métier dupliquée.
- Aucune sortie contenant un ancien chemin absolu ou des données sensibles.
- Assertions légères sur les résultats clés.
- Texte expliquant le sens métier avant les graphiques.
- Sorties nettoyées ou régénérées dans le workflow de documentation.

## Relation code → tests → notebook

Pour chaque nouvelle capacité :

1. définir le contrat et les critères d'acceptation dans les notes ;
2. implémenter la fonction dans `src/` ;
3. ajouter tests unitaires puis intégration ;
4. appeler la fonction dans le notebook correspondant ;
5. documenter les résultats et limites ;
6. mettre à jour `JOURNAL.md`.
