# Analyse critique

## Forces à conserver

### Séparation des cas d'usage

Le projet distingue correctement risque à la demande et risque comportemental. Cette séparation évite qu'une bonne métrique obtenue avec des informations futures soit présentée comme une performance disponible à l'octroi.

### Pipeline unique entraînement-inférence

Le feature engineering, l'imputation, l'encodage et XGBoost sont sérialisés ensemble. C'est une bonne protection contre le training-serving skew.

### Seuil choisi hors du test final

Le seuil n'est pas optimisé sur le jeu de test. L'utilisation de probabilités out-of-fold du train est conceptuellement meilleure qu'un seuil arbitraire de 0,5.

### Organisation du dépôt

Le découpage `data/features/models/evaluation/pipelines/inference` est lisible sans être excessivement abstrait. Les notebooks jouent déjà le rôle de consommateurs du package.

## Écarts critiques — priorité P0

### 1. Le contrat d'inférence contredit `application_risk`

`inference_input_schema()` retire uniquement `default_flag` de `required_columns`. Il continue donc d'exiger `repayment_delay_days`, `missed_payments`, `risk_score` et `customer_segment`, alors que ces colonnes sont explicitement indisponibles avant l'octroi et exclues du modèle `application_risk`.

Conséquence : une vraie demande neuve conforme au cas d'usage documenté sera rejetée avant même la prédiction. Les tests ne détectent pas ce défaut, car ils construisent les entrées d'inférence à partir du dataset complet en retirant seulement la cible.

Correction attendue : construire le schéma d'inférence depuis le `risk_scope` et les dépendances réelles de la pipeline ou, mieux, depuis un contrat d'entrée persisté avec chaque modèle.

### 2. Le registre `latest` n'est pas portable

`artifacts/models/latest.json` contient un chemin absolu vers l'ancien emplacement `/Users/surelmanda/BNPL-Credit-Risk/...`. Le dépôt se trouve maintenant sous `3-Mlops-Databricks-Projects`, donc `latest` ne peut plus charger l'artefact local.

Correction attendue : ne persister que la version ou un chemin relatif validé sous `models_dir`. Le résolveur doit reconstruire le chemin et refuser toute sortie du registre attendu.

### 3. Le workflow nocturne ne peut pas fonctionner après un checkout propre

Les modèles et les fichiers de `data/processed/` sont ignorés par Git. Le workflow installe le package puis lance directement `predict-batch` avec `data/processed/applications_to_score.csv`, sans télécharger de modèle ni récupérer un batch depuis une source externe.

Conséquence : le terme « nightly batch scoring » est trompeur dans l'état actuel. Il s'agit d'un squelette, pas d'une orchestration opérationnelle.

Correction attendue : définir une source de données, un registre de modèles et une destination des prédictions. Tant que ces interfaces ne sont pas décidées, le workflow doit être présenté comme exemple désactivé ou remplacé par une CI de test.

### 4. La réévaluation n'est pas liée au contexte de l'artefact

`run_evaluation_pipeline()` recharge la pipeline sauvegardée, mais reconstruit le split avec la configuration courante et le dataset courant. Il n'utilise pas systématiquement le `risk_scope`, le split, la graine et la version des données persistés avec l'artefact.

Conséquence : après une modification YAML ou des données, la commande peut produire un rapport présenté comme une réévaluation du même modèle alors que le protocole a changé. Un artefact comportemental peut aussi être évalué avec la configuration par défaut du modèle applicatif.

Correction attendue : persister un manifeste complet d'entraînement et une empreinte du dataset, puis faire échouer clairement une réévaluation incompatible ou exiger un mode explicite de comparaison sur nouvelles données.

## Écarts majeurs — priorité P1

### Validation croisée non temporelle

Le holdout final est temporel, mais les prédictions out-of-fold utilisent `StratifiedKFold(shuffle=True)`. Le seuil et le score CV sont donc produits avec des folds qui mélangent passé et futur dans le train.

Pour un modèle déployé dans le futur, préférer des folds temporels croissants ou une validation walk-forward. Il faut aussi gérer explicitement les périodes, les classes présentes dans chaque fold et, si nécessaire, une période d'embargo.

### Incohérence lorsque la calibration est activée

Le seuil est choisi sur les probabilités out-of-fold non calibrées, puis la pipeline finale peut être remplacée par une version calibrée. Le même nombre de seuil ne représente alors plus la même décision.

Correction attendue : générer des probabilités out-of-fold du processus complet calibré, ou choisir le seuil sur un jeu de validation séparé après calibration.

### Seuil métier encore fictif

La matrice de coût `FN=5` et `FP=1` est exprimée en unités arbitraires. Le seuil `best_f1` prédit 76,85 % des demandes comme défaut, avec une spécificité de 34,90 %. Sans taux d'acceptation cible, marge, exposition, LGD, coûts opérationnels et politique de revue manuelle, ce seuil ne peut pas servir de règle de décision réelle.

Il faut distinguer probabilité de défaut, décision d'acceptation, limite proposée et stratégie de revue.

### Explicabilité insuffisante

L'importance XGBoost globale ne fournit ni raison individuelle, ni stabilité, ni explication exploitable pour une décision défavorable. Ajouter SHAP ne suffit pas à lui seul : il faut une taxonomie de motifs compréhensibles et un contrôle de cohérence.

### Fairness et conformité absentes

Le modèle touche au crédit. Aucun audit de disparité n'est implémenté par âge, emploi ou localisation ; aucune mesure de TPR/FPR, calibration, taux d'acceptation ou adverse impact ratio par groupe n'est produite.

Une revue juridique par juridiction reste nécessaire avant toute automatisation d'une décision réelle.

### Artefact incomplet pour la traçabilité

Les métadonnées contiennent le commit Git et des paramètres, mais il manque notamment :

- hash et version logique du dataset ;
- versions Python et bibliothèques ;
- configuration complète figée ;
- contrat exact des entrées brutes ;
- provenance des données ;
- signature/empreinte de la pipeline ;
- model card propre à la version.

Le type `ArtifactBundle` sait sauvegarder une model card, mais la pipeline d'entraînement ne lui en fournit pas.

## Écarts de qualité et d'exploitation — priorité P2

- Couverture globale à 68 %, avec `cli.py`, `evaluation_pipeline.py`, `validation_pipeline.py`, `realtime.py`, les schémas temps réel et plusieurs fonctions de rapport peu ou pas couverts.
- Aucun seuil minimal de couverture dans la CI.
- Les chemins heureux sont bien testés, mais les fichiers corrompus, formats non supportés, artefacts partiels, pointeurs invalides et stratégies `warn/quarantine` le sont moins.
- Les écritures de prédiction sont atomiques, mais les artefacts et `latest.json` ne le sont pas.
- Deux entraînements lancés dans la même seconde peuvent partager la même version.
- Pas de promotion `candidate -> staging -> production`, de rollback ni de comparaison champion/challenger.
- Pas de monitoring de qualité d'entrée, drift, stabilité des scores, distribution des décisions ou performance quand les labels arrivent.
- Pas de boucle de collecte des outcomes ni de définition robuste de la cible et de sa fenêtre de maturité.
- Le Dockerfile n'impose pas de tests de santé et le compose attend un `.env` qui n'existe pas forcément.
- Le parallélisme `n_jobs=-1` et le backend graphique ne sont pas adaptés à tous les environnements CI/headless.
- Les sorties de notebooks contiennent d'anciens chemins absolus ; elles sont informatives mais vieillissantes.
- `04_model_training.ipynb` utilise `../configs/...`, dépendant du répertoire courant du kernel.
- Les données créées dans `06_batch_inference.ipynb` contiennent encore les signaux post-octroi, ce qui masque le défaut du contrat applicatif.

## Conclusion critique

Le projet possède déjà une architecture crédible et un pipeline testé. Le travail manquant n'est pas d'ajouter immédiatement plus de modèles. Il faut d'abord aligner le contrat de données avec le cas d'usage, rendre les artefacts reproductibles et portables, établir un protocole d'évaluation temporel cohérent, puis brancher de vraies interfaces d'exploitation et de gouvernance.
