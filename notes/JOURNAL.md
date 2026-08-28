# Journal du projet

## 2026-08-25 — Audit initial

### Actions

- Inventaire du dépôt hors `Dev/`.
- Lecture de l'architecture, des configurations, du package, des scripts, des tests, des notebooks et des métadonnées d'artefacts.
- Vérification Ruff : réussie.
- Vérification mypy : réussie sur 44 fichiers.
- Vérification pytest : 51 tests réussis, couverture globale 68 %.
- Analyse de la structure et des sorties des sept notebooks : aucun output d'erreur enregistré.
- Création du présent dossier de notes.

### Décisions proposées

- Conserver `application_risk` comme scope de décision par défaut.
- Maintenir la règle « logique dans le package, démonstration dans les notebooks ».
- Traiter d'abord le contrat d'inférence et la portabilité des artefacts.
- Ne pas considérer le workflow nocturne comme opérationnel avant connexion à une source de batch et à un registre de modèles.
- Reporter l'ajout de nouveaux algorithmes après consolidation du protocole temporel et métier.

### Anomalies prioritaires

- L'inférence applicative exige des colonnes post-octroi.
- `latest.json` contient un ancien chemin absolu.
- La réévaluation dépend de la configuration et des données courantes.
- La CV est stratifiée et mélangée malgré un holdout temporel.
- Le seuil n'est plus cohérent si la calibration est activée.
- La CI nocturne ne possède ni modèle ni batch d'entrée après un checkout propre.

### Prochaine session recommandée

Commencer l'étape 1 de la feuille de route : rendre les tests et l'environnement reproductibles, puis ajouter une vraie CI de qualité. Aucun changement de logique métier ne devrait être mélangé à cette première étape.

## 2026-08-25 — Visualisation du rapport de qualité

- Ajout de `plot_data_quality_report(report)` dans le package de visualisation.
- Le dashboard affiche les dimensions, doublons, valeurs manquantes, distribution de la cible et types de colonnes.
- Ajout du visuel à `00_data_discovery.ipynb` juste après la construction du rapport.
- Ajout de tests pour un rapport contenant des anomalies et pour un rapport propre.

## 2026-08-25 — Réparation de l'environnement Jupyter

- Réinstallation editable du package depuis le chemin actuel du dépôt.
- Vérification que `.venv/bin/python` importe le package depuis `src/bnpl_credit_risk`.
- Enregistrement du kernel `bnpl-credit-risk` avec le nom `Python (BNPL Credit Risk)`.
- Association des sept notebooks au kernel du projet.

## 2026-08-25 — API notebook du rapport de qualité

- Ajout de la façade `DataQualityReport(...).show(df)` pour une utilisation plus naturelle dans les notebooks.
- Conservation de `plot_data_quality_report(report)` comme fonction bas niveau réutilisable et testable dans la couche de visualisation.
- Import paresseux de Matplotlib : les pipelines qui utilisent seulement `build()` ou `log()` ne chargent pas la visualisation.
- Correction du double rendu Jupyter : `show()` affiche explicitement une fois et retourne `None`.
- Remplacement du `print(report)` brut par un résumé lisible des dimensions, doublons, valeurs manquantes et classes de la cible.

## 2026-08-25 — Profil détaillé des colonnes

- Ajout du profil détaillé des colonnes dans la couche de qualité des données,
  exposé désormais par `DataQualityReport.profile_dataset(df)`.
- Profil par colonne : type, non-nuls, missingness, cardinalité, unicité, zéros, infinis, outliers IQR, statistiques descriptives et exemples.
- Ajout d'une cellule dédiée dans `00_data_discovery.ipynb`.
- Refactorisation : calculs déplacés dans `data/profiling.py`, options validées par une dataclass et tests de qualité isolés dans `test_data_quality.py`.
- Simplification du tableau : suppression de `Skewness`, `Top Value`, `Top Frequency` et `Memory (KB)`.

## 2026-08-27 — Export stratifié train/test

- Ajout d'un split reproductible et stratifié 90 % train / 10 % test.
- Ajout d'une sauvegarde atomique vers `data/raw/train.csv` et `data/raw/test.csv`.
- Ajout d'un graphique groupé comparant la distribution de la cible dans les deux partitions.
- Ajout d'un pipeline, d'un script et de tests unitaires/intégration dédiés.
- Correction du contrat du pipeline : le DataFrame à découper est désormais un
  argument obligatoire ; aucun CSV n'est chargé implicitement dans le pipeline.
- Ajout de la classe `StratifiedDataSplitPipeline`, initialisée avec la
  configuration puis exécutée avec `split_pipeline.run(df)`.
- Ajout de cette API orientée objet et de l'affichage de la figure dans
  `01_data_validation.ipynb`.
- Centralisation de la proportion, de la graine, des fichiers de sortie et de
  la configuration du graphique dans `configs/data_split.yaml`.
- Simplification du notebook : `StratifiedDataSplitPipeline` reçoit uniquement
  `settings` et le `ProjectConfig` déjà chargé ; `run()` reçoit uniquement `df`.
- Déplacement de la démonstration du schéma d'inférence depuis
  `01_data_validation.ipynb` vers `06_batch_inference.ipynb`, où elle valide
  désormais un véritable lot d'applications sans cible.

## 2026-08-27 — Feature engineering sur le train

- `03_feature_engineering.ipynb` charge désormais `train.csv` depuis la
  configuration du split au lieu de retraiter le dataset brut complet.
- Ajout de trois variables pré-octroi : `installment_amount`,
  `installment_to_income_ratio` et `credit_score_band`.
- Centralisation de leurs règles dans `configs/features.yaml` et extension de
  `BNPLFeatureBuilder`, réutilisé à l'identique par le preprocessing du modèle.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Contrôle explicite du data leakage

- Formalisation des raisons métier de fuite dans `configs/features.yaml`.
- Ajout de `BNPLLeakageGuard` pour auditer puis retirer les colonnes non
  disponibles au checkout dans le scope `application_risk`.
- Ajout de `LeakageControlPipeline` pour appliquer la même règle au train et au
  test, puis sauvegarder des copies sûres dans `data/interim`.
- Passage du notebook `02` à une EDA exclusivement fondée sur le train, suivi
  de l'audit et de la matérialisation des datasets nettoyés.
- Lecture de `train_application.csv` par le notebook `03` avant feature
  engineering.
- Ajout d'une garde équivalente dans le pipeline d'entraînement et désactivation
  des features dérivées fuyardes pour `application_risk`.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Réorganisation du notebook de découverte

- Réorganisation de `00_data_discovery.ipynb` selon le flux : setup et
  chargement, aperçu, profil des colonnes, résumé qualité, dashboard visuel.
- Initialisation de `DataQualityReport` avant sa première utilisation.
- Suppression des appels redondants à `df.head()` et `df.dtypes`, déjà couverts
  par l'aperçu et le profil détaillé.
- Suppression des anciennes sorties enregistrées pour repartir d'un notebook
  propre, sans exécuter ses cellules.
- Configuration de la taille du dashboard qualité dans `configs/data.yaml` et
  transmission à `DataQualityReport` lors de son initialisation.
- Réorganisation du graphique : groupes `Train` et `Test` séparés, avec les
  barres des classes de la cible rapprochées dans chaque groupe.

## 2026-08-27 — Réorganisation du notebook EDA

- Réorganisation de `02_exploratory_data_analysis.ipynb` en huit sections :
  chargement, cible, variables numériques, variables catégorielles, plan BNPL,
  corrélations, diagnostic du leakage et matérialisation des datasets sûrs.
- Centralisation de la taille de toutes les figures EDA dans
  `configs/data.yaml`, sous `exploratory_analysis.visualization.size`.
- Extension des fonctions de `visualization/eda.py` avec le paramètre nommé
  `size`, transmis depuis la configuration au début du notebook.
- Suppression des anciennes sorties enregistrées du notebook.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Harmonisation des notebooks 00 à 03

- Uniformisation des quatre notebooks sur le kernel `.venv` (`python3`).
- Suppression de toutes les anciennes sorties, erreurs et informations
  d'exécution enregistrées dans les fichiers notebooks.
- Suppression de la cellule temporaire `print(sys.executable)` du notebook `01`.
- Simplification de la lecture de la taille EDA : la valeur typée de la
  configuration est utilisée directement, sans conversion `tuple(...)`.
- Ajout de messages explicites dans `02` et `03` lorsque les fichiers produits
  par l'étape précédente sont absents.
- Clarification de l'enchaînement `00` → `01` → `02` → `03` dans les sections
  Markdown des notebooks.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Simplification du notebook EDA

- Retrait des visualisations des anciennes sections `3`, `4` et `5` : profil
  numérique, profil catégoriel et structure du plan BNPL.
- Conservation de la distribution de la cible, de la matrice de corrélation,
  des diagnostics de leakage et du pipeline de matérialisation des données sûres.
- Renumérotation et clarification des sections restantes du notebook `02`.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Façade orientée objet pour l'analyse du leakage

- Ajout de `BNPLLeakageAnalysis` dans la couche `visualization`.
- Centralisation de `target_column`, des colonnes numériques et de la taille
  des figures lors de l'initialisation de la classe.
- Conservation des fonctions de tracé existantes comme primitives internes
  réutilisables.
- Simplification du notebook `02` avec trois appels métier : distribution de
  la cible, corrélations et diagnostics du leakage.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Taille effective de la heatmap de corrélation

- Passage de la heatmap à `imshow(..., aspect="auto")` afin qu'elle utilise
  réellement la largeur et la hauteur configurées.
- Suppression de l'effet visuel de taille fixe causé par les cellules carrées
  et le recadrage automatique des sorties Jupyter.
- Aucun notebook, script ou test exécuté après ce changement.

## 2026-08-27 — Retrait des graphiques de diagnostic du leakage

- Suppression de `show_leakage_diagnostics()` et des trois graphiques qui
  illustraient sans démontrer le leakage.
- Retrait des primitives inutilisées associées au score de risque, aux
  paiements manqués et aux intervalles de retard.
- Remplacement de la section visuelle par une évaluation sémantique fondée sur
  la disponibilité des variables au moment du checkout.
- Conservation de la corrélation comme simple signal exploratoire.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Feature engineering métier BNPL

- Retrait de `income_to_purchase_ratio`, redondante avec la colonne source
  `debt_to_income_ratio`, qui vaut exactement `purchase_amount / monthly_income`
  dans le dataset actuel.
- Renommage de `installment_to_income_ratio` en
  `installment_burden_ratio` pour expliciter son sens métier.
- Ajout de `affordability_band`, `income_after_installment` et
  `installment_term`.
- Remplacement du mois ordinal `txn_month` par les composantes cycliques
  `transaction_month_sin` et `transaction_month_cos`.
- Extension des groupes d'âge jusqu'à 100 ans, conformément au contrat de
  données.
- Mise à jour de la configuration, du builder, du transformer sklearn, des
  tests de contrat, de la documentation et du notebook `03`.
- Ajout dans le notebook d'un catalogue métier, d'un profil qualité des
  nouvelles features et des hypothèses d'utilisation responsable.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Matérialisation des features pour la modélisation

- Ajout de `FeatureEngineeringPipeline` pour transformer de manière identique
  les partitions train et test puis les sauvegarder.
- Ajout des sorties configurées `data/processed/train_features.csv` et
  `data/processed/test_features.csv`.
- Le notebook `03` charge désormais les deux partitions sûres, applique le
  pipeline de features, affiche l'analyse sur le train et sauvegarde les deux
  résultats pour l'étape suivante.
- Le notebook `04` charge explicitement ces fichiers et les transmet au
  pipeline d'entraînement sans effectuer un nouveau split.

## 2026-08-27 — Benchmark initial des trois modèles de boosting

- Réorganisation du notebook `04_model_training.ipynb` autour de trois jeux :
  train, validation stratifiée et test réservé.
- Réutilisation de `StratifiedDataSplitPipeline` pour séparer
  `train_features.csv`, avec des noms de sortie distincts afin de ne pas
  écraser les partitions précédentes.
- Ajout de trois cellules indépendantes pour XGBoost, LightGBM et CatBoost
  avec leurs hyperparamètres par défaut.
- Ajout, pour chaque modèle, des courbes d'apprentissage train/validation,
  d'une matrice de confusion avec courbe ROC et d'une comparaison des métriques
  validation/test (precision, recall, F1, MCC et ROC-AUC).
- Ajout d'une matrice finale et d'un classement fondé sur la validation.
- Mise en commentaire temporaire de l'ancien `run_training_pipeline`.
- Aucun notebook, modèle, script ou test exécuté après ces changements.

## 2026-08-27 — Contrat de split pour les données préparées

- Identification de la cause de l'échec du notebook `04` : le pipeline de
  split appliquait le contrat du CSV brut à un dataset application-risk dont
  les quatre colonnes de leakage avaient volontairement été supprimées.
- Ajout d'un contrat `prepared_training_input_schema` qui exige la cible et
  les features du scope actif sans réintroduire les variables postérieures à
  la décision de crédit.
- Ajout du paramètre explicite `input_stage="prepared"` au pipeline de split ;
  le comportement `raw` reste la valeur par défaut pour le notebook `01`.
- Adaptation du notebook `04` au nouveau contrat.
- Aucun notebook, modèle, script ou test exécuté après cette correction.

## 2026-08-27 — Catégories natives dans les modèles de boosting

- Retrait du préprocesseur OneHot commun du notebook `04`.
- Sélection explicite des features configurées afin d'exclure l'identifiant,
  la date et la cible des entrées des modèles.
- Configuration de XGBoost avec `enable_categorical=True` et l'algorithme
  d'arbres `hist`.
- Transmission de colonnes pandas `category` à LightGBM.
- Transmission des colonnes catégorielles originales à CatBoost via
  `cat_features`.
- Alignement des catégories XGBoost/LightGBM à partir du train uniquement ;
  aucune catégorie de validation ou de test n'est apprise en amont.
- Aucun notebook, modèle, script ou test exécuté après ces changements.

## 2026-08-27 — Diagnostic approfondi des performances

- Audit complet des transitions entre les notebooks `00` et `04`.
- Vérification de l'intégrité des partitions, de la conservation des lignes,
  des distributions de cible, des formules de features et des catégories.
- Mesure du drift aléatoire et chronologique, sans drift expliquant la baisse
  de performance.
- Tests diagnostiques CatBoost sans sauvegarde d'artefact pour isoler l'effet
  de la taille, du feature engineering, du leakage et du seuil.
- Identification d'une baisse de généralisation causée par les features
  redondantes et d'une forte sensibilité au seuil fixe `0.5`.
- Recommandation de remplacer le holdout de développement fixe par une
  validation croisée, tout en conservant le test fermé jusqu'au choix final.
- Rapport détaillé ajouté dans `notes/MODEL_PERFORMANCE_DIAGNOSIS.md`.

## 2026-08-27 — Implémentation du protocole de benchmark corrigé

- Ajout de `DataDriftReport` avec PSI, distance KS, taux de catégories inconnues,
  dérive de la cible et comparaison chronologique.
- Ajout des seuils de drift et de la fraction récente dans `configs/data.yaml`.
- Réorganisation du notebook `02` pour mesurer le drift avant le contrôle
  sémantique du leakage.
- Reconstruction du notebook `04` autour d'une validation croisée stratifiée à
  cinq folds sur l'ensemble des 9 310 lignes de développement.
- Utilisation d'une baseline application-time sans les features engineered
  redondantes, tout en les conservant pour une future étude d'ablation.
- Traitement natif des catégories dans XGBoost, LightGBM et CatBoost avec
  vocabulaire appris séparément dans chaque fold.
- Ajout de l'early stopping par fold et du nombre médian d'itérations pour le
  réentraînement final.
- Ajout des métriques ROC-AUC, PR-AUC, precision, recall, F1, MCC, Brier score
  et log loss avec moyenne et écart-type des folds.
- Sélection du seuil sur les prédictions out-of-fold selon la politique de
  `configs/training.yaml`.
- Le test n'est chargé qu'après le choix du modèle et n'évalue que le candidat
  final.
- Aucun notebook, entraînement ou test automatisé exécuté après ces changements ;
  seuls des contrôles statiques de JSON, syntaxe et lint ont été effectués.

## 2026-08-27 — Split de validation en mémoire et API LightGBM 4.7

- Suppression des anciens fichiers matérialisés `train_model.csv` et
  `validation_model.csv` : la validation croisée du notebook `04` ne sauvegarde
  aucun fold sur disque.
- Remplacement de l'argument LightGBM déprécié `eval_set` par `eval_X` et
  `eval_y`, conformément à l'API installée. Un unique jeu de validation est
  transmis à LightGBM pour éviter l'interprétation incorrecte d'une liste de
  DataFrames ; la courbe train est reconstruite à chaque itération retenue.
- Aucun notebook, fold ou modèle exécuté après cette correction.
- Le pipeline d'entraînement accepte maintenant des partitions préparées tout
  en conservant `BNPLFeatureBuilder` dans l'artefact sklearn afin de garantir
  la cohérence entre entraînement et inférence.
- Ajout d'un test d'intégration de matérialisation et isolation de ses chemins
  de sortie dans les fixtures.
- Aucun notebook, script ou test exécuté après ces changements.

## 2026-08-27 — Analyse métier du seuil de décision

- Ajout au notebook `04` d'une grille de seuils évaluée exclusivement sur les
  probabilités out-of-fold du modèle sélectionné, sans ouvrir le test final.
- Traduction de la matrice de confusion en indicateurs BNPL : défauts détectés,
  défauts manqués et bons payeurs signalés à tort.
- Ajout du coût métier configuré et du ratio marginal de défauts supplémentaires
  détectés pour 100 faux signaux supplémentaires lorsque le seuil est abaissé.
- Ajout d'un visuel commun precision/recall et impact opérationnel, avec le seuil
  retenu clairement matérialisé.
- Aucun notebook ni entraînement exécuté ; contrôles statiques uniquement.

## 2026-08-27 — Séparation entraînement / évaluation des modèles de boosting

- Extraction de la logique volumineuse du notebook `04` vers les classes
  `BoostingBenchmark`, `BoostingBenchmarkPipeline` et
  `BoostingBenchmarkVisualizer`.
- Centralisation dans `configs/model.yaml` des modèles comparés, paramètres
  LightGBM/CatBoost, métrique de sélection, grille de seuil et tailles des
  visualisations.
- Le notebook `04` ne fait désormais que charger les données de développement,
  lancer la comparaison, afficher les diagnostics identiques des trois modèles,
  sélectionner le meilleur et sauvegarder le benchmark.
- Déplacement du choix du seuil vers le notebook `05`, après le diagnostic de
  calibration et avant toute ouverture du test final.
- Ajout dans le notebook `05` des courbes de calibration, seuil métier,
  lift/gains cumulés, importance native et quatre vues TreeSHAP : importance
  globale, beeswarm, explication locale et dépendance.
- Ajout d'un artefact de développement `artifacts/benchmarks/<version>` afin de
  transmettre le modèle sélectionné et les prédictions OOF de `04` à `05` sans
  dépendre de l'état mémoire du kernel.
- Aucun entraînement ni notebook exécuté ; contrôles statiques uniquement.

## 2026-08-27 — Inférence batch alignée sur le modèle approuvé

- Ajout à la fin du notebook `05` d'une publication explicite du modèle après
  acceptation de son évaluation finale.
- L'artefact publié contient le modèle natif sélectionné, son vocabulaire
  catégoriel, le seuil figé, les bandes de risque, les métriques et le contrat
  de features.
- Reconstruction du notebook `06` autour de `BatchInferencePipeline` : contrôle
  de l'artefact, exemple optionnel sans cible, scoring et inspection des sorties.
- Refus des modèles non marqués `approved_for_inference` afin d'éviter que
  `latest` utilise silencieusement un ancien artefact.
- Vérification du contrat de features et de l'ordre configuré des colonnes de
  sortie avant l'écriture atomique.
- Ajout de `BatchInferenceVisualizer` pour regrouper bandes de risque,
  probabilités et décisions dans une figure configurée.
- Aucun notebook, entraînement ou scoring exécuté ; contrôles statiques uniquement.

## 2026-08-28 — Pointeur de modèle relocalisable

- Correction de `models/registry.py` : la résolution de `latest` utilise
  désormais `models_dir / version` et ignore les anciens chemins absolus.
- Simplification des nouveaux fichiers `artifacts/models/latest.json` afin de
  ne sauvegarder que l'identifiant de version.
- Réparation du pointeur existant qui référençait encore
  `/Users/surelmanda/BNPL-Credit-Risk` avant le déplacement du dépôt.
- L'ancien modèle est maintenant localisable mais reste volontairement non
  approuvé ; le notebook `05` doit publier le gagnant avant le scoring dans `06`.
- Aucun modèle, notebook ou scoring exécuté.

## 2026-08-28 — Submission du test

- Correction du nom en `submission.csv` et configuration de sa destination
  dans `data/output`.
- Suppression de `data/processed/applications_to_score.csv` et du mécanisme de
  création d'un échantillon de démonstration.
- Le notebook `06` utilise maintenant la totalité de
  `data/processed/test_features.csv`.
- La vraie cible est isolée avant l'inférence, puis réconciliée par `user_id`
  après la prédiction ; elle n'est jamais transmise au modèle.
- Le fichier final contient exactement `predicted_label` et `true_label` ; les
  probabilités et bandes de risque restent disponibles en mémoire.
- Aucun modèle, notebook ou scoring exécuté ; `submission.csv` sera matérialisé
  après publication du modèle approuvé et exécution de la section 3 du notebook
  `06`.

## 2026-08-28 — Sélection CatBoost et suivi MLflow

- Le contrat affichant `xgboost / legacy` provenait de l'ancien pointeur
  `artifacts/models/latest.json`, et non du benchmark courant où CatBoost est
  classé premier.
- Le notebook `05` publie exclusivement le gagnant contenu dans le benchmark,
  puis déplace atomiquement `latest` après la réussite du suivi MLflow.
- Ajout d'un tracker central : notebook `04` journalise un run parent et trois
  runs imbriqués comparables ; notebook `05` journalise le gagnant approuvé, les
  métriques OOF/test, le seuil, la calibration, lift/gains, importances, SHAP et
  l'artefact complet.
- Le projet est raccordé à la plateforme MLflow centrale
  `/Users/surelmanda/.mlflow` : base SQLite partagée, artefacts partagés et
  expérience dédiée `bnpl_credit_risk`. Aucun `mlruns` n'est créé dans le dépôt.
- MLflow est une dépendance standard et son activation est centralisée dans
  `configs/training.yaml`. Aucun entraînement ou notebook n'a été exécuté.
- La version du client est alignée sur MLflow `3.14`, version ayant produit le
  schéma Alembic actuel de la base centrale. Un client `2.22.5` ne reconnaissait
  pas la révision `b7e4c1a90f23` et ne pouvait donc pas s'y connecter.
- Le benchmark est instrumenté pendant l'entraînement : run parent, runs
  imbriqués par algorithme, ressources système, métriques par fold et par
  itération, prédictions OOF, tableaux et figures comparables.
- Les datasets development/test sont enregistrés comme entrées MLflow avec
  source, schéma, profil et digest, et les CSV sont copiés comme artefacts.
- Le gagnant est sauvegardé en PyFunc avec signature et exemple d'entrée, puis
  la version approuvée est publiée dans le Model Registry avec l'alias
  `champion`. Les configurations YAML, le lockfile et le commit Git assurent la
  reproductibilité.
