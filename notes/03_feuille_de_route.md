# Feuille de route progressive

La règle proposée est simple : une étape n'est terminée que lorsque son code, ses tests, sa documentation et son notebook de démonstration sont cohérents.

## Étape 0 — Figer le diagnostic

Statut : terminé par les notes présentes dans ce dossier.

Livrables : inventaire, analyse critique, priorités et plan des notebooks.

## Étape 1 — Réparer le socle reproductible

Objectif : obtenir un dépôt qui fonctionne de façon identique après un nouveau clone.

- Recréer proprement `.venv` plutôt que conserver les anciens shebangs.
- Configurer Matplotlib en mode headless pour tests et pipelines.
- Rendre le nombre de jobs CV configurable avec une valeur CI sûre.
- Ajouter une CI sur pull request : installation, Ruff, mypy, pytest et seuil de couverture.
- Tester le build et l'exécution Docker.
- Corriger les chemins relatifs des notebooks et nettoyer les sorties obsolètes.

Critère de sortie : un clone propre passe les mêmes contrôles sans variable manuelle cachée.

## Étape 2 — Corriger le contrat d'inférence et le registre

Objectif : scorer une vraie demande applicative ne contenant aucun signal post-octroi.

- Définir explicitement le contrat brut minimal par `risk_scope`.
- Persister ce contrat dans chaque artefact.
- Valider l'entrée avec le contrat de l'artefact chargé, pas seulement avec la config courante.
- Ajouter un test d'inférence `application_risk` sans les quatre colonnes fuyantes.
- Remplacer le chemin absolu de `latest.json` par une version ou un chemin relatif.
- Ajouter tests de déplacement du dépôt, pointeur corrompu et tentative de path traversal.
- Rendre atomiques la sauvegarde d'un bundle et la mise à jour du pointeur.

Critère de sortie : `predict-batch` fonctionne après déplacement du dépôt avec une entrée disponible au moment de l'octroi.

## Étape 3 — Garantir la reproductibilité ML

Objectif : pouvoir expliquer exactement comment chaque modèle a été produit.

- Ajouter hash du dataset, plage temporelle, schéma, versions logicielles et configuration complète au manifeste.
- Sauvegarder la model card versionnée.
- Séparer `reproduce-evaluation` et `evaluate-on-new-data`.
- Faire reconstruire le split depuis le manifeste de l'artefact.
- Ajouter une validation de compatibilité avant évaluation.
- Rendre les identifiants de version uniques au-delà de la seconde.

Critère de sortie : le même artefact et le même dataset reproduisent les métriques attendues ou échouent avec un message précis.

## Étape 4 — Consolider la validation temporelle et la calibration

Objectif : produire une estimation réaliste de la performance future.

- Remplacer la CV stratifiée mélangée par une validation walk-forward pour `time_based`.
- Documenter les fenêtres train/validation/test et la maturité de la cible.
- Choisir le seuil sur des probabilités cohérentes avec la pipeline finale calibrée.
- Comparer sans calibration, Platt et isotonic sur Brier, log-loss et courbes de fiabilité.
- Ajouter intervalles de confiance ou bootstrap temporel des métriques.
- Vérifier la stabilité par mois et par marché.

Critère de sortie : aucune observation future ne participe à une décision passée, directement ou via le seuil.

## Étape 5 — Passer d'une métrique ML à une politique de crédit

Objectif : relier les scores à des décisions économiques explicites.

- Définir exposition, marge, LGD, coût du défaut, coût du refus et coût de revue.
- Produire courbe coût/profit, taux d'acceptation, défaut attendu et capacité de revue selon le seuil.
- Introduire trois zones : acceptation, revue manuelle, refus.
- Définir les risk bands à partir de la politique métier et non de quartiles fixes arbitraires.
- Comparer à des baselines : décision naïve, régression logistique et score simple.

Critère de sortie : le seuil retenu est justifié par un scénario métier documenté et sensible aux hypothèses.

## Étape 6 — Explicabilité, fairness et gouvernance

Objectif : rendre les décisions auditables et surveiller les disparités.

- Ajouter explications globales et locales avec SHAP.
- Transformer les explications locales en motifs contrôlés et compréhensibles.
- Mesurer taux d'acceptation, TPR, FPR, précision et calibration par groupe pertinent.
- Ajouter adverse impact ratio et intervalles d'incertitude.
- Documenter limites, populations exclues, usages interdits et procédure d'appel humain.
- Faire valider les exigences réglementaires par les personnes compétentes.

Critère de sortie : chaque décision peut être expliquée et les écarts entre groupes sont visibles avant promotion.

## Étape 7 — Monitoring et cycle de vie

Objectif : détecter la dégradation après déploiement.

- Journaliser modèle, schéma, timestamp, score et décision sans exposer inutilement les données personnelles.
- Surveiller qualité d'entrée, catégories inconnues, missingness, drift des features, drift des scores et taux de décision.
- Joindre ultérieurement les outcomes matures pour suivre performance et calibration.
- Définir alertes, propriétaire, SLA et playbook de rollback.
- Mettre en place registre et promotion `candidate/staging/production`.
- Définir cadence de réentraînement basée sur le drift et la maturité des labels.

Critère de sortie : une dérive mesurable produit une alerte et une action documentée.

## Étape 8 — Orchestration réelle

Objectif : remplacer le workflow démonstratif par un flux exploitable.

- Choisir la source des demandes et la destination des prédictions.
- Choisir le registre d'artefacts et la gestion des accès/secrets.
- Ajouter idempotence, reprise, audit, quarantaine et contrôle des doublons.
- Séparer CI, entraînement planifié, promotion et scoring batch.
- N'ajouter une API temps réel que si le besoin, la latence et les contrôles sont définis.

Critère de sortie : un environnement propre récupère une version de modèle approuvée, score un batch réel et publie le résultat avec traçabilité.

## Ordre recommandé pour le prochain travail de code

1. Étape 1 : reproductibilité et CI.
2. Étape 2 : contrat d'inférence et registre portable.
3. Étape 3 : manifeste et réévaluation fidèle.
4. Étape 4 : validation temporelle et calibration.

Il est préférable de ne pas commencer SHAP, FastAPI ou un nouveau modèle avant d'avoir terminé ces quatre blocs.
