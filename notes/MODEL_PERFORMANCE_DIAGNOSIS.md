# Diagnostic des performances — notebooks 00 à 04

Date : 2026-08-27

## Verdict

La performance modérée ne vient ni d'une corruption des données, ni d'un
drift entre les partitions, ni principalement du nombre de lignes perdues dans
le split train/validation. Les causes dominantes sont :

1. le signal réellement disponible avant l'octroi est limité ;
2. le feature engineering actuel ajoute surtout des transformations
   redondantes et augmente l'overfitting ;
3. le seuil fixe `0.5` dégrade fortement F1, recall et MCC ;
4. le protocole montre le test pendant la comparaison des modèles, ce qui
   transforme progressivement le test en jeu de développement ;
5. un seul split de validation produit un classement trop incertain pour
   conclure que CatBoost est réellement meilleur que LightGBM.

Une AUC autour de `0.71–0.73` est cohérente avec les variables application-time
disponibles. L'AUC proche de `0.79` obtenue en réintroduisant les informations
postérieures au paiement serait meilleure numériquement, mais ne serait pas un
modèle valide pour décider d'accorder le BNPL au checkout.

## Résultats de l'audit des données

| Partition | Lignes | Défauts | Taux de défaut |
|---|---:|---:|---:|
| Dataset complet | 10 345 | 4 040 | 39,0527 % |
| Train initial | 9 310 | 3 636 | 39,0548 % |
| Train modèle | 8 379 | 3 272 | 39,0500 % |
| Validation | 931 | 364 | 39,0977 % |
| Test réservé | 1 035 | 404 | 39,0338 % |

Contrôles réussis :

- aucun doublon de `user_id` ;
- aucune intersection d'utilisateurs entre train, validation et test ;
- toutes les lignes du dataset brut sont conservées après le premier split ;
- toutes les lignes du train sont conservées après le second split ;
- aucune valeur manquante dans les fichiers feature-engineered ;
- les colonnes retirées sont exactement `repayment_delay_days`,
  `missed_payments`, `risk_score` et `customer_segment` ;
- les formules de toutes les features numériques créées sont exactes à la
  précision flottante.

Il n'y a donc pas de preuve qu'une étape 00–03 ait mélangé, perdu ou corrompu
les données.

## Audit notebook par notebook

### 00 — Data discovery

La structure, les types, les valeurs manquantes et la cible sont correctement
inspectés. Il manque toutefois un contrôle de provenance et de cohérence
temporelle. Le dataset n'a que douze mois distincts : janvier–juin 2023 puis
juillet–décembre 2024. Toute la période juillet 2023–juin 2024 est absente.
Cette rupture est un indice fort de dataset synthétique ou assemblé et doit
être documentée.

### 01 — Validation et premier split

Le nettoyage est déterministe : types, espaces et doublons. Il n'apprend aucune
statistique sur le dataset complet ; l'effectuer avant le split ne crée donc
pas de fuite dans sa forme actuelle.

Le split stratifié 90/10 est intègre. En revanche, un split aléatoire mesure la
capacité à généraliser vers d'autres lignes issues de la même distribution, pas
la capacité à généraliser vers le futur. Le bon protocole dépend du cas métier :

- benchmark du dataset synthétique : split stratifié ou validation croisée ;
- déploiement futur réel : backtest chronologique et suivi du drift.

### 02 — EDA et leakage

L'EDA fondée sur le train uniquement est correcte. La suppression sémantique du
leakage est également correcte pour un modèle d'octroi : retards et paiements
manqués n'existent pas au checkout.

Le retrait de `risk_score` est prudent et justifié tant que sa provenance n'est
pas démontrée. Cette variable est fortement liée à `credit_score`, aux retards,
aux paiements manqués et au ratio d'endettement. Ses déciles font passer le taux
de défaut de 1,7 % à 72,9 %. Elle ressemble donc à un score synthétique construit
à partir de variables pré- et post-événement.

L'étape manquante est une analyse de stabilité dédiée : drift des features,
drift de la cible, catégories inconnues et stabilité temporelle.

### 03 — Feature engineering

Les formules sont correctement implémentées, mais plusieurs features dupliquent
presque entièrement une information déjà disponible :

| Paire | Corrélation de Spearman |
|---|---:|
| `monthly_income` / `income_after_installment` | 0,9995 |
| `monthly_income` / `credit_score` | 0,9177 |
| `credit_score` / `income_after_installment` | 0,9174 |
| `debt_to_income_ratio` / `installment_burden_ratio` | 0,8136 |

Les bandes catégorielles dupliquent également leurs variables continues :
`credit_score_band`, `age_group`, `affordability_band` et `installment_term`.
Les arbres n'ont généralement pas besoin que la même information soit fournie
simultanément sous sa forme continue et sous plusieurs découpages arbitraires.

Le test d'ablation contrôlé avec CatBoost donne :

| Jeu de features | AUC train | AUC validation | AUC test |
|---|---:|---:|---:|
| Variables application brutes sûres | 0,8111 | 0,7282 | 0,7213 |
| Variables sûres + features actuelles | 0,8291 | 0,7208 | 0,7095 |
| Variables sûres + informations post-événement | 0,8462 | 0,7930 | 0,7909 |
| Informations post-événement uniquement | 0,7845 | 0,7731 | 0,7763 |

Le feature engineering actuel augmente l'ajustement au train mais réduit la
généralisation. Il doit être simplifié et validé feature par feature, pas ajouté
en bloc.

### 04 — Modélisation

Le benchmark par défaut est utile comme baseline, mais les paramètres par
défaut ne constituent pas une configuration finale. Le seuil `0.5` n'est pas
non plus une règle métier.

Pour CatBoost avec les features actuelles :

| Politique | Validation F1 | Validation MCC | Test F1 | Test MCC |
|---|---:|---:|---:|---:|
| Seuil fixe 0,5 | 0,5135 | 0,2489 | 0,5302 | 0,2592 |
| Seuil F1 choisi sur validation, 0,175 | 0,6617 | 0,4000 | 0,6413 | 0,3453 |

Le seuil optimisé augmente fortement le recall mais réduit la précision ; il ne
doit donc pas être adopté automatiquement. En risque de crédit, le seuil doit
être choisi avec le coût des faux négatifs, le coût des faux positifs, la
capacité opérationnelle et une calibration des probabilités.

Le principal défaut du notebook 04 est l'affichage du test pour les trois
modèles avant le choix final. Même si le classement écrit utilise validation,
un humain voit les résultats test et les utilise implicitement. Le test n'est
alors plus une mesure finale totalement indépendante.

## Diagnostic du drift

Le drift entre les partitions aléatoires n'explique pas les performances :

- PSI maximal train/validation : `0,024` ;
- PSI maximal train/test : `0,014` ;
- aucune catégorie inconnue en validation ou test ;
- taux de défaut presque identiques dans les trois partitions ;
- statistiques KS numériques faibles.

Selon les seuils heuristiques courants, ces PSI correspondent à des
distributions très stables. Un backtest chronologique séparant 70 % train,
10 % validation et les 20 % les plus récents produit une AUC de `0,7222`, très
proche du split aléatoire.

Les PSI chronologiques de `transaction_month_sin` et
`transaction_month_cos` sont très élevés, mais c'est une conséquence mécanique
du changement de mois entre fenêtres temporelles. Ces deux features ont une AUC
univariée proche de `0,50` et n'apportent pratiquement aucun signal. Elles sont
des candidates prioritaires au retrait.

Conclusion : une infrastructure de drift manque au projet, mais le drift observé
n'est pas la cause de l'AUC actuelle.

## Taille des données et stratégie de split

La courbe diagnostique CatBoost avec les variables brutes sûres donne :

| Lignes train | AUC validation |
|---:|---:|
| 1 675 | 0,6989 |
| 3 351 | 0,7297 |
| 5 027 | 0,7270 |
| 6 703 | 0,7165 |
| 8 379 | 0,7282 |

La courbe fluctue autour de `0,72–0,73` dès environ 3 300 lignes. Elle ne montre
pas une amélioration régulière qui indiquerait que les 931 lignes de validation
sont la cause principale. Un réentraînement non optimisé sur les 9 310 lignes
donne une AUC test de `0,7120`, contre `0,7213` avec 8 379 lignes : ajouter ces
lignes ne crée pas mécaniquement du signal.

Il faut conserver trois rôles logiques, mais pas nécessairement trois fichiers
fixes pendant tout le développement :

1. conserver les 1 035 lignes test complètement fermées ;
2. utiliser une validation croisée stratifiée à 5 folds sur les 9 310 lignes ;
3. choisir modèle, features et hyperparamètres uniquement avec les folds ;
4. réentraîner le pipeline choisi sur les 9 310 lignes ;
5. ouvrir le test une seule fois pour l'évaluation finale.

Cette stratégie exploite davantage les données et produit une estimation plus
stable qu'un unique jeu de validation de 931 lignes.

## Incertitude sur le classement des modèles

Avec 364 défauts en validation, l'intervalle approximatif à 95 % de l'AUC
CatBoost `0,7208` est `[0,6863 ; 0,7553]`. Celui de LightGBM `0,7070` est
`[0,6720 ; 0,7420]`. Ils se recouvrent largement. La différence observée de
`0,0138` ne suffit pas pour affirmer que CatBoost est réellement supérieur.

Une validation croisée répétée doit fournir moyenne, écart-type et résultats
par fold pour ROC-AUC, PR-AUC, MCC, F1, recall, precision, Brier score et log
loss.

## Causes priorisées

1. **Signal application-time limité** : cause principale de l'AUC autour de
   0,72. `credit_score` atteint déjà seul environ 0,678 d'AUC.
2. **Features redondantes et overfitting** : cause démontrée de la baisse
   CatBoost de 0,728 à 0,721 en validation.
3. **Seuil 0,5 non adapté** : cause principale des faibles F1, recall et MCC
   affichés, mais pas de l'AUC.
4. **Protocole d'évaluation fragile** : split unique et consultation répétée du
   test ; le classement des modèles n'est pas statistiquement solide.
5. **Variables de leakage retirées** : baisse attendue mais nécessaire pour un
   modèle d'octroi honnête.
6. **Drift** : hypothèse non confirmée sur ce dataset.
7. **Quantité de données** : facteur secondaire ; davantage de vraies données
   historiques pourrait aider, mais récupérer seulement les 931 lignes de
   validation ne résout pas le problème.

## Plan d'amélioration recommandé

### Étape 1 — Corriger l'évaluation

- fermer le test pendant tout le développement ;
- remplacer le split train/validation fixe par une validation croisée
  stratifiée à 5 folds ;
- comparer les modèles avec moyenne et écart-type ;
- ajouter PR-AUC, Brier score, log loss et intervalles de confiance ;
- conserver un benchmark chronologique séparé.

### Étape 2 — Revenir à une baseline minimale

- entraîner les trois modèles sur les variables application brutes sûres ;
- retirer temporairement toutes les features créées ;
- réintroduire une feature à la fois par ablation cross-validée ;
- supprimer en priorité `income_after_installment`, les deux features de mois
  et les bandes qui dupliquent une variable continue, sauf gain cross-validé.

### Étape 3 — Optimiser sans fuite

- vérifier one-hot versus catégories natives pour XGBoost ;
- utiliser early stopping dans chaque fold ;
- optimiser séparément les hyperparamètres des trois bibliothèques ;
- sélectionner le seuil sur prédictions out-of-fold selon une fonction de coût
  métier ;
- calibrer les probabilités si Brier/log loss et courbes de calibration le
  justifient.

### Étape 4 — Formaliser la stabilité

- ajouter PSI/KS et catégories inconnues entre référence et production ;
- suivre le taux de défaut et la calibration dans le temps ;
- définir des fenêtres temporelles et des seuils d'alerte ;
- documenter la rupture temporelle du dataset et sa provenance synthétique.

### Étape 5 — Séparer les objectifs métier

- conserver un modèle `application_risk` sans information post-paiement ;
- créer éventuellement un second modèle `behavioral_risk` pour collections ou
  surveillance après octroi ;
- ne jamais comparer leurs performances comme s'ils répondaient à la même
  décision métier.

## Conclusion

Le pipeline précédent est techniquement cohérent et n'a pas détruit les
données. La performance actuelle est surtout une performance honnête sur un
signal pré-octroi limité, aggravée par un feature engineering trop redondant et
un seuil de décision mal adapté. La prochaine étape ne doit pas être de supprimer
le test ou d'ajouter davantage de features au hasard, mais de reconstruire le
benchmark avec validation croisée, baseline minimale, ablation et seuil métier.

## Statut d'implémentation

Les recommandations prioritaires ont été intégrées le 2026-08-27 :

- rapport de drift PSI/KS/catégories inconnues dans l'architecture ;
- drift aléatoire et chronologique dans le notebook `02` ;
- validation croisée stratifiée à cinq folds dans le notebook `04` ;
- baseline limitée aux variables brutes application-time ;
- early stopping isolé dans chaque fold ;
- comparaison par moyenne et écart-type avec ROC-AUC, PR-AUC, MCC, F1,
  precision, recall, Brier score et log loss ;
- sélection du seuil sur les probabilités out-of-fold ;
- chargement du test uniquement après le choix du modèle et du seuil ;
- évaluation test limitée au seul modèle sélectionné.

L'ablation des features métier reste volontairement l'étape suivante : elle
doit être exécutée une feature à la fois avec les mêmes folds, après validation
de cette nouvelle baseline.
