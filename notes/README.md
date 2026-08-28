# Notes du projet BNPL Credit Risk

Ce dossier centralise les constats, décisions et prochaines étapes du projet. Il a été créé le 25 août 2026 à partir d'un audit en lecture seule du dépôt, en excluant strictement le contenu de `Dev/`.

## Documents

- [01_etat_des_lieux.md](01_etat_des_lieux.md) : inventaire de ce qui existe et niveau de maturité réel.
- [02_analyse_critique.md](02_analyse_critique.md) : forces, écarts, risques et priorités argumentées.
- [03_feuille_de_route.md](03_feuille_de_route.md) : plan progressif pour terminer le projet sans brûler les étapes.
- [04_plan_des_notebooks.md](04_plan_des_notebooks.md) : rôle attendu de chaque notebook par rapport au package Python.
- [JOURNAL.md](JOURNAL.md) : journal des audits et décisions à maintenir au fil du projet.

## Règles de travail retenues

1. Ne jamais modifier un fichier sous `Dev/`.
2. Le code métier réutilisable vit dans `src/bnpl_credit_risk/`.
3. Les notebooks démontrent, expliquent et valident les fonctions du package ; ils ne dupliquent pas la logique métier.
4. Chaque étape de la feuille de route doit être testée avant de passer à la suivante.
5. Une métrique technique ne suffit pas : toute décision de crédit doit être reliée à un coût métier, à la calibration, à l'explicabilité et à l'équité.

## Statut synthétique

Le dépôt constitue un bon prototype MLOps local, nettement plus avancé qu'un simple notebook. Il n'est cependant pas encore prêt pour une mise en production réelle. Les blocages prioritaires sont le contrat d'inférence, la portabilité du registre d'artefacts, la fidélité de la réévaluation et l'orchestration nocturne.
