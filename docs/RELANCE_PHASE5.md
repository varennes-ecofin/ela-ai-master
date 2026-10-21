# ELA AI — Prompt de relance : Phase 5 (nouveau backoffice) puis Phase 6 (production université)

> À coller en premier message d'une nouvelle discussion dans le projet « ELA AI ».
> Date de rédaction : 21/10/2026, à la fin de la consolidation (fin de Phase 4).

## 0. Ce que tu dois lire avant de répondre

Dans le projet Claude (outil Projects) : `claude/PLAN_MODERNISATION.md` (v2.0, plan complet et statut), puis ce document.
Dans le dépôt (dossier `E:\ELA AI`, lié à la conversation) : `README.md`, `CHANGELOG.md` (sections Unreleased : Phases 4a → 4e + correctifs + consolidation), `docs/DEPLOIEMENT_VPS.md` (§ 6 : la migration réelle faite le 19–21/10), `docker-compose.yml`, `deploy/`, `apps/admin/` (backoffice Streamlit actuel : `app.py`, `db.py`, `views/*.py`), `src/ela/db/` (engine, migrations 0001 → 0008, `quiz.py`, `exercises.py`, `activity.py`, `progress.py`, `review_sheets.py`, `code_runs.py`), `src/ela/config.py`, `src/ela/cli/users.py`, `src/ela/eval/` (`runner.py`, `answer.py`, `judge.py`).
Ne propose rien avant d'avoir lu ces fichiers ; pose tes questions, puis attends le « go » avant d'écrire du code.

## 1. Où nous en sommes

- Phases 0 → 4 terminées et **validées** (jalon J4 : parcours étudiant complet). Version `2.0.0a11`, 194 tests offline + 12 PostgreSQL verts, CI GitHub Actions verte sur `main` (dépôt public `varennes-ecofin/ela-ai-master`, branche de travail `modernisation` fusionnée dans `main`).
- Fonctionnel : agent LangChain 1.4 / LangGraph (`search_course`, `get_course_outline`, `get_slide`, `generate_quiz`, `generate_exercise`, `generate_review_sheet`, `run_python`), citations `[S#]` vérifiées, quiz 2.0 (persistance, difficulté adaptative, relance ciblée, lots de 5), exercices socratiques (`check_step`), `/bilan`, fiches de révision par chapitre (cache `review_sheets`), suggestions de suivi, Atelier Code 2.0 (sandbox `apps/sandbox`, quota 10/jour, correction automatique, interprétation de la sortie réelle), backoffice Streamlit avec onglet Pédagogie (quiz, exercices, questions, hors corpus, code).
- **Déployé sur le VPS** le 21/10 : `https://ela.varennes-vps.com` (chat) et `/admin` (Streamlit, authentification basique Caddy + compte `admin`/`supervisor` en base). Détails serveur : code dans `/home/debian/ela-ai-master` (suit `main`), Caddy **dans Docker** (`/opt/caddy`, réseau Docker externe `proxy`, partagé avec l'application Échecs Junior), surcouche `docker-compose.override.yml` (copie de `deploy/docker-compose.override.vps.yml` : `app` et `admin` sur le réseau `proxy`, `admin` aussi sur `default`), base PostgreSQL conservée de l'ancienne version (volume `ela-ai-master_postgres_data`, image `pgvector/pgvector:pg15`, migrations 0001 → 0008 appliquées, collation rafraîchie), `RETRIEVAL_BACKEND=pgvector`, embeddings `bge-m3` locaux (1 466 chunks ingérés), `SANDBOX_URL=http://sandbox:8080`. Mise à jour courante : `git pull && docker compose build && docker compose up -d` (+ `docker compose exec app ela-db upgrade` s'il y a une migration).
- Modèles : Cerebras `gpt-oss-120b` (chat, effort low) / `qwen-3.8-27b` (vision, repli) ; Atelier Code avec `CODE_REASONING_EFFORT=medium`, `CODE_MAX_TOKENS=4096`. ILaaS : clé disponible, URL du service de chat encore inconnue.
- Éval : `ela-eval answer --engine agent` (dernière mesure 18/10 : fidélité 0,86, 0 citation fantôme, 0 refus indu, `multihop_two_searches_rate` = 0) — **à rejouer** après les changements de prompts du 18–21/10.

## 2. Points en suspens (hors Phase 5, à traiter quand ils bloquent)

- `system_v2.md` + `PROMPT_VERSION=v2` pour les questions multi-étapes (deux recherches), avec comparaison `ela-eval`.
- Fiabilité des scripts de l'Atelier Code : dépend surtout du modèle ; idée validée en principe : variable `CODE_MODEL` dédiée (modèle de code distinct du chat) et **évaluation de l'atelier** (`ela-eval code` : sujets → génération → exécution → taux de réussite sans / avec correction) pour comparer les modèles objectivement. Les règles de prompt sur les noms de paramètres (`arch` : `mu`/`omega`/`alpha[1]`/`beta[1]`, moyenne AR : `Const`/`y[1]` ; statsmodels : `const`/`ar.L1`/`ma.L1`/`sigma2`) sont dans `src/ela/prompts.py`.
- Reportés : flashcards à révision espacée, profils de chat (Cours / Entraînement / Atelier), export PDF des fiches.
- Sur le PC de Gilles, `pre-commit` est réduit aux deux hooks ruff (Norton bloque les autres) ; les commits se font avec `--no-verify` si besoin. Les fichiers `.github/workflows/*` ne peuvent pas être écrits à distance : les livrer en pièce jointe à copier.

## 3. Manière de travailler (inchangée, à respecter)

- Gilles (économiste, responsable du master ESA, Université d'Orléans) programme en Python ; répondre en **français**, de façon concise, sans options annexes ; **proposer un plan et attendre le « go »** avant de coder ; poser les questions avant.
- Code Python avec **docstrings en anglais**, rigoureux, testé : Claude code et teste **dans sa sandbox** (venv `.venv`, `PYTHONPATH=src`, pytest ; PostgreSQL embarqué `pgserver` pour les tests d'intégration), puis écrit les fichiers dans `E:\ELA AI` et donne les commandes à lancer sous VS Code (PowerShell, venv `.venv`, Python 3.12). Docker n'est **pas** installé sur le PC ; la pile Docker n'existe que sur le VPS. Sous VS Code, le `.env` est injecté au terminal (le rouvrir après modification). Tests hermétiques (`tests/conftest.py` purge les variables ELA).
- Jamais besoin des clés API ni des mots de passe. `.env`, `.env.*`, `secrets.toml`, `vpslog.txt`, `Other/`, `data/`, `*.jsonl`, `docker-compose.override.yml` restent hors git ; le dépôt est public et ne contient aucun secret ni donnée étudiante. Jamais de données étudiantes vers un cloud (Langfuse désactivé en production).
- Livraison : liste des fichiers écrits, version `pyproject.toml` incrémentée (`2.0.0aN`), `CHANGELOG.md` et `README.md` mis à jour, `PLAN_MODERNISATION.md` mis à jour (disque + projet), commandes à lancer, points à tester. Ruff (`line-length 120`, règles D) propre.
- Sur le serveur : ne jamais `source`-er le `.env` dans le shell (ses variables prendraient le pas sur le fichier pour `docker compose`) ; `docker compose` force `DATABASE_URL` et `ADMIN_DATABASE_URL` vers le service `db`.

## 4. Phase 5 — Nouveau backoffice FastAPI + Jinja2 + HTMX (décision prise le 21/10)

Objectif : remplacer Streamlit par un backoffice prêt pour la production universitaire (plusieurs enseignants, rôles, authentification institutionnelle à venir, ingestions sans SSH, journaux exploitables), construit **à côté** de Streamlit, page par page, en réutilisant `ela.db` et les requêtes SQL existantes ; Streamlit retiré seulement à parité.

Périmètre attendu (à découper en lots, chacun livré testé) :
1. **Socle** : application `apps/backoffice/` (FastAPI, Jinja2, HTMX, CSS léger), sessions signées, connexion par compte `users` (rôle `admin`), structure prête pour un fournisseur d'identité externe (CAS/LDAP/OIDC) sans le brancher encore ; rôles (`admin`, `teacher` à introduire ?) ; service `backoffice` dans `docker-compose.yml` derrière Caddy sur `/admin` (remplacement progressif de Streamlit).
2. **Utilisateurs et quotas** : CRUD, création en masse (CSV), réinitialisation de mot de passe, activation, quotas journaliers (reprend `ela-users`).
3. **Cours et ingestion** : arborescence `data/latex/{M1|M2|Commun}/<Cours>`, dépôt de fichiers `.tex`, lancement d'un **job d'ingestion** en tâche de fond avec progression en direct (table `jobs`, HTMX polling ou SSE), état de la base de connaissances (chunks par cours/niveau, empreinte du corpus), invalidation du cache des fiches.
4. **Activité, feedbacks, lecteur de conversations** : parité avec les vues Streamlit (`apps/admin/views/`), pédagogie (quiz, exercices, questions, hors corpus, code).
5. **Évaluation** : lancer `ela-eval answer` / `retrieval` depuis l'interface (job de fond), historique des rapports (`eval/reports/*.json`), comparaison de deux configurations ; **nouvelle évaluation de l'Atelier Code** (`ela-eval code`) pour comparer les modèles.
6. **Exploitation** : sauvegardes PostgreSQL automatisées (service ou cron dans compose), healthchecks exposés, journal applicatif centralisé (JSON) consultable ; documentation d'exploitation mise à jour (`docs/DEPLOIEMENT_VPS.md`).

Questions à poser à Gilles avant de commencer (exemples) : rôle `teacher` dès maintenant ou seulement `admin` ? authentification cible à l'université (CAS, LDAP, OIDC) pour préparer la bonne abstraction ; `/admin` sur le même sous-domaine ou sous-domaine dédié ; conservation de Streamlit jusqu'à quel lot ; ordre des lots.

## 5. Phase 6 — Production université / ILaaS (après la Phase 5)

Profil `docker-compose.prod.yml`, variables ILaaS (URL du service de chat à obtenir), **mesure de la qualité du tool calling et des sorties structurées sur les modèles ILaaS** (c'est le risque principal : l'agent, les quiz, les fiches et l'atelier en dépendent), test de charge 30–60 étudiants, documentation d'exploitation, procédure de mise à jour des cours.
