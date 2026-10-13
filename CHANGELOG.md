# Changelog — ELA AI

Toutes les évolutions notables du projet. Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/).
Le plan de modernisation complet est dans `PLAN_MODERNISATION.md`.

## [2.0.0a2] — 2026-10-13 — Phase 1b : migrations, montée de versions, Docker

### Ajouté
- **Alembic** embarqué dans le paquet (`src/ela/db/migrations/`) et commande `ela-db` (`upgrade`, `current`, `history`, `downgrade`). Migration `0001_baseline` idempotente : tables Chainlit 2.12 (dont `elements.autoPlay/playerConfig`, `steps.command`, `feedbacks.threadId`), colonnes de comptes ELA, index, vue `v_daily_usage`. `ela-users migrate` délègue à Alembic. Remplace le SQL manuel du README/GUIDE.
- Test d'intégration de la data layer personnalisée sous Chainlit 2.12 (`tests/test_data_layer.py`) : cycle complet utilisateur → fil → étape → image stockée localement → feedback → reprise.

### Modifié
- **Versions** : Chainlit 2.9.6 → 2.12.0 ; LangChain core 1.2.7 → 1.6.10, openai 1.7.1, community 0.4.2, classic 1.0.8, huggingface 1.2.3 ; chromadb 1.4.1 → 1.5.9 ; sentence-transformers 5.7.0 ; pydantic 2.14.1 ; SQLAlchemy 2.0.54 ; bcrypt 5.0 ; Streamlit 1.65. Jeu de versions résolu avec `uv`, suite de tests verte.
- `.chainlit/config.toml` migré au format 2.12 (sections MCP) ; `cl.Action` sans arguments obsolètes.
- **Dockerfiles multi-stage** (builder → runtime sans compilateur, utilisateur non-root, healthchecks, torch CPU-only) ; caches des modèles HF et FlashRank dans `data/` (volume) : téléchargés une seule fois.
- **Images de `public/`** optimisées : fond de connexion 7,9 Mo → 0,24 Mo, avatar 0,7 → 0,12 Mo, bannière 4,5 → 0,28 Mo, logos réduits. Les fichiers non référencés (`ela2.png` 19 Mo, `etudiant.png` 8 Mo, `custom-background.png`, `logo.pdf`) sont retirés du dépôt.

## [2.0.0a1] — 2026-10-13 — Phase 1a : fondations (structure, ressources partagées, outillage)

### Ajouté
- `pyproject.toml` : le projet devient le paquet installable `ela-ai` (`pip install -e ".[admin,dev]"`), avec les commandes `ela-ingest`, `ela-users`, `ela-hash`. `requirements.txt` supprimé (une seule source de vérité).
- `src/ela/` : `engine.py` (moteur de session léger), `retrieval/` (ressources partagées + retriever scopé), `db/` (moteur SQL partagé avec pool, comptes, threads), `history.py` (historique borné), `prompts.py`, `guard.py`, `log.py`, `ingestion/`, `cli/`.
- `apps/chat/` (Chainlit : `app.py`, `data_layer.py`) et `apps/admin/` (Streamlit, déplacé).
- Tests : 39 tests (configuration, historique, retrieval avec composants simulés, moteur avec LLM simulé, intégration PostgreSQL optionnelle via `TEST_DATABASE_URL`).
- CI GitHub Actions (ruff + pytest avec PostgreSQL), `.pre-commit-config.yaml`.

### Modifié
- **Performance** : embeddings, base vectorielle, index BM25 (un par niveau) et reranker sont chargés **une fois par processus** (préchargement en arrière-plan au démarrage, `WARMUP_ON_START`) et rechargés automatiquement si la base Chroma est reconstruite ; ouvrir une session ne recharge plus rien.
- **Base de données** : un seul moteur SQLAlchemy async (pool) au lieu d'un moteur par requête ; bcrypt vérifié hors de la boucle d'événements ; une seule `DATABASE_URL` (le pilote asyncpg/psycopg2 est dérivé automatiquement, `ADMIN_DATABASE_URL` devient optionnelle).
- **Historique** : fenêtre bornée (`HISTORY_MAX_TURNS`, `HISTORY_MAX_CHARS`) au lieu du fil complet.
- Configuration : le `.env` de la racine est trouvé depuis `apps/admin` et les CLI (`ELA_ENV_FILE` pour forcer un chemin) ; nouveaux paramètres `RETRIEVAL_*`, `RERANK_*`, `CHUNK_*`, `LOG_JSON`, `STORAGE_DIR`.
- `print` remplacés par `logging` (texte en dev, JSON en Docker).
- Dockerfiles réécrits pour la nouvelle structure (`pip install .` du paquet) ; `docker-compose.yml` passe `DATABASE_URL` aux deux services.
- README réécrit.

### Supprimé
- `app.py`, `main_ela.py`, `rag_guard.py`, `ingest.py`, `manage_users.py`, `generate_hash.py`, `requirements.txt`, `admin/` (contenu déplacé).

## [2.0.0a0] — 2026-10-06 — Phase 0 : remise en marche et sécurisation

### Ajouté
- Paquet `ela/` : configuration centralisée `ela.config` (pydantic-settings, lecture `.env`) et fabrique de modèles `ela.llm` (client OpenAI-compatible unique pour Cerebras, ILaaS et Groq ; modèle de secours ; routage vision).
- `.env.example` documenté (fournisseurs, modèles, base de données, backoffice). Une variable vide prend sa valeur par défaut ; `LLM_MODEL_FALLBACK=none` et `LLM_REASONING_EFFORT=omit` désactivent explicitement.
- Reverse proxy **Caddy** (`Caddyfile`, service `caddy`) : HTTPS automatique, en-têtes de sécurité, `/admin` protégé par authentification basique.
- `tests/test_llm.py` : 12 tests hors-ligne (fabrique, fallback, encodage d'images, payload, valeurs `.env` vides).
- `generate_hash.py --caddy` : génère le hash `ADMIN_BASIC_AUTH_HASH` sans Docker.
- `manage_users.py` : CLI de gestion des comptes (`check`, `migrate`, `create`, `set-password`, `deactivate`, `list`) — testé contre PostgreSQL.
- `.dockerignore`, `CHANGELOG.md`, `PLAN_MODERNISATION.md`.

### Modifié
- **LLM** : abandon de Groq / Llama 4 Scout (modèle arrêté le 17/07/2026) au profit de Cerebras (`gpt-oss-120b` texte, `qwen-3.8-27b` vision). `main_ela.py` et `rag_guard.py` n'importent plus `langchain_groq`.
- **Vision** : plusieurs images par message, MIME réel transmis (jpeg/png/webp/gif), limite `MAX_IMAGES_PER_MESSAGE`.
- `docker-compose.yml` : `app` et `admin` ne publient plus de port (accès via Caddy) ; Streamlit servi sous `/admin` ; image Postgres `pgvector/pgvector:pg15` (même version majeure, extension disponible pour la phase 2) ; `admin` attend que la base soit saine.
- `.chainlit/config.toml` : `unsafe_allow_html = false`, `allow_origins` restreint, uploads limités aux images.
- `requirements.txt` : `langchain-groq` → `langchain-openai==1.1.7`.
- `README.md` : sections configuration, LLM, stack et arborescence mises à jour.

### Sécurité
- `.gitignore` durci : secrets (`.env*`, `secrets.toml`, `vpslog.txt`, historiques de hash), dossier `Other/`, données runtime, cours LaTeX.
- Les mots de passe VPS, Postgres et comptes applicatifs présents en clair dans d'anciens fichiers doivent être **changés** (voir PLAN_MODERNISATION.md, phase 0).
