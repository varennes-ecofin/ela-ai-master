# Changelog — ELA AI

Toutes les évolutions notables du projet. Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/).
Le plan de modernisation complet est dans `PLAN_MODERNISATION.md`.

## [Unreleased] — Phase 2 : retrieval 2.0

### Ajouté
- **Parseur LaTeX v2** (`ela.ingestion.texclean`, `ela.ingestion.latex`) : suppression correcte des commentaires (les frames commentées ne sont plus ingérées), expansion des macros sans argument du cours, chunks typés — `slide`, `definition`/`theorem`/`proposition`/`lemma`/`corollary`/`example`/`remark`/`model`/`assumption` (rattachés à leur slide par `parent_id`) et `text` (manuscrits découpés par chapitre/section/paragraphes) — avec `doc_kind` (`slides`/`manuscript`), notes `\note{}` de l'enseignant, tableaux, figures, colonnes, chapitres beamer personnalisés ; identifiants stables par hash de contenu.
- **Embeddings** `BAAI/bge-m3` (`ela.retrieval.embeddings`) : local (sentence-transformers) ou API OpenAI-compatible (ILaaS), repli local automatique sur le même modèle.
- **Store pgvector** (`ela.retrieval.store.PgVectorStore`, migration `0002_rag_chunks`) : table `rag_chunks` avec index HNSW et `tsvector` français, recherche hybride vecteur + plein texte fusionnée par RRF en une requête SQL ; `ChromaStore` (Chroma + BM25) derrière la même interface.
- **Ingestion incrémentale** : `ela-ingest` ne vectorise que les chunks nouveaux ou modifiés et supprime ceux qui ont disparu ; `--full`, `--dump chunks.jsonl` ; backoffice : dry run / mise à jour incrémentale / reconstruction complète.
- **Reranker** optionnel (`ela.retrieval.rerank`) : FlashRank ou CrossEncoder `BAAI/bge-reranker-v2-m3`. **Désactivé par défaut** (`RERANK_PROVIDER=none`) : mesuré le 17/10 sur 65 questions (Chroma + bge-m3), l'ordre de fusion seul donne any-hit@5 = 0,94 / MRR = 0,74 / 0,25 s, contre 0,88 / 0,57 avec `ms-marco-MiniLM-L-12-v2` et 0,59 / 0,29 avec `ms-marco-MultiBERT-L-12` ; le rappel du pool de 40 candidats est de 1,00. Baseline Phase 1 : 0,63 / 0,43.
- **Retriever** : slides recherchées d'abord, manuscrits en quota (`RETRIEVAL_MANUSCRIPT_K`) et plafonnés dans le top-n final (`RETRIEVAL_MANUSCRIPT_MAX`), union des deux branches (vecteur, plein texte) comme pool de candidats, expansion des définitions/théorèmes vers leur slide parente avec re-remplissage après dédoublonnage, scores de fusion et de reranking dans les métadonnées.
- `ela-eval retrieval` mesure aussi le **rappel du pool** (slide attendue parmi les candidats avant reranking) pour distinguer un défaut de recherche d'un défaut de reranking.
- `ela.mathfmt.normalize_math` : les réponses en `\(…\)`, `\[…\]` ou `equation`/`align` nus sont réécrites en `$…$` / `$$…$$` pour KaTeX (le LaTeX brut n'apparaît plus dans le chat).
- Tests : parseur (fixtures beamer/manuscrit), stores (pgvector en intégration, Chroma), retriever, pipeline incrémental, normalisation LaTeX, configuration.
- `eval/questions.yaml` : jeu d'évaluation v1 — 71 questions (UTS 17, MTS 17, HFE 15, SA 9, NCE 7, cloisonnement 1, hors corpus 2, hors domaine 3) avec slides attendues et points de réponse, rédigées à partir des `.tex` ; chaque slide attendue a été vérifiée contre le corpus ingéré. **À valider par l'enseignant.**
- `ela-eval retrieval` : mesure hit@k, any-hit@k, MRR et latence du retriever par niveau et par cours, rapport JSON dans `eval/reports/` ; `ela-eval compare` pour comparer deux rapports (avant/après).
- `docker-compose.yml` : le Caddy d'ELA devient un profil optionnel (`--profile standalone`) ; par défaut app/admin n'écoutent que sur 127.0.0.1 pour être servis par un reverse proxy existant (`deploy/caddy-ela.snippet`).

### Modifié
- Configuration : `EMBEDDING_PROVIDER/MODEL/DIM/BASE_URL/API_KEY/LOCAL_FALLBACK`, `RETRIEVAL_BACKEND`, `RETRIEVAL_MANUSCRIPT_K`, `RETRIEVAL_LEXICAL_WEIGHT`, `RETRIEVAL_RRF_K`, `RETRIEVAL_FTS_CONFIG`, `RETRIEVAL_EXPAND_ELEMENTS`, `RERANK_PROVIDER/MODEL/MAX_LENGTH`, `CHUNK_TARGET_CHARS`, `CHUNK_MAX_CHARS`, `INGEST_INCLUDE_NOTES` remplacent `EMBEDDING_MODEL_NAME`, `RETRIEVAL_BM25_WEIGHT`, `RERANK_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.
- Le contexte RAG transmis au modèle conserve les sauts de ligne (listes, formules en bloc) et indique l'emplacement de chaque extrait (chapitre › section › slide).
- Prompt : consigne explicite sur les délimiteurs LaTeX et la reprise des notations du cours.
- Dépendances : `langchain-community`, `langchain-classic`, `langchain-chroma`, `langchain-text-splitters` retirées (Chroma utilisé via `chromadb` directement).

### Supprimé
- Swap atomique de dossiers Chroma dans le backoffice (remplacé par l'ingestion incrémentale, visible sans redémarrage).

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
