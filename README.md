# ELA AI — Econometrics Learning Assistant 🎓

![ELA AI Banner](public/ela_banner.png)

ELA AI est un assistant pédagogique en économétrie (séries temporelles, économétrie financière) destiné aux étudiants du master ESA. Il répond **à partir des supports de cours LaTeX** de l'enseignant (RAG), analyse des graphiques ou équations en image, génère des quiz et des scripts Python/R.

> Projet en cours de modernisation — voir `PLAN_MODERNISATION.md` (phases) et `CHANGELOG.md`.

## Fonctionnalités

- 🧠 **RAG** : recherche hybride (vecteurs + BM25) avec reranking, cloisonnée par niveau (M1 / M2 / Commun).
- 👁️ **Vision** : analyse d'images (plusieurs par message) par un modèle multimodal.
- 🎓 **Quiz** et 💻 **Atelier Code** (Python / R) ancrés dans le cours.
- 🔐 **Comptes** en base PostgreSQL (bcrypt), rôles, niveau, quota journalier.
- 🛠️ **Backoffice** (Streamlit) : utilisateurs, base de connaissances, activité, feedbacks, conversations.
- 🔌 **Fournisseurs LLM interchangeables** (Cerebras, ILaaS, Groq — API compatibles OpenAI) par simple configuration.

## Architecture

```
ela-ai/
├── pyproject.toml          # paquet `ela-ai`, dépendances, outils (ruff, pytest)
├── src/ela/                # cœur applicatif (installé en paquet)
│   ├── config.py           #   configuration (pydantic-settings, .env)
│   ├── llm.py              #   fabrique de modèles (Cerebras / ILaaS / Groq, fallback, vision)
│   ├── engine.py           #   moteur de session (ask / quiz / code)
│   ├── prompts.py, guard.py, history.py
│   ├── retrieval/          #   ressources partagées, retriever scopé par niveau
│   ├── ingestion/          #   parseur LaTeX + pipeline `ela-ingest`
│   ├── db/                 #   moteur SQL partagé, comptes, threads
│   └── cli/                #   `ela-users`, `ela-hash`
├── apps/chat/              # application Chainlit (app.py, data_layer.py)
├── apps/admin/             # backoffice Streamlit
├── tests/                  # tests unitaires + intégration PostgreSQL
├── .chainlit/, public/, chainlit.md   # configuration et assets Chainlit (racine)
├── data/latex/             # cours .tex  (non versionné)   data/chroma_db/  base vectorielle
├── Caddyfile, docker-compose.yml, Dockerfile, apps/admin/Dockerfile
└── .env.example            # modèle documenté de configuration
```

## Installation (développement)

Prérequis : Python 3.11 ou 3.12, PostgreSQL local, une clé API Cerebras.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate sous Linux/macOS)
pip install -e ".[admin,dev]"     # paquet ela + backoffice + outils de dev
copy .env.example .env            # puis renseigner les valeurs (voir ci-dessous)
```

Variables minimales du `.env` : `CHAINLIT_AUTH_SECRET` (`chainlit create-secret`), `DATABASE_URL`, `CEREBRAS_API_KEY`. Toutes les autres sont documentées dans `.env.example`.

> Sous VS Code, le `.env` est injecté dans le terminal intégré à son ouverture : après toute modification, fermez et rouvrez le terminal.

### Base de données et comptes

```bash
ela-users check                                   # connexion + schéma
ela-users migrate                                 # ajoute les colonnes de comptes (idempotent)
ela-users create admin --role admin --level ALL   # mot de passe demandé au clavier
ela-users create etudiant1 --level M1 --quota 50
ela-users list
```

La base Chainlit (tables `users`, `threads`, `steps`, `elements`, `feedbacks`) doit exister ; le script SQL d'initialisation est dans `GUIDE_backoffice.md` (remplacé par Alembic en phase 1b).

### Base de connaissances

Déposez vos fichiers `.tex` dans `data/latex/{M1|M2|Commun}/{NomDuCours}/` puis :

```bash
ela-ingest --dry-run      # vérifie la structure et affiche le rapport
ela-ingest                # calcule les embeddings et (re)construit data/chroma_db
```

### Lancer

```bash
chainlit run apps/chat/app.py -w         # chat : http://localhost:8000
cd apps/admin && streamlit run app.py    # backoffice : http://localhost:8501
```

### Qualité

```bash
pytest -q                                 # tests (intégration DB si TEST_DATABASE_URL est défini)
ruff check src apps tests && ruff format src apps tests
pre-commit install                        # hooks ruff au commit
```

## Déploiement (Docker, serveur)

```bash
docker compose up -d --build
```

Services : `caddy` (HTTPS automatique, ports 80/443), `app` (Chainlit), `admin` (Streamlit sous `/admin`, protégé par `ADMIN_BASIC_AUTH_*`), `db` (PostgreSQL 15 + pgvector). `SITE_ADDRESS` fixe l'adresse servie : un nom de domaine donne un certificat Let's Encrypt, une IP ou `localhost` un certificat auto-signé. Compose force `DATABASE_URL` vers le service `db` : le `.env` reste identique en local et sur le serveur.

Premier démarrage : créer les tables Chainlit puis `docker compose exec app ela-users migrate` et `docker compose exec -it app ela-users create admin --role admin --level ALL`.

## Configuration des modèles

Aucune modification de code : tout se règle dans `.env` (voir `src/ela/config.py` et `src/ela/llm.py`).

```ini
LLM_PROVIDER=cerebras            # cerebras | ilaas | groq
LLM_MODEL_CHAT=gpt-oss-120b      # réponses, quiz, atelier code
LLM_MODEL_VISION=qwen-3.8-27b    # utilisé automatiquement quand une image est jointe
LLM_MODEL_FALLBACK=qwen-3.8-27b  # modèle de secours (`none` pour désactiver)
LLM_REASONING_EFFORT=low         # gpt-oss : low|medium|high ; qwen : none|low|... ; omit = non envoyé
```

## Stack

| Composant | Technologie |
| --- | --- |
| Front chat | Chainlit 2.9 |
| LLM | Cerebras (`gpt-oss-120b`, `qwen-3.8-27b`) — ILaaS / Groq en option |
| Orchestration | LangChain 1.x |
| Vector store | ChromaDB (pgvector prévu en phase 2) |
| Embeddings / rerank | sentence-transformers, FlashRank (bge-m3 + reranker multilingue prévus en phase 2) |
| Base de données | PostgreSQL 15 + pgvector, SQLAlchemy (asyncpg / psycopg2) |
| Backoffice | Streamlit (FastAPI + HTMX prévu en phase 5) |
| Reverse proxy | Caddy 2 |

## Licence & contact

Projet développé dans un cadre pédagogique — © 2026 Gilles de Truchis — [github.com/varennes-ecofin](https://github.com/varennes-ecofin)
