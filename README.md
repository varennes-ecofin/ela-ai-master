# ELA AI — Econometrics Learning Assistant 🎓

![ELA AI Banner](public/ela_banner.png)

ELA AI est un assistant pédagogique en économétrie (séries temporelles, économétrie financière) destiné aux étudiants du master ESA. Il répond **à partir des supports de cours LaTeX** de l'enseignant (RAG), analyse des graphiques ou équations en image, génère des quiz et des scripts Python/R.

> Projet en cours de modernisation — voir `PLAN_MODERNISATION.md` (phases) et `CHANGELOG.md`.

## Fonctionnalités

- 🧠 **RAG** : cours LaTeX découpés en chunks typés (slides, définitions, théorèmes, manuscrits), embeddings `bge-m3`, recherche hybride (vecteurs + plein texte français) dans PostgreSQL/pgvector avec reranking multilingue, cloisonnée par niveau (M1 / M2 / Commun).
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
│   ├── prompts.py, guard.py, history.py, mathfmt.py
│   ├── retrieval/          #   embeddings, store (pgvector | chroma), reranker, retriever scopé par niveau
│   ├── ingestion/          #   parseur LaTeX (texclean.py, latex.py) + pipeline incrémental `ela-ingest`
│   ├── eval/               #   jeu de questions + `ela-eval` (métriques de retrieval)
│   ├── db/                 #   moteur SQL partagé, comptes, threads, migrations Alembic (`ela-db`)
│   └── cli/                #   `ela-users`, `ela-hash`
├── apps/chat/              # application Chainlit (app.py, data_layer.py)
├── apps/admin/             # backoffice Streamlit
├── tests/                  # tests unitaires + intégration PostgreSQL
├── .chainlit/, public/, chainlit.md   # configuration et assets Chainlit (racine)
├── eval/questions.yaml     # questions de référence (rapports dans eval/reports/)
├── data/latex/             # cours .tex (non versionné) ; data/hf-cache, data/flashrank : modèles téléchargés
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
ela-db upgrade                                    # crée/met à jour le schéma (Alembic, idempotent)
ela-users check                                   # connexion + schéma
ela-users create admin --role admin --level ALL   # mot de passe demandé au clavier
ela-users create etudiant1 --level M1 --quota 50
ela-users list
```

`ela-db upgrade` crée les tables Chainlit (`users`, `threads`, `steps`, `elements`, `feedbacks`) et les colonnes de comptes ELA sur une base vide, et complète sans rien casser une base existante. Les migrations sont dans `src/ela/db/migrations/` (`ela-db history`, `ela-db current`).

### Base de connaissances

Déposez vos fichiers `.tex` dans `data/latex/{M1|M2|Commun}/{NomDuCours}/` (un fichier maître avec ses `\include{}` est résolu automatiquement) puis :

```bash
ela-ingest --dry-run                  # analyse les cours et affiche le rapport (chunks par cours et par type)
ela-ingest --dry-run --dump chunks.jsonl   # idem + export de chaque chunk pour relecture
ela-ingest                            # incrémental : ne vectorise que les chunks nouveaux ou modifiés
ela-ingest --full                     # vide le store puis réingère tout
```

Le parseur produit des chunks typés : une **slide** par `frame` (titre, contenu lisible, notes `\note{}`), un chunk par **définition / théorème / proposition…** rattaché à sa slide, et des paragraphes de **manuscrit** (HDR, articles) regroupés par section. Les manuscrits sont marqués `doc_kind=manuscript` et ne remontent qu'en complément des slides.

Le premier lancement télécharge `BAAI/bge-m3` (~2,2 Go) dans `data/hf-cache/` ; comptez quelques minutes de vectorisation sur CPU pour ~1 500 chunks.

**Store** : par défaut `RETRIEVAL_BACKEND=pgvector` — les chunks vont dans la table `rag_chunks` de `DATABASE_URL` (extension `vector`). Sur un PostgreSQL Windows sans pgvector, deux options :

- installer pgvector (binaires communautaires pour Windows ou compilation, voir [pgvector — installation](https://github.com/pgvector/pgvector#installation)) puis, en superuser, `CREATE EXTENSION vector;` dans la base ELA ;
- ou garder `RETRIEVAL_BACKEND=chroma` en local (`data/chroma_db/`, aucune extension requise) — même parseur, mêmes embeddings, même reranker ; seule la recherche plein texte est remplacée par BM25.

Sur le serveur, l'image `pgvector/pgvector:pg15` du compose inclut l'extension.

### Lancer

```bash
chainlit run apps/chat/app.py -w         # chat : http://localhost:8000
cd apps/admin && streamlit run app.py    # backoffice : http://localhost:8501
```

### Évaluation du RAG

```bash
ela-eval retrieval --label before     # hit@5 / MRR du retriever sur eval/questions.yaml → eval/reports/
ela-eval retrieval --label after      # même mesure après changement de configuration
ela-eval compare eval/reports/<avant>.json eval/reports/<après>.json
```

### Qualité

```bash
pytest -q                                 # tests (intégration DB si TEST_DATABASE_URL est défini)
ruff check src apps tests && ruff format src apps tests
pre-commit install                        # hooks ruff au commit
```

## Déploiement (Docker, serveur)

Deux modes, selon que le serveur a déjà un reverse proxy ou non :

```bash
# Serveur avec un Caddy (ou nginx) déjà en place — cas du VPS :
docker compose up -d --build
#   → app sur 127.0.0.1:8000, backoffice sur 127.0.0.1:8501 ; copier deploy/caddy-ela.snippet
#     dans le Caddyfile existant (sous-domaine dédié ou chemin /ela) puis `caddy reload`.

# Serveur dédié, sans reverse proxy :
docker compose --profile standalone up -d --build
#   → le service `caddy` d'ELA prend les ports 80/443 (HTTPS automatique via SITE_ADDRESS).
```

Services : `app` (Chainlit), `admin` (Streamlit, protégé par authentification basique dans Caddy), `db` (PostgreSQL 15 + pgvector), `caddy` (profil `standalone` seulement). Compose force `DATABASE_URL` vers le service `db` : le `.env` reste identique en local et sur le serveur.

Premier démarrage : `docker compose exec app ela-db upgrade`, `docker compose exec -it app ela-users create admin --role admin --level ALL`, puis `docker compose exec app ela-ingest` (les cours sont dans `./data/latex`, monté dans le conteneur). Les modèles (embeddings, reranker) sont téléchargés au premier lancement dans `data/` (volume), donc une seule fois.

## Configuration des modèles

Aucune modification de code : tout se règle dans `.env` (voir `src/ela/config.py`, `.env.example`).

```ini
LLM_PROVIDER=cerebras            # cerebras | ilaas | groq
LLM_MODEL_CHAT=gpt-oss-120b      # réponses, quiz, atelier code
LLM_MODEL_VISION=qwen-3.8-27b    # utilisé automatiquement quand une image est jointe
LLM_MODEL_FALLBACK=qwen-3.8-27b  # modèle de secours (`none` pour désactiver)
LLM_REASONING_EFFORT=low         # gpt-oss : low|medium|high ; qwen : none|low|... ; omit = non envoyé

EMBEDDING_PROVIDER=local         # local (sentence-transformers) | ilaas | openai (API compatible)
EMBEDDING_MODEL=BAAI/bge-m3      # 1024 dims ; EMBEDDING_LOCAL_FALLBACK=true recharge le même modèle en local si l'API tombe
RETRIEVAL_BACKEND=pgvector       # pgvector | chroma
RERANK_PROVIDER=none             # none (ordre de fusion, meilleur mesuré) | flashrank (ms-marco-MiniLM-L-12-v2) | local (bge-reranker-v2-m3)
```

Pour comparer deux configurations (par ex. les deux rerankers), lancez `ela-eval retrieval --label <nom>` avec chacune puis `ela-eval compare`.

## Stack

| Composant | Technologie |
| --- | --- |
| Front chat | Chainlit 2.12 |
| LLM | Cerebras (`gpt-oss-120b`, `qwen-3.8-27b`) — ILaaS / Groq en option |
| Orchestration | LangChain 1.x |
| Base de connaissances | PostgreSQL + pgvector (hybride vecteur + plein texte français, RRF) — ChromaDB + BM25 en repli |
| Embeddings / rerank | `BAAI/bge-m3` (local ou API ILaaS) ; reranking optionnel (FlashRank, `BAAI/bge-reranker-v2-m3`) — désactivé par défaut, la fusion RRF suffit |
| Base de données | PostgreSQL 15 + pgvector, SQLAlchemy (asyncpg / psycopg2), Alembic |
| Backoffice | Streamlit (FastAPI + HTMX prévu en phase 5) |
| Reverse proxy | Caddy 2 |

## Licence & contact

Projet développé dans un cadre pédagogique — © 2026 Gilles de Truchis — [github.com/varennes-ecofin](https://github.com/varennes-ecofin)
