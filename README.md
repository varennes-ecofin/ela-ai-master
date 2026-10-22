# ELA AI — Econometrics Learning Assistant 🎓

![ELA AI Banner](public/ela_banner.png)

ELA AI est un assistant pédagogique en économétrie (séries temporelles, économétrie financière) destiné aux étudiants du master ESA. Il répond **à partir des supports de cours LaTeX** de l'enseignant (RAG), analyse des graphiques ou équations en image, génère des quiz et des scripts Python/R.

> Projet en cours de modernisation — voir `PLAN_MODERNISATION.md` (phases) et `CHANGELOG.md`.

## Fonctionnalités

- 🤖 **Agent** (LangChain 1.x / LangGraph) : le modèle décide quand et comment chercher dans le cours (`search_course`, `get_course_outline`, `get_slide`), décompose les questions composées en plusieurs recherches, et affiche chaque étape dans le chat.
- 🧠 **RAG** : cours LaTeX découpés en chunks typés (slides, définitions, théorèmes, manuscrits), embeddings `bge-m3`, recherche hybride (vecteurs + plein texte français) dans PostgreSQL/pgvector, cloisonnée par niveau (M1 / M2 / Commun).
- 📚 **Sources vérifiées** : chaque extrait lu reçoit un identifiant `[S1]`, `[S2]`… ; la réponse ne peut citer que ces extraits (les citations fantômes sont retirées) et un panneau latéral montre cours › chapitre › slide avec l'extrait rendu en LaTeX.
- 👁️ **Vision** : les images jointes (graphiques, équations, sorties logicielles) sont transcrites par un modèle multimodal puis traitées par l'agent.
- 🎓 **Quiz** (persistants, difficulté adaptative, relance ciblée sur les erreurs), ✏️ **Exercices socratiques** (« Vérifier mon étape », indices progressifs, corrigé), 📄 **Fiches de révision** par chapitre (`/fiche`, générées depuis les slides, téléchargeables en Markdown, partagées entre étudiants), 📈 **Bilan** de l'étudiant (`/bilan`), 💻 **Atelier Code** (Python / R, exécution des scripts Python dans un bac à sable isolé avec sortie et graphiques, correction automatique, quota journalier) et, après chaque réponse, des **suggestions de suivi** (quiz, exercice, fiche, exemple Python) — tous ancrés dans le cours.
- 🔐 **Comptes** en base PostgreSQL (bcrypt), rôles, niveau, quota journalier.
- 🛠️ **Backoffice** (Streamlit) : utilisateurs, base de connaissances, activité, **pédagogie** (réussite aux quiz, exercices, questions par chapitre, sujets hors corpus), feedbacks, conversations. Un **nouveau backoffice** (FastAPI + HTMX, `/backoffice`) le remplace page par page : connexion par compte (`admin`, `supervisor`), journal d'administration, tableau de bord (phase 5).
- 🔌 **Fournisseurs LLM interchangeables** (Cerebras, ILaaS, Groq — API compatibles OpenAI) par simple configuration.

## Architecture

```
ela-ai/
├── pyproject.toml          # paquet `ela-ai`, dépendances, outils (ruff, pytest)
├── src/ela/                # cœur applicatif (installé en paquet)
│   ├── config.py           #   configuration (pydantic-settings, .env)
│   ├── llm.py              #   fabrique de modèles (Cerebras / ILaaS / Groq, fallback, vision)
│   ├── engine.py           #   moteur de session : un message étudiant → flux d'événements (tokens, étapes, sources)
│   ├── agent/              #   agent create_agent : tools.py, middleware.py, citations.py, outline.py, prompts/*.md
│   ├── pedagogy/           #   quiz et exercices (sorties structurées Pydantic)
│   ├── prompts.py, history.py (fenêtre + résumé), mathfmt.py, structured.py, tracing.py (Langfuse optionnel)
│   ├── retrieval/          #   embeddings, store (pgvector | chroma), reranker, retriever scopé par niveau
│   ├── ingestion/          #   parseur LaTeX (texclean.py, latex.py) + pipeline incrémental `ela-ingest`
│   ├── eval/               #   jeu de questions, `ela-eval retrieval` (hit@k) et `ela-eval answer` (juge LLM, citations)
│   ├── db/                 #   moteur SQL partagé, comptes, threads, audit, migrations Alembic (`ela-db`)
│   ├── auth/               #   fournisseurs d'identité (local ; CAS à venir)
│   └── cli/                #   `ela-users`, `ela-hash`
├── apps/chat/              # application Chainlit (app.py, ui.py : étapes, panneau de sources, quiz, exercice)
├── apps/admin/             # backoffice Streamlit (en cours de remplacement)
├── apps/backoffice/        # nouveau backoffice FastAPI + Jinja2 + HTMX (phase 5)
├── tests/                  # tests unitaires + intégration PostgreSQL
├── .chainlit/, public/, chainlit.md   # configuration et assets Chainlit (racine)
├── eval/questions.yaml     # questions de référence (rapports dans eval/reports/)
├── data/latex/             # cours .tex (non versionné) ; data/hf-cache, data/flashrank : modèles téléchargés
├── Caddyfile, docker-compose.yml, Dockerfile, apps/admin/Dockerfile, apps/backoffice/Dockerfile
└── .env.example            # modèle documenté de configuration
```

## Installation (développement)

Prérequis : Python 3.11 ou 3.12, PostgreSQL local, une clé API Cerebras.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate sous Linux/macOS)
pip install -e ".[admin,backoffice,dev]"   # paquet ela + backoffices + outils de dev
copy .env.example .env            # puis renseigner les valeurs (voir ci-dessous)
```

Variables minimales du `.env` : `CHAINLIT_AUTH_SECRET` (`chainlit create-secret`), `DATABASE_URL`, `CEREBRAS_API_KEY`, `BACKOFFICE_SECRET_KEY` (`python -c "import secrets; print(secrets.token_urlsafe(48))"`). Toutes les autres sont documentées dans `.env.example`.

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
python apps/backoffice/main.py --reload  # nouveau backoffice : http://localhost:8502/backoffice/
cd apps/admin && streamlit run app.py    # backoffice Streamlit : http://localhost:8501
```

### Backoffice (FastAPI + HTMX)

`apps/backoffice/` remplace progressivement Streamlit (phase 5) : FastAPI + Jinja2 + HTMX, icônes Lucide (sprite SVG inclus dans les pages), aucun actif externe. Connexion avec un compte de la table `users` de rôle `admin` (tout) ou `supervisor` (consultation) ; les étudiants sont refusés. Sessions dans un cookie signé (`BACKOFFICE_SECRET_KEY`, 8 h par défaut), jeton CSRF sur chaque formulaire, verrouillage temporaire après cinq échecs, compte revérifié à chaque requête (désactiver ou rétrograder un compte prend effet immédiatement). Chaque action est inscrite dans la table `admin_audit` (migration `0009`), visible sur l'accueil pour les `admin`. L'application est servie sous `BACKOFFICE_ROOT_PATH` (`/backoffice`) en local comme derrière Caddy ; `/backoffice/healthz` (`?deep=1` pour vérifier la base) sert de sonde. Le fournisseur d'identité est une abstraction (`ela.auth`, `AUTH_PROVIDER=local`) prévue pour accueillir le CAS de l'université avec création automatique des comptes.

### Évaluation

```bash
ela-eval retrieval --label before     # hit@5 / MRR du retriever sur eval/questions.yaml → eval/reports/
ela-eval retrieval --label after      # même mesure après changement de configuration
ela-eval compare eval/reports/<avant>.json eval/reports/<après>.json

ela-eval answer --engine legacy --label before   # chaîne fixe de la phase 2 (mesure « avant »)
ela-eval answer --label agent                    # agent : latence, recherches, citations, fidélité (juge LLM)
ela-eval answer --ids UTS-17,MTS-05 --no-judge   # sous-ensemble, sans juge
ela-eval compare eval/reports/<before>-answer.json eval/reports/<agent>-answer.json
```

`ela-eval answer` répond à chaque question comme l'étudiant la recevrait et mesure : latence médiane/p90, recherches et appels d'outils, citations valides ou fantômes, fidélité aux `answer_points` (juge LLM, modèle `EVAL_JUDGE_MODEL`), refus hors domaine, étiquetage « connaissances générales » hors corpus, questions multi-étapes résolues avec ≥ 2 recherches. Le rapport JSON contient chaque réponse pour relecture.

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
#   → app sur 127.0.0.1:8000, Streamlit sur 127.0.0.1:8501, backoffice sur 127.0.0.1:8502 ; copier deploy/caddy-ela.snippet
#     dans le Caddyfile existant (sous-domaine dédié ou chemin /ela) puis `caddy reload`.

# Serveur dédié, sans reverse proxy :
docker compose --profile standalone up -d --build
#   → le service `caddy` d'ELA prend les ports 80/443 (HTTPS automatique via SITE_ADDRESS).
```

Services : `app` (Chainlit), `admin` (Streamlit, protégé par authentification basique dans Caddy), `backoffice` (FastAPI, `/backoffice`, connexion dans l'application), `db` (PostgreSQL 15 + pgvector), `sandbox` (exécution des scripts, réseau interne), `caddy` (profil `standalone` seulement). Compose force `DATABASE_URL` vers le service `db` : le `.env` reste identique en local et sur le serveur.

Premier démarrage : `docker compose exec app ela-db upgrade`, `docker compose exec -it app ela-users create admin --role admin --level ALL`, puis `docker compose exec app ela-ingest` (les cours sont dans `./data/latex`, monté dans le conteneur). Les modèles (embeddings, reranker) sont téléchargés au premier lancement dans `data/` (volume), donc une seule fois.

## Agent

Chaque message étudiant est une exécution sans état de l'agent (`src/ela/agent/`) : prompt système rendu depuis `prompts/system_<PROMPT_VERSION>.md` (niveau M1/M2, cours visibles, résumé des échanges anciens), outils `search_course` / `get_course_outline` / `get_slide` / `generate_quiz` / `generate_exercise`, limites d'appels (`AGENT_MAX_*`), modèle de secours. L'historique est borné (`HISTORY_MAX_TURNS`, `HISTORY_MAX_CHARS`) et les tours évincés sont résumés une fois par session. Les prompts sont des fichiers Markdown versionnés : pour tester une variante, copier `system_v1.md` en `system_v2.md` et régler `PROMPT_VERSION=v2`, puis comparer avec `ela-eval answer`.

Un étudiant ne voit que les cours de son niveau (`--level M1` par défaut dans `ela-users create`) et les documents `Commun` ; `ALL` et `admin` voient tout. Quiz et exercices ne sont générés que si les extraits retournés mentionnent le sujet demandé (`ela.pedagogy.relevance`) ; un sujet traité dans un cours d'un autre niveau est signalé comme tel (« traité dans Financial_Econometrics (M2) »). Les quiz sont générés par lots de 5 questions (10 au plus). Les fiches de révision sont construites à partir des slides du chapitre (plan du cours) et mises en cache dans `review_sheets` par (cours, chapitre, niveau, `PROMPT_VERSION`, empreinte du corpus) : une ré-ingestion ou un changement de prompt les régénère.

### Atelier Code 2.0 — exécution des scripts

Le service `apps/sandbox/` exécute les scripts Python des étudiants dans un conteneur isolé (réseau interne sans accès Internet, non-root, système de fichiers en lecture seule, limites CPU/mémoire/processus/temps). Il est activé côté application par `SANDBOX_URL=http://sandbox:8080` dans le `.env` (`docker compose up -d --build sandbox`) ; vide, les scripts sont générés mais pas exécutés. `SANDBOX_DAILY_QUOTA` (10 par défaut) limite les exécutions par étudiant et par jour ; elles sont journalisées dans `code_runs` (backoffice, onglet Pédagogie › Code). Les scripts sont générés avec un budget et un effort de raisonnement dédiés (`CODE_MAX_TOKENS`, `CODE_REASONING_EFFORT`), vérifiés syntaxiquement avant affichage, et corrigés automatiquement une fois en cas d'échec d'exécution (`CODE_AUTO_FIX`). Après une exécution réussie, ELA interprète la sortie réelle (tableau des résultats, commentaire relié au cours avec citations) ; un script qui n'imprime rien de chiffré se voit proposer un complément. Quand le bac à sable est configuré, l'agent dispose aussi de l'outil `run_python` (un appel par réponse). Pour essayer le flux sur un poste de développement sans Docker : `python apps/sandbox/server.py` (nécessite numpy, pandas, statsmodels, matplotlib dans l'environnement) et `SANDBOX_URL=http://127.0.0.1:8080` — sans isolation réseau, donc pour vos propres scripts uniquement.

Traçage optionnel avec [Langfuse](https://langfuse.com) : `pip install -e ".[tracing]"` puis `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (et `LANGFUSE_HOST` pour une instance à vous). Désactivé tant que les clés sont vides ; l'identifiant étudiant n'est transmis que si `LANGFUSE_SEND_USER_ID=true`.

## Configuration des modèles

Aucune modification de code : tout se règle dans `.env` (voir `src/ela/config.py`, `.env.example`).

```ini
LLM_PROVIDER=cerebras            # cerebras | ilaas | groq
LLM_MODEL_CHAT=gpt-oss-120b      # agent (appels d'outils), quiz, exercices, résumé, atelier code
LLM_MODEL_VISION=qwen-3.8-27b    # transcrit les images jointes pour l'agent
LLM_MODEL_FALLBACK=qwen-3.8-27b  # modèle de secours de l'agent (`none` pour désactiver)
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
| Orchestration | LangChain 1.4 (`create_agent`) + LangGraph 1.2 ; traçage Langfuse optionnel |
| Base de connaissances | PostgreSQL + pgvector (hybride vecteur + plein texte français, RRF) — ChromaDB + BM25 en repli |
| Embeddings / rerank | `BAAI/bge-m3` (local ou API ILaaS) ; reranking optionnel (FlashRank, `BAAI/bge-reranker-v2-m3`) — désactivé par défaut, la fusion RRF suffit |
| Base de données | PostgreSQL 15 + pgvector, SQLAlchemy (asyncpg / psycopg2), Alembic |
| Backoffice | FastAPI + Jinja2 + HTMX (phase 5, en cours) ; Streamlit jusqu'à parité |
| Reverse proxy | Caddy 2 |

## Licence & contact

Projet développé dans un cadre pédagogique — © 2026 Gilles de Truchis — [github.com/varennes-ecofin](https://github.com/varennes-ecofin)
