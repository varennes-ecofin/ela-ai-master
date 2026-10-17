# ELA AI — Prompt de relance pour une nouvelle conversation (Phase 3)

> À coller tel quel en premier message d'une nouvelle discussion du projet **ELA AI**.
> Les documents du projet (`claude/PLAN_MODERNISATION.md`, `claude/RELANCE_PHASE3.md`) et le dossier `E:\ELA AI`
> (lié via l'application de bureau) contiennent les détails ; ce prompt fixe le contexte et la règle du jeu.

---

Je suis Gilles, économiste (Université d'Orléans, responsable du master ESA). Nous modernisons ensemble **ELA AI**, mon assistant pédagogique en économétrie (RAG sur mes cours LaTeX, étudiants M1/M2). Le dossier `E:\ELA AI` est lié à cette conversation. Commence par lire les documents du projet `claude/PLAN_MODERNISATION.md` (phases, décisions, statut) et `claude/RELANCE_PHASE3.md` (ce brief), ainsi que `README.md` et `CHANGELOG.md` dans le dossier, puis propose-moi le **plan détaillé de la Phase 3** (agent) que je validerai avant toute modification de code. Pose tes questions si nécessaire.

## Règles de travail (inchangées)

- Tu codes et testes dans ta sandbox, puis tu écris les fichiers dans `E:\ELA AI` et tu me donnes les **commandes à lancer sous VS Code** (PowerShell, venv `.venv` activé, Python 3.12). Docker n'est **pas** installé sur mon PC ; la pile Docker ne sert que sur le VPS.
- Python avec docstrings en anglais, code rigoureux, réponses en français, concises ; pas d'options annexes superflues. Tu n'as jamais besoin de mes clés API ni de mes mots de passe.
- Sécurité : `.env`, `secrets.toml`, `vpslog.txt`, `Other/`, `data/` restent hors git (déjà dans `.gitignore`). Le dépôt GitHub (`varennes-ecofin`) est public mais ne contient aucun secret.
- Sous VS Code, le `.env` est injecté dans le terminal à son ouverture : après modification, fermer/rouvrir le terminal. Une variable d'environnement prime sur le `.env`.

## État du projet (17/10/2026) — version `2.0.0a3`

**Phases 0, 1 et 2 livrées et validées.** Stack : Chainlit 2.12 (front + data layer PostgreSQL maison), LangChain 1.x (`langchain-core`, `langchain-openai`, `langchain-huggingface` uniquement), pydantic-settings, SQLAlchemy (asyncpg/psycopg2) + Alembic, pytest (68 tests verts), ruff, paquet `ela-ai` en `src/` layout installé en `-e`.

Arborescence utile :
- `src/ela/config.py` (Settings, tous les paramètres `.env`), `llm.py` (fabrique Cerebras/ILaaS/Groq, vision, fallback), `engine.py` (`ElaEngine.ask/generate_quiz/generate_code`, `format_context`), `guard.py` (RAGGuard, à remplacer en Phase 3), `prompts.py`, `history.py`, `mathfmt.py` (normalisation LaTeX → `$…$`).
- `src/ela/ingestion/texclean.py` + `latex.py` (parseur v2 : chunks typés `slide`/`definition`/`theorem`/…/`text`, `doc_kind` slides|manuscript, `parent_id`, `chunk_id` par hash) ; `pipeline.py` (`ela-ingest`, incrémental, `--full`, `--dry-run`, `--dump`).
- `src/ela/retrieval/` : `embeddings.py` (bge-m3 local | ILaaS | OpenAI-compatible, repli local), `store.py` (`ChunkStore` : `PgVectorStore` hybride SQL + `ChromaStore` + BM25, RRF), `rerank.py` (optionnel), `resources.py` (singletons process), `retriever.py` (`ScopedRetriever` : union des deux branches ≈ 40 candidats, quota manuscrit, sélection top-n avec plafond manuscrit, expansion élément → slide parente, métadonnées `score`/`rerank_score`/`matched_element`, `last_hits`).
- `src/ela/eval/` : `eval/questions.yaml` (71 questions, `expected_sources`, `answer_points`), `ela-eval retrieval|compare` (hit@k, any-hit, MRR, rappel du pool, latence, rapports JSON dans `eval/reports/`).
- `src/ela/db/` : moteurs, comptes (`ela-users`), threads, migrations `0001_baseline`, `0002_rag_chunks` (`ela-db upgrade`).
- `apps/chat/app.py` (Chainlit, actions quiz/code, streaming + `normalize_math`), `apps/admin/` (Streamlit : utilisateurs, base de connaissances avec dry run / incrémental / complet, activité, feedbacks, conversations).
- `docker-compose.yml` (app/admin sur 127.0.0.1, `pgvector/pgvector:pg15`, Caddy en profil `standalone`), `deploy/caddy-ela.snippet`, `Dockerfile`, `apps/admin/Dockerfile`.

**Résultats Phase 2** (`ela-eval`, k=5, 65 questions, Chroma + bge-m3 local, sans reranker) : any-hit@5 **0,94** (baseline 0,63), hit@5 0,90 (0,60), MRR **0,74** (0,43), latence 0,25 s (1,55 s) ; rappel du pool de candidats 1,00 ; échecs restants UTS-16, MTS-02, SA-06, NCE-05 (rangs 8–19). Les rerankers FlashRank dégradent (MiniLM 0,88, MultiBERT 0,59) → `RERANK_PROVIDER=none` par défaut ; `bge-reranker-v2-m3` (`local`) disponible mais non mesuré.

## Mon environnement

- Windows, PostgreSQL **18** local (base `chainlit_db`, rôle ELA désormais propriétaire de la base, du schéma, des tables et des vues), pgAdmin. **pgvector n'est pas installé** sur le PC : `RETRIEVAL_BACKEND=chroma` en local (store `data/chroma_db`, 1 466 chunks ingérés) ; pgvector est prévu sur le VPS (image Docker) — installer sur le PC reste une option (binaires communautaires PostgreSQL 18).
- Modèles en cache : `BAAI/bge-m3` (`C:\Users\…\.cache\huggingface`), FlashRank dans `data/flashrank`.
- LLM : Cerebras `gpt-oss-120b` (texte) et `qwen-3.8-27b` (vision/fallback), `LLM_REASONING_EFFORT=low`. ILaaS : clé disponible, embeddings `bge-m3` à `https://rag-api.ilaas.fr/v1`, **URL du service de chat encore inconnue** (modèles annoncés : qwen3.8-27b, gemma-4-31b, mistral-small-4-119b).
- VPS (démo, 3–4 utilisateurs) : une autre application y tourne déjà derrière **Caddy** (ports 80/443 pris) → ELA en `docker compose up -d` sans profil + snippet dans le Caddyfile existant (sous-domaine ou chemin `/ela`, à décider). Production ultérieure sur serveurs de l'université puis ILaaS (30–60 utilisateurs).

## Reste à faire (hors Phase 3)

- Commit/push de la Phase 2 (`git add . && git commit -m "Phase 2 : parseur LaTeX v2, bge-m3, store hybride pgvector/Chroma, ingestion incrémentale, rendu LaTeX, eval" && git push`), déplacer `ci.github-workflow.yml` → `.github/workflows/ci.yml` et `pre-commit-config.example.yaml` → `.pre-commit-config.yaml`.
- Valider `eval/questions.yaml` (71 questions rédigées par Claude à partir des `.tex`).
- Vérifier dans Chainlit le rendu LaTeX (plus de `\(…\)` bruts) et le backoffice (onglet « Mise à jour »).
- Rotation des mots de passe VPS/Postgres/comptes (recommandée depuis la Phase 0).
- Plus tard : juge LLM de fidélité (Phase 2.6, `answer_points`), job d'ingestion asynchrone avec journal (Phase 5), mesure `ela-eval` en pgvector sur le VPS.

## Phase 3 — ce que j'attends

Objectif du plan : remplacer la chaîne fixe `RAGGuard → prompt → LLM` par un **agent** (`langchain.agents.create_agent`, LangChain 1.x / LangGraph) qui décide quand et comment chercher, avec :
1. outils `search_course(query, course=None, chapter=None, k)`, `get_course_outline(course)`, `get_slide(chunk_id | course + slide_title)`, `generate_quiz`, `generate_exercise` (sortie structurée) ; décomposition des questions composées en plusieurs recherches ;
2. middleware : limites d'appels, retry/fallback modèle, résumé d'historique, garde-fou de domaine léger, injection du niveau M1/M2 ;
3. UI Chainlit : appels d'outils visibles (`cl.Step`), **panneau de sources** (cours › chapitre › slide, extrait rendu en LaTeX, scores déjà présents dans les métadonnées), contrôle des citations fantômes ;
4. suppression de `guard.py`, prompts versionnés ; traçage Langfuse optionnel ;
5. évaluation de bout en bout (fidélité par juge LLM, citations correctes, latence) avec `eval/questions.yaml`.

Critères : question multi-étapes résolue avec 2 recherches visibles ; latence médiane ≤ actuelle ; aucune citation fantôme sur le jeu de test ; refus propre hors domaine. Présente le plan par étapes livrables et testables, avec pour chacune les fichiers touchés, les tests prévus et les commandes de vérification. Attends mon « go » avant de coder.
