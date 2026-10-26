# ELA AI — Plan de modernisation (v2)

> **Version** : 2.1 — 2026-10-21
> **Auteur** : Gilles de Truchis, avec assistance Claude
> **Statut** : Phases 0 à 4 validées (jalon J4, 19/10) · **Consolidation faite le 21/10** : dépôt `main` à jour avec CI verte, nouvelle pile déployée sur le VPS (`https://ela.varennes-vps.com`, Caddy Docker sur réseau `proxy`, base migrée, sandbox actif) · **Phase 5 cadrée le 21/10** (décisions ci-dessous) : lots 5.1 (socle) et 5.2 (utilisateurs) validés · **lot 5.3 (cours, ingestion en tâche de fond) livré le 26/10, à tester** · lots 5.4 → 5.6 puis « Annales » à suivre, puis Phase 6
> **Périmètre** : refonte architecture + passage à un RAG agentique + nouvelles fonctionnalités étudiant

---

## 0. Décisions actées

| Sujet | Décision |
|---|---|
| Front | **Chainlit** (mise à jour 2.9.6 → 2.12.x) |
| Fournisseur LLM principal | **Cerebras** — `gpt-oss-120b` (texte, agent) et `qwen-3.8-27b` (vision, agent, sorties structurées) |
| Fournisseur LLM secondaire | **ILaaS** — `qwen3.8-27b`, `gemma-4-31b`, `mistral-small-4-119b` (production université) |
| Groq | abandonné (modèle Llama 4 Scout arrêté le 17/07/2026) |
| Embeddings | **`bge-m3`** via ILaaS (`https://rag-api.ilaas.fr/v1`, API compatible OpenAI) avec repli **local** (`BAAI/bge-m3`, même modèle ⇒ mêmes vecteurs, index interchangeable) |
| Cibles de déploiement | VPS (démo, 3–4 utilisateurs) → serveurs université (30–60 utilisateurs) → ILaaS |
| Git | dépôt GitHub existant, non synchronisé ; remise en place d'un flux `commit → push` sur branche `modernisation` |
| Mode de travail | Claude code et teste dans son environnement, écrit ensuite les fichiers dans `E:\ELA AI`, et fournit les commandes à lancer sous VS Code |

### Décisions confirmées le 06/10/2026

1. **Stockage vectoriel** : **pgvector** (voir §2.3), derrière une interface `VectorStore` conservant Chroma en repli.
2. **Backoffice** : **FastAPI + Jinja2 + HTMX** en phase 5 (voir §2.5), Streamlit conservé jusqu'à parité.
3. **Atelier Code 2.0** (exécution sandboxée) : **inclus** en phase 4.
4. **ILaaS** : URL du service d'inférence et disponibilité d'un reranker **à préciser** (variable `ILAAS_CHAT_BASE_URL` prévue, vide en attendant ; reranker local par défaut).
5. Dépôt GitHub **public** ; la version en ligne n'expose aucun secret (vérifié par Gilles). Les secrets ayant figuré en clair dans le dossier local sont tout de même à renouveler.

---

## 1. Rappel des constats d'audit (résumé)

- **Bloquant** : modèle LLM arrêté ; secrets en clair dans le dossier projet (corrigé côté `.gitignore` le 06/10, rotation des mots de passe à faire) ; pas de HTTPS.
- **Qualité RAG** : embedding tronqué à 128 tokens (le contenu mathématique des slides n'est pas vectorisé), reranker anglophone, BM25 naïf, parseur LaTeX par regex, aucune évaluation.
- **Architecture** : moteur RAG + index BM25 reconstruits **à chaque session utilisateur**, moteur SQL recréé à chaque requête, historique non borné, pipeline rigide à 2–4 appels LLM par message, quiz non persistants, dette outillage (pas de tests/CI/migrations/logs).

---

## 2. Architecture cible

### 2.1 Vue d'ensemble

```
                 ┌──────────── Caddy (HTTPS, Let's Encrypt) ────────────┐
                 │  /            → app (Chainlit)                        │
                 │  /admin       → admin (Streamlit → FastAPI/HTMX)      │
                 └───────────────────────────────────────────────────────┘
                              │                          │
        ┌─────────────────────▼──────────┐   ┌───────────▼────────────┐
        │  app  — Chainlit 2.12          │   │  admin                  │
        │  ├─ auth / data layer (PG)     │   │  ├─ utilisateurs/quotas │
        │  ├─ ela.agent (LangChain 1.x)  │   │  ├─ cours & ingestion   │
        │  │   outils: search_course,    │   │  │   (jobs asynchrones) │
        │  │   get_outline, get_slide,   │   │  ├─ activité/feedbacks  │
        │  │   quiz, exercise, (run_py)  │   │  └─ évaluation RAG      │
        │  └─ UI: steps, sources, quiz   │   └───────────┬────────────┘
        └──────────┬──────────┬──────────┘               │
                   │          │                           │
     ┌─────────────▼──┐  ┌────▼──────────────────────────▼─────────────┐
     │ LLM providers  │  │  PostgreSQL 16 + pgvector                    │
     │ Cerebras/ILaaS │  │  chainlit (users, threads, steps, feedbacks) │
     │ (OpenAI-compat)│  │  ela (courses, chunks+vecteurs+tsvector,     │
     └────────────────┘  │       quiz_results, progress, ingestion_jobs)│
                         └──────────────────────────────────────────────┘
     Embeddings : ILaaS bge-m3 (API) | local bge-m3 (repli)
     Reranker   : bge-reranker-v2-m3 (local, CPU)
     Traces     : Langfuse (optionnel, auto-hébergé)
```

### 2.2 Arborescence cible

```
ela-ai/
├── pyproject.toml              # uv ; extras : chat, admin, ingest-local, dev
├── docker-compose.yml          # caddy, db (pgvector), app, admin
├── Caddyfile
├── .env.example                # toutes les variables documentées, sans valeurs
├── alembic/                    # migrations versionnées (remplace pgAdmin)
├── src/ela/
│   ├── config.py               # pydantic-settings
│   ├── llm.py                  # fabrique de modèles (provider/modèle/fallback)
│   ├── db/                     # engine async partagé (pool), modèles, dépôts
│   ├── ingestion/              # parser LaTeX, chunker, embedder, indexeur incrémental
│   ├── retrieval/              # VectorStore (pgvector | chroma), hybride, reranker
│   ├── agent/                  # create_agent, outils, middleware, prompts
│   ├── pedagogy/               # quiz, exercices, fiches, progression
│   └── eval/                   # jeu de test, métriques, rapport
├── apps/
│   ├── chat/                   # Chainlit : app.py, data_layer.py, ui/
│   └── admin/                  # Streamlit (phase 1–4) puis FastAPI+HTMX (phase 5)
├── scripts/                    # ela-ingest, ela-eval, ela-create-user
└── tests/                      # unit (sans réseau) + integration (PG en conteneur)
```

### 2.3 Pourquoi pgvector plutôt que Chroma (réponse à la question)

Avec ~2 000–3 000 chunks, **Chroma suffit en termes de performance** ; la qualité de recherche dépend des embeddings et du reranker, pas du moteur. L'intérêt de pgvector est **opérationnel** :

| Besoin | Chroma (actuel) | pgvector + FTS Postgres |
|---|---|---|
| Recherche lexicale (BM25) | index en mémoire reconstruit par processus (et aujourd'hui par session) | `tsvector` config `french` indexé en base, partagé, zéro RAM applicative |
| Mise à jour d'un cours en direct (démo) | reconstruction complète + swap de dossiers | `UPSERT` transactionnel des seuls chunks modifiés, l'app voit le changement immédiatement |
| Filtrage niveau/cours/chapitre | `where` Chroma | `WHERE` SQL, jointures avec `users`, `courses` |
| Sauvegarde / restauration | dossier + base PG à synchroniser | un seul `pg_dump` |
| Multi-instances (université, 30–60 users) | un dossier par conteneur | base centrale partagée |
| Dépendance | chromadb (+ sqlite) | extension `pgvector` sur le Postgres cible |

**Recommandation** : pgvector, avec l'image Docker `pgvector/pgvector:pg16`. **Risque** : si les serveurs de l'université fournissent un Postgres sans l'extension, repli sur Chroma grâce à l'interface `VectorStore` (l'ingestion et l'agent ne changent pas).

### 2.4 Fournisseurs LLM — une seule voie de code

Cerebras, ILaaS et (au besoin) Groq exposent tous une API compatible OpenAI. ELA utilisera un unique client (`ChatOpenAI(base_url=..., api_key=...)`) paramétré par l'environnement, ce qui permet de changer de fournisseur **sans toucher au code** :

```ini
LLM_PROVIDER=cerebras                     # cerebras | ilaas | groq
CEREBRAS_API_KEY=
CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
ILAAS_API_KEY=
ILAAS_CHAT_BASE_URL=                      # à fournir
ILAAS_EMBED_BASE_URL=https://rag-api.ilaas.fr/v1
LLM_MODEL_CHAT=gpt-oss-120b               # agent, explications
LLM_MODEL_VISION=qwen-3.8-27b             # routé automatiquement si une image est jointe
LLM_MODEL_FALLBACK=qwen-3.8-27b
EMBEDDING_PROVIDER=ilaas                  # ilaas | local
EMBEDDING_MODEL=bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3    # vide = désactivé
VECTOR_STORE=pgvector                     # pgvector | chroma
```

Points d'attention : `gpt-oss-120b` n'a pas de vision (les messages avec image passent par `qwen-3.8-27b`) ; `qwen-3.8-27b` raisonne par défaut (`reasoning_effort` réglé bas en chat pour la latence, haut pour la génération de quiz).

### 2.5 Backoffice — remplacer Streamlit ?

Streamlit a permis d'aller vite mais souffre de : session non sécurisée (pas de CSRF/cookies signés, pas de rôles), rerun complet à chaque clic, ingestion lancée en `subprocess` bloquant, difficile à placer derrière un proxy, impossible d'exposer une API réutilisable.

Options :

| Option | Pour | Contre |
|---|---|---|
| **FastAPI + Jinja2 + HTMX + Chart.js** (recommandé) | même stack que Chainlit (FastAPI), auth session/CSRF propre, jobs en arrière-plan, API REST réutilisable (futur front, scripts), peu de JavaScript | plus de code que Streamlit pour les écrans |
| FastAPI + front React/Next | UI riche | build JS, deux langages, surdimensionné ici |
| NiceGUI / Reflex | tout-Python | frameworks jeunes, même lock-in que Streamlit |
| Garder Streamlit | zéro coût | limites ci-dessus |

Le backoffice FastAPI réutilise `src/ela` (dépôts SQL, ingestion, évaluation) : il n'y a donc pas de logique dupliquée, uniquement des vues.

---

## 3. Phases

Chaque phase est livrable seule, avec ses critères d'acceptation et les commandes à lancer sous VS Code. Les estimations sont en **sessions de travail** (une session ≈ une demi-journée de votre côté pour tester/valider).

### Phase 0 — Remise en marche et sécurisation (1–2 sessions)

**Objectif** : ELA fonctionne à nouveau, sans fuite de secrets, en HTTPS.

Tâches :
1. ✅ `.gitignore` durci ; `.env.example` documenté ; `.dockerignore`.
2. ⏳ **Rotation** des mots de passe VPS (SSH), Postgres et comptes `etudiant`/`supervisor` — à faire par Gilles.
3. ✅ `ela/config.py` + `ela/llm.py` : fabrique de modèles compatible OpenAI (Cerebras par défaut, ILaaS, Groq, fallback) ; `main_ela.py` et `rag_guard.py` sans dépendance Groq.
4. ✅ Routage vision : image jointe ⇒ `LLM_MODEL_VISION`, MIME réel transmis, images multiples acceptées.
5. ✅ **Caddy** dans `docker-compose` : HTTPS automatique, `/admin` protégé (auth basique), Chainlit `allow_origins` restreint, `unsafe_allow_html=false`, uploads limités aux images.
6. ⏳ Flux git : branche `modernisation`, premier commit propre, push vers GitHub — à faire par Gilles (commandes fournies).

Critères d'acceptation : une question M1 reçoit une réponse sourcée via Cerebras ; une image est analysée via `qwen-3.8-27b` ; `git status` ne montre aucun fichier sensible ; l'app répond en HTTPS sur le VPS.

### Phase 1 — Fondations techniques (3–4 sessions)

**Objectif** : une base de code maintenable et testable, sans changement fonctionnel visible.

Livrée en deux lots.

**Lot 1a — livré le 13/10/2026**
1. ✅ Restructuration en paquet `src/ela` + `apps/` (§2.2) ; `pyproject.toml` (`pip install -e .`, `uv` compatible) ; `ruff` ; `pytest` (39 tests) ; `pre-commit` ; GitHub Actions (lint + tests avec PostgreSQL).
2. ✅ `pydantic-settings` pour toute la configuration (`.env` racine trouvé depuis les sous-dossiers) ; `logging` (JSON en Docker) à la place des `print`.
3. ✅ **Ressources partagées** : embeddings, store, index BM25 par niveau, reranker et moteur SQL (pool async) chargés **une fois par processus**, préchargés au démarrage, rechargés si la base est reconstruite ; `ElaEngine` est un objet léger par session.
4. ✅ Historique borné (fenêtre par tours et par caractères).
5. ✅ Dockerfiles adaptés à la nouvelle structure ; `.dockerignore`.

**Lot 1b — livré le 13/10/2026**
6. ✅ **Alembic** embarqué (`ela-db upgrade`), migration de base idempotente (tables Chainlit 2.12 + colonnes comptes), remplace le SQL manuel ; les tables `courses`, `chunks`, `quiz_results`, `student_progress`, `ingestion_jobs` arrivent avec les phases qui les utilisent.
7. ✅ Dockerfiles multi-stage, non-root, healthchecks, torch CPU ; caches de modèles persistants dans `data/`. L'image `admin` garde torch tant que l'ingestion calcule les embeddings en local (phase 2).
8. ✅ Chainlit 2.12 / LangChain 1.6 / chromadb 1.5 ; data layer custom validée par un test d'intégration complet.
9. ✅ Images de `public/` optimisées (≈ 45 Mo → < 1 Mo pour les fichiers utilisés ; fichiers inutilisés retirés).
10. Persistance du quiz en base → reportée en phase 4 (Quiz 2.0) pour ne concevoir la table qu'une fois.

Critères d'acceptation : `uv run pytest` vert ; `ruff check` vert ; démarrage d'une session < 1 s après le premier chargement ; 4 sessions simultanées sur le VPS sans augmentation de RAM par session ; `alembic upgrade head` idempotent.

### Phase 2 — Retrieval 2.0 (4–5 sessions) — *gain qualité principal*

**Objectif** : une recherche qui trouve réellement le contenu des slides, mesurée.

Tâches :
1. ✅ **Jeu d'évaluation** : 71 questions (`eval/questions.yaml`), `ela-eval retrieval` / `ela-eval compare`. ✅ **Mesure « avant »** (14/10) : any-hit@5 = 0,63, MRR = 0,43 ; M1 ≈ 0,88, M2 entre 0,29 et 0,40 — échecs M2 dus aux blocs de texte du manuscrit HDR qui saturent le top-5, échecs M1 = slides voisines (troncature 128 tokens). Juge LLM de fidélité à venir.
2. ✅ **Embeddings** `bge-m3` : `EMBEDDING_PROVIDER=local|ilaas|openai` (`ela.retrieval.embeddings`), 1024 dims, repli local automatique sur le même modèle si l'API tombe.
3. ✅ **Reranker** : mesuré inutile avec bge-m3 — désactivé par défaut (`RERANK_PROVIDER=none`) ; FlashRank MiniLM/MultiBERT dégradent (0,88 / 0,59 contre 0,94 sans), `bge-reranker-v2-m3` reste disponible (`local`) et non mesuré.
4. ✅ **Store** pgvector : table `rag_chunks` (migration `0002`, HNSW cosinus + `tsvector` français GIN), recherche hybride RRF en une requête SQL (`ela.retrieval.store.PgVectorStore`) ; Chroma + BM25 conservé derrière la même interface (`RETRIEVAL_BACKEND=chroma`, utile sur un PC Windows sans pgvector).
5. ✅ **Ingestion** : parseur maison (`texclean.py` + `latex.py`, sans dépendance LaTeX) — commentaires, macros du cours, environnements typés (définition, théorème, proposition, lemme, corollaire, exemple, remarque, modèle, hypothèse), colonnes/overlays beamer, tableaux, figures, notes `\note{}`, manuscrits découpés par chapitre/section/paragraphes ; en-tête contextuel ; **incrémental par `content_hash`** (`ela-ingest`, `--full`, `--dump`) ; backoffice : dry run / incrémental / complet. ⏳ Job asynchrone avec journal en base → phase 5 (backoffice FastAPI).
6. ✅ **Retriever** : slides d'abord + quota de manuscrit (`RETRIEVAL_MANUSCRIPT_K`), expansion définition → slide parente, scores exposés dans les métadonnées (panneau sources en phase 3).
7. ✅ Rendu LaTeX dans le chat : normalisation des délimiteurs (`ela.mathfmt`) + consigne de prompt.
8. ✅ **Mesure « après »** (17/10, Chroma + bge-m3, k=5, 65 questions) : any-hit@5 **0,94** (avant 0,63), hit@5 0,90 (0,60), MRR **0,74** (0,43), latence 0,25 s (1,55 s) ; par cours any-hit : UTS 0,94, MTS 0,94, HFE 1,00, SA 0,89, NCE 0,86. Rappel du pool (40 candidats) = 1,00 ; les 4 échecs restants (UTS-16, MTS-02, SA-06, NCE-05) sont aux rangs 8–19 du pool. Objectif ≥ 0,85 atteint. À refaire en pgvector (plein texte français) sur le VPS.
9. Sous-questions : décomposition des questions composées → phase 3 (agent).

Critères d'acceptation : hit@5 et MRR en hausse significative sur le jeu de test (objectif indicatif : hit@5 ≥ 0,85) ; ajout d'un fichier `.tex` ⇒ seuls ses chunks sont ré-encodés, visible en < 1 min sur le VPS via ILaaS ; démonstration « ajout d'un cours en direct » possible.

### Phase 3 — ELA agentique (5–6 sessions) — *validée le 18/10/2026*

**Objectif** : remplacer la chaîne fixe `RAGGuard → prompt → LLM` par un agent qui décide quand et comment chercher.

Décisions prises le 17/10 : citations numérotées `[S1]`, `[S2]`… reliées au panneau de sources (+ `[Connaissances générales]` pour le hors-cours) ; agent **sans état** reconstruit depuis l'historique Chainlit (fenêtre bornée + résumé en session, pas de checkpointer LangGraph avant la phase 4) ; garde-fou de domaine **sans appel LLM supplémentaire** (prompt + contrôle mécanique des citations), pré-classifieur seulement si l'évaluation montre des fuites ; Langfuse optionnel hors compose, jamais de données étudiantes vers le cloud en production ; images **transcrites** par le modèle vision puis traitées par l'agent texte (aucune dépendance au tool calling du modèle vision) ; quiz avec LaTeX dans question/explication et options en texte brut ; exercice minimal Indice/Solution.

Tâches :
1. ✅ **Agent** `langchain.agents.create_agent` (`langchain` 1.4.6 + `langgraph` 1.2.16), `src/ela/agent/` : `agent.py` (graphe compilé partagé par le processus), `context.py` (`TurnContext` : niveau, registre de sources, quiz/exercice du tour), `tools.py`, `outline.py`, `citations.py`, `events.py`, `vision.py`, `middleware.py`, `prompts/` (`system_v1.md`, `quiz_v1.md`, `exercise_v1.md`, `transcribe_v1.md`, `summary_v1.md`, `PROMPT_VERSION`).
   - `search_course(query, course=None, chapter=None, k)` — retriever hybride scopé au niveau (filtre cours côté store, chapitre sur les candidats), extraits étiquetés `[S1]`… dans un `SourceRegistry` par tour ;
   - `get_course_outline(course=None)` — plan chapitre › section › slides (ou liste des cours), construit depuis le store et mis en cache par génération ;
   - `get_slide(source_id | slide_title [, course])` — relecture d'un extrait ou d'une slide du plan ;
   - `generate_quiz(topic, n, difficulty)` / `generate_exercise(topic)` — sorties structurées Pydantic (`ela/pedagogy/`), laissées dans le contexte du tour pour l'UI ;
   - décomposition des questions composées : consigne de prompt (une recherche par notion) + appels d'outils parallèles.
2. ✅ **Middleware** : prompt système dynamique (niveau M1/M2, cours visibles, résumé d'historique) ; `ModelFallbackMiddleware` (modèle de secours) ; `ModelCallLimitMiddleware` et `ToolCallLimitMiddleware` (global, `search_course`, quiz, exercice) ; `ToolErrorMiddleware`. Résumé d'historique maison (`HistorySummarizer`, un appel modèle quand des tours sortent de la fenêtre, mis en cache par session). Les retries réseau restent dans le client `ChatOpenAI`.
3. ✅ **UI Chainlit** (`apps/chat/ui.py`) : un `cl.Step` par appel d'outil (« 🔎 Recherche : « test ADF » → 5 extrait(s) »), réponse streamée, **panneau de sources** = éléments latéraux nommés `S1`, `S2`… (cours › chapitre › slide, score, extrait rendu en KaTeX) vers lesquels les citations de la réponse pointent ; citations fantômes retirées avant affichage ; quiz et exercice (boutons Indice / Solution) déclenchés par l'agent ou par `/start_quiz`, `/exercise`.
4. ✅ `guard.py` supprimé du chemin de production ; la chaîne de la phase 2 est **gelée** dans `ela/eval/legacy_pipeline.py` uniquement pour la mesure « avant » (`ela-eval answer --engine legacy`), à supprimer après comparaison.
5. ✅ Traçage **Langfuse** optionnel (`ela/tracing.py`, variables `LANGFUSE_*`, extra `pip install -e ".[tracing]"`), désactivé par défaut, hors `docker-compose` (Langfuse v3 auto-hébergé exige ClickHouse/Redis/MinIO).
6. ✅ **Évaluation de bout en bout** `ela-eval answer` (`ela/eval/answer.py`, `judge.py`) : latence médiane/p90, recherches et appels d'outils par question, citations valides / fantômes, fidélité par juge LLM (`answer_points`), refus hors domaine, étiquetage hors corpus, multi-étapes avec ≥ 2 recherches ; `ela-eval compare` étendu aux rapports `answer`. ✅ **Mesures du 18/10** (71 questions, Cerebras gpt-oss-120b, juge gpt-oss-120b, effort medium) — chaîne fixe → agent : fidélité 0,80 → **0,86**, citations fantômes 3 → **0**, refus indus 9 → **0** (les 9 questions du cours *Duration_Models* étaient refusées à tort par l'ancien garde-fou), refus hors domaine 1,00 → 1,00, hors corpus étiqueté 0,33 → 0,67, latence médiane 1,10 s → **1,09 s** (p90 1,62 → 1,49 s). Point faible : les questions multi-étapes du jeu (UTS-17, MTS-05, HFE-15) sont résolues en une seule recherche (fidélité 1,00 malgré tout) ; `multihop_two_searches_rate` = 0 — à retravailler via un `system_v2.md`. Bugs corrigés après le test en situation : colonne `steps.autoCollapse` manquante (migration 0003, réponses invisibles à la reprise des fils), persistance des éléments texte, mélange des options de quiz, délimiteurs `$$` multi-lignes.

Critères d'acceptation : question multi-étapes résolue avec 2 recherches visibles (atteint en chat sur « ADF vs Phillips-Perron », pas sur les 3 questions multi-hop du jeu) ; latence médiane ≤ celle de la chaîne fixe ✅ ; aucune citation fantôme sur le jeu de test ✅ ; refus propre hors domaine ✅.

### Phase 4 — Expérience étudiant (4–6 sessions, modulable)

1. ✅ **Quiz 2.0** (lot 4a, livré le 18/10) : persistant (`quiz_results`, migration 0004), LaTeX dans questions/options (options rendues dans le message, boutons A/B/C/D), difficulté adaptative par session, relance ciblée (« M'expliquer davantage » via l'agent, « Refaire un quiz sur mes erreurs » avec questions reformulées), score enregistré.
2. ✅ **Mode exercice / socratique** (lot 4b, livré le 18/10) : ELA guide sans donner la solution ; boutons « Vérifier mon étape » (évaluation structurée de la tentative contre le corrigé caché, question d'orientation), « Indice », « Solution » ; tentatives persistées (`exercise_attempts`, migration 0005).
3. ✅ **Fiches de révision** par chapitre (lot 4d, livré le 18/10) : `/fiche` et outil `generate_review_sheet`, générées depuis les slides du chapitre (plan du cours, parties fusionnées pour les longs chapitres), sortie structurée, Markdown + KaTeX, fichier `.md` téléchargeable, cache partagé `review_sheets` (migration 0007) régénéré sur changement de corpus/prompt. PDF reporté (pas de pandoc/LaTeX dans la pile).
4. **Flashcards** avec révision espacée (table `student_progress`).
5. ✅ **Tableau de bord étudiant** (lot 4c, livré le 18/10) : `/bilan` dans Chainlit (chapitres travaillés, scores, points faibles, exercices, suggestions cliquables) sur `student_activity` (migration 0006) + `quiz_results` + `exercise_attempts` ; onglet « Pédagogie » du backoffice (réussite par sujet, questions ratées, exercices, questions par chapitre, hors corpus).
6. ✅ **Suggestions de suivi** cliquables en fin de réponse (lot 4d) : « Me tester », « Exercice », « Fiche », « Exemple Python », calculées depuis les extraits cités sans appel LLM.
7. **Profils de chat** : Cours / Entraînement / Atelier Code.
8. ✅ **Atelier Code 2.0** (lot 4e, livré le 18/10) : service `apps/sandbox` (sous-processus `python -I` avec limites `resource`, figures capturées), conteneur sur réseau interne sans Internet, non-root, lecture seule ; bouton « Exécuter » sur clic, « Corriger le script » (une passe), outil d'agent `run_python` (1 appel/réponse), quota 10 exécutions/jour/étudiant (`code_runs`, migration 0008), onglet « Code » du backoffice ; interprétation de la sortie réelle après exécution (tableau des résultats + commentaire cité, « Compléter le script » si rien n'est imprimé). R en génération seule. Validé en local le 18/10 ; à valider sur le VPS (Docker).

Correctifs du 18/10 (retours de test avec un profil étudiant M1) : sorties structurées réellement utilisées en production (le modèle avec repli n'exposait pas `with_structured_output` → JSON texte tronqué à 2 048 tokens, antislashs LaTeX non échappés), quiz générés par lots de 5 (10 questions possibles), bouton « Réessayer le quiz » après un échec de l'outil, garde de pertinence sans LLM (`ela.pedagogy.relevance`) contre la dérive de sujet (quiz ARMA pour une demande GARCH), prompt système et messages informés du cloisonnement par niveau (« traité dans Financial_Econometrics (M2) »), bilan comptant les quiz dans les chapitres travaillés.

Critères d'acceptation : un étudiant M1 enchaîne quiz → erreur → explication → exercice guidé sans quitter le fil ; ses résultats apparaissent dans son tableau de bord et dans le backoffice.

### Phase 5 — Backoffice et observabilité (3–4 sessions) — *cadrée le 21/10/2026*

**Décisions du 21/10** : rôle enseignant = `supervisor` existant (consultation) et `admin` (tout) ; authentification cible **CAS** (`auth.univ-orleans.fr/cas`) avec **création automatique des comptes étudiants** au premier login (règle de niveau M1/M2 à partir des attributs CAS, sinon niveau par défaut corrigé dans le backoffice), abstraction `ela.auth` partagée avec Chainlit, seul le fournisseur local implémenté pour l'instant ; **import CSV** des étudiants (identifiant, niveau) conservé au lot 5.2 au cas où la DSI contraindrait l'usage du CAS ; nouveau backoffice sur `ela.varennes-vps.com/backoffice` (même sous-domaine), Streamlit maintenu sur `/admin` jusqu'à parité ; **pas** d'authentification basique Caddy devant le nouveau backoffice (connexion dans l'application, incompatible avec CAS sinon) ; une seule image `backoffice` (torch CPU inclus) qui exécutera les jobs d'ingestion en processus ; journal d'audit dès le socle ; sessions en cookie signé ; HTMX vendu dans le dépôt. Phase de test de la production limitée aux étudiants M1 et M2 du master ESA.

Lots (chacun livré testé) :

1. ✅ **5.1 Socle** (livré le 21/10, validé le 22/10, version `2.0.0a12`) : `apps/backoffice/` (FastAPI + Jinja2 + HTMX), connexion sur les comptes `users` (`admin`, `supervisor`), cookie de session signé, CSRF, verrouillage après échecs, compte revérifié à chaque requête, `ela.auth` (`AuthProvider`, `Identity`, `ProvisioningPolicy`, `AUTH_PROVIDER=local`), table `admin_audit` (migration `0009`), accueil (indicateurs, dernières conversations, journal), `/backoffice/healthz`, service `backoffice` dans compose + extraits Caddy, 21 tests hors ligne + 3 PostgreSQL.
2. ✅ **5.2 Utilisateurs et quotas** (livré le 22/10, validé le 26/10, version `2.0.0a13`) : liste filtrable, fiche avec activité, création (mot de passe saisi ou généré, affiché une seule fois), rôle/niveau/quota, réinitialisation, désactivation, purge des comptes sans activité, **import CSV** avec prévisualisation et identifiants remis une seule fois (`ela-users import` en CLI), mot de passe personnel, politique de mots de passe (8 caractères, chiffre, spécial), audit de chaque action. Décisions du 22/10 : identifiant = partie de l'adresse universitaire avant le `@` ; pas de suppression physique sauf compte sans activité ; `supervisor` en lecture seule sans quotas ni mots de passe ; changement de mot de passe des étudiants dans Chainlit reporté (lot CAS).
3. ✅ **5.3 Cours et ingestion** (livré le 26/10, version `2.0.0a14`) : arborescence `data/latex`, dépôt de fichiers ou de `.zip`, suppression, **déplacement d'un cours entre niveaux** (renommage + ré-indexation automatique), état de la base (chunks par cours, diagnostic disque/base), ingestion en tâche de fond dans un processus enfant (table `jobs`, migration `0010`, progression HTMX, historique), racine `data/annales` préparée. Décisions du 26/10 : `.tex/.sty/.cls/.bib` + zip ; un job à la fois ; Streamlit conservé jusqu'au 5.4.
4. **5.4 Activité, pédagogie, feedbacks, conversations** : parité avec les vues Streamlit (Chart.js), puis retrait de Streamlit.
5. **5.5 Évaluation** : `ela-eval answer` / `retrieval` depuis l'interface (job), historique des rapports, comparaison ; `ela-eval code` (évaluation de l'Atelier Code pour comparer les modèles).
6. **5.6 Exploitation** : sauvegardes PostgreSQL automatisées, healthchecks exposés, journal applicatif consultable, documentation.
7. **Annales** (décidé le 26/10, après le 5.4) : ingestion des sujets d'examen et corrigés LaTeX de `data/annales/{M1|M2}/<Cours>/` en chunks `exam` (énoncé + corrigé), tenus à l'écart des recherches de cours ; `generate_exercise` et `check_step` s'appuient sur de vrais exercices et corrigés ; PDF/Word des collègues en seconde étape (énoncés seulement, formules dégradées à l'extraction).
8. Ensuite : fournisseur **CAS** (`AUTH_PROVIDER=cas`) + provisioning automatique côté Chainlit et backoffice.

### Phase 6 — Préparation production université / ILaaS (2 sessions, le moment venu)

Profil `docker-compose.prod.yml`, variables ILaaS, test de charge 30–60 utilisateurs, documentation d'exploitation, procédure de mise à jour des cours.

---

## 4. Ordre et jalons

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
 remise      fondations   retrieval    agent       étudiant     backoffice   prod
 en marche                + éval
```

Variante « résultat visible vite » : 0 → 2 (embeddings/reranker uniquement) → 3 → 1 → 4 → 5.

Jalons démontrables : **J1** (fin P0) ELA répond à nouveau en HTTPS ; **J2** (fin P2) ajout d'un cours en direct + rapport d'évaluation avant/après ; **J3** (fin P3) agent avec étapes et sources visibles — livré le 17/10 ; **J4** (fin P4) parcours étudiant complet — validé le 19/10 ; **J5** (fin P5) backoffice FastAPI à parité, Streamlit retiré.

---

## 5. Risques et parades

| Risque | Parade |
|---|---|
| pgvector indisponible sur le Postgres de l'université | interface `VectorStore`, repli Chroma |
| Quotas/latence ILaaS embeddings | repli local `bge-m3` (mêmes vecteurs) |
| Changements de modèles chez Cerebras (comme Groq) | modèles en variables d'environnement, fallback, test de fumée au démarrage |
| Rupture Chainlit 2.12 sur la data layer custom | tests d'intégration de la data layer, montée de version isolée en phase 1 |
| Appels d'outils instables sur petits modèles | strict mode (schémas JSON), `ToolCallLimit`, validation Pydantic, fallback `gpt-oss-120b` |
| Régression qualité non détectée | jeu d'évaluation exécuté à chaque phase |
| Exécution de code (si retenue) | conteneur jetable, sans réseau, limites CPU/RAM/temps, liste blanche de paquets |
| VPS : un Caddy sert déjà une autre application (ports 80/443 pris) | le `caddy` d'ELA est un profil optionnel (`--profile standalone`) ; par défaut app/admin n'écoutent que sur 127.0.0.1 et le Caddy existant les sert via `deploy/caddy-ela.snippet` (sous-domaine ou chemin `/ela`) |

---

## 6. Mode de travail

1. Claude développe et teste dans son environnement (tests unitaires avec modèles simulés, Postgres en conteneur pour l'intégration).
2. Les fichiers sont écrits dans `E:\ELA AI` ; chaque livraison est accompagnée des **commandes VS Code** (`uv sync`, `uv run pytest`, `docker compose up`, etc.) et d'une liste de points à vérifier.
3. Gilles valide, commit et push ; déploiement sur le VPS pour la démo.
4. Un `CHANGELOG.md` et ce plan (statut par phase) sont tenus à jour dans le dépôt.
