# ELA AI — Plan de modernisation (v2)

> **Version** : 1.3 — 2026-10-13
> **Auteur** : Gilles de Truchis, avec assistance Claude
> **Statut** : Phase 0 validée · Phase 1a validée (13/10) · Phase 1b livrée (en test) · Phase 2 à suivre
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
1. **Jeu d'évaluation** : 40–60 questions de référence réparties par niveau/cours (UTS, MTS, HFE, SA, Research), chacune avec les slides attendues. Claude génère une première version à partir des `.tex`, Gilles valide/corrige. Script `ela-eval` : hit@5, MRR, et juge LLM de fidélité. **Mesure « avant »** sur le système actuel.
2. **Embeddings** `bge-m3` (ILaaS en priorité, local en repli), 1024 dims, chunks jusqu'à 8k tokens — fin de la troncature à 128 tokens.
3. **Reranker** multilingue `bge-reranker-v2-m3` (CPU, ~1 s pour 20 candidats).
4. **Store** pgvector : table `chunks(id, course_id, level, chapter, section, slide_title, kind, content, content_hash, embedding vector(1024), tsv tsvector)` ; recherche hybride = RRF(cosinus, `ts_rank` config `french`) en une requête SQL ; filtres niveau/cours. (Si Chroma conservé : mêmes embeddings/reranker, tokeniseur BM25 corrigé.)
5. **Ingestion** : parseur LaTeX robuste (`pylatexenc`/`TexSoup`) ; unités typées `definition`, `theorem`, `proposition`, `example`, `remark`, `frame` ; conservation du LaTeX mathématique brut ; en-tête contextuel (cours › chapitre › section › slide) ; **incrémental** par `content_hash` (seuls les fichiers modifiés sont ré-encodés) ; exécution en job asynchrone avec journal en base ; CLI `ela-ingest [--course …] [--dry-run]`.
6. Sous-questions : décomposition des questions composées en plusieurs recherches (sera repris par l'agent en phase 3).
7. **Mesure « après »**, comparaison dans le rapport d'évaluation.

Critères d'acceptation : hit@5 et MRR en hausse significative sur le jeu de test (objectif indicatif : hit@5 ≥ 0,85) ; ajout d'un fichier `.tex` ⇒ seuls ses chunks sont ré-encodés, visible en < 1 min sur le VPS via ILaaS ; démonstration « ajout d'un cours en direct » possible.

### Phase 3 — ELA agentique (5–6 sessions)

**Objectif** : remplacer la chaîne fixe `RAGGuard → prompt → LLM` par un agent qui décide quand et comment chercher.

Tâches :
1. `langchain.agents.create_agent` (LangGraph) avec les **outils** :
   - `search_course(query, course=None, chapter=None, k=8)` — hybride + rerank, renvoie chunks + métadonnées ;
   - `get_course_outline(course)` — table des matières (chapitres/sections/titres de slides) ;
   - `get_slide(course, slide_title | chunk_id)` — lecture exacte d'une slide citée ;
   - `generate_quiz(topic, n, difficulty)` — sortie structurée Pydantic, LaTeX autorisé ;
   - `generate_exercise(topic)` — énoncé + indices + corrigé (révélation progressive) ;
   - (option) `run_python(code)` — voir phase 4.
2. **Middleware** : limite d'appels modèle/outils ; retry + `ModelFallbackMiddleware` ; `SummarizationMiddleware` (historique) ; sélection dynamique du modèle (vision) ; garde-fou de domaine léger en `before_model` ; injection du niveau M1/M2 et du profil pédagogique dans le system prompt.
3. **UI Chainlit** : chaque appel d'outil affiché en `cl.Step` (« Recherche dans UTS_C2 → 5 slides ») ; **sources en panneau latéral** (cours, chapitre, titre de slide, extrait rendu en LaTeX) ; vérification que chaque `[Source: …]` cité correspond à un chunk réellement récupéré, sinon correction.
4. Suppression de `rag_guard.py` ; prompts versionnés dans `ela/agent/prompts/`.
5. Traçage **Langfuse** (auto-hébergé dans le compose, optionnel) : coût, latence, chaîne d'outils, feedbacks reliés aux traces.
6. Évaluation de bout en bout sur le jeu de test (fidélité, taux de citation correcte, latence moyenne).

Critères d'acceptation : question multi-étapes (« compare le test ADF du chapitre 2 au KPSS ») résolue avec 2 recherches visibles ; latence médiane ≤ celle de l'actuel malgré l'agentivité ; aucune citation fantôme sur le jeu de test ; refus propre hors domaine.

### Phase 4 — Expérience étudiant (4–6 sessions, modulable)

1. **Quiz 2.0** : persistant, LaTeX dans questions/options (options rendues dans le message, boutons A/B/C/D), difficulté adaptative, relance ciblée sur les erreurs, score enregistré.
2. **Mode exercice / socratique** : ELA guide sans donner la solution ; boutons « Indice », « Vérifier mon étape », « Solution ».
3. **Fiches de révision** par chapitre (générées depuis les slides, exportables Markdown/PDF).
4. **Flashcards** avec révision espacée (table `student_progress`).
5. **Tableau de bord étudiant** (dans Chainlit) : notions vues, scores, points faibles, suggestions.
6. **Suggestions de suivi** cliquables en fin de réponse (« voir la démonstration », « me tester », « exemple Python »).
7. **Profils de chat** : Cours / Entraînement / Atelier Code.
8. (Option, à confirmer) **Atelier Code 2.0** : exécution du script généré dans un service Docker isolé (sans réseau, CPU/mémoire/temps limités, `numpy/pandas/statsmodels/matplotlib`), sortie et graphique affichés dans le chat.

Critères d'acceptation : un étudiant M1 enchaîne quiz → erreur → explication → exercice guidé sans quitter le fil ; ses résultats apparaissent dans son tableau de bord et dans le backoffice.

### Phase 5 — Backoffice et observabilité (3–4 sessions)

1. Nouveau backoffice **FastAPI + Jinja2 + HTMX** (si confirmé) : auth session, rôles, CRUD utilisateurs/quotas, gestion des cours et **jobs d'ingestion** (progression en direct), activité, feedbacks, lecteur de conversations.
2. Pages nouvelles : **Évaluation** (lancer `ela-eval`, comparer deux configurations) ; **Pédagogie** (questions fréquentes par chapitre, notions échouées aux quiz, sujets hors corpus demandés ⇒ signal pour les cours).
3. Sauvegardes Postgres automatisées, healthchecks, journal applicatif centralisé.
4. Streamlit retiré une fois la parité atteinte.

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

Jalons démontrables : **J1** (fin P0) ELA répond à nouveau en HTTPS ; **J2** (fin P2) ajout d'un cours en direct + rapport d'évaluation avant/après ; **J3** (fin P3) agent avec étapes et sources visibles ; **J4** (fin P4) parcours étudiant complet.

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

---

## 6. Mode de travail

1. Claude développe et teste dans son environnement (tests unitaires avec modèles simulés, Postgres en conteneur pour l'intégration).
2. Les fichiers sont écrits dans `E:\ELA AI` ; chaque livraison est accompagnée des **commandes VS Code** (`uv sync`, `uv run pytest`, `docker compose up`, etc.) et d'une liste de points à vérifier.
3. Gilles valide, commit et push ; déploiement sur le VPS pour la démo.
4. Un `CHANGELOG.md` et ce plan (statut par phase) sont tenus à jour dans le dépôt.
