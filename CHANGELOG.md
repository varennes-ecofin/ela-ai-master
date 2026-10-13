# Changelog — ELA AI

Toutes les évolutions notables du projet. Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/).
Le plan de modernisation complet est dans `PLAN_MODERNISATION.md`.

## [2.0.0a0] — 2026-10-06 — Phase 0 : remise en marche et sécurisation

### Ajouté
- Paquet `ela/` : configuration centralisée `ela.config` (pydantic-settings, lecture `.env`) et fabrique de modèles `ela.llm` (client OpenAI-compatible unique pour Cerebras, ILaaS et Groq ; modèle de secours ; routage vision).
- `.env.example` documenté (fournisseurs, modèles, base de données, backoffice). Une variable vide prend sa valeur par défaut ; `LLM_MODEL_FALLBACK=none` et `LLM_REASONING_EFFORT=omit` désactivent explicitement.
- Reverse proxy **Caddy** (`Caddyfile`, service `caddy`) : HTTPS automatique, en-têtes de sécurité, `/admin` protégé par authentification basique.
- `tests/test_llm.py` : 12 tests hors-ligne (fabrique, fallback, encodage d'images, payload, valeurs `.env` vides).
- `generate_hash.py --caddy` : génère le hash `ADMIN_BASIC_AUTH_HASH` sans Docker.
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
