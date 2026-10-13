# ELA AI - Econometrics Learning Assistant 🎓

![ELA AI Banner](public/ela_banner.png)

## 📖 Description

ELA AI est un assistant d'apprentissage intelligent spécialisé en économétrie, développé avec Chainlit et LangChain. Il combine deux approches puissantes :
1.  **RAG (Retrieval-Augmented Generation)** : Pour répondre aux questions théoriques en se basant exclusivement sur des supports de cours LaTeX.
2.  **Vision par Ordinateur** : Pour analyser et expliquer des graphiques, tableaux ou équations manuscrites via des modèles multimodaux.

Le projet est désormais conçu pour la production avec une architecture conteneurisée (**Docker**) et une base de données persistante (**PostgreSQL**).

### ✨ Fonctionnalités principales

- 🧠 **RAG Expert** : Recherche hybride (BM25 + Sémantique) sourcée exclusivement dans vos documents.
- 👁️ **Vision IA** : Analyse d'images (courbes, matrices, scans) via un modèle multimodal (`qwen-3.8-27b`), plusieurs images par message.
- 💾 **Persistance SQL** : Historique des conversations et feedbacks stockés durablement dans PostgreSQL.
- ⚡ **Reranking intelligent** : Utilisation de FlashRank pour optimiser la pertinence des résultats.
- 🔐 **Authentification** : Système multi-utilisateurs (Étudiant / Superviseur) sécurisé.
- 📐 **Support LaTeX** : Affichage natif des formules mathématiques.

---

## 🚀 Installation & Déploiement

Vous avez deux modes d'installation : **Production (Docker)** ou **Développement (Local)**.

### Prérequis

- Git
- Une clé API **Cerebras** ([cloud.cerebras.ai](https://cloud.cerebras.ai)) — ILaaS et Groq sont également supportés (API compatibles OpenAI)
- **Mode Docker** : Docker Desktop & Docker Compose
- **Mode Local** : Python 3.11+ et PostgreSQL installé localement

### Option A : Déploiement Docker (Recommandé)

C'est la méthode la plus simple pour lancer l'application avec sa base de données.

1.  **Cloner le repository**
    ```bash
    git clone [https://github.com/varennes-ecofin/ela-ai-master.git](https://github.com/varennes-ecofin/ela-ai-master.git)
    cd ela-ai-master
    ```

2.  **Configuration**
    Copiez `.env.example` en `.env` et renseignez au minimum :
    ```ini
    CHAINLIT_AUTH_SECRET=   # chainlit create-secret
    POSTGRES_PASSWORD=
    CEREBRAS_API_KEY=
    SITE_ADDRESS=           # domaine, IP du serveur ou localhost
    ADMIN_BASIC_AUTH_HASH=  # python generate_hash.py --caddy
    ```
    Toutes les variables (fournisseurs LLM, modèles, chemins) sont documentées dans `.env.example`.

3.  **Lancer les services**
    ```bash
    docker compose up -d --build
    ```

4.  **Initialiser la Base de Données (Premier lancement uniquement)**
    ```bash
    docker compose exec db psql -U chainlit_user -d chainlit_db -c "
    CREATE TABLE IF NOT EXISTS users (id UUID PRIMARY KEY, identifier TEXT UNIQUE, \"createdAt\" TEXT, metadata JSONB);
    CREATE TABLE IF NOT EXISTS threads (id UUID PRIMARY KEY, name TEXT, \"createdAt\" TEXT, \"userId\" UUID REFERENCES users(id), \"userIdentifier\" TEXT, tags TEXT[], metadata JSONB);
    CREATE TABLE IF NOT EXISTS steps (id UUID PRIMARY KEY, name TEXT, type TEXT, \"threadId\" UUID REFERENCES threads(id), \"parentId\" UUID, \"disableFeedback\" BOOLEAN, streaming BOOLEAN, \"waitForAnswer\" BOOLEAN, \"isError\" BOOLEAN, metadata JSONB, tags TEXT[], input TEXT, output TEXT, \"createdAt\" TEXT, start TEXT, \"end\" TEXT, generation JSONB, \"showInput\" TEXT, language TEXT, indent INT, \"defaultOpen\" BOOLEAN);
    CREATE TABLE IF NOT EXISTS elements (id UUID PRIMARY KEY, \"threadId\" UUID REFERENCES threads(id), type TEXT, url TEXT, \"chainlitKey\" TEXT, name TEXT, display TEXT, \"objectKey\" TEXT, size TEXT, page INT, language TEXT, \"forId\" UUID, mime TEXT, props JSONB);
    CREATE TABLE IF NOT EXISTS feedbacks (id UUID PRIMARY KEY, \"forId\" UUID REFERENCES steps(id), value INT, comment TEXT);
    INSERT INTO users (id, identifier, \"createdAt\") VALUES (gen_random_uuid(), 'etudiant', NOW()) ON CONFLICT (identifier) DO NOTHING;
    "
    ```

L'application est servie en HTTPS par Caddy : **https://SITE_ADDRESS/** (chat) et **https://SITE_ADDRESS/admin/** (backoffice, authentification basique). Avec une IP ou `localhost`, le certificat est auto-signé (avertissement navigateur) ; avec un nom de domaine, Let's Encrypt est utilisé automatiquement.

### Option B : Installation Locale (Développement)

1.  **Environnement virtuel**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
    pip install -r requirements.txt
    ```

2.  **Configuration .env**
    Attention à l'URL de la base de données qui doit pointer vers votre localhost :
    ```ini
    DATABASE_URL=postgresql+asyncpg://chainlit_user:securepassword@localhost:****/chainlit_db
    ```

3.  **Lancer l'application**
    ```bash
    chainlit run app.py -w
    ```

---

## 📚 Configuration de la base de connaissances

Pour que le RAG fonctionne, vous devez ingérer vos cours.

1.  **Préparer vos fichiers**
    Placez vos fichiers `.tex` dans le dossier `./latex/`.

2.  **Lancer l'ingestion**
    ```bash
    python ingest.py
    ```
    *Cela va générer la base vectorielle dans le dossier `./chroma_db/`.*

---

## 🎮 Utilisation

### Identifiants par défaut

| Utilisateur | Mot de passe | Rôle |
|-------------|--------------|------|
| `etudiant` | *(défini en base)* | Accès standard |
| `supervisor` | *(défini en base)* | Accès complet (admin) |

*Les comptes sont stockés dans la table `users` (hash bcrypt) et gérés depuis le backoffice ; `generate_hash.py` permet de créer un hash manuellement.*

### Modes d'interaction

- **💬 Chat & Vision** : Posez une question ou glissez-déposez une image (graphique, équation) pour obtenir une explication.
- **🎓 Quiz Interactif** : Cliquez sur le starter "Générer un quiz" (ou tapez `/start_quiz`). ELA vous demandera un sujet et lancera une série de questions interactives.
- **💻 Atelier Code** : Cliquez sur le starter "Atelier Code" (ou tapez `/code_workshop`). Choisissez votre langage (**Python** ou **R**) et le modèle à implémenter. ELA générera un script complet avec données synthétiques.

---

## 🏗️ Architecture du projet

```text
ela-ai-master/
├── .chainlit/              # Configuration Chainlit et traductions
├── .files_ela/             # Stockage physique des images uploadées (Volume Docker)
├── chroma_db/              # Base de données vectorielle (RAG - Embeddings des cours)
├── latex/                  # Fichiers sources .tex (Cours d'économétrie)
├── public/                 # Assets statiques (Logos, icônes des starters)
├── ela/                    # Paquet partagé : config (pydantic-settings), fabrique LLM
├── tests/                  # Tests hors-ligne (pytest)
├── .env                    # Variables d'environnement (Secrets, API Keys, DB URL)
├── .env.example            # Modèle documenté du .env
├── Caddyfile               # Reverse proxy HTTPS (chat + /admin)
├── app.py                  # Interface Chat (Chainlit) + Gestion Base de Données
├── main_ela.py             # Cerveau RAG : Orchestration LangChain et Vision
├── rag_guard.py            # Contrôle qualité : Grader de pertinence et filtrage de domaine
├── chainlit.md             # Message d'accueil (Markdown)
├── docker-compose.yml      # Orchestration des services (App + PostgreSQL)
├── Dockerfile              # Configuration de l'image Docker Python
├── ingest.py               # Script d'indexation des documents vers ChromaDB
└── requirements.txt        # Liste des dépendances Python
```
---

## 🔧 Configuration avancée

### Modifier le fournisseur ou le modèle LLM

Aucune modification de code : tout se règle dans `.env` (voir `ela/config.py` et `ela/llm.py`).

```ini
LLM_PROVIDER=cerebras            # cerebras | ilaas | groq (API compatibles OpenAI)
LLM_MODEL_CHAT=gpt-oss-120b      # texte : réponses, quiz, atelier code
LLM_MODEL_VISION=qwen-3.8-27b    # utilisé automatiquement quand une image est jointe
LLM_MODEL_FALLBACK=qwen-3.8-27b  # modèle de secours (texte), `none` pour désactiver
LLM_REASONING_EFFORT=low         # gpt-oss : low|medium|high ; qwen : none|low|...
```

### Stockage des fichiers

Les images uploadées sont stockées localement via la classe `LocalStorageClient` dans `app.py`. En production Docker, ce dossier est monté via un volume pour ne pas perdre les données au redémarrage.

---

## 🐛 Résolution de problèmes

### Erreur "getaddrinfo failed" (Docker vs Local)

Si vous passez du serveur à votre PC local, n'oubliez pas de changer `DATABASE_URL` dans le `.env` :

* Serveur Docker : `@db:****`
* PC Local : `@localhost:****`

### Erreur d'affichage des icônes (Starters)

Assurez-vous que le dossier `public` est bien monté dans le `docker-compose.yml` et videz le cache de votre navigateur.

---

## 📊 Stack technique

| Composant | Technologie |
| --- | --- |
| **Frontend/Backend** | Chainlit 2.9.6 |
| **LLM Engine** | Cerebras (`gpt-oss-120b`, `qwen-3.8-27b`) — ILaaS / Groq en option |
| **Reverse proxy** | Caddy 2 (HTTPS automatique) |
| **Database** | PostgreSQL 15 + pgvector + AsyncPG |
| **Vector Store** | ChromaDB (migration pgvector prévue en phase 2) |
| **Orchestration** | Docker Compose |
| **Framework IA** | LangChain |

---

## 📝 License & Contact

Ce projet est développé dans un cadre pédagogique.
© 2026 - Gilles de Truchis

**GitHub** : [github.com/varennes-ecofin](https://github.com/varennes-ecofin)
