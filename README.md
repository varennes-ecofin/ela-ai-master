# ELA AI - Econometrics Learning Assistant 🎓

![ELA AI Banner](public/ela_banner.png)

## 📖 Description

ELA AI est un assistant d'apprentissage intelligent spécialisé en économétrie, développé avec Chainlit et LangChain. Il combine deux approches puissantes :
1.  **RAG (Retrieval-Augmented Generation)** : Pour répondre aux questions théoriques en se basant exclusivement sur des supports de cours LaTeX.
2.  **Vision par Ordinateur** : Pour analyser et expliquer des graphiques, tableaux ou équations manuscrites via des modèles multimodaux.

Le projet est conçu pour la production avec une architecture conteneurisée (**Docker**) et une base de données persistante (**PostgreSQL**).

### ✨ Fonctionnalités principales

- 🧠 **RAG Expert** : Recherche hybride (BM25 + Sémantique) sourcée exclusivement dans vos documents.
- 👁️ **Vision IA** : Analyse d'images (courbes, matrices, scans) via Llama 4 Scout / Llama 3.2 Vision.
- 📂 **Galerie "Mes Contenus"** : Espace dédié pour retrouver toutes les images et graphiques envoyés.
- 💾 **Persistance SQL** : Historique des conversations et feedbacks stockés durablement dans PostgreSQL.
- ⚡ **Reranking intelligent** : Utilisation de FlashRank pour optimiser la pertinence des résultats.
- 🔐 **Authentification** : Système multi-utilisateurs (Étudiant / Superviseur) sécurisé.
- 📐 **Support LaTeX** : Affichage natif des formules mathématiques.

---

## 🚀 Installation & Déploiement

Vous avez deux modes d'installation : **Production (Docker)** ou **Développement (Local)**.

### Prérequis

- Git
- Une clé API Groq (gratuite sur [console.groq.com](https://console.groq.com))
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
    Créez un fichier `.env` à la racine :
    ```ini
    GROQ_API_KEY=gsk_votre_cle_ici
    CHAINLIT_AUTH_SECRET=votre_secret_aleatoire
    ELA_AUTH_DATA=etudiant:password,supervisor:password
    
    # Configuration Docker (ne pas toucher pour le mode Docker)
    DATABASE_URL=postgresql+asyncpg://chainlit_user:securepassword@db:5432/chainlit_db
    ```

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

L'application est accessible sur : **http://localhost:80** (ou l'IP de votre serveur).

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
    DATABASE_URL=postgresql+asyncpg://chainlit_user:securepassword@localhost:5432/chainlit_db
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
| `etudiant` | `password` | Accès standard + Galerie |
| `supervisor` | `password` | Accès complet (futur admin) |

*Ces identifiants sont configurables dans la variable `ELA_AUTH_DATA` du fichier `.env`.*

### Commandes Chat
- **Upload d'image** : Glissez-déposez une image pour qu'ELA l'analyse.
- **Bouton "Ma Galerie"** : Crée une conversation affichant l'historique de vos images.

---

## 🏗️ Architecture du projet

```text
ela-ai-master/
├── .files_ela/             # Stockage physique des images (Persistance Docker)
├── chroma_db/              # Base vectorielle (Embeddings des cours)
├── latex/                  # Sources .tex des cours
├── public/                 # Assets (Logos, icônes)
├── app.py                  # Application principale (Chainlit + DB + Galerie)
├── main_ela.py             # Cerveau IA (LangChain, Vision, RAG)
├── ingest.py               # Script d'ingestion des données
├── docker-compose.yml      # Orchestration Docker
├── Dockerfile              # Image système
├── requirements.txt        # Dépendances Python
└── .env                    # Secrets (Non commité)

---

## 🔧 Configuration avancée

### Modifier le modèle LLM

Dans `main_ela.py`, vous pouvez ajuster le modèle utilisé :

```python
# Modèle Vision & Texte
self.llm = ChatGroq(
    model="llama-3.2-90b-vision-preview", # ou "llama-4-scout-..."
    temperature=0.0,
    max_tokens=2048
)

```

### Stockage des fichiers

Les images uploadées sont stockées localement via la classe `LocalStorageClient` dans `app.py`. En production Docker, ce dossier est monté via un volume pour ne pas perdre les données au redémarrage.

---

## 🐛 Résolution de problèmes

### Erreur "getaddrinfo failed" (Docker vs Local)

Si vous passez du serveur à votre PC local, n'oubliez pas de changer `DATABASE_URL` dans le `.env` :

* Serveur Docker : `@db:5432`
* PC Local : `@localhost:5432`

### Erreur d'affichage des icônes (Starters)

Assurez-vous que le dossier `public` est bien monté dans le `docker-compose.yml` et videz le cache de votre navigateur.

---

## 📊 Stack technique

| Composant | Technologie |
| --- | --- |
| **Frontend/Backend** | Chainlit 2.9.6 |
| **LLM Engine** | Groq (Llama 3.2 Vision / Llama 3.3) |
| **Database** | PostgreSQL 15 + AsyncPG |
| **Vector Store** | ChromaDB |
| **Orchestration** | Docker Compose |
| **Framework IA** | LangChain |

---

## 📝 License & Contact

Ce projet est développé dans un cadre pédagogique.
© 2026 - Gilles de Truchis

**GitHub** : [github.com/varennes-ecofin](https://github.com/varennes-ecofin)
