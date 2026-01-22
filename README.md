# ELA AI - Econometrics Learning Assistant 🎓

![ELA AI Banner](ela.png)

## 📖 Description

ELA AI est un assistant d'apprentissage intelligent spécialisé en économétrie, développé avec Chainlit et LangChain. Il utilise une architecture RAG (Retrieval-Augmented Generation) avancée pour répondre aux questions des étudiants en se basant exclusivement sur des supports de cours au format LaTeX.

### ✨ Fonctionnalités principales

- 🔍 **Recherche Hybride** : Combine BM25 (recherche par mots-clés) et recherche vectorielle sémantique
- ⚡ **Reranking intelligent** : Utilise FlashRank pour optimiser la pertinence des résultats
- 🎯 **Réponses sourcées** : Chaque réponse cite précisément les slides et fichiers sources
- 🔐 **Authentification** : Système de connexion sécurisé pour étudiants et professeurs
- 📐 **Support LaTeX** : Affichage natif des formules mathématiques
- 🇫🇷 **Multilingue** : Optimisé pour le français et l'anglais technique

---

## 🚀 Installation

### Prérequis

- Python 3.9+
- pip
- Git

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/varennes-ecofin/ela-ai-master.git
cd ela-ai-master
```

2. **Créer un environnement virtuel**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OU
.venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**

Créez un fichier `.env` à la racine du projet :
```env
GROQ_API_KEY=votre_clé_api_groq
```

Pour obtenir une clé API Groq gratuite : [https://console.groq.com](https://console.groq.com)

---

## 📚 Configuration de la base de connaissances

### 1. Préparer vos fichiers LaTeX

Placez vos fichiers `.tex` dans le dossier `./latex/` :
```bash
mkdir latex
cp /chemin/vers/vos/cours/*.tex ./latex/
```

### 2. Ingérer les documents

Lancez le script d'ingestion pour créer la base vectorielle :
```bash
python ingest.py
```

Ce script va :
- Parser vos fichiers LaTeX (frames, sections, contenu)
- Nettoyer le balisage LaTeX
- Générer les embeddings multilingues
- Stocker les vecteurs dans ChromaDB (`./chroma_db/`)

**⏱️ Temps estimé** : 2-5 minutes selon le nombre de fichiers

---

## 🎮 Utilisation

### Lancer l'application Chainlit

```bash
chainlit run app.py -w
```

L'interface sera accessible à : **http://localhost:8000**

### Identifiants par défaut

| Utilisateur | Mot de passe |
|-------------|--------------|
| `etudiant` | `*********` |
| `professeur` | `*********` |

### Mode CLI (optionnel)

Pour tester sans interface web :
```bash
python main_ela.py
```

---

## 🏗️ Architecture du projet

```
ela-ai-master/
├── app.py                  # Application Chainlit (interface web)
├── main_ela.py            # Logique RAG core
├── ingest.py              # Pipeline d'ingestion LaTeX
├── requirements.txt       # Dépendances Python
├── config.toml           # Configuration Chainlit
├── chainlit.md           # Page d'accueil
├── .env                  # Variables d'environnement (à créer)
├── .gitignore            # Fichiers à exclure de Git
├── latex/                # Dossier des fichiers .tex sources
├── chroma_db/            # Base vectorielle (générée)
└── .chainlit/            # Cache Chainlit (généré)
```

---

## 🔧 Configuration avancée

### Modifier le modèle LLM

Dans `main_ela.py`, ligne 95 :
```python
self.llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Changez ici
    temperature=0.0,
    max_tokens=2048
)
```

Modèles Groq disponibles : `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`

### Ajuster le nombre de résultats

Dans `main_ela.py`, ligne 124 :
```python
compressor = FlashRankCompressor(top_n=5)  # Augmentez pour plus de contexte
```

### Personnaliser le prompt système

Éditez le template dans `main_ela.py`, méthode `_build_chain()`, ligne 139.

---

## 🐛 Résolution de problèmes

### Erreur "GROQ_API_KEY non définie"
Vérifiez que votre fichier `.env` existe et contient la clé.

### Erreur "Le dossier 'chroma_db' n'existe pas"
Lancez d'abord `python ingest.py` pour créer la base.

### FlashRank pas disponible
```bash
pip install flashrank
```

### Problèmes d'encodage LaTeX
Assurez-vous que vos fichiers `.tex` sont en UTF-8.

---

## 📊 Stack technique

| Composant | Technologie |
|-----------|-------------|
| **Framework web** | Chainlit 2.9.6 |
| **LLM** | Groq (Llama 3.3 70B) |
| **Embeddings** | HuggingFace Multilingual MPNet |
| **Vector DB** | ChromaDB |
| **Retrievers** | BM25 + Semantic (Ensemble) |
| **Reranker** | FlashRank (ms-marco-MiniLM) |
| **Orchestration** | LangChain |

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout fonctionnalité X'`)
4. Pushez vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

---

## 📝 License

Ce projet est développé dans un cadre pédagogique.  
© 2026 - Gilles de Truchis

---

## 📧 Contact

Pour toute question ou suggestion :
- **Auteur** : Gilles de Truchis
- **GitHub** : [github.com/varennes-ecofin](https://github.com/varennes-ecofin)

---

## 🙏 Remerciements

- LangChain pour l'infrastructure RAG
- Groq pour l'accès gratuit aux LLMs
- Chainlit pour l'interface conversationnelle
- La communauté HuggingFace pour les modèles d'embeddings
