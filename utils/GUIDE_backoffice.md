# GUIDE DE DÉVELOPPEMENT — Backoffice ELA AI

> **Version** : 2.0  
> **Date** : 2026-02-13  
> **Auteur** : Gilles de Truchis (avec assistance Claude)  
> **Statut** : En cours — Phase 1

---

## Contexte du projet

ELA AI (Econometrics Learning Assistant) est un assistant pédagogique RAG déployé sur Debian 12 avec Docker (Chainlit + PostgreSQL + ChromaDB). L'application atteint un stade POC mature et doit évoluer pour :

1. Accueillir des **beta testeurs** avec une gestion d'accès propre.
2. Permettre au superadmin de **monitorer l'activité**.
3. **Cloisonner le contenu** par niveau académique (M1/M2).
4. **Contrôler la consommation** via des quotas.

Le backoffice est une application **Streamlit** déployée comme service Docker séparé, connectée à la même base PostgreSQL qu'ELA.

---

## Architecture cible

```
docker-compose.yml
├── app       (Chainlit ELA)           → port 80
├── db        (PostgreSQL 15)          → port **** (interne)
└── admin     (Streamlit backoffice)   → port ****
```

Tous les services partagent le même réseau Docker. Le backoffice `admin` accède à `db:****` directement.

---

## Schéma DB — Table `users` enrichie

La table `users` existante (créée par Chainlit) est étendue avec de nouvelles colonnes. Les colonnes d'origine (`id`, `identifier`, `createdAt`, `metadata`) ne sont **pas modifiées** pour préserver la compatibilité Chainlit.

| Colonne | Type | Default | Description |
|---------|------|---------|-------------|
| `id` | UUID | PK | **Existante** — Chainlit |
| `identifier` | TEXT UNIQUE | — | **Existante** — Login username |
| `createdAt` | TEXT | — | **Existante** — Chainlit format |
| `metadata` | JSONB | — | **Existante** — Chainlit metadata |
| `password_hash` | TEXT | NULL | **Nouvelle** — Hash bcrypt |
| `role` | TEXT | `'student'` | **Nouvelle** — `student`, `supervisor`, `admin` |
| `level` | TEXT | `'M1'` | **Nouvelle** — `M1`, `M2`, `ALL` |
| `is_active` | BOOLEAN | TRUE | **Nouvelle** — Désactivation sans suppression |
| `daily_quota` | INTEGER | 50 | **Nouvelle** — Messages/jour (NULL = illimité) |
| `last_login` | TIMESTAMP | NULL | **Nouvelle** — Dernière connexion |

---

## Phases de développement

### Phase 1 — Migration schéma DB ✅

**Objectif** : Enrichir la table `users` sans casser Chainlit.

**Actions** :
- `ALTER TABLE users ADD COLUMN ...` pour chaque nouvelle colonne.
- Générer les hash bcrypt pour les utilisateurs existants.
- Créer une vue `v_quota_usage` pour le suivi des quotas.

**Fichiers modifiés** : Aucun (migration SQL pure via PgAdmin).

**Vérification** : L'app ELA doit continuer à fonctionner normalement après la migration (les nouvelles colonnes ont des valeurs par défaut, Chainlit les ignore).

---

### Phase 2 — Backoffice Streamlit + CRUD utilisateurs

**Objectif** : Déployer le conteneur admin et gérer les utilisateurs.

**Actions** :
- Créer le service `admin` dans `docker-compose.yml`.
- Implémenter `admin/app.py` (login admin).
- Implémenter `admin/db.py` (connexion PostgreSQL partagée).
- Implémenter `admin/pages/users.py` (CRUD complet).

**Structure fichiers** :
```
admin/
├── Dockerfile
├── requirements.txt        # streamlit, psycopg2-binary, pandas, bcrypt
├── app.py                  # Point d'entrée + authentification admin
├── db.py                   # Pool de connexion PostgreSQL
├── pages/
│   ├── users.py            # Gestion utilisateurs (CRUD, rôles, quotas)
│   └── ...
└── .streamlit/
    └── config.toml         # Configuration Streamlit
```

**Fonctionnalités `users.py`** :
- Tableau de tous les utilisateurs (st.dataframe éditable).
- Formulaire de création (identifier, password, role, level, quota).
- Modification inline (rôle, niveau, quota, is_active).
- Suppression (soft-delete via `is_active = FALSE`).
- Hash bcrypt automatique à la création/modification du mot de passe.

---

### Phase 2b — Adaptation authentification `app.py`

**Objectif** : Remplacer le login env-var par un login DB avec bcrypt.

**Actions** :
- Supprimer `load_users_from_env()` et la variable `ELA_AUTH_DATA`.
- Modifier `auth_callback()` pour requêter la table `users` (bcrypt verify).
- Stocker le profil complet (role, level, quota) dans `cl.user_session`.
- Ajouter un check quota avant chaque appel à `ela_bot.ask()`.
- Mettre à jour `last_login` à chaque connexion.

**Fichiers modifiés** : `app.py` uniquement.

**Logique du check quota** :
```
1. Récupérer daily_quota du profil en session
2. Si NULL → pas de limite (admin)
3. Sinon → COUNT steps WHERE userIdentifier = X AND type = 'user_message' AND date = aujourd'hui
4. Si count >= quota → message d'avertissement, pas d'appel LLM
```

---

### Phase 3 — Refonte ingestion multi-niveaux (`ingest.py`)

**Objectif** : Scanner récursivement `latex/` et injecter level/course dans les métadonnées ChromaDB.

**Nouvelle structure source** :
```
latex/
├── M1/
│   ├── Econometrie_Base/
│   │   └── cours_chap1.tex
│   └── Micro_Eco/
│       └── cours_micro.tex
├── M2/
│   ├── Series_Temporelles/
│   │   └── cours_var_vecm.tex
│   └── Econometrie_Financiere/
│       └── cours_garch.tex
└── Commun/
    └── Rappels_Maths/
        └── algebre.tex
```

**Métadonnées extraites** par fichier :
```python
{
    "source": "cours_chap1.tex",
    "level": "M1",                  # Premier sous-dossier
    "course": "Econometrie_Base",   # Deuxième sous-dossier
    "section": "...",               # Extrait du LaTeX
    "slide_title": "...",           # Extrait du LaTeX
    "type": "slide"
}
```

**Fichiers modifiés** : `ingest.py`.

**Points d'attention** :
- Le dossier `Commun` produit `level = "Commun"` (accessible à tous).
- Reconstruction complète de `chroma_db/` (pas d'ingestion incrémentale).

---

### Phase 4 — Retriever scopé par niveau (`main_ela.py`)

**Objectif** : Cloisonner les résultats RAG selon le profil utilisateur.

**Actions** :
- Modifier `ELA_Bot.__init__()` pour accepter un paramètre `user_level`.
- Appliquer un filtre ChromaDB `where` sur les métadonnées :
  ```python
  filter = {"$or": [{"level": user_level}, {"level": "Commun"}]}
  ```
- Pré-construire un `BM25Retriever` par niveau au démarrage (dict caché).
- Adapter `app.py` pour passer le `level` de la session à `ELA_Bot`.

**Fichiers modifiés** : `main_ela.py`, `app.py`.

**Option prompt adapté** :
Ajuster `ELA_BASE_INSTRUCTIONS` selon le niveau :
- M1 : ton pédagogique, définitions rappelées, pas de raccourcis.
- M2 : ton avancé, références croisées entre concepts, notations compactes.

---

### Phase 5 — Gestion RAG depuis le backoffice

**Objectif** : Upload de fichiers `.tex` et reconstruction de la base vectorielle depuis Streamlit.

**Actions** :
- Implémenter `admin/pages/rag_management.py`.
- Vue arborescence du dossier `latex/` (st.tree ou récursion manuelle).
- Upload `.tex` avec sélection du niveau et du cours (dropdowns).
- Bouton "Reconstruire la base" → lance `ingest.py` via subprocess.
- Affichage du log d'ingestion (nombre de chunks, durée, erreurs).

**Volume Docker** : Le dossier `latex/` doit être monté comme volume partagé entre `app` et `admin`.

**Stratégie de rebuild** :
1. Ingestion dans `chroma_db_new/`.
2. Swap atomique : `mv chroma_db chroma_db_old && mv chroma_db_new chroma_db`.
3. Les nouvelles sessions ELA chargent automatiquement la nouvelle base.
4. Nettoyage : `rm -rf chroma_db_old` après validation.

---

### Phase 6 — Monitoring d'activité

**Objectif** : Dashboard de suivi de l'utilisation.

**Actions** :
- Implémenter `admin/pages/activity.py`.
- Métriques : messages/jour par utilisateur, conversations totales, répartition quiz/chat/code.
- Graphiques temporels (Plotly via Streamlit).
- Vue consommation quotas (barres de progression par utilisateur).

**Source de données** : Tables `threads`, `steps` existantes + `users` enrichie.

**Requêtes clés** :
- Activité par jour : `COUNT(steps) GROUP BY DATE(createdAt), userIdentifier`
- Type d'activité : détection par nom du thread (`🎓 Quiz`, `💻 Code`, etc.)
- Quota restant : `daily_quota - COUNT(steps du jour)`

---

### Phase 7 — Feedbacks et conversations

**Objectif** : Qualité des réponses et debug.

**Actions** :
- `admin/pages/feedbacks.py` : liste des feedbacks (thumbs up/down) avec contexte.
- `admin/pages/conversations.py` : lecture d'une conversation complète (steps ordonnés).
- Filtrage par utilisateur, date, score feedback.

**Source de données** : Tables `feedbacks`, `steps`, `threads`.

---

## Notes techniques transversales

### Compatibilité Chainlit

Chainlit gère ses propres tables (`users`, `threads`, `steps`, `elements`, `feedbacks`). Toute modification du schéma doit :
- Uniquement **ajouter** des colonnes (jamais renommer/supprimer les existantes).
- Utiliser des `DEFAULT` pour que Chainlit puisse continuer à `INSERT` sans connaître les nouvelles colonnes.
- Ne pas modifier les types des colonnes existantes.

### Sécurité

- Les mots de passe sont hashés avec **bcrypt** (coût 12).
- Le backoffice Streamlit est protégé par login + restriction réseau (reverse proxy ou IP whitelist).
- Les secrets (DATABASE_URL, ADMIN_PASSWORD) sont dans `.streamlit/secrets.toml` ou variables d'environnement Docker.

### Variables d'environnement

Après la phase 2b, la variable `ELA_AUTH_DATA` disparaît du `.env`. Les credentials sont en DB.

Variables conservées :
```ini
GROQ_API_KEY=gsk_...
CHAINLIT_AUTH_SECRET=...
DATABASE_URL=postgresql+asyncpg://chainlit_user:securepw*****@db:****/chainlit_db
```

Variables ajoutées pour le backoffice :
```ini
ADMIN_DATABASE_URL=postgresql://chainlit_user:securepw*****@db:****/chainlit_db
ADMIN_USERNAME=superadmin
ADMIN_PASSWORD_HASH=<bcrypt hash>
```

---

## Checklist de validation par phase

| Phase | Test de non-régression |
|-------|----------------------|
| 1 | ELA démarre normalement, login fonctionne, conversations OK |
| 2 | Backoffice accessible, CRUD utilisateurs fonctionnel |
| 2b | Login ELA via DB, anciens comptes fonctionnent, quota bloquant |
| 3 | `ingest.py` produit des métadonnées level/course correctes |
| 4 | Étudiant M1 ne voit que les docs M1 + Commun |
| 5 | Upload .tex + rebuild depuis Streamlit, nouvelle base active |
| 6 | Dashboard affiche les bonnes métriques |
| 7 | Feedbacks et conversations lisibles |
