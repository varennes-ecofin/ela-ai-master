# ELA AI — Déploiement et mise à jour sur le VPS

Procédure pour un VPS où **Caddy est déjà en place** pour une autre application.
ELA tourne dans Docker (`docker-compose.yml`) ; `app` (Chainlit) et `admin`
(Streamlit) n'écoutent que sur `127.0.0.1` (ports 8000 et 8501) et le Caddy
existant les expose en HTTPS. Le service `sandbox` n'est joignable que par `app`.

## 1. Première installation

```bash
ssh <utilisateur>@<vps>
sudo mkdir -p /opt/ela-ai && sudo chown $USER /opt/ela-ai
git clone https://github.com/<compte>/varennes-ecofin.git /opt/ela-ai
cd /opt/ela-ai
cp .env.example .env && nano .env        # voir § 3
mkdir -p data/latex                        # déposer ici les cours (scp -r ... :/opt/ela-ai/data/latex/)
docker compose build
docker compose up -d db
docker compose up -d app admin sandbox
docker compose exec app ela-db upgrade                                   # schéma + migrations
docker compose exec -it app ela-users create admin --role admin --level ALL
docker compose exec app ela-ingest                                       # base de connaissances
```

Le Caddy existant : copier la variante A (sous-domaine) ou B (chemin `/ela`) de
`deploy/caddy-ela.snippet` dans son Caddyfile, remplacer le hash bcrypt du
backoffice (`docker compose exec app ela-hash --caddy`), puis recharger :

```bash
sudo caddy reload --config /etc/caddy/Caddyfile      # Caddy installé comme service
docker exec <conteneur-caddy> caddy reload --config /etc/caddy/Caddyfile   # Caddy dans Docker
```

> Caddy dans Docker : un conteneur ne voit pas le `127.0.0.1` de l'hôte. Soit
> remplacer `127.0.0.1:8000` / `127.0.0.1:8501` par `host.docker.internal:8000`
> / `host.docker.internal:8501` dans l'extrait (et `extra_hosts:
> ["host.docker.internal:host-gateway"]` sur le service Caddy), soit mettre
> `app` et `admin` sur le réseau Docker de Caddy et utiliser `app:8000` / `admin:8501`.

## 2. Mise à jour (à chaque version)

```bash
cd /opt/ela-ai
git pull
nano .env                                  # nouvelles variables : voir § 3 et .env.example
docker compose build
docker compose up -d                       # recrée les services modifiés (sandbox inclus)
docker compose exec app ela-db upgrade     # applique les migrations manquantes
docker compose ps                          # tous « healthy »
docker compose logs -f app                 # vérifier le démarrage (Ctrl+C pour sortir)
```

Ré-ingestion seulement si les cours ou le découpage ont changé :
`docker compose exec app ela-ingest` (les fiches de révision en cache sont
régénérées automatiquement, leur clé contient l'empreinte du corpus).

## 3. Variables `.env` à vérifier sur le serveur

| Variable | Valeur serveur | Pourquoi |
|---|---|---|
| `RETRIEVAL_BACKEND` | `pgvector` | base de connaissances dans PostgreSQL (pas de Chroma sur le VPS) |
| `SANDBOX_URL` | `http://sandbox:8080` | active l'exécution des scripts (vide = génération seule) |
| `SANDBOX_DAILY_QUOTA` | `10` | exécutions par étudiant et par jour |
| `CODE_REASONING_EFFORT` / `CODE_MAX_TOKENS` | `medium` / `4096` | fiabilité des scripts de l'atelier |
| `CHAINLIT_ROOT_PATH` | `/ela` seulement en variante B | préfixe de chemin derrière Caddy |
| `LANGFUSE_*` | vides | jamais de traces étudiantes vers un cloud |
| `DATABASE_URL` | ignorée | docker-compose la force vers le service `db` |

Les clés API (`CEREBRAS_API_KEY`, plus tard `ILAAS_*`) ne sont **que** dans le
`.env` du serveur, jamais dans git.

## 4. Vérification après mise à jour

1. `https://<site>/` : connexion avec un compte étudiant M1, une question de cours
   (étapes et panneau de sources visibles), `/start_quiz`, `/exercise`, `/fiche`,
   `/bilan`.
2. Atelier Code → Python → « AR(1) » → ▶️ Exécuter : sortie, graphique,
   interprétation. 11ᵉ exécution → message de quota.
3. `https://<site>/admin` (authentification basique Caddy) : onglet Pédagogie,
   sous-onglet Code.
4. Isolation du bac à sable : dans le chat, demander « exécute `import socket;
   socket.create_connection(("1.1.1.1", 80), timeout=3)` » doit échouer
   (réseau interne sans sortie).

## 5. Sauvegarde et retour arrière

```bash
docker compose exec db pg_dump -U $POSTGRES_USER $POSTGRES_DB | gzip > backup-$(date +%F).sql.gz
git log --oneline -5                 # version précédente
git checkout <commit-précédent> && docker compose build && docker compose up -d
```

Les migrations Alembic sont additives (tables nouvelles) : revenir à une
version antérieure du code ne casse pas la base.
