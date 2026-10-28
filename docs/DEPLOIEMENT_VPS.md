# ELA AI — Déploiement et mise à jour sur le VPS

Procédure pour un VPS où **Caddy est déjà en place** pour une autre application.
ELA tourne dans Docker (`docker-compose.yml`) ; `app` (Chainlit) et `backoffice`
(FastAPI) n'écoutent que sur `127.0.0.1` (ports 8000 et 8502) et le Caddy
existant les expose en HTTPS. Le service `admin` (Streamlit) a disparu au lot 5.4b (§ 9). Le service
`sandbox` n'est joignable que par `app`.

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
docker compose up -d app backoffice sandbox
docker compose exec app ela-db upgrade                                   # schéma + migrations
docker compose exec -it app ela-users create admin --role admin --level ALL
docker compose exec app ela-ingest                                       # base de connaissances
```

Le Caddy existant : copier la variante A (sous-domaine) ou B (chemin `/ela`) de
`deploy/caddy-ela.snippet` dans son Caddyfile, puis recharger :

```bash
sudo caddy reload --config /etc/caddy/Caddyfile      # Caddy installé comme service
docker exec <conteneur-caddy> caddy reload --config /etc/caddy/Caddyfile   # Caddy dans Docker
```

> Caddy dans Docker : un conteneur ne voit pas le `127.0.0.1` de l'hôte. Soit
> remplacer `127.0.0.1:8000` / `127.0.0.1:8502` par `host.docker.internal:8000`
> / `host.docker.internal:8502` dans l'extrait (et `extra_hosts:
> ["host.docker.internal:host-gateway"]` sur le service Caddy), soit mettre
> `app` et `backoffice` sur le réseau Docker de Caddy et utiliser `app:8000` / `backoffice:8080`.

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
| `BACKOFFICE_SECRET_KEY` | 48 caractères aléatoires (`python -c "import secrets; print(secrets.token_urlsafe(48))"`) | signe les sessions du nouveau backoffice ; le service ne démarre pas sans |
| `BACKOFFICE_ROOT_PATH` | `/backoffice` (`/ela/backoffice` en variante B) | préfixe d'URL du nouveau backoffice |
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
3. `https://<site>/backoffice/` : connexion avec le compte `admin` (tableau de
   bord, journal d'administration), refus d'un compte étudiant, déconnexion ;
   `https://<site>/backoffice/healthz?deep=1` renvoie `"database": "ok"` ;
   onglet Activité → Pédagogie → Code (exécutions de l'étape 2).
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

## 6. Migration depuis l'ancienne version (VPS avec Caddy dans Docker)

> Procédure historique (migration du 19–21/10/2026, version `2.0.0a11`) : elle mentionne encore
> le service `admin` (Streamlit) et son authentification basique, retirés depuis (§ 9).

Situation de départ : l'ancien ELA (`app.py`, `main_ela.py`) tourne depuis
`/home/debian/ela-ai-master` avec son propre `docker-compose.yml` (application
publiée sur le port 80 en HTTP, backoffice sur 0.0.0.0:8501) ; Caddy tourne en
conteneur (`/opt/caddy`) sur le réseau Docker `proxy` et ne publie que le 443.

Cible : Caddy publie 80 et 443 et sert `https://ela.<domaine>` (application) et
`https://ela.<domaine>/admin` (backoffice, authentification basique) en joignant
`app:8000` et `admin:8501` sur le réseau `proxy`. La base PostgreSQL existante
(volume `ela-ai-master_postgres_data`) est conservée.

Séquence (chaque étape indique ce qu'elle doit afficher) :

1. **Sauvegarde** de la base, ancienne pile encore en marche :
   `docker compose exec db pg_dump -U $POSTGRES_USER $POSTGRES_DB | gzip > ~/ela-backup-$(date +%F).sql.gz`
   (variables lues avec `set -a; . ./.env; set +a` au préalable) — fichier de quelques centaines de Ko.
2. **Arrêt** de l'ancienne pile : `docker compose down` (sans `-v` : les volumes restent). `docker ps` ne montre plus que `caddy` et `echecs`.
3. **Code** : `git fetch origin && git reset --hard origin/main` ; `ls` montre `apps/ src/ pyproject.toml docker-compose.yml deploy/` et plus `app.py`.
4. **Surcouche compose** : `cp deploy/docker-compose.override.vps.yml docker-compose.override.yml`.
5. **Cours** : la nouvelle ingestion lit `data/latex/{M1|M2|Commun}/<Cours>/*.tex`. Vérifier `find data -maxdepth 3 -type d` et réorganiser si besoin (`mkdir -p data/latex/M1 data/latex/M2 data/latex/Commun`, puis déplacer chaque dossier de cours dans son niveau).
6. **`.env`** : repartir de `.env.example` (`cp .env .env.ancien && cp .env.example .env`) et reporter depuis l'ancien : `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` (identiques, sinon la base ne s'ouvre plus), `CHAINLIT_AUTH_SECRET`, `CEREBRAS_API_KEY`. Régler `ELA_ENV=prod`, `RETRIEVAL_BACKEND=pgvector`, `EMBEDDING_PROVIDER=local`, `SANDBOX_URL=http://sandbox:8080`, `LOG_JSON` est forcé par compose. Laisser `DATABASE_URL` et `ADMIN_DATABASE_URL` vides ou tels quels : compose les écrase vers le service `db`.
7. **Construction et démarrage** : `docker compose build` (5–10 min la première fois) puis `docker compose up -d db` ; `docker compose ps` : `db` healthy.
8. **Migrations** : `docker compose run --rm app ela-db upgrade` — liste `0001_baseline` → `0008_code_runs`.
9. **Démarrage** : `docker compose up -d` ; `docker compose ps` : `app`, `admin`, `sandbox`, `db` ; `docker compose logs -f app` jusqu'à « Your app is available at http://0.0.0.0:8000 » (le premier démarrage télécharge bge-m3, ≈ 2,2 Go, dans `data/hf-cache`).
10. **Ingestion** : `docker compose exec app ela-ingest` — nombre de fichiers, de chunks, par niveau.
11. **Caddy** (`/opt/caddy`) : ajouter `DOMAINE_ELA=ela.<domaine>` au `.env` de Caddy ; coller `deploy/caddy-ela.docker.snippet` à la fin du `Caddyfile` avec le hash de `docker compose exec app ela-hash --caddy` ; dans son `docker-compose.yml`, ajouter `"80:80"` aux ports de Caddy et retirer `disable_http_challenge` du bloc échecs (le port 80 est libre maintenant) ; `docker compose up -d` dans `/opt/caddy` puis `docker logs caddy --tail 20` : certificat obtenu pour le sous-domaine (le DNS doit déjà répondre : `dig +short ela.<domaine>`).
12. **Vérifications** du § 4, puis rotation des mots de passe (§ 3 du README : `ela-users create …` réécrit un compte).

## 7. Ajout du nouveau backoffice (phase 5, lot 5.1) sur le VPS

1. `git pull` puis, dans le `.env` du serveur, ajouter `BACKOFFICE_SECRET_KEY=<48 caractères aléatoires>`
   (laisser `BACKOFFICE_ROOT_PATH=/backoffice`, valeur par défaut).
2. `cp deploy/docker-compose.override.vps.yml docker-compose.override.yml` (le fichier ajoute
   `backoffice` au réseau `proxy`).
3. `docker compose build backoffice && docker compose up -d backoffice` ; `docker compose exec app ela-db upgrade`
   (migration `0009_admin_audit`) ; `docker compose ps` : `ela-backoffice` healthy.
4. Caddy (`/opt/caddy`) : dans le bloc `{$DOMAINE_ELA}` du `Caddyfile`, ajouter avant le `handle` final
   le bloc `@backoffice` de `deploy/caddy-ela.docker.snippet` (`reverse_proxy backoffice:8080`, sans
   `basic_auth`), puis `docker exec caddy caddy reload --config /etc/caddy/Caddyfile`.
5. Vérification § 4, point 3.

## 8. Lots 5.2 et 5.3 (comptes, cours, ingestion depuis le backoffice)

1. `git pull`, `docker compose build backoffice && docker compose up -d backoffice`, puis
   `docker compose exec app ela-db upgrade` (migration `0010_jobs`).
2. `https://<site>/backoffice/users` : comptes et import CSV ; `https://<site>/backoffice/courses` :
   cours, dépôt de fichiers, ingestion (Vérifier / Mise à jour incrémentale / Reconstruction complète).
   L'ingestion tourne dans un processus enfant du conteneur `backoffice` ; `./data` (cours, cache
   `hf-cache` du modèle d'embeddings) est partagé avec `app`, donc aucun second téléchargement du modèle.
3. Le chat voit les nouveaux chunks à la question suivante (store partagé) ; les fiches de révision en
   cache sont régénérées automatiquement (empreinte du corpus). `docker compose exec app ela-ingest`
   reste utilisable en ligne de commande.

Retour arrière : `docker compose down` dans `/home/debian/ela-ai-master`, `git checkout <ancien commit>` (ou le `.env.ancien` + `docker-compose.yml.backup`), `docker compose up -d`. La base n'est pas modifiée de façon destructive par les migrations (tables ajoutées seulement).

## 9. Lot 5.4 : backoffice à parité, retrait de Streamlit

Le lot 5.4a (`2.0.0a15`) porte les dernières pages de Streamlit (activité, pédagogie,
feedbacks, conversations) dans le backoffice FastAPI ; le lot 5.4b (`2.0.0a16`) supprime
`apps/admin/`, le service `admin` du compose, l'extra `admin` du paquet et les blocs
`/admin` de Caddy (plus d'authentification basique). Sur le VPS, dans l'ordre :

1. `git pull` (les deux lots) puis, **avant toute commande compose**, remplacer la surcouche :
   `cp deploy/docker-compose.override.vps.yml docker-compose.override.yml`. L'ancienne copie déclare
   encore `admin:` ; comme `docker-compose.yml` ne définit plus ce service, compose refuserait tout le
   projet (« service "admin" has neither an image nor a build context specified »).
   `docker compose config --services` doit lister `app backoffice caddy db sandbox`.
2. `docker compose build backoffice` puis `docker compose up -d --remove-orphans` : `ela-backoffice`
   est reconstruit, le conteneur `ela-admin` (orphelin) est arrêté et supprimé ; `docker compose ps`
   ne liste plus `admin`. Aucune migration dans ces lots (`ela-db upgrade` reste sans effet ; si le
   backoffice n'avait jamais tourné sur le serveur, appliquer d'abord le § 7 : `BACKOFFICE_SECRET_KEY`
   et migrations `0009` / `0010`).
3. Caddy (`/opt/caddy`) : dans le bloc `{$DOMAINE_ELA}` du `Caddyfile`, supprimer le bloc `@admin`
   (avec son `basic_auth` et `reverse_proxy admin:8501`), garder `@backoffice` et le `handle` final —
   le résultat doit être identique à `deploy/caddy-ela.docker.snippet` ; puis
   `docker exec caddy caddy reload --config /etc/caddy/Caddyfile`. `https://<site>/admin` renvoie
   désormais sur Chainlit (404 côté application) : c'est attendu.
4. `.env` du serveur : les lignes `ADMIN_BASIC_AUTH_USER` / `ADMIN_BASIC_AUTH_HASH` peuvent être
   supprimées (ignorées sinon). `ADMIN_DATABASE_URL` reste forcée par compose pour les outils en
   ligne de commande.
5. Nettoyage : `docker image prune -f` (image `ela-ai-master-admin` devenue inutile, ≈ 2 Go avec torch).
6. Vérification § 4, point 3, plus `https://<site>/backoffice/activity`, `/pedagogy`, `/feedbacks`,
   `/conversations` avec le compte `admin` puis `supervisor` (lecture seule) ; un export CSV s'ouvre
   dans Excel avec les accents.

Retour arrière : `git checkout 2.0.0a15` (ou le commit du lot 5.4a) remet `apps/admin/` et le service
`admin` ; `docker compose up -d --build admin` puis rétablir le bloc `@admin` dans Caddy.

## 10. Lot 5.5a : évaluation depuis le backoffice

Le lot 5.5a (`2.0.0a17`) ajoute l'onglet **Évaluation** : `ela-eval retrieval` / `answer` lancés
en tâche de fond par le conteneur `backoffice`, rapports JSON dans `eval/reports/`, comparaison.
Aucune migration. Le service `backoffice` monte désormais `./eval` (jeu de questions et rapports)
et rejoint `sandbox_net` (préparation de `ela-eval code`, lot 5.5b). Sur le VPS, dans l'ordre :

1. `git pull` puis, **avant toute commande compose**, remplacer la surcouche (elle ajoute
   `sandbox_net` au service `backoffice`) :
   `cp deploy/docker-compose.override.vps.yml docker-compose.override.yml` ;
   `docker compose config --services` doit toujours lister `app backoffice caddy db sandbox`.
2. Le dossier `eval/` doit être lisible et `eval/reports/` **inscriptible par l'uid 1000**
   (utilisateur `ela` du conteneur). Vérifier :
   ```bash
   id -u                      # attendu : 1000 (utilisateur debian)
   mkdir -p eval/reports && ls -ln eval   # les lignes doivent montrer 1000 1000
   ```
   Si l'uid affiché n'est pas 1000 : `sudo chown -R 1000:1000 eval`. (`eval/questions.yaml` est
   versionné, `eval/reports/*.json` ne l'est pas ; les rapports contiennent les questions du jeu et
   les sorties du modèle, jamais de données étudiantes.)
3. `docker compose build backoffice && docker compose up -d backoffice` : attendu
   `Container ela-backoffice  Started` puis, après ~30 s, `docker compose ps` → `ela-backoffice`
   `healthy`. Le conteneur doit voir les deux réseaux :
   `docker inspect ela-backoffice --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}'`
   → `ela-ai-master_default ela-ai-master_sandbox_net proxy`.
4. `.env` du serveur (facultatif) : `EVAL_JUDGE_MODEL` / `EVAL_JUDGE_PROVIDER` fixent le juge
   (vides = modèle et fournisseur du chat). Les clés `CEREBRAS_API_KEY`, `ILAAS_API_KEY` (+
   `ILAAS_CHAT_BASE_URL`), `GROQ_API_KEY` présentes déterminent les fournisseurs proposés dans le
   formulaire. Après modification : `docker compose up -d backoffice` (relecture du `.env`).
5. Vérification : `https://<site>/backoffice/evaluation` avec le compte `admin` → formulaire visible,
   « Juge fixe : … » affiché, fournisseurs proposés ; lancer **Évaluation du retrieval**
   (étiquette `vps-retrieval`, sans appel LLM, ~1 min) : la page du job montre `[i/71] …` puis
   « réussi » avec le lien du rapport ; `ls -l eval/reports/` sur le serveur liste le fichier
   (propriétaire 1000). Lancer ensuite **Évaluation des réponses** avec la limite 20 et le juge
   (≈ 40 appels LLM) ; comparer avec un rapport précédent (`Comparer`). Avec le compte
   `supervisor` : rapports lisibles, pas de formulaire ni de bouton Supprimer.
6. En ligne de commande sur le serveur, utiliser le conteneur `backoffice`, qui monte `./eval` (le
   service `app` ne le monte pas) : `docker compose exec backoffice ela-eval retrieval --label cli`
   écrit dans le même `eval/reports/` que le backoffice, et le rapport apparaît dans l'historique.

Retour arrière : `git checkout 2.0.0a16`, `cp deploy/docker-compose.override.vps.yml
docker-compose.override.yml`, `docker compose build backoffice && docker compose up -d backoffice`.
Les rapports déjà écrits restent dans `eval/reports/`.
