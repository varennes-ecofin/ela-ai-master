# Changelog — ELA AI

Toutes les évolutions notables du projet. Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/).
Le plan de modernisation complet est dans `PLAN_MODERNISATION.md`.

## [2.0.0a17] — 2026-10-28 — Phase 5, lot 5.5a : évaluation depuis le backoffice

### Ajouté
- **Backoffice → Évaluation** (onglet activé) : **lancement** d'une évaluation `retrieval` (hit@k, MRR, sans appel LLM) ou `answer` (réponses de l'agent : latence, recherches, citations, fidélité par juge LLM) en **tâche de fond** — formulaire réservé à `admin` (étiquette, cours, type de question, limite — 20 par défaut, vide = toutes —, `k`, juge activé par défaut) avec estimation du nombre d'appels LLM (`2 × N` avec juge, `N` sans) et confirmation avant lancement ; le job réutilise `JobRunner` / table `jobs` (`kind = eval`, un seul job à la fois, ingestion ou évaluation), processus enfant `python -I -u -X utf8 -m ela.eval.runner answer|retrieval … --result`, progression `[i/n]` en direct (fragment HTMX 2 s), page du job avec le rapport et ses métriques clés ; audit `eval.start`.
- **Choix du modèle évalué** (fournisseur + modèle, champ libre avec suggestions) passé au processus enfant par `LLM_MODEL_CHAT` / `LLM_PROVIDER` : seuls les fournisseurs dont la clé est présente sur le serveur sont proposés ; le **juge reste fixe** (`EVAL_JUDGE_MODEL` / nouveau `EVAL_JUDGE_PROVIDER`, résolus depuis les réglages du serveur) et le modèle de secours est désactivé pendant l'évaluation d'un autre modèle ; modèle et fournisseur enregistrés dans le rapport (`model`, `provider`, `judge_model`, `judge_provider`, `prompt_version`, `kind`, aussi pour `retrieval`).
- **Historique des rapports** `eval/reports/*.json` (source unique, partagée avec la ligne de commande) : date, étiquette, type, modèle, questions, métriques clés, nombre de questions à regarder, filtre par type ; **page d'un rapport** (métriques, tableau par cours, tableau par question avec les échecs en tête — non trouvée, erreur, citation fantôme, fidélité < 0,5, refus indu — et le détail dépliable : question, réponse, sources, verdict du juge point par point), export CSV du tableau, téléchargement du JSON, suppression (`admin`, audit `eval.delete`) ; `supervisor` en lecture seule.
- **Comparaison** de deux rapports de même type (`/backoffice/evaluation/compare`) : tableau avant / après avec delta coloré selon le sens de la métrique, questions gagnées / perdues (`retrieval`), fidélité en hausse / en baisse (`answer`), questions présentes dans un seul rapport ; même logique en ligne de commande (`ela.eval.compare`, `ela-eval compare`).
- `ela.eval.reports` (index des rapports, noms de fichiers validés, cache par mtime, fichiers illisibles signalés) ; `ela.eval.compare` ; `ela-eval retrieval` accepte `--type`, `--limit` et `--result` ; `ela-eval answer` accepte `--result` ; réglages `EVAL_JUDGE_PROVIDER`, `EVAL_DATASET`, `EVAL_REPORTS_DIR`.
- `docker-compose.yml` : volume `./eval:/app/eval` (jeu de questions et rapports) et réseau `sandbox_net` sur `backoffice` (préparation de `ela-eval code`) ; `eval/reports/.gitkeep` ; `docs/DEPLOIEMENT_VPS.md` § 10.
- Tests : comparaison et index (8), runner `ela-eval` sans modèle (8), jobs d'évaluation (3), pages Évaluation hors ligne (10), job `eval` sur PostgreSQL (1).

### Supprimé
- `ela-eval answer --engine legacy` et `src/ela/eval/legacy_pipeline.py` (chaîne fixe de la Phase 2, gelée pour la mesure « avant » faite le 18/10), `LegacyEngineRunner`, les constantes de prompt qui ne servaient qu'à elle ; `verify_legacy_citations` (format `[Source: fichier]`) reste dans `ela.agent.citations`.

### Modifié
- Messages « Une ingestion est déjà en cours » → « Un job est déjà en cours (ingestion ou évaluation) » ; historique des jobs commun aux ingestions et aux évaluations.

## [2.0.0a16] — 2026-10-27 — Phase 5, lot 5.4b : retrait du backoffice Streamlit

### Supprimé
- `apps/admin/` (application Streamlit, vues, Dockerfile, configuration) : le backoffice FastAPI est à parité depuis le lot 5.4a.
- Extra `admin` de `pyproject.toml` (`streamlit`, `pandas`, `plotly`) et ses exceptions ruff ; installation : `pip install -e ".[backoffice,dev]"`.
- Service `admin` de `docker-compose.yml` (port 8501), variables `ADMIN_BASIC_AUTH_USER` / `ADMIN_BASIC_AUTH_HASH` et blocs `/admin` avec `basic_auth` du `Caddyfile`, de `deploy/caddy-ela.snippet`, de `deploy/caddy-ela.docker.snippet` et de `deploy/docker-compose.override.vps.yml` ; `ela-hash --caddy` (le backoffice authentifie lui-même ses utilisateurs).

### Modifié
- CI : installation `.[dev,backoffice]` ; README, `.env.example` et `docs/DEPLOIEMENT_VPS.md` (§ 9 : procédure de retrait sur le VPS, `docker compose up -d --remove-orphans`, bloc Caddy, nettoyage des images). `ADMIN_DATABASE_URL` reste une option des outils en ligne de commande.

## [2.0.0a15] — 2026-10-27 — Phase 5, lot 5.4a : activité, pédagogie, feedbacks, conversations

### Ajouté
- **Backoffice → Activité** (onglet « Activité » de la navigation, sous-navigation Activité / Pédagogie / Feedbacks / Conversations, lecture seule pour `admin` et `supervisor`) : indicateurs (comptes actifs, conversations, messages du jour et de la veille, messages de la période), **messages par jour**, **conversations par type** (anneau + aires empilées par jour), **classement des utilisateurs** (barres + tableau, liens vers la fiche et les conversations), **consommation des quotas du jour** (barres de progression, « quota atteint »). Filtre utilisateur sur les graphiques quotidiens.
- **Filtre de période** commun aux quatre pages : 30 derniers jours par défaut (J−30 → J), raccourcis 7 / 30 / 90 jours, dates libres, portées dans l'URL (`?from=&to=`) ; paramètres invalides (début après fin, plus de trois ans) signalés et remplacés par la période par défaut. Les jours sont ceux des horodatages en base (UTC), comme pour les quotas et l'accueil.
- **Types de conversation** déduits du nom du fil (`ela.db.threads.classify_thread` / `THREAD_KIND_SQL`) : chat, quiz (`🎓`), exercice (`✏️`), fiche (`📄`), code (`💻`).
- **Pédagogie** : cinq onglets — Quiz (indicateurs, réussite par sujet, questions les plus ratées), Exercices (par sujet), Questions (par cours et chapitre, dernières questions avec filtre cours et lien vers le fil), Hors corpus (refusées, connaissances générales, sans extrait cité), Code (par statut et origine, erreurs fréquentes, par étudiant).
- **Feedbacks** : statistiques globales (total, positifs, négatifs, taux), liste filtrée (période, utilisateur, score) paginée par 50 avec la réponse notée, le commentaire et, à la demande (fragment HTMX), l'échange autour de la réponse (3 messages avant, 1 après).
- **Conversations** : liste filtrée (période, utilisateur, type) paginée par 50 avec nombre de messages et de feedbacks ; **lecteur** d'un fil en bulles (texte brut, retours à la ligne conservés, LaTeX tel quel), feedback attaché aux réponses, erreurs signalées, **étapes d'outils** de l'agent (recherches, exécutions) affichables (`?tools=1`).
- **Export CSV** de chaque tableau avec les filtres de la page (`;`, UTF-8 avec BOM, dates ISO, booléens oui/non) : activité par jour / par utilisateur / quotas, les neuf tableaux de la pédagogie, feedbacks, conversations (métadonnées, pas les messages).
- **Graphiques SVG générés côté serveur** (`apps/backoffice/charts.py` : courbe, aires empilées, anneau, barres horizontales ; couleurs par classes CSS, thème clair/sombre, infobulles natives), sans bibliothèque JavaScript.
- Liens croisés : les dernières conversations de l'accueil ouvrent le lecteur ; la fiche d'un compte mène à ses conversations, son activité et ses feedbacks.
- `ela.db.stats`, `ela.db.pedagogy`, `ela.db.feedbacks`, `ela.db.conversations` (requêtes async réutilisables, jamais d'exception vers la vue) ; `apps/backoffice/periods.py`, `paging.py`, `csvexport.py` ; neuf icônes Lucide ajoutées au sprite.
- Tests : pages, filtres, pagination, contexte, CSV, périodes, graphiques et classification hors ligne (15, `tests/test_backoffice_activity.py`) ; requêtes des quatre modules et pages réelles sur PostgreSQL (3).

### Modifié
- `ela.db.threads` porte la classification des conversations partagée par les statistiques et le lecteur.

## [2.0.0a14] — 2026-10-26 — Phase 5, lot 5.3 : cours et ingestion

### Ajouté
- **Backoffice → Cours** : arborescence `data/latex/{M1|M2|Commun}/<Cours>/` avec, par cours, fichiers `.tex`, taille, chunks en base, dernière indexation et état (indexé / à ingérer / vide) ; indicateurs de la base de connaissances (chunks par niveau, backend) ; **diagnostic disque vs base** (cours à ingérer, chunks orphelins d'un cours supprimé ou déplacé, dossiers vides) ; création d'un cours.
- **Page d'un cours** : liste des fichiers (téléchargement, suppression), **dépôt** de fichiers `.tex`, `.sty`, `.cls`, `.bib` ou d'une archive `.zip` (sous-dossiers conservés, dossier englobant retiré, fichiers parasites et chemins sortants ignorés, 20 Mo max), statistiques du cours en base (par type et par document), **déplacement vers un autre niveau** (le dossier est renommé puis une mise à jour incrémentale ré-indexe le cours ; refusé pendant une ingestion), suppression du cours.
- **Ingestion en tâche de fond** : boutons Vérifier / Mise à jour incrémentale / Reconstruction complète (confirmation) ; un seul job à la fois ; table `jobs` (migration `0010` : type, mode, cible, état, lancé par, début/fin, progression, journal, résultat JSON) ; l'ingestion tourne dans un **processus enfant** (`python -I -u -m ela.ingestion.pipeline --result …`, le modèle d'embeddings n'occupe la mémoire que pendant le job) dont la sortie est copiée en base au fil de l'eau ; **progression en direct** (barre et dernière ligne, fragment HTMX toutes les 2 s) ; page de chaque job avec résultat (analysés, vectorisés, inchangés, supprimés, total) et journal complet ; **historique des jobs** ; jobs laissés « en cours » par un redémarrage marqués interrompus.
- **Annales** : racine `data/annales/` (`ANNALES_SOURCE_DIR`) avec la même arborescence, dépôt et gestion des fichiers possibles, pas encore d'indexation (lot « Annales » à venir).
- `ela-ingest --result fichier.json` écrit le résultat de l'ingestion ; `stats()` sur les deux stores (chunks par niveau, cours, document, type) ; `ela.courses.catalog`, `ela.courses.state`, `ela.db.jobs`, `ela.jobs.runner`.
- `supervisor` : arborescence, état, jobs en lecture seule. Audit : `course.create`, `course.upload`, `course.delete_file`, `course.delete`, `course.move`, `ingest.start`.
- Tests : catalogue et diagnostic (6), exécuteur de jobs avec un processus enfant simulé (5), pages cours/jobs hors ligne (7), table `jobs` et état de la base sur PostgreSQL avec un corpus ingéré (2).

## [2.0.0a13] — 2026-10-22 — Phase 5, lot 5.2 : utilisateurs et quotas

### Ajouté
- **Backoffice → Utilisateurs** : tableau filtrable (recherche, rôle, niveau, état ; fragment HTMX) avec quota, consommation du jour (« quota atteint »), état du mot de passe et dernière connexion ; fiche de compte (profil, résumé d'activité : conversations, questions, quiz, exercices, exécutions de code) ; création (mot de passe saisi ou **généré et affiché une seule fois**) ; modification du rôle, du niveau et du quota ; réinitialisation du mot de passe ; désactivation / réactivation ; **suppression** réservée aux comptes sans aucune activité (confirmation). Garde-fous : impossible de changer son propre rôle, de désactiver ou supprimer son propre compte, de rétrograder le dernier administrateur actif. `supervisor` : lecture seule, sans quotas ni mots de passe.
- **Import CSV** (`/backoffice/users/import`) : dépôt du fichier (`;` ou `,`, UTF-8 avec ou sans BOM, 2 Mo max), niveau et quota par défaut, option « mettre à jour les comptes existants » (cellules vides = valeurs actuelles conservées) ; **prévisualisation** ligne par ligne (création / mise à jour / ignoré / erreur, identifiant normalisé — une adresse `@etu.univ-orleans.fr` est tronquée avant le `@`) puis confirmation ; mots de passe générés pour les lignes sans mot de passe, **remis une seule fois** (liste à l'écran + lien de téléchargement CSV à usage unique, 15 min, jamais stockés). Même moteur en ligne de commande : `ela-users import fichier.csv [--dry-run] [--level M1] [--quota 50] [--update-existing] [--credentials mdp.csv]`.
- **Mon mot de passe** (`/backoffice/account/password`) pour les comptes `admin` et `supervisor` (ancien mot de passe exigé).
- **Politique de mots de passe** (`ela.auth.passwords`) : 8 caractères minimum, un chiffre et un caractère spécial — appliquée au backoffice, à l'import et à `ela-users create` / `set-password` ; génération de mots de passe lisibles (sans caractères ambigus).
- `ela.db.users` : `Account`, `list_accounts`, `get_account`, `create_account`, `update_account`, `set_password_hash`, `count_admins`, `activity_summary`, `purge_account`, normalisation et validation des identifiants (`^[a-z0-9][a-z0-9._-]{1,63}$`) ; `ela.accounts.importer` (`plan_import` / `apply_import`).
- Audit : `user.create`, `user.update`, `user.password_reset`, `user.activate`, `user.deactivate`, `user.purge`, `users.import`, `account.password` (jamais de mot de passe dans les détails).
- Tests : politique et génération (3), importateur (6), pages utilisateurs hors ligne (13), administration et import réels sur PostgreSQL + CLI (3).

### Modifié
- Jeton HTMX `X-CSRF-Token` honoré sur toutes les requêtes ; `OneShotStore` en mémoire (jetons à usage unique, 15 min) pour les prévisualisations d'import et les mots de passe générés.

## [2.0.0a12] — 2026-10-21 — Phase 5, lot 5.1 : socle du nouveau backoffice (FastAPI + Jinja2 + HTMX)

### Ajouté
- **Nouveau backoffice** `apps/backoffice/` (FastAPI, Jinja2, HTMX vendu dans `static/`, CSS léger clair/sombre), construit à côté de Streamlit et servi sous `BACKOFFICE_ROOT_PATH` (`/backoffice`) en local comme derrière Caddy. Lancement : `python apps/backoffice/main.py --reload` ; Docker : service `backoffice` (`127.0.0.1:8502`, `uvicorn --factory`), extraits Caddy (`/backoffice*`, sans authentification basique) et surcouche VPS mis à jour.
- **Connexion et sessions** : formulaire de connexion sur les comptes `users` (`admin` : tout ; `supervisor` : consultation ; étudiants refusés), cookie de session signé (`itsdangerous`, `BACKOFFICE_SECRET_KEY`, `BACKOFFICE_SESSION_HOURS`, `HttpOnly`/`SameSite=Lax`/`Secure` en prod), jeton CSRF (cookie signé + champ ou en-tête `X-CSRF-Token`, aussi sur le formulaire de connexion), verrouillage d'un identifiant après `BACKOFFICE_LOGIN_MAX_FAILURES` échecs pendant `BACKOFFICE_LOGIN_LOCK_MINUTES`, compte revérifié à chaque requête (désactivation ou changement de rôle immédiats), redirections `next` limitées à l'application, en-têtes de sécurité et `Cache-Control: no-store`.
- **Abstraction d'authentification** `ela.auth` (`AuthProvider`, `Credentials`, `Identity`, `ProvisioningPolicy`, `AUTH_PROVIDER=local`) : le fournisseur local enveloppe `ela.db.users.authenticate` ; le CAS de l'université (création automatique des comptes étudiants au premier login) s'y branchera sans toucher aux vues.
- **Journal d'administration** : table `admin_audit` (migration `0009`, `ela.db.audit`) — `login.ok`, `login.failed`, `login.locked`, `logout` pour l'instant, affiché sur l'accueil pour les `admin` (heure locale `BACKOFFICE_TIMEZONE`).
- **Accueil** : indicateurs (comptes actifs, étudiants, étudiants actifs sur 7 jours, conversations, messages du jour et de la veille, chunks indexés) et dernières conversations, rafraîchis par HTMX toutes les 60 s (`ela.db.dashboard`) ; sonde `/backoffice/healthz` (`?deep=1` vérifie la base).
- **Interface** : icônes vectorielles Lucide (sprite SVG inclus dans le gabarit, `templates/_icons.html`, 40 icônes, licence ISC) via la macro Jinja `icon()`, plus aucun émoji ; marque ELA, barre de navigation (libellés repliés en icônes sous 1060 px), badges de rôle (« Administrateur », « Enseignant »), cartes d'indicateurs, tableaux, états vides et messages homogènes, thème clair/sombre ; logo ELA (`public/logo_light.png` / `logo_dark.png`) en marque et sur la page de connexion, `public/favicon.png` en icône de site (dossier `public/` servi sous `/backoffice/brand/`, copié dans l'image Docker) ; macros partagées (`_macros.html` : `icon`, `badge`, `page_head`, `empty`) pour les lots suivants.
- Extra `pip install -e ".[backoffice]"` (fastapi, uvicorn, jinja2, python-multipart, itsdangerous — versions déjà imposées par Chainlit) ; `httpx` en dev.
- Tests : 21 tests hors ligne (`tests/test_backoffice.py` : fabrique, préfixe, sonde, redirections, CSRF, échecs et verrouillage, rôles, cookie altéré ou étranger, compte désactivé/rétrogradé, déconnexion, primitives de session, fournisseur local) et 3 tests PostgreSQL (`tests/test_backoffice_db.py` : migration, audit, tableau de bord, connexion réelle bcrypt).

### Modifié
- `tests/conftest.py` purge aussi `AUTH_*` et `BACKOFFICE_*` ; `.env.example`, README et `docs/DEPLOIEMENT_VPS.md` documentent le nouveau service.

### Corrigé
- Tests hermétiques : `ela.config` ne fixait plus le chemin du `.env` à l'import du module mais à l'appel de `get_settings()` (`Settings(_env_file=find_env_file())`). Un module de test important `ela.config` au chargement (`test_sandbox.py`) faisait lire le `.env` réel du poste (`SANDBOX_URL` défini → trois tests de l'atelier en échec).
- `test_runner_enforces_memory_limit` séparé et ignoré sous Windows (les limites mémoire du bac à sable reposent sur `resource.setrlimit`, POSIX seulement).

## [Unreleased] — Consolidation

### Corrigé
- `docker-compose.yml` : chemin du Dockerfile du backoffice (`apps/admin/Dockerfile`).
- CI : installation identique aux images Docker (`.[dev,admin]`, torch CPU, matplotlib pour les tests du bac à sable), Python 3.12 ; les tests ignorent aussi les variables `AGENT_*`, `PROMPT_*`, `LANGFUSE_*`, `EVAL_*`, `SANDBOX_*`, `CODE_*` du terminal.
- `.gitignore` : exports `*.jsonl` (contenu des cours) hors dépôt.

### Ajouté
- `docs/DEPLOIEMENT_VPS.md` : installation, mise à jour, variables serveur, vérifications, sauvegarde/retour arrière, intégration à un Caddy existant.

## [Unreleased] — Phase 4e : Atelier Code 2.0 (exécution sandboxée)

### Ajouté
- **Service d'exécution** `apps/sandbox/` (image `ela-sandbox` : Python 3.12 + numpy, pandas, scipy, statsmodels, matplotlib, arch ; serveur HTTP en bibliothèque standard, `POST /run`). Chaque script tourne dans un sous-processus `python -I` jetable, répertoire temporaire, limites CPU / mémoire / processus / taille de fichier (`resource`), délai mural ; les figures matplotlib sont capturées (backend Agg) et renvoyées en PNG. Dans `docker-compose.yml`, le service `sandbox` est sur un réseau **interne** (aucun accès Internet), non-root, `read_only`, `cap_drop: ALL`, `no-new-privileges`, `mem_limit`, `pids_limit` ; seul `app` peut l'atteindre.
- **Client et service** (`ela.sandbox`) : `SANDBOX_URL` (vide = scripts générés mais jamais exécutés, comportement précédent), `SANDBOX_TIMEOUT_SECONDS`, **quota** `SANDBOX_DAILY_QUOTA` (10 exécutions par étudiant et par jour ; 0 = illimité) ; chaque exécution est journalisée dans `code_runs` (migration `0008`).
- **Atelier Code** : bouton **▶️ Exécuter** sous le script (Python, bac à sable configuré) ; sortie, erreurs (fin de trace) et graphiques affichés dans le fil ; en cas d'erreur, bouton **🔧 Corriger le script** (une passe de correction par le modèle, prompt `fix_code_v1.md`) puis nouvelle exécution. R reste en génération seule.
- **Agent** : outil `run_python(code)` (un appel par réponse, présent seulement si `SANDBOX_URL` est défini) pour calculer, simuler ou vérifier numériquement ; la sortie et les figures sont affichées à l'étudiant, le modèle les commente ; même quota.
- **Backoffice → Pédagogie → onglet « 💻 Code »** : exécutions par statut et origine, erreurs les plus fréquentes, exécutions par étudiant.
- Le prompt de l'atelier demande des scripts autonomes (graine fixée, pas de `plt.show()`, ni réseau, ni fichiers, ni `input()`) **qui impriment leurs résultats** (tableau des estimations : paramètre, vraie valeur si simulée, estimation, écart-type, t-stat ; statistiques utiles) ; la réponse initiale décrit « ce qu'il faudra regarder » sans annoncer de résultat (le script n'a pas encore tourné).
- **Fiabilité des scripts** : modèle dédié à l'atelier (`CODE_MAX_TOKENS=4096`, `CODE_REASONING_EFFORT=medium` ; le budget de 2 048 tokens du chat tronquait les scripts longs) ; vérification syntaxique (`ast.parse`) avant affichage avec correction silencieuse, sans exécution ni quota ; la correction reçoit l'historique des erreurs déjà rencontrées sur le script (pas de régression) ; `CODE_AUTO_FIX=true` : après un échec d'exécution, une correction automatique puis ré-exécution avant d'afficher l'erreur (une exécution de quota supplémentaire ; la trace reste visible si la seconde tentative échoue, avec le bouton « Corriger »).
- **Interprétation après exécution** : après une exécution réussie, un appel modèle (`interpret_run_v1.md`, `ela.sandbox.service.interpret_run`) produit un tableau Markdown des résultats recopiés de la sortie réelle et un commentaire relié au cours (citations `[S#]` vérifiées, panneau de sources, titres des figures). Si le script n'a rien imprimé de chiffré, ELA le dit et propose **🔧 Compléter le script** (une passe qui ajoute les `print()`).

## [Unreleased] — Phase 4d : fiches de révision et suggestions de suivi

### Ajouté
- **Fiches de révision par chapitre** : `/fiche` (starter « Fiche de révision », choix du cours puis du chapitre par boutons, ou « *UTS, chapitre 2* » en texte) et outil d'agent `generate_review_sheet(course, chapter)` (« fais-moi une fiche sur le chapitre 2 »). La fiche est générée depuis les slides du chapitre prises dans le plan du cours (pas de recherche vectorielle, donc pas de dérive ; chapitres longs résumés par parties puis fusionnés — `ela.pedagogy.review`), en sortie structurée `ReviewSheet` (objectifs, sections : notions clés / définitions / formules / pièges, « à retenir », titres des slides ; prompt `review_sheet_v1.md`). Affichée en Markdown + KaTeX dans le fil avec un fichier `.md` téléchargeable et les boutons « Me tester sur ce chapitre » / « Un exercice ».
- **Cache partagé des fiches** : table `review_sheets` (migration `0007`, `ela.db.review_sheets`) — une fiche par (cours, chapitre, niveau, version de prompt, empreinte du corpus), resservie à tous les étudiants et régénérée quand les slides ou le prompt changent. Aucune donnée étudiante ; la consultation est journalisée dans `student_activity` (bilan, backoffice).
- **Suggestions de suivi** en fin de réponse, sans appel au modèle (`ui.followup_actions`) : « 🎓 Me tester : *notion* » (quiz restreint au chapitre dominant des extraits cités), « ✏️ Exercice : *notion* », « 📄 Fiche : *chapitre* » et « 💻 Exemple Python » quand la question parle de code. Pas de suggestion sur un refus, une réponse sans extrait cité, ou pendant un quiz/exercice.
- `generate_quiz` accepte des filtres `course` / `chapter` (quiz sur un chapitre précis depuis une fiche ou une suggestion).

## [Unreleased] — Correctifs quiz (retours de test 4c)

### Corrigé
- **Sorties structurées en production** : le modèle partagé est enveloppé d'un modèle de secours (`RunnableWithFallbacks`), qui n'expose pas `with_structured_output` ; quiz, exercices et vérification d'étape passaient donc toujours par le repli « JSON en texte » avec le budget `LLM_MAX_TOKENS` (2 048). `ainvoke_structured` déballe le modèle principal pour la voie structurée, conserve les replis pour la voie texte, accepte un budget `max_tokens` dédié (4 096 pour la génération) et répare les antislashs LaTeX non échappés (`\hat{\rho}` → formules rouges, options brouillées).
- **Quiz longs** : génération par lots de 5 questions (une requête par lot, les questions des lots précédents sont exclues) ; un quiz de 10 questions tient désormais ; les questions mal formées (options vides ou brouillées, doublons) sont écartées ; si un lot échoue, les questions déjà obtenues sont conservées et le nombre réellement produit est affiché.
- **Reprise après échec** : quand l'outil `generate_quiz` de l'agent échoue, la réponse porte un bouton « 🔁 Réessayer le quiz » qui regénère avec le même sujet, le même nombre de questions et la même difficulté (un nouveau tour d'agent oubliait le nombre demandé) ; `ElaEngine.generate_quiz` / `generate_exercise` distinguent l'échec du modèle (`GenerationError`) du sujet absent (`None`). L'en-tête de la première question affiche le sujet, le nombre réel de questions et la difficulté.
- **Dérive de sujet** : `generate_quiz` et `generate_exercise` vérifient que les extraits retournés mentionnent le sujet demandé (`ela.pedagogy.relevance` : acronymes obligatoires, moitié des autres termes, sans appel au modèle) ; un étudiant M1 qui demande un quiz GARCH n'obtient plus un quiz ARMA.
- **Niveau de l'étudiant** : le prompt système indique que seuls les cours de son niveau lui sont accessibles ; quand un sujet est traité dans un cours d'un autre niveau (`courses_elsewhere`), l'outil et les messages de `/start_quiz`, `/exercise` et du bilan le disent (« ce sujet est traité dans Financial_Econometrics (M2) ») au lieu de « le cours ne traite pas ce sujet ».
- **Bilan** : les chapitres travaillés comptent aussi les quiz (`chapters_worked` : questions **et** quiz par cours › chapitre) ; un chapitre déjà testé par quiz n'est plus proposé en « Me tester ».

## [Unreleased] — Phase 4c : suivi de l'étudiant

### Ajouté
- **Journal d'activité** : chaque question répondue par l'agent est enregistrée (`student_activity`, migration `0006` : cours/chapitre dominants des extraits cités, nombre de recherches et d'extraits, réponse hors cours `[Connaissances générales]`, refus) — uniquement pour un étudiant connecté.
- **`/bilan`** dans le chat (et starter « Mon bilan ») : questions posées, chapitres travaillés, réussite aux quiz (points faibles < 60 %, points forts ≥ 80 %), exercices commencés/terminés, puis jusqu'à trois suggestions cliquables (quiz sur un point faible, reprise d'un exercice inachevé, quiz sur un chapitre travaillé mais jamais testé). `ela.db.progress` (`student_report`, `render_report`, `build_suggestions`).
- **Backoffice — onglet « 🎓 Pédagogie »** (`apps/admin/views/pedagogy.py`) : réussite des quiz par sujet et questions les plus ratées, exercices par sujet (taux d'achèvement, étapes correctes), questions par cours/chapitre et dernières questions, **hors corpus** (questions refusées, répondues en connaissances générales ou sans extrait — signal pour les supports), le tout filtré par période.

## [Unreleased] — Phase 4b : mode exercice socratique

### Ajouté
- **« ✍️ Vérifier mon étape »** sur chaque exercice : l'étudiant écrit son raisonnement, ELA l'évalue contre le corrigé caché (`ela.pedagogy.exercise.check_step`, sortie structurée `StepFeedback` : verdict correct / partiel / incorrect / hors sujet, commentaire, question d'orientation) sans jamais donner la suite ; les tentatives précédentes et les indices déjà révélés sont transmis au modèle ; `stop` rend la main au chat ; l'exercice est marqué terminé quand le résultat attendu est atteint (le corrigé reste consultable). Prompt `check_step_v1.md`.
- **Persistance** des tentatives : table `exercise_attempts` (migration `0005`), `ela.db.exercises` (`record_attempt`, `exercise_summary`) — matière pour le bilan étudiant (lot 4c).
- `Exercise` porte `exercise_id` et `topic`.

### Corrigé
- Console : le warning Pydantic `PydanticSerializationUnexpectedValue` émis par `langchain-openai` à chaque sortie structurée réussie est filtré (`ela.log`).

## [Unreleased] — Phase 4a : Quiz 2.0

### Ajouté
- **Persistance des quiz** : table `quiz_results` (migration `0004`), une ligne par question répondue (étudiant, fil, quiz, sujet, difficulté, cours/chapitre, options, réponse, extraits du cours) ; `ela.db.quiz` (`record_answer`, `recent_errors`, `user_stats`, `next_difficulty`). Les écritures n'interrompent jamais le quiz.
- **Relance ciblée** : après une mauvaise réponse, bouton « 💬 M'expliquer davantage » (l'agent explique la bonne réponse à partir du cours, dans le fil) ; en fin de quiz, bouton « 🔁 Refaire un quiz sur mes erreurs » (nouveau quiz ciblé sur les notions ratées, questions reformulées — paramètre `focus` de `generate_quiz`).
- **Difficulté adaptative** par session : ≥ 80 % de bonnes réponses → niveau supérieur, ≤ 40 % → niveau inférieur ; affichée dans le message de score et appliquée au quiz suivant (`/start_quiz` ou relance).
- `Quiz` porte désormais `quiz_id`, `difficulty`, `source_labels`, `course`, `chapter`.
- La difficulté adaptative s'applique aussi aux quiz demandés à l'agent en langage naturel (outil `generate_quiz` : niveau de la session sauf demande explicite) ; `/start_quiz` accepte « sujet, 5 questions » (1 à 10, 3 par défaut) ; les prompts quiz/exercice ne montrent plus les identifiants `S#` au modèle (les questions ne citent plus « l'extrait [S4] »).

### Corrigé
- Data layer : un pas (étape d'outil, réponse) terminé rapidement pouvait être écrasé par son insertion initiale (tâches concurrentes de Chainlit) ; `output`, `input`, `end` ne reculent plus lors de l'upsert.

## [2.0.0a4] — 2026-10-18 — Phase 3 : ELA agentique

### Ajouté
- **Agent** (`ela.agent`, LangChain 1.4 `create_agent` sur LangGraph) : le modèle décide quand et comment chercher avec les outils `search_course` (hybride, scopé au niveau, filtres cours/chapitre), `get_course_outline` (plan chapitre › section › slides), `get_slide` (relecture d'un extrait ou d'une slide), `generate_quiz` et `generate_exercise` (sorties structurées Pydantic, `ela.pedagogy`). Questions composées traitées par plusieurs recherches (appels d'outils parallèles).
- **Citations vérifiées** : chaque extrait lu par le modèle reçoit un identifiant `[S1]`, `[S2]`… (`SourceRegistry`) ; les citations de la réponse sont contrôlées mécaniquement et les citations fantômes retirées ; `[Connaissances générales]` pour le hors-cours.
- **Middleware** : prompt système dynamique (niveau, cours visibles, résumé d'historique), modèle de secours, limites d'appels modèle/outils, récupération des erreurs d'outils. Résumé d'historique par session (`HistorySummarizer`) pour les tours qui sortent de la fenêtre.
- **Prompts versionnés** (`ela/agent/prompts/*_v1.md`, `PROMPT_VERSION`), inscrits dans les traces et rapports.
- **Images** : transcription fidèle par le modèle vision (équations LaTeX, tableaux, sorties logicielles, graphiques) puis traitement par l'agent texte ; visible comme une étape « Lecture de l'image jointe ».
- **UI Chainlit** (`apps/chat/ui.py`) : appels d'outils affichés en étapes (`cl.Step`), panneau de sources latéral (cours › chapitre › slide, score, extrait rendu en KaTeX) relié aux citations `[S1]`, quiz avec LaTeX dans les questions/explications (options en texte brut), exercice avec boutons **Indice** / **Solution** ; nouvelle commande `/exercise`.
- **Évaluation de bout en bout** `ela-eval answer` (`ela.eval.answer`, juge LLM `ela.eval.judge`) : latence médiane/p90, recherches par question, citations valides/fantômes, fidélité (points attendus), refus hors domaine, étiquetage hors corpus, multi-étapes ; `--engine legacy` rejoue la chaîne fixe de la phase 2 (gelée dans `ela.eval.legacy_pipeline`) pour la mesure « avant » ; `ela-eval compare` accepte les rapports `answer`.
- **Traçage Langfuse** optionnel (`ela.tracing`, variables `LANGFUSE_*`, extra `tracing`), désactivé par défaut, identifiant étudiant non transmis sauf `LANGFUSE_SEND_USER_ID=true`.
- Store/retriever : filtre par cours (`courses`) dans les deux backends, filtre chapitre et `top_n` par appel dans `ScopedRetriever.invoke`.
- Tests : citations, sorties structurées, juge et métriques, outils et plan de cours, agent de bout en bout avec modèle scripté (recherche simple, question composée, citation fantôme, refus, limite d'appels, image, modèle de secours, quiz, résumé d'historique), rendu Chainlit avec module simulé, traçage.

### Corrigé
- Migration `0003_chainlit_212_steps` : colonnes `steps.autoCollapse` et `steps.modes` écrites par Chainlit 2.12 ; sans elles, l'enregistrement de chaque étape d'outil échouait silencieusement et les étapes disparaissaient à la reprise d'un fil.
- Stockage local : les éléments texte (panneau de sources) sont écrits en UTF-8 (`.md`, `text/markdown`) au lieu d'échouer en binaire ; test d'intégration « tour de l'agent persisté puis repris » ajouté.
- Quiz : les options sont mélangées après génération (la bonne réponse n'est plus systématiquement A) ; elles sont rendues dans le message (LaTeX/KaTeX autorisé), les boutons ne portent plus que la lettre.
- Rendu LaTeX : un bloc `$$…$$` multi-lignes (`$$\begin{aligned}`) reçoit ses délimiteurs sur des lignes propres, sans quoi remark-math l'affichait en rouge ; le panneau de sources n'affiche plus l'en-tête contextuel (COURS/CHAPITRE/…) déjà porté par le fil d'Ariane.

### Modifié
- `ElaEngine.ask` renvoie un flux d'événements typés (`ela.agent.events`) consommé par Chainlit et par l'évaluation ; `generate_quiz` / `generate_exercise` renvoient des objets Pydantic ; l'historique complet est transmis au moteur qui le borne et le résume.
- Configuration : `PROMPT_VERSION`, `AGENT_MAX_MODEL_CALLS`, `AGENT_MAX_TOOL_CALLS`, `AGENT_MAX_SEARCHES`, `AGENT_SEARCH_K`, `AGENT_SEARCH_K_MAX`, `AGENT_OUTLINE_MAX_CHARS`, `AGENT_SUMMARY_ENABLED`, `LANGFUSE_*`, `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_REASONING_EFFORT`.
- Dépendances : `langchain==1.4.6`, `langgraph==1.2.16` ; extra `tracing` (`langfuse`).

### Supprimé
- `src/ela/guard.py` (RAGGuard) et l'appel de notation systématique avant chaque réponse ; `prompts.quiz_prompt` (remplacé par `prompts/quiz_v1.md`).

## [2.0.0a3] — 2026-10-17 — Phase 2 : retrieval 2.0

### Ajouté
- **Parseur LaTeX v2** (`ela.ingestion.texclean`, `ela.ingestion.latex`) : suppression correcte des commentaires (les frames commentées ne sont plus ingérées), expansion des macros sans argument du cours, chunks typés — `slide`, `definition`/`theorem`/`proposition`/`lemma`/`corollary`/`example`/`remark`/`model`/`assumption` (rattachés à leur slide par `parent_id`) et `text` (manuscrits découpés par chapitre/section/paragraphes) — avec `doc_kind` (`slides`/`manuscript`), notes `\note{}` de l'enseignant, tableaux, figures, colonnes, chapitres beamer personnalisés ; identifiants stables par hash de contenu.
- **Embeddings** `BAAI/bge-m3` (`ela.retrieval.embeddings`) : local (sentence-transformers) ou API OpenAI-compatible (ILaaS), repli local automatique sur le même modèle.
- **Store pgvector** (`ela.retrieval.store.PgVectorStore`, migration `0002_rag_chunks`) : table `rag_chunks` avec index HNSW et `tsvector` français, recherche hybride vecteur + plein texte fusionnée par RRF en une requête SQL ; `ChromaStore` (Chroma + BM25) derrière la même interface.
- **Ingestion incrémentale** : `ela-ingest` ne vectorise que les chunks nouveaux ou modifiés et supprime ceux qui ont disparu ; `--full`, `--dump chunks.jsonl` ; backoffice : dry run / mise à jour incrémentale / reconstruction complète.
- **Reranker** optionnel (`ela.retrieval.rerank`) : FlashRank ou CrossEncoder `BAAI/bge-reranker-v2-m3`. **Désactivé par défaut** (`RERANK_PROVIDER=none`) : mesuré le 17/10 sur 65 questions (Chroma + bge-m3), l'ordre de fusion seul donne any-hit@5 = 0,94 / MRR = 0,74 / 0,25 s, contre 0,88 / 0,57 avec `ms-marco-MiniLM-L-12-v2` et 0,59 / 0,29 avec `ms-marco-MultiBERT-L-12` ; le rappel du pool de 40 candidats est de 1,00. Baseline Phase 1 : 0,63 / 0,43.
- **Retriever** : slides recherchées d'abord, manuscrits en quota (`RETRIEVAL_MANUSCRIPT_K`) et plafonnés dans le top-n final (`RETRIEVAL_MANUSCRIPT_MAX`), union des deux branches (vecteur, plein texte) comme pool de candidats, expansion des définitions/théorèmes vers leur slide parente avec re-remplissage après dédoublonnage, scores de fusion et de reranking dans les métadonnées.
- `ela-eval retrieval` mesure aussi le **rappel du pool** (slide attendue parmi les candidats avant reranking) pour distinguer un défaut de recherche d'un défaut de reranking.
- `ela.mathfmt.normalize_math` : les réponses en `\(…\)`, `\[…\]` ou `equation`/`align` nus sont réécrites en `$…$` / `$$…$$` pour KaTeX (le LaTeX brut n'apparaît plus dans le chat).
- Tests : parseur (fixtures beamer/manuscrit), stores (pgvector en intégration, Chroma), retriever, pipeline incrémental, normalisation LaTeX, configuration.
- `eval/questions.yaml` : jeu d'évaluation v1 — 71 questions (UTS 17, MTS 17, HFE 15, SA 9, NCE 7, cloisonnement 1, hors corpus 2, hors domaine 3) avec slides attendues et points de réponse, rédigées à partir des `.tex` ; chaque slide attendue a été vérifiée contre le corpus ingéré. **À valider par l'enseignant.**
- `ela-eval retrieval` : mesure hit@k, any-hit@k, MRR et latence du retriever par niveau et par cours, rapport JSON dans `eval/reports/` ; `ela-eval compare` pour comparer deux rapports (avant/après).
- `docker-compose.yml` : le Caddy d'ELA devient un profil optionnel (`--profile standalone`) ; par défaut app/admin n'écoutent que sur 127.0.0.1 pour être servis par un reverse proxy existant (`deploy/caddy-ela.snippet`).

### Modifié
- Configuration : `EMBEDDING_PROVIDER/MODEL/DIM/BASE_URL/API_KEY/LOCAL_FALLBACK`, `RETRIEVAL_BACKEND`, `RETRIEVAL_MANUSCRIPT_K`, `RETRIEVAL_LEXICAL_WEIGHT`, `RETRIEVAL_RRF_K`, `RETRIEVAL_FTS_CONFIG`, `RETRIEVAL_EXPAND_ELEMENTS`, `RERANK_PROVIDER/MODEL/MAX_LENGTH`, `CHUNK_TARGET_CHARS`, `CHUNK_MAX_CHARS`, `INGEST_INCLUDE_NOTES` remplacent `EMBEDDING_MODEL_NAME`, `RETRIEVAL_BM25_WEIGHT`, `RERANK_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.
- Le contexte RAG transmis au modèle conserve les sauts de ligne (listes, formules en bloc) et indique l'emplacement de chaque extrait (chapitre › section › slide).
- Prompt : consigne explicite sur les délimiteurs LaTeX et la reprise des notations du cours.
- Dépendances : `langchain-community`, `langchain-classic`, `langchain-chroma`, `langchain-text-splitters` retirées (Chroma utilisé via `chromadb` directement).

### Supprimé
- Swap atomique de dossiers Chroma dans le backoffice (remplacé par l'ingestion incrémentale, visible sans redémarrage).

## [2.0.0a2] — 2026-10-13 — Phase 1b : migrations, montée de versions, Docker

### Ajouté
- **Alembic** embarqué dans le paquet (`src/ela/db/migrations/`) et commande `ela-db` (`upgrade`, `current`, `history`, `downgrade`). Migration `0001_baseline` idempotente : tables Chainlit 2.12 (dont `elements.autoPlay/playerConfig`, `steps.command`, `feedbacks.threadId`), colonnes de comptes ELA, index, vue `v_daily_usage`. `ela-users migrate` délègue à Alembic. Remplace le SQL manuel du README/GUIDE.
- Test d'intégration de la data layer personnalisée sous Chainlit 2.12 (`tests/test_data_layer.py`) : cycle complet utilisateur → fil → étape → image stockée localement → feedback → reprise.

### Modifié
- **Versions** : Chainlit 2.9.6 → 2.12.0 ; LangChain core 1.2.7 → 1.6.10, openai 1.7.1, community 0.4.2, classic 1.0.8, huggingface 1.2.3 ; chromadb 1.4.1 → 1.5.9 ; sentence-transformers 5.7.0 ; pydantic 2.14.1 ; SQLAlchemy 2.0.54 ; bcrypt 5.0 ; Streamlit 1.65. Jeu de versions résolu avec `uv`, suite de tests verte.
- `.chainlit/config.toml` migré au format 2.12 (sections MCP) ; `cl.Action` sans arguments obsolètes.
- **Dockerfiles multi-stage** (builder → runtime sans compilateur, utilisateur non-root, healthchecks, torch CPU-only) ; caches des modèles HF et FlashRank dans `data/` (volume) : téléchargés une seule fois.
- **Images de `public/`** optimisées : fond de connexion 7,9 Mo → 0,24 Mo, avatar 0,7 → 0,12 Mo, bannière 4,5 → 0,28 Mo, logos réduits. Les fichiers non référencés (`ela2.png` 19 Mo, `etudiant.png` 8 Mo, `custom-background.png`, `logo.pdf`) sont retirés du dépôt.

## [2.0.0a1] — 2026-10-13 — Phase 1a : fondations (structure, ressources partagées, outillage)

### Ajouté
- `pyproject.toml` : le projet devient le paquet installable `ela-ai` (`pip install -e ".[admin,dev]"`), avec les commandes `ela-ingest`, `ela-users`, `ela-hash`. `requirements.txt` supprimé (une seule source de vérité).
- `src/ela/` : `engine.py` (moteur de session léger), `retrieval/` (ressources partagées + retriever scopé), `db/` (moteur SQL partagé avec pool, comptes, threads), `history.py` (historique borné), `prompts.py`, `guard.py`, `log.py`, `ingestion/`, `cli/`.
- `apps/chat/` (Chainlit : `app.py`, `data_layer.py`) et `apps/admin/` (Streamlit, déplacé).
- Tests : 39 tests (configuration, historique, retrieval avec composants simulés, moteur avec LLM simulé, intégration PostgreSQL optionnelle via `TEST_DATABASE_URL`).
- CI GitHub Actions (ruff + pytest avec PostgreSQL), `.pre-commit-config.yaml`.

### Modifié
- **Performance** : embeddings, base vectorielle, index BM25 (un par niveau) et reranker sont chargés **une fois par processus** (préchargement en arrière-plan au démarrage, `WARMUP_ON_START`) et rechargés automatiquement si la base Chroma est reconstruite ; ouvrir une session ne recharge plus rien.
- **Base de données** : un seul moteur SQLAlchemy async (pool) au lieu d'un moteur par requête ; bcrypt vérifié hors de la boucle d'événements ; une seule `DATABASE_URL` (le pilote asyncpg/psycopg2 est dérivé automatiquement, `ADMIN_DATABASE_URL` devient optionnelle).
- **Historique** : fenêtre bornée (`HISTORY_MAX_TURNS`, `HISTORY_MAX_CHARS`) au lieu du fil complet.
- Configuration : le `.env` de la racine est trouvé depuis `apps/admin` et les CLI (`ELA_ENV_FILE` pour forcer un chemin) ; nouveaux paramètres `RETRIEVAL_*`, `RERANK_*`, `CHUNK_*`, `LOG_JSON`, `STORAGE_DIR`.
- `print` remplacés par `logging` (texte en dev, JSON en Docker).
- Dockerfiles réécrits pour la nouvelle structure (`pip install .` du paquet) ; `docker-compose.yml` passe `DATABASE_URL` aux deux services.
- README réécrit.

### Supprimé
- `app.py`, `main_ela.py`, `rag_guard.py`, `ingest.py`, `manage_users.py`, `generate_hash.py`, `requirements.txt`, `admin/` (contenu déplacé).

## [2.0.0a0] — 2026-10-06 — Phase 0 : remise en marche et sécurisation

### Ajouté
- Paquet `ela/` : configuration centralisée `ela.config` (pydantic-settings, lecture `.env`) et fabrique de modèles `ela.llm` (client OpenAI-compatible unique pour Cerebras, ILaaS et Groq ; modèle de secours ; routage vision).
- `.env.example` documenté (fournisseurs, modèles, base de données, backoffice). Une variable vide prend sa valeur par défaut ; `LLM_MODEL_FALLBACK=none` et `LLM_REASONING_EFFORT=omit` désactivent explicitement.
- Reverse proxy **Caddy** (`Caddyfile`, service `caddy`) : HTTPS automatique, en-têtes de sécurité, `/admin` protégé par authentification basique.
- `tests/test_llm.py` : 12 tests hors-ligne (fabrique, fallback, encodage d'images, payload, valeurs `.env` vides).
- `generate_hash.py --caddy` : génère le hash `ADMIN_BASIC_AUTH_HASH` sans Docker.
- `manage_users.py` : CLI de gestion des comptes (`check`, `migrate`, `create`, `set-password`, `deactivate`, `list`) — testé contre PostgreSQL.
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
