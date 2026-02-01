# app.py
import os
import re
import sys
import asyncio
import aiofiles

# --- FIX WINDOWS (OBLIGATOIRE POUR POSTGRES/ASYNCPG) ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from langchain_core.messages import HumanMessage, AIMessage

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Import de votre logique RAG
from main_ela import ELA_Bot

# chainlit run app.py -w 

load_dotenv()

# --- CLASSE DE STOCKAGE LOCAL (Custom Storage) ---
class LocalStorageClient:
    """
    Simule un stockage Cloud (S3/Azure) mais sauvegarde
    les fichiers dans un dossier local sur votre ordinateur.
    """
    def __init__(self, base_path: str = "stockage_fichiers"):
        self.base_path = base_path
        # Crée le dossier racine s'il n'existe pas
        os.makedirs(self.base_path, exist_ok=True)

    async def upload_file(self, object_key: str, data: bytes, mime: str = "application/octet-stream", overwrite: bool = True):
        # On définit le chemin complet du fichier
        file_path = os.path.join(self.base_path, object_key)
        
        # Création des sous-dossiers
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Écriture du fichier
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data)
            
        return {"object_key": object_key, "url": str(file_path)}

    async def get_read_url(self, object_key: str) -> str:
        """Retourne le chemin du fichier pour que Chainlit puisse l'afficher."""
        file_path = os.path.join(self.base_path, object_key)
        return str(file_path)

    # Suppression des médias associées aux discussions
    async def delete_file(self, object_key: str):
        """Supprime le fichier physique ET les dossiers vides parents."""
        file_path = os.path.join(self.base_path, object_key)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Fichier supprimé : {file_path}")
                
                # --- NETTOYAGE DES DOSSIERS VIDES ---
                # On remonte l'arborescence pour supprimer les dossiers devenus vides
                # Structure : .files_ela / ThreadID / ElementID / image.png
                
                directory = os.path.dirname(file_path) # Dossier ElementID
                
                # On tente de supprimer le dossier courant et son parent (ThreadID)
                # os.rmdir ne fonctionne QUE si le dossier est vide, donc c'est sécurisé
                for _ in range(2): 
                    try:
                        os.rmdir(directory)
                        print(f"📂 Dossier vide supprimé : {directory}")
                        # On remonte d'un cran
                        directory = os.path.dirname(directory)
                        
                        # Sécurité : ne jamais supprimer le dossier racine de stockage
                        if os.path.abspath(directory) == os.path.abspath(self.base_path):
                            break
                    except OSError:
                        # Le dossier n'est pas vide (contient d'autres images), on s'arrête
                        break
                        
        except Exception as e:
            print(f"⚠️ Erreur lors de la suppression : {e}")
            

# --- GESTION UTILISATEURS (Depuis .env) ---
def load_users_from_env():
    """Charge le dictionnaire user:password depuis le .env"""
    users_dict = {}
    raw_data = os.getenv("ELA_AUTH_DATA", "")
    if not raw_data:
        return users_dict
    for pair in raw_data.split(","):
        if ":" in pair:
            username, password = pair.split(":", 1)
            users_dict[username.strip()] = password.strip()
    return users_dict

USERS = load_users_from_env()

# --- 1. ACTIVATION DU DATA LAYER (SIDEBAR NATIVE) ---
# Récupère l'URL sécurisée depuis le .env
@cl.data_layer
def get_data_layer():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("❌ DATABASE_URL manquante dans le fichier .env")
    
    # On instancie notre stockage local
    storage = LocalStorageClient(base_path=".files_ela")
    
    # On injecte le stockage dans le DataLayer
    return SQLAlchemyDataLayer(
        conninfo=database_url, 
        storage_provider=storage
    )

# --- 2. AUTHENTIFICATION ---
@cl.password_auth_callback
def auth_callback(username, password):
    """Vérifie les identifiants par rapport au .env"""
    if username in USERS and USERS[username] == password:
        return cl.User(identifier=username)
    return None

# --- 3. DÉMARRAGE DE SESSION ---
@cl.on_chat_start
async def start():
    """Initialise le bot quand une nouvelle session démarre."""
    print("🚀 Démarrage nouvelle session")
    
    # On instancie le bot RAG
    ela_instance = ELA_Bot()
    cl.user_session.set("ela_bot", ela_instance)
    
    # # Message de bienvenue
    # welcome_msg = cl.Message(
    #     content="👋 Bonjour ! Je suis **ELA**, votre assistant en économétrie.\n\nPosez-moi une question sur vos cours ou consultez l'historique dans la barre latérale.",
    #     author="ELA 🤖"
    # )
    # await welcome_msg.send()

# --- 4. REPRISE DE CONVERSATION (Clic dans la Sidebar) ---
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Appelé quand l'utilisateur clique sur une ancienne conversation.
    Chainlit charge l'historique visuel automatiquement.
    Nous devons juste réinitialiser le bot.
    """
    print(f"🔄 Reprise de la conversation {thread['id']}")
    ela_instance = ELA_Bot()
    cl.user_session.set("ela_bot", ela_instance)

# --- 5. GESTION DES MESSAGES ---
@cl.on_message
async def main(message: cl.Message):
    ela_bot = cl.user_session.get("ela_bot")
    
    # 1. GESTION DES COMMANDES SPÉCIALES

    # A. Démarrage du Quiz
    if message.content == "/start_quiz":
        # --- RENOMMAGE 1 : Immédiat ---
        await rename_current_thread("🎓 Nouveau Quiz") 
        
        await cl.Message(content="🎓 **Mode Quiz activé !**\nSur quel concept du cours voulez-vous vous tester ? (ex: *MCO, Séries Temporelles, Tests de racine unitaire...*)").send()
        cl.user_session.set("quiz_mode", "waiting_topic")
        return
    
    # B. Galerie
    if message.content.strip().lower() == "/gallery":
        await show_user_gallery()
        return
    
    # C. Atelier Code
    if message.content == "/code_workshop":
        # Renommage
        await rename_current_thread("💻 Atelier Code")
        
        # Actions pour choisir le langage
        actions = [
            cl.Action(name="code_lang", value="Python", label="Python", payload={"value": "Python"}),
            cl.Action(name="code_lang", value="R", label="R", payload={"value": "R"})
        ]
        
        await cl.Message(
            content="💻 **Bienvenue dans l'Atelier Code !**\n\nJe peux générer pour vous des exemples pratiques basés sur vos cours.\nQuel langage souhaitez-vous utiliser ?",
            actions=actions
        ).send()
        return
    
    # 2. LOGIQUE DU QUIZ (Machine à états)
    # ------------------------------------
    quiz_mode = cl.user_session.get("quiz_mode")

    # ÉTAT 2.1 : L'utilisateur donne le sujet
    if quiz_mode == "waiting_topic":
        topic = message.content
        
        # --- RENOMMAGE 2 : Avec le sujet ---
        # On limite la taille du titre pour que ça rentre dans la sidebar
        safe_topic = (topic[:25] + '..') if len(topic) > 25 else topic
        await rename_current_thread(f"🎓 Quiz : {safe_topic}")

        msg_wait = cl.Message(content=f"🔍 Analyse de vos cours sur **{topic}** et génération des questions...")
        await msg_wait.send()
        
        # Génération via le Bot
        quiz_data = await ela_bot.generate_quiz_json(topic, num_questions=3)
        
        if not quiz_data:
            await msg_wait.update(content="⚠️ Je n'ai pas trouvé assez d'informations dans le cours pour ce sujet. Essayez un autre terme.")
            return

        # On stocke le quiz
        cl.user_session.set("quiz_data", quiz_data)
        cl.user_session.set("quiz_index", 0) 
        cl.user_session.set("quiz_score", 0)
        cl.user_session.set("quiz_mode", "active")
        
        # On lance la première question
        await ask_next_question()
        return

    # ÉTAT 2.2 : Si l'utilisateur répond à une question en tapant du texte
    if quiz_mode == "active":
        # Note : On n'a pas besoin de récupérer quiz_data ici car on renvoie juste un message d'aide
        
        await cl.Message(content="💡 Utilisez les boutons ci-dessus pour répondre !").send()
        return
    
    # 3. GESTION DU CODE WORKSHOP
    if cl.user_session.get("code_mode") == "waiting_topic":
        topic = message.content
        language = cl.user_session.get("code_lang_choice")
        
        await rename_current_thread(f"💻 Code : {topic}")
        
        msg_load = cl.Message(content=f"⚙️ Génération du script **{language}** pour **{topic}**...")
        await msg_load.send()
        
        response = await ela_bot.generate_practical_code(topic, language)
        
        msg_load.content = response
        await msg_load.update()
        
        # Fin du mode code, retour au chat normal
        cl.user_session.set("code_mode", None)
        return
    
    # 4. GESTION DES IMAGES
    image_path = None
    
    # 4.1 Vérifier s'il y a des fichiers attachés
    if message.elements:
        # On prend le premier fichier (on pourrait gérer une boucle pour plusieurs)
        file = message.elements[0]
        
        # FILTRE STRICT : Uniquement les images
        if "image" in file.mime:
            image_path = file.path # Chainlit a déjà téléchargé le fichier ici
        else:
            # Si l'utilisateur envoie un PDF ou autre ici
            await cl.Message(content="⚠️ Désolé, je n'accepte que les images (.png, .jpg, .jpeg).").send()
            return

    # 4.2 Reconstruction Historique (inchangé)
    context_messages = cl.chat_context.get()
    history_langchain = []
    
    for msg in context_messages:
        if msg.id == message.id: 
            continue
        if msg.type == "user_message":
            # Note : On ne remet pas l'image dans l'historique texte pour économiser les tokens
            # On garde juste le texte
            history_langchain.append(HumanMessage(content=msg.content))
        elif msg.type == "assistant_message":
            history_langchain.append(AIMessage(content=msg.content))
    
    msg = cl.Message(content="", author="ELA 🤖")
    
    # 4.3 Appel à ELA avec l'image (si présente)
    # On passe le chemin local de l'image
    response = await ela_bot.ask(
        question=message.content, 
        chat_history=history_langchain,
        image_path=image_path
    )
    
    msg.content = response
    await msg.send()
    
# --- FONCTIONS UTILITAIRES POUR LA GALLERIE ---
async def show_user_gallery():
    """Affiche la galerie et renomme la conversation."""
    user = cl.user_session.get("user")
    
    # Configuration DB
    db_url = os.getenv("DATABASE_URL")
    engine = create_async_engine(db_url)
    AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    try:
        # --- 1. RENOMMAGE AUTOMATIQUE DU THREAD ---
        # On récupère l'ID de la conversation actuelle
        thread_id = cl.context.session.thread_id
        
        if thread_id:
            async with AsyncSessionLocal() as session:
                # Requête pour renommer
                update_query = text('UPDATE threads SET name = :name WHERE id = :id')
                # On utilise ::uuid pour être sûr que Postgres comprenne le format
                await session.execute(update_query, {
                    "name": "🖼️ Mes contenus médias", 
                    "id": thread_id
                })
                await session.commit() # Important pour valider le changement

        # --- 2. AFFICHAGE DE LA GALERIE ---
        msg = cl.Message(content=f"📂 **Galerie de {user.identifier}**\nRecherche de vos images...", author="ELA 🤖")
        await msg.send()

        query = text("""
            SELECT e.url, e.name 
            FROM elements e
            JOIN threads t ON e."threadId" = t.id
            WHERE t."userIdentifier" = :user_id 
            AND e.mime LIKE 'image/%'
        """)
        
        images_found = []
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(query, {"user_id": user.identifier})
            rows = result.fetchall()
            
            for row in rows:
                if row[0] and os.path.exists(row[0]):
                    images_found.append(
                        cl.Image(path=row[0], name=row[1], display="inline")
                    )
        
        if not images_found:
            msg.content = "Aucune image trouvée dans votre historique."
            await msg.update()
        else:
            msg.content = f"Voici les **{len(images_found)} images** retrouvées dans vos archives :"
            msg.elements = images_found
            await msg.update()
            
    except Exception as e:
        # En cas d'erreur, on l'affiche mais on ne bloque pas tout
        print(f"Erreur Galerie : {str(e)}")
        await cl.Message(content=f"❌ Une erreur est survenue : {str(e)}").send()
        
    finally:
        await engine.dispose()

# (Optionnel) Fonction pour renommer automatiquement le chat après le 1er message
@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "ELA", "Assistant": "ELA 🤖"}
    return rename_dict.get(orig_author, orig_author)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Mes images",
            message="/gallery",
            icon="/public/picture.svg",
        ),
        # cl.Starter(
        #     label="Etudier",
        #     message="Qu'est-ce qu'un modèle AR(1) ?",
        #     icon="/public/study.svg",
        # ),
        cl.Starter(
            label="Générer un quiz",
            message="/start_quiz",
            icon="/public/quiz.svg",
        ),
        cl.Starter(
            label="Atelier Code",
            message="/code_workshop",
            icon="/public/terminal.svg",
        )

    ]
    
# --- FONCTIONS UTILITAIRES POUR LE QUIZ ---
async def ask_next_question():
    """Affiche la question actuelle sous forme de message avec boutons."""
    quiz_data = cl.user_session.get("quiz_data")
    index = cl.user_session.get("quiz_index")
    
    if index >= len(quiz_data):
        score = cl.user_session.get("quiz_score")
        await cl.Message(content=f"🏁 **Quiz terminé !**\nVotre score : {score}/{len(quiz_data)}\n\nPosez une autre question ou tapez `/start_quiz` pour recommencer.").send()
        cl.user_session.set("quiz_mode", None)
        return

    q = quiz_data[index]
    
    actions = []
    letters = ["A", "B", "C", "D"]
    num_options = min(len(q["options"]), 4) 

    for i in range(num_options):
        raw_option = q["options"][i]
        
        # --- CORRECTION DOUBLES LETTRES ---
        # On utilise une expression régulière (Regex) pour supprimer "A)", "A.", "1." au début
        # Cela garde juste le texte de la réponse.
        clean_option = re.sub(r'^[A-D0-9][\)\.]\s*', '', raw_option).strip()
        
        actions.append(
            cl.Action(
                name="quiz_answer",
                payload={"value": str(i)}, # On stocke bien la valeur ici
                label=f"{letters[i]}) {clean_option}", 
                description="Cliquez pour choisir"
            )
        )

    await cl.Message(
        content=f"**Question {index + 1}/{len(quiz_data)}**\n\n{q['question']}",
        actions=actions
    ).send()
    
@cl.action_callback("quiz_answer")
async def on_quiz_answer(action: cl.Action):
    """Gère le clic sur un bouton de réponse."""
    quiz_data = cl.user_session.get("quiz_data")
    index = cl.user_session.get("quiz_index")
    score = cl.user_session.get("quiz_score")
    
    # --- CORRECTION DU CRASH ---
    # Au lieu de action.value, on lit le payload
    user_idx = int(action.payload["value"])
    
    current_q = quiz_data[index]
    correct_idx = current_q["correct_index"]
    
    # Feedback
    if user_idx == correct_idx:
        score += 1
        cl.user_session.set("quiz_score", score)
        feedback = f"✅ **Correct !**\n_{current_q['explanation']}_"
    else:
        letters = ["A", "B", "C", "D"]
        # On nettoie aussi l'affichage de la bonne réponse ici pour éviter les doublons
        raw_correct = current_q['options'][correct_idx]
        clean_correct = re.sub(r'^[A-D0-9][\)\.]\s*', '', raw_correct).strip()
        
        feedback = f"❌ **Incorrect.**\nLa bonne réponse était **{letters[correct_idx]}** : {clean_correct}.\n\n_{current_q['explanation']}_"

    # Supprime les boutons précédents
    await action.remove()
    
    await cl.Message(content=feedback).send()
    
    # Question suivante
    cl.user_session.set("quiz_index", index + 1)
    await asyncio.sleep(1)
    await ask_next_question()
    

async def rename_current_thread(new_name: str):
    """Renomme la conversation actuelle dans la base de données."""
    thread_id = cl.context.session.thread_id
    if not thread_id:
        return

    db_url = os.getenv("DATABASE_URL")
    # Création moteur temporaire
    engine = create_async_engine(db_url)
    AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    try:
        async with AsyncSessionLocal() as session:
            query = text('UPDATE threads SET name = :name WHERE id = :id')
            await session.execute(query, {"name": new_name, "id": thread_id})
            await session.commit()
    except Exception as e:
        print(f"⚠️ Erreur renommage thread : {e}")
    finally:
        await engine.dispose()
        
@cl.action_callback("code_lang")
async def on_code_lang(action: cl.Action):
    lang = action.payload["value"]
    cl.user_session.set("code_lang_choice", lang)
    cl.user_session.set("code_mode", "waiting_topic")
    
    await action.remove()
    
    await cl.Message(content=f"C'est noté pour **{lang}** !\n\nQuel modèle ou concept voulez-vous implémenter ? (ex: *MCO, VAR, ARCH, Test de Student...*)").send()