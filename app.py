import os
import chainlit as cl

from main_ela import ELA_Bot

# chainlit run app.py -w 

def load_users_from_env():
    """
    Charge les utilisateurs depuis la variable d'env ELA_AUTH_DATA.
    Format attendu : "user1:pass1,user2:pass2"
    """
    users_dict = {}
    # On récupère la chaîne brute, ou une chaîne vide si elle n'existe pas
    raw_data = os.environ.get("ELA_AUTH_DATA", "")
    
    if not raw_data:
        print("⚠️ AVERTISSEMENT : Aucun utilisateur configuré dans .env")
        return users_dict

    # On découpe par virgule pour avoir chaque couple
    pairs = raw_data.split(",")
    
    for pair in pairs:
        # On découpe chaque couple par les deux-points
        if ":" in pair:
            # .strip() enlève les espaces accidentels
            username, password = pair.split(":", 1) 
            users_dict[username.strip()] = password.strip()
            
    return users_dict

# --- 1. CONFIGURATION DE L'AUTHENTIFICATION ---
USERS = load_users_from_env()

@cl.author_rename
def rename(orig_author: str):
    """Renomme l'assistant dans l'interface"""
    rename_dict = {
        "LLMMathChain": "ELA",
        "Chatbot": "ELA"
    }
    return rename_dict.get(orig_author, orig_author)

@cl.password_auth_callback
def auth_callback(username, password):
    """
    Fonction appelée lors du login.
    """
    if username in USERS and USERS[username] == password:
        return cl.User(identifier=username)
    return None

# --- 2. DÉMARRAGE D'UNE SESSION ---
@cl.on_chat_start
async def start():
    """
    S'exécute à chaque fois qu'un utilisateur ouvre une session.
    """
    msg = cl.Message(content="🚀 Initialisation d'ELA AI... Chargement des cours...", author="Système")
    await msg.send()
    
    try:
        ela_instance = ELA_Bot()
        cl.user_session.set("ela_bot", ela_instance)
        
        user = cl.user_session.get('user')
        
        msg.content = f"Bonjour {user.identifier} ! Je suis ELA, votre assistante en économétrie. Posez-moi une question sur vos cours."
        msg.author = "ELA 🤖"
        await msg.update()
        
    except Exception as e:
        msg.content = f"❌ Erreur critique lors du chargement : {str(e)}"
        await msg.update()

# --- 3. RÉCEPTION D'UN MESSAGE ---
@cl.on_message
async def main(message: cl.Message):
    """
    S'exécute quand l'utilisateur envoie une question.
    """
    bot = cl.user_session.get("ela_bot")
    msg = cl.Message(content="", author="ELA 🤖")
    
    try:
        response = await cl.make_async(bot.ask)(message.content)
        msg.content = response
        await msg.send()
        
    except Exception as e:
        msg.content = f"Oups, une erreur est survenue : {str(e)}"
        await msg.send()