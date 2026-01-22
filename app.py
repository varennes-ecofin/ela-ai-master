import chainlit as cl

from main_ela import ELA_Bot

# chainlit run app.py -w 

# --- 1. CONFIGURATION DE L'AUTHENTIFICATION ---
USERS = {
    "etudiant": "master_esa_2025",
    "professeur": "admin123"
}

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