import os
from dotenv import load_dotenv
# ... imports ...
load_dotenv()

# --- DIAGNOSTIC TEMPORAIRE (A SUPPRIMER ENSUITE) ---
api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    print(f"✅ Clé trouvée : {api_key[:4]}...{api_key[-4:]}") # Affiche début/fin pour vérifier
    print(f"📏 Longueur de la clé : {len(api_key)}")
else:
    print("❌ AUCUNE CLÉ GROQ TROUVÉE DANS L'ENVIRONNEMENT")
# ---------------------------------------------------
