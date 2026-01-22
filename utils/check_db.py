import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def check_database():
    print(f"🕵️‍♂️ Inspection de la base vectorielle : {DB_PATH}")

    if not os.path.exists(DB_PATH):
        print("❌ Erreur : Le dossier 'chroma_db' n'existe pas. Lancez ingest.py d'abord.")
        return

    # 1. Chargement de la base
    print("   Chargement du modèle d'embedding (identique à l'ingestion)...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    except Exception as e:
        print(f"❌ Erreur critique au chargement : {e}")
        return

    # 2. Vérification du volume
    # On récupère tous les IDs pour compter
    collection_data = db.get() 
    count = len(collection_data['ids'])
    
    print("\n📊 STATISTIQUES :")
    print(f"   Nombre total de segments (slides) : {count}")
    
    if count == 0:
        print("⚠️ Attention : La base est vide !")
        return

    # 3. Test de récupération (Sanity Check)
    # On pose une question technique pour voir si on remonte le bon cours
    query = "stationnarité faible"
    print(f"\n🔍 TEST DE RECHERCHE : '{query}'")
    print("   Recherche des 3 segments les plus proches...")
    
    results = db.similarity_search(query, k=3)

    print("-" * 40)
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"RESULTAT #{i+1}")
        print(f"📂 Source     : {meta.get('source', 'N/A')}")
        print(f"📑 Section    : {meta.get('section', 'N/A')}")
        print(f"📺 Slide      : {meta.get('slide_title', 'N/A')}")
        print(f"📄 Type       : {meta.get('type', 'N/A')}")
        print(f"📝 Extrait    : {doc.page_content[:150].replace(chr(10), ' ')}...") # On affiche les 150 premiers caractères
        print("-" * 40)

    print("\n✅ Si les extraits ci-dessus correspondent à de l'économétrie et que les métadonnées sont justes, tout est OK !")

if __name__ == "__main__":
    check_database()