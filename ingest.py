import os
import re
import glob
import shutil
from typing import List

# --- IMPORTS LANGCHAIN ---
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
SOURCE_DIR = "./latex"       # Dossier contenant vos .tex
DB_PATH = "./chroma_db"      # Dossier de la base vectorielle
# Modèle d'embedding performant pour le français/anglais technique
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- CLASSE PARSER LATEX ---
class LatexIngestor:
    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    def clean_latex_content(self, text: str) -> str:
        """
        Nettoie le balisage LaTeX pour ne garder que le sens mathématique et textuel.
        """
        # 1. Supprimer les commentaires (sauf si échappés \%)
        text = re.sub(r'(?<!\\)%.*', '', text)
        
        # 2. Simplifier les inclusions d'images
        text = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', '[FIGURE]', text)
        
        # 3. Supprimer les commandes de mise en page pure (bruit)
        commands_to_remove = [
            r'\\centering', r'\\medskip', r'\\bigskip', r'\\small', 
            r'\\footnotesize', r'\\tiny', r'\\vfill', r'\\newpage', 
            r'\\tableofcontents', r'\\pause'
        ]
        for cmd in commands_to_remove:
            text = re.sub(cmd, '', text)
            
        # 4. Remplacer les itemize par des puces pour aider le LLM
        text = text.replace(r'\item', '\n* ')
        
        # 5. Nettoyer les espaces multiples
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    def parse_file(self, file_path: str) -> List[Document]:
        """Découpe un fichier .tex en documents basés sur les Frames (Slides)"""
        filename = os.path.basename(file_path)
        print(f"   📖 Lecture de : {filename}")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        file_docs = []
        
        # Découpage par SECTIONS (\section{Title})
        # Capture le titre court optionnel [..] et le titre principal {..}
        section_pattern = re.compile(r'\\section(?:\[.*?\])?\{(.*?)\}', re.DOTALL)
        parts = section_pattern.split(content)
        
        # Le premier morceau est l'intro (avant la 1ère section)
        current_section = "Introduction / Général"
        if parts[0].strip():
            self._extract_frames(parts[0], filename, current_section, file_docs)

        # Ensuite on alterne : Titre Section -> Contenu -> Titre -> Contenu
        for i in range(1, len(parts), 2):
            section_title = parts[i].strip()
            section_content = parts[i+1]
            self._extract_frames(section_content, filename, section_title, file_docs)
            
        return file_docs

    def _extract_frames(self, content: str, filename: str, section: str, docs_list: List[Document]):
        """Extrait chaque environnement frame"""
        # Regex pour capturer \begin{frame}{Titre}... \end{frame}
        frame_pattern = re.compile(r'\\begin\{frame\}(?:\[.*?\])?(?:\{(.*?)\})?(.*?)\\end\{frame\}', re.DOTALL)
        
        matches = list(frame_pattern.finditer(content))
        
        if not matches and content.strip():
            # Cas où il n'y a pas de frames (ex: syllabus ou article), on prend le bloc entier
            clean_body = self.clean_latex_content(content)
            if len(clean_body) > 10:
                doc = Document(
                    page_content=f"SECTION: {section}\nCONTENU:\n{clean_body}",
                    metadata={"source": filename, "section": section, "type": "text_block"}
                )
                docs_list.append(doc)
            return

        for match in matches:
            frame_title = match.group(1) if match.group(1) else "Sans titre"
            frame_body = match.group(2)
            clean_body = self.clean_latex_content(frame_body)
            
            # Contexte enrichi pour le RAG
            full_content = (
                f"SOURCE: {filename}\n"
                f"SECTION: {section}\n"
                f"TITRE SLIDE: {frame_title}\n"
                f"CONTENU:\n{clean_body}"
            )
            
            doc = Document(
                page_content=full_content,
                metadata={
                    "source": filename,
                    "section": section,
                    "slide_title": frame_title,
                    "type": "slide"
                }
            )
            docs_list.append(doc)

    def load_documents(self) -> List[Document]:
        tex_files = glob.glob(os.path.join(self.source_dir, "*.tex"))
        all_docs = []
        print(f"📂 Recherche dans {self.source_dir}...")
        for tex_file in tex_files:
            all_docs.extend(self.parse_file(tex_file))
        return all_docs

# --- FONCTION PRINCIPALE ---
def main():
    # 1. Nettoyage de l'ancienne base pour éviter les doublons
    if os.path.exists(DB_PATH):
        print(f"🗑️  Suppression de l'ancienne base : {DB_PATH}")
        shutil.rmtree(DB_PATH)

    # 2. Chargement et Parsing Latex
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"⚠️ Dossier '{SOURCE_DIR}' créé. Mettez vos .tex dedans et relancez !")
        return

    ingestor = LatexIngestor(SOURCE_DIR)
    raw_docs = ingestor.load_documents()

    if not raw_docs:
        print("❌ Aucun document trouvé ou extrait. Vérifiez vos fichiers .tex.")
        return

    print(f"✅ {len(raw_docs)} slides/segments extraits.")

    # 3. Sécurité : Splitter secondaire (si un slide est vraiment trop long)
    # On garde une taille large car on veut préserver l'unité du slide
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(raw_docs)
    
    # 4. Génération des Embeddings et Stockage Chroma
    print("🧠 Calcul des vecteurs (Embeddings)... Patientez.")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print(f"🎉 Ingestion terminée ! Base vectorielle créée dans '{DB_PATH}'")

if __name__ == "__main__":
    main()