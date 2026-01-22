C'est une excellente idée. Claude, avec son accès aux outils (MCP) et à la documentation dynamique, sera très efficace pour résoudre ce "puzzle" des imports LangChain qui changent souvent entre les versions `core`, `community` et `partners`.

Voici le document Markdown complet et structuré. Vous pouvez enregistrer ce contenu dans un fichier nommé **`PROJECT_CONTEXT.md`** et le donner à Claude.

---

# Documentation Technique : Projet ELA AI (Econometrics Learning Assistant)

## 1. Contexte du Projet

**ELA AI** est un assistant pédagogique RAG (Retrieval-Augmented Generation) spécialisé en économétrie niveau Master/PhD.
Contrairement aux RAGs classiques basés sur des PDF, ce projet ingère directement des sources **LaTeX (`.tex`)** pour préserver la sémantique mathématique et la structure des cours (Slides/Beamer).

**Objectif actuel :** Faire tourner le script du Chatbot (`main_bot.py`) en local avec une architecture RAG avancée (Recherche Hybride + Re-ranking).

## 2. Stack Technique

* **Langage :** Python 3.10+
* **Orchestration :** LangChain (Versions récentes v0.2/v0.3 avec séparation `core`/`community`).
* **LLM :** Groq (Modèle `llama-3.3-70b-versatile`).
* **Vector Store :** ChromaDB (Local).
* **Embeddings :** HuggingFace (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`).
* **Recherche :** Hybride (BM25 + Chroma Vector) + Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`).

## 3. Structure du Dossier

```text
ela_ai_project/
├── chroma_db/         # Base vectorielle générée (contient ~1100 slides)
├── latex/             # Fichiers sources .tex
├── ingest.py          # Script d'ingestion (FONCTIONNEL)
├── check_db.py        # Script de diagnostic (FONCTIONNEL)
└── main_bot.py        # Script du Chatbot (PROBLÈMES D'IMPORTS LANGCHAIN)

```

## 4. Codes Actuels

### A. Script d'Ingestion (`ingest.py`) - **État : OK**

Ce script parse les fichiers LaTeX, nettoie le bruit visuel, découpe par "Frame/Slide" et crée les embeddings.

```python
import os
import re
import glob
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCE_DIR = "./latex"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

class LatexIngestor:
    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    def clean_latex_content(self, text: str) -> str:
        text = re.sub(r'(?<!\\)%.*', '', text)
        text = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', '[FIGURE]', text)
        commands_to_remove = [r'\\centering', r'\\medskip', r'\\bigskip', r'\\small', r'\\footnotesize', r'\\tiny', r'\\vfill', r'\\newpage', r'\\tableofcontents', r'\\pause']
        for cmd in commands_to_remove:
            text = re.sub(cmd, '', text)
        text = text.replace(r'\item', '\n* ')
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    def parse_file(self, file_path: str) -> List[Document]:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        file_docs = []
        section_pattern = re.compile(r'\\section(?:\[.*?\])?\{(.*?)\}', re.DOTALL)
        parts = section_pattern.split(content)
        current_section = "Introduction / Général"
        if parts[0].strip():
             self._extract_frames(parts[0], filename, current_section, file_docs)
        for i in range(1, len(parts), 2):
            section_title = parts[i].strip()
            section_content = parts[i+1]
            self._extract_frames(section_content, filename, section_title, file_docs)
        return file_docs

    def _extract_frames(self, content: str, filename: str, section: str, docs_list: List[Document]):
        frame_pattern = re.compile(r'\\begin\{frame\}(?:\[.*?\])?(?:\{(.*?)\})?(.*?)\\end\{frame\}', re.DOTALL)
        matches = list(frame_pattern.finditer(content))
        if not matches and content.strip():
            clean_body = self.clean_latex_content(content)
            if len(clean_body) > 10:
                doc = Document(page_content=f"SECTION: {section}\nCONTENU:\n{clean_body}", metadata={"source": filename, "section": section, "type": "text_block"})
                docs_list.append(doc)
            return
        for match in matches:
            frame_title = match.group(1) if match.group(1) else "Sans titre"
            frame_body = match.group(2)
            clean_body = self.clean_latex_content(frame_body)
            full_content = f"SOURCE: {filename}\nSECTION: {section}\nTITRE SLIDE: {frame_title}\nCONTENU:\n{clean_body}"
            doc = Document(page_content=full_content, metadata={"source": filename, "section": section, "slide_title": frame_title, "type": "slide"})
            docs_list.append(doc)

    def load_documents(self) -> List[Document]:
        tex_files = glob.glob(os.path.join(self.source_dir, "*.tex"))
        all_docs = []
        for tex_file in tex_files:
            all_docs.extend(self.parse_file(tex_file))
        return all_docs

def main():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    ingestor = LatexIngestor(SOURCE_DIR)
    raw_docs = ingestor.load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(raw_docs)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=DB_PATH)

if __name__ == "__main__":
    main()

```

### B. Script du Bot (`main_bot.py`) - **État : Erreurs d'Imports**

C'est ici que j'ai besoin d'aide. Je veux implémenter une recherche hybride (BM25 + Vector) suivie d'un Reranking (Cross-Encoder).
Les imports LangChain semblent conflictuels ou dépréciés selon les versions installées (`langchain`, `langchain-community`, `langchain-core`).

**Voici la logique souhaitée (Draft Code) :**

```python
import os
import sys

# --- IMPORTS À VÉRIFIER/CORRIGER ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# C'est ici que ça casse souvent : localisation de BM25, Ensemble, CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class ELA_Bot:
    def __init__(self):
        # 1. Load Chroma
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_db = Chroma(persist_directory=DB_PATH, embedding_function=self.embedding_model)
        
        # 2. Build Retrievers
        self.retriever = self._build_retrievers()
        
        # 3. LLM Groq
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
        
        # 4. Chain
        self.rag_chain = self._build_chain()

    def _build_retrievers(self):
        # Vector Retriever
        chroma_retriever = self.vector_db.as_retriever(search_kwargs={"k": 20})
        
        # BM25 Retriever (from documents in memory)
        all_docs_data = self.vector_db.get()
        docs_list = [Document(page_content=txt, metadata=meta) for txt, meta in zip(all_docs_data['documents'], all_docs_data['metadatas'])]
        bm25_retriever = BM25Retriever.from_documents(docs_list)
        bm25_retriever.k = 20
        
        # Ensemble
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5])
        
        # Reranking
        model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
        compressor = CrossEncoderReranker(model=model, top_n=5)
        final_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        
        return final_retriever

    def _build_chain(self):
        template = """Context: {context} \n Question: {question} \n Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        return ({"context": self.retriever | format_docs, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser())

    def ask(self, question):
        return self.rag_chain.invoke(question)

if __name__ == "__main__":
    bot = ELA_Bot()
    while True:
        q = input("Question: ")
        print(bot.ask(q))

```

## 5. Demande pour Claude

Utilise ton MCP LangChain ou tes connaissances à jour pour :

1. Identifier les **imports exacts et valides** pour `BM25Retriever`, `EnsembleRetriever`, `CrossEncoderReranker` et `HuggingFaceCrossEncoder` dans la version la plus récente de LangChain.
2. Réécrire le fichier `main_bot.py` pour qu'il soit exécutable sans erreur d'import (`ImportError` ou `ModuleNotFoundError`).
3. Vérifier la compatibilité des packages `langchain-community` vs `langchain` core.