# main_ela.py
import os
import base64
from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Configuration
load_dotenv()
DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Vérification FlashRank
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

class FlashRankCompressor:
    """Custom compressor using FlashRank for reranking documents."""
    def __init__(self, top_n: int = 5):
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        self.top_n = top_n
    
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents: 
            return []
        passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(documents)]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        return [documents[result["id"]] for result in results[:self.top_n]]

class ELA_Bot:
    """
    ELA (Econometrics Learning Assistant) - Version Stateless pour Chainlit DataLayer.
    """
    
    def __init__(self):
        print("🤖 Initialisation du moteur RAG ELA...")
        
        if not os.path.exists(DB_PATH):
            print(f"❌ Erreur : Le dossier '{DB_PATH}' n'existe pas.")
            # En prod, on pourrait lever une erreur, ici on print juste
        
        # 1. Embeddings & VectorDB
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_db = Chroma(persist_directory=DB_PATH, embedding_function=self.embedding_model)
        
        # 2. Retrievers
        self.retriever = self._build_retrievers()
        
        # 3. LLM
        if "GROQ_API_KEY" not in os.environ:
            print("⚠️ GROQ_API_KEY non définie.")
        
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            max_tokens=2048
        )
        
        # 4. Chain
        self.rag_chain = self._build_chain()
        print("✅ Moteur ELA prêt !")

    def _build_retrievers(self):
        chroma_retriever = self.vector_db.as_retriever(search_kwargs={"k": 20})
        
        # BM25 nécessite les documents bruts
        all_docs_data = self.vector_db.get()
        docs_list = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(all_docs_data['documents'], all_docs_data['metadatas'])
        ]
        
        # Sécurité si la DB est vide
        if not docs_list:
            return chroma_retriever

        bm25_retriever = BM25Retriever.from_documents(docs_list)
        bm25_retriever.k = 20
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        
        if FLASHRANK_AVAILABLE:
            compressor = FlashRankCompressor(top_n=5)
            class CompressorWrapper:
                def __init__(self, compressor, base_retriever):
                    self.compressor = compressor
                    self.base_retriever = base_retriever
                def invoke(self, query):
                    docs = self.base_retriever.invoke(query)
                    return self.compressor.compress_documents(docs, query)
            return CompressorWrapper(compressor, ensemble_retriever)
        
        return ensemble_retriever

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es ELA (Econometrics Learning Assistant), un assistant expert pour étudiants de master et strictement limité au contenu pédagogique fourni.
DIRECTIVE CRITIQUE :
Tu ne possèdes AUCUNE connaissance en dehors des informations fournies ci-dessous (Contexte et Image).
Tu es amnésique concernant l'histoire, la géographie ou la culture générale.

RÈGLES ABSOLUES :
1. Tu dois répondre en utilisant **UNIQUEMENT** les informations présentes dans le CONTEXTE DU COURS ci-dessous ou dans l'image fournie.
2. **Maths** : Utilise `$...$` (inline) et `$$...$$` (bloc).
3. **Sources** : Cite [Source: Fichier.tex, Slide: Titre].
4. **Incertitude** : Si la réponse à la question n'est pas explicitement dans le contexte ou l'image, tu dois dire : "Je ne trouve pas cette information dans vos documents de cours."
5. **Conversation** : Utilise UNIQUEMENT l'historique de conversation, les informations présentes dans le CONTEXTE DU COURS ci-dessous ou dans l'image fournie.
6. **Contexte** : N'utilise JAMAIS tes connaissances externes pour combler un manque d'information (pas d'hallucination).
7. **Style** : Pédagogique, précis, rigoureux.
CONTEXTE DU COURS (Source unique) :
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        def format_docs(docs):
            formatted = []
            for doc in docs:
                src = doc.metadata.get('source', 'Inconnu')
                slide = doc.metadata.get('slide_title', 'Sans titre')
                # Nettoyage basique
                content = doc.page_content.replace("\n", " ")
                formatted.append(f">> [Source: {src} | Slide: {slide}]\n{content}")
            return "\n\n".join(formatted)

        # On utilise une méthode simple pour combiner le contexte et la question
        chain = (
            {
                "context": lambda x: format_docs(self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    # NOUVELLE MÉTHODE pour préparer le contexte texte (RAG)
    def _get_rag_context(self, question: str):
        docs = self.retriever.invoke(question)
        formatted = []
        for doc in docs:
            src = doc.metadata.get('source', 'Inconnu')
            content = doc.page_content.replace("\n", " ")
            formatted.append(f">> [Source: {src}]\n{content}")
        return "\n\n".join(formatted)

    # MODIFICATION MAJEURE de la méthode ask
    async def ask(self, question: str, chat_history: list = None, image_path: str = None) -> str:
        if chat_history is None: 
            chat_history = []
        
        try:
            # 1. Récupérer le contexte RAG (Textuel)
            context_text = self._get_rag_context(question)
            
            # 2. Préparer le message Système
            system_prompt = f"""Tu es ELA, assistant expert en économétrie.
            RÈGLES :
            1. Analyse l'image fournie si présente (graphique, équation, tableau).
            2. Utilise le CONTEXTE DU COURS ci-dessous pour t'aider.
            3. Réponds en français avec rigueur mathématique (LaTeX pour les formules).
            
            CONTEXTE DU COURS :
            {context_text}"""

            messages = [SystemMessage(content=system_prompt)]
            
            # Ajouter l'historique de conversation
            messages.extend(chat_history)

            # 3. Construire le message Utilisateur (Texte + Image potentielle)
            content_blocks = [{"type": "text", "text": question}]
            
            if image_path:
                # Encodage de l'image en Base64 pour l'API Groq
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                })
                print("🖼️ Image détectée et envoyée à Groq Vision")

            # Ajouter le message utilisateur final
            messages.append(HumanMessage(content=content_blocks))

            # 4. Appel direct au LLM (On contourne la chaine rigide pour la flexibilité Vision)
            response = await self.llm.ainvoke(messages)
            
            return response.content
            
        except Exception as e:
            return f"❌ Erreur ELA Vision : {str(e)}"