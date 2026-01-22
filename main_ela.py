import os
import sys
from dotenv import load_dotenv

# --- 1. CORE IMPORTS ---
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 2. INTEGRATIONS ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- 3. RETRIEVERS (from langchain-classic) ---
# For LangChain v1, these are in langchain-classic
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# from langchain_classic.retrievers import ContextualCompressionRetriever

# --- 4. RERANKER (FlashRank is simpler than cross-encoder) ---
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    print("⚠️  FlashRank not installed. Install with: pip install flashrank")
    FLASHRANK_AVAILABLE = False


# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
load_dotenv()

class FlashRankCompressor:
    """
    Custom compressor using FlashRank for reranking documents.
    FlashRank is faster and simpler than sentence-transformers cross-encoder.
    """
    
    def __init__(self, top_n: int = 5):
        """
        Initialize FlashRank reranker.
        
        Args:
            top_n: Number of top documents to return after reranking
        """
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        self.top_n = top_n
    
    def compress_documents(self, documents: list[Document], query: str) -> list[Document]:
        """
        Rerank documents using FlashRank.
        
        Args:
            documents: List of retrieved documents
            query: Original query string
            
        Returns:
            List of reranked documents (top_n)
        """
        if not documents:
            return []
        
        # Prepare passages for reranking
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        
        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Return top_n documents
        reranked_docs = []
        for result in results[:self.top_n]:
            original_doc = documents[result["id"]]
            reranked_docs.append(original_doc)
        
        return reranked_docs


class ELA_Bot:
    """
    ELA (Econometrics Learning Assistant) - RAG chatbot with hybrid retrieval and reranking.
    """
    
    def __init__(self):
        """Initialize the ELA bot with ChromaDB, retrievers, and LLM."""
        print("🤖 Initialisation d'ELA AI...")
        
        # A. Load vector database
        print("   📚 Chargement de la mémoire vectorielle...")
        if not os.path.exists(DB_PATH):
            print(f"❌ Erreur : Le dossier '{DB_PATH}' n'existe pas.")
            print("   -> Lancez 'python ingest.py' d'abord.")
            sys.exit(1)
        
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_db = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=self.embedding_model
        )
        
        # B. Build retrieval system
        print("   🔍 Construction du Retriever Hybride...")
        self.retriever = self._build_retrievers()
        
        # C. Connect to LLM
        print("   🧠 Connexion au cerveau (Groq)...")
        if "GROQ_API_KEY" not in os.environ:
            print("⚠️  GROQ_API_KEY non définie. Définissez-la avant de continuer.")
            sys.exit(1)
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,  # Mathematical rigor
            max_tokens=2048
        )
        
        # D. Build RAG chain
        self.rag_chain = self._build_chain()
        print("✅ ELA AI est prêt !")

    def _build_retrievers(self):
        """
        Build hybrid retrieval system with reranking.
        
        Architecture:
        1. Retrieve 20 docs: BM25 (keywords) + Chroma (semantic)
        2. Rerank to top 5: FlashRank
        
        Returns:
            Final retriever with reranking
        """
        # 1. Vector retriever (semantic search)
        chroma_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 20}
        )
        
        # 2. BM25 retriever (keyword search)
        print("   📖 Indexation BM25...")
        try:
            all_docs_data = self.vector_db.get()
            docs_list = [
                Document(page_content=txt, metadata=meta)
                for txt, meta in zip(
                    all_docs_data['documents'], 
                    all_docs_data['metadatas']
                )
            ]
        except Exception as e:
            print(f"❌ Erreur chargement documents pour BM25 : {e}")
            sys.exit(1)
        
        if not docs_list:
            print("❌ La base de données est vide.")
            sys.exit(1)
        
        bm25_retriever = BM25Retriever.from_documents(docs_list)
        bm25_retriever.k = 20
        
        # 3. Ensemble retriever (fusion)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        
        # 4. Add reranking
        if FLASHRANK_AVAILABLE:
            print("   ⚡ Ajout du reranking FlashRank...")
            compressor = FlashRankCompressor(top_n=5)
            
            # Custom wrapper to pass query to compressor
            class CompressorWrapper:
                def __init__(self, compressor, base_retriever):
                    self.compressor = compressor
                    self.base_retriever = base_retriever
                
                def invoke(self, query):
                    docs = self.base_retriever.invoke(query)
                    return self.compressor.compress_documents(docs, query)
            
            return CompressorWrapper(compressor, ensemble_retriever)
        else:
            print("   ⚠️  Reranking désactivé (FlashRank non disponible)")
            return ensemble_retriever

    def _build_chain(self):
        """
        Build the RAG chain with expert prompt.
        
        Returns:
            Complete RAG chain (retriever + prompt + LLM)
        """
        template = """Tu es ELA (Econometrics Learning Assistant), un assistant expert pour doctorants en économétrie.
Ta tâche est d'expliquer des concepts mathématiques en te basant UNIQUEMENT sur les extraits de cours fournis.

RÈGLES IMPORTANTES :
1. **Maths & Formatage (CRUCIAL)** : 
   - Pour les expressions mathématiques dans le texte, utilise les délimiteurs `$` (dollars simples). Exemple : "Soit $\\alpha$ le paramètre".
   - Pour les équations importantes ou centrées, utilise les délimiteurs `$$` (dollars doubles). Exemple :
    $$ Y_t = \\beta X_t + \\epsilon_t $$
2. **Sources** : Cite l'origine de tes informations sous la forme [Source: Fichier.tex, Slide: Titre].
3. **Incertitude** : Si le contexte ne contient pas la réponse, dis "Je ne trouve pas cette information dans vos cours". Ne l'invente pas.
4. **Style** : Sois pédagogique, précis et rigoureux.

CONTEXTE :
{context}

QUESTION : 
{question}

RÉPONSE :"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            """Format retrieved documents with metadata."""
            formatted = []
            for doc in docs:
                src = doc.metadata.get('source', 'Inconnu')
                slide = doc.metadata.get('slide_title', 'Sans titre')
                content = doc.page_content.replace("\n", " ")
                formatted.append(
                    f">> [Source: {src} | Slide: {slide}]\n{content}"
                )
            return "\n\n".join(formatted)

        # Build chain
        chain = (
            {
                "context": lambda x: format_docs(self.retriever.invoke(x)),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain

    def ask(self, question: str) -> str:
        """
        Ask a question to ELA.
        
        Args:
            question: User question
            
        Returns:
            ELA's response
        """
        print(f"\n🤔 ELA réfléchit à : '{question}'...")
        try:
            response = self.rag_chain.invoke(question)
            return response
        except Exception as e:
            return f"❌ Erreur : {e}"


# --- MAIN LOOP ---
if __name__ == "__main__":
    try:
        bot = ELA_Bot()
        print("\n" + "="*60)
        print("🎓 ELA AI - Assistant Économétrie (Hybrid RAG + Rerank)")
        print("   Tapez 'q' pour quitter.")
        print("="*60)

        while True:
            q = input("\n💬 Vous : ")
            if q.lower() in ['q', 'quit', 'exit']:
                print("👋 Au revoir !")
                break
            
            if not q.strip():
                continue
            
            resp = bot.ask(q)
            print(f"\n🤖 ELA :\n{resp}")
            print("-" * 60)
            
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur au démarrage : {e}")
        import traceback
        traceback.print_exc()