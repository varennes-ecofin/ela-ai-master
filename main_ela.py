# main_ela.py
"""
ELA (Econometrics Learning Assistant) — RAG engine with level-scoped retrieval.

Phase 4: The retriever filters ChromaDB and BM25 results based on the
user's academic level (M1, M2, ALL). Documents tagged 'Commun' or 'legacy'
are always included.
"""

import os
import json
import base64
from dotenv import load_dotenv
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False

from rag_guard import RAGGuard

# Configuration
load_dotenv()
DB_PATH = "./data/chroma_db"   # au lieu de "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Levels that grant access to ALL documents (no filtering)
_UNFILTERED_LEVELS = {"ALL", "admin"}

# Levels always included regardless of user level
_UNIVERSAL_LEVELS = {"Commun", "legacy"}


# --- SYSTEM PROMPTS ---

ELA_BASE_INSTRUCTIONS = """
Tu es ELA (Econometrics Learning Assistant), un assistant expert pédagogique.

PROTOCOL DE CITATION STRICT (OBLIGATOIRE) :
Tu dois impérativement indiquer l'origine de chaque information donnée. Distingue visuellement les sources :

1. **SOURCE DOCUMENT (Priorité Absolue)** : 
    - Si l'information vient du CONTEXTE DU COURS, termine la phrase/paragraphe par : `[Source: Nom_Du_Fichier]`.
    - Ne paraphrase pas sans citer.

2. **SOURCE EXTERNE (Complément IA)** :
    - Si tu utilises tes propres connaissances (autorisé UNIQUEMENT selon les règles du MODE EXPERT ci-dessous), termine le paragraphe par : `[Source: Connaissances Générales]`.
    - Si une réponse mixe les deux, chaque partie doit avoir son étiquette distincte.

DIRECTIVES DE COMPORTEMENT (MODE EXPERT) :

1. **Hiérarchie des Connaissances** :
    - **PRIORITÉ 1 (Le Cours)** : En PRIORITÉ ABSOLUE, base-toi sur le CONTEXTE DU COURS, les images et l'historique. Si le cours définit une notation ou une méthode spécifique, tu dois la suivre impérativement.
    - **PRIORITÉ 2 (Savoir Spécialisé)** : Si le contexte est muet, tu es autorisé à utiliser tes connaissances (en taguant `[Source: Connaissances Générales]`) UNIQUEMENT si le sujet concerne :
        * **Séries Temporelles** : ARIMA, VAR, VECM, Stationnarité, Cointégration, Racine Unitaire, Bruit Blanc...
        * **Économétrie Financière** : Volatilité (ARCH/GARCH), Rendements, Gestion des risques financiers.
        * **Code** : Syntaxe Python/R appliquée à ces sujets spécifiques.

2. **Frontières Strictes (Liste d'Exclusion)** :
    Tu dois REFUSER de traiter les sujets suivants s'ils ne sont pas dans le contexte :
    - **Micro-économétrie** : Panel, Logit/Probit, Tobit, IV (sauf si contexte Séries Temporelles).
    - **Machine Learning Généraliste** : Classification, Clustering, NLP, Vision.
    - **Culture Générale** : Histoire, Politique, etc.

   *Réaction* : Si l'utilisateur pose une question interdite, réponds : "Je suis spécialisé en Séries Temporelles et Économétrie Financière. Ce sujet sort du cadre du cours."

3. **Style & Format** :
    - Pédagogique, universitaire, rigoureux.
    - Utilise impérativement LaTeX : $...$ (inline) et $$...$$ (bloc).
"""

# Level-specific prompt supplements
_LEVEL_PROMPT_SUPPLEMENT = {
    "M1": (
        "\nNIVEAU ÉTUDIANT : Master 1 (M1).\n"
        "Adapte ton discours : rappelle les définitions fondamentales, "
        "explique les intuitions avant les formules, évite les raccourcis "
        "de notation. Privilégie les exemples concrets et les analogies."
    ),
    "M2": (
        "\nNIVEAU ÉTUDIANT : Master 2 (M2).\n"
        "Adapte ton discours : tu peux supposer les bases acquises, "
        "utiliser des notations compactes, faire des références croisées "
        "entre concepts (ex: lien entre cointégration et VECM). "
        "Privilégie la rigueur formelle et les subtilités théoriques."
    ),
}


def _get_level_instructions(user_level: str) -> str:
    """Return the level-specific prompt supplement.

    Args:
        user_level: The user's academic level (M1, M2, ALL, etc.).

    Returns:
        Additional prompt instructions, or empty string for ALL/admin.
    """
    return _LEVEL_PROMPT_SUPPLEMENT.get(user_level, "")


# --- CHROMA FILTER BUILDER ---

def _build_chroma_filter(user_level: str) -> Optional[dict]:
    """Build a ChromaDB `where` filter for level-scoped retrieval.

    Args:
        user_level: The user's academic level.

    Returns:
        A ChromaDB where-filter dict, or None if no filtering needed.
    """
    if user_level in _UNFILTERED_LEVELS:
        return None

    # Include user's level + universal levels
    allowed = [user_level] + list(_UNIVERSAL_LEVELS)
    return {"level": {"$in": allowed}}


def _filter_docs_by_level(docs: List[Document], user_level: str) -> List[Document]:
    """Filter a list of Documents by academic level.

    Used for BM25 results which don't support native metadata filtering.

    Args:
        docs: List of Documents to filter.
        user_level: The user's academic level.

    Returns:
        Filtered list of Documents.
    """
    if user_level in _UNFILTERED_LEVELS:
        return docs

    allowed = {user_level} | _UNIVERSAL_LEVELS
    return [
        doc for doc in docs
        if doc.metadata.get("level", "legacy") in allowed
    ]


# --- FLASHRANK COMPRESSOR ---

class FlashRankCompressor:
    """Custom compressor using FlashRank for reranking documents."""

    def __init__(self, top_n: int = 5):
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Rerank documents by relevance to the query.

        Args:
            documents: Candidate documents.
            query: The user's query.

        Returns:
            Top-N reranked documents.
        """
        if not documents:
            return []
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        return [documents[result["id"]] for result in results[:self.top_n]]


# --- SCOPED RETRIEVER WRAPPER ---

class ScopedRetriever:
    """Retriever that applies level filtering across Chroma + BM25 + reranking.

    Combines a ChromaDB retriever (with native where-filter) and a BM25
    retriever (with post-hoc filtering) via EnsembleRetriever, then
    optionally reranks with FlashRank.

    Args:
        vector_db: The Chroma vector store instance.
        user_level: Academic level for scoping (M1, M2, ALL).
        embedding_model: The embedding model (needed for Chroma).
    """

    def __init__(self, vector_db: Chroma, user_level: str):
        self.user_level = user_level
        self.vector_db = vector_db

        # 1. Chroma retriever with native level filter
        chroma_filter = _build_chroma_filter(user_level)
        search_kwargs = {"k": 20}
        if chroma_filter:
            search_kwargs["filter"] = chroma_filter

        self.chroma_retriever = vector_db.as_retriever(
            search_kwargs=search_kwargs,
        )

        # 2. BM25 retriever on level-filtered documents
        all_docs_data = vector_db.get()
        all_docs = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(
                all_docs_data["documents"], all_docs_data["metadatas"],
            )
        ]
        scoped_docs = _filter_docs_by_level(all_docs, user_level)

        self.bm25_retriever = None
        self.ensemble_retriever = None

        if scoped_docs:
            def case_insensitive_tokenizer(text):
                return text.lower().split()

            self.bm25_retriever = BM25Retriever.from_documents(
                scoped_docs, preprocess_func=case_insensitive_tokenizer,
            )
            self.bm25_retriever.k = 20

            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.chroma_retriever],
                weights=[0.5, 0.5],
            )

        # 3. Optional FlashRank reranker
        self.compressor = None
        if FLASHRANK_AVAILABLE:
            self.compressor = FlashRankCompressor(top_n=5)

        level_label = user_level if chroma_filter else "ALL (no filter)"
        doc_count = len(scoped_docs) if scoped_docs else 0
        print(f"   🔍 ScopedRetriever: level={level_label}, docs={doc_count}")

    def invoke(self, query: str) -> List[Document]:
        """Retrieve and optionally rerank documents for the given query.

        Args:
            query: The search query.

        Returns:
            List of relevant Documents, scoped to the user's level.
        """
        # Use ensemble if available, otherwise fall back to Chroma only
        base = self.ensemble_retriever or self.chroma_retriever
        docs = base.invoke(query)

        if self.compressor:
            docs = self.compressor.compress_documents(docs, query)

        return docs


# --- ELA BOT ---

class ELA_Bot:
    """ELA (Econometrics Learning Assistant) — Stateless RAG bot for Chainlit.

    Args:
        user_level: Academic level of the current user (M1, M2, ALL).
            Defaults to 'ALL' (no filtering) for backward compatibility.
    """

    def __init__(self, user_level: str = "ALL"):
        self.user_level = user_level
        print(f"🤖 Initialisation du moteur RAG ELA (niveau: {user_level})...")

        if not os.path.exists(DB_PATH):
            print(f"❌ Erreur : Le dossier '{DB_PATH}' n'existe pas.")

        # 1. Embeddings & VectorDB
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
        )
        self.vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embedding_model,
        )

        # 2. Level-scoped retriever
        self.retriever = ScopedRetriever(self.vector_db, user_level)

        # 3. LLM
        if "GROQ_API_KEY" not in os.environ:
            print("⚠️ GROQ_API_KEY non définie.")

        self.llm = ChatGroq(
            model=LLM_MODEL_NAME,
            temperature=0.2,
            max_tokens=2048,
        )

        # 4. Chain (kept for potential direct usage)
        self.rag_chain = self._build_chain()
        print("✅ Moteur ELA prêt !")

        # 5. RAG Guard
        self.guard = RAGGuard(llm=self.llm, max_retries=1)

    def _build_chain(self):
        """Build the LangChain LCEL chain for direct RAG queries."""
        level_supplement = _get_level_instructions(self.user_level)
        system_prompt_text = ELA_BASE_INSTRUCTIONS + level_supplement + """
        
        CONTEXTE DU COURS (Source unique) :
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            formatted = []
            for doc in docs:
                src = doc.metadata.get("source", "Inconnu")
                slide = doc.metadata.get("slide_title", "Sans titre")
                content = doc.page_content.replace("\n", " ")
                formatted.append(f">> [Source: {src} | Slide: {slide}]\n{content}")
            return "\n\n".join(formatted)

        chain = (
            {
                "context": lambda x: format_docs(self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def _get_rag_context(self, question: str) -> str:
        """Retrieve and format RAG context for a question.

        Args:
            question: The user's question.

        Returns:
            Formatted context string.
        """
        docs = self.retriever.invoke(question)
        return self._format_rag_context(docs)

    # --- MAIN ENTRY POINT (STREAMING + GUARD) ---
    async def ask(
        self,
        question: str,
        chat_history: list = None,
        image_path: str = None,
    ):
        """Answer a user question with RAG guard and streaming.

        Async generator that yields response tokens.

        Args:
            question: The user's question.
            chat_history: List of previous messages (LangChain format).
            image_path: Optional path to an uploaded image.

        Yields:
            Response text chunks (str).
        """
        if chat_history is None:
            chat_history = []

        try:
            # 1. Retrieve + Grade
            docs, grade = await self._retrieve_and_grade(question)

            # 2. Route: refuse if out of domain
            if grade.action == "refuse_domain":
                yield (
                    "Je suis spécialisé en **Séries Temporelles** et "
                    "**Économétrie Financière**. Ce sujet sort du cadre du cours.\n\n"
                    f"_({grade.reasoning})_"
                )
                return

            # 3. Build context from graded docs
            context_text = self._format_rag_context(docs)

            # 4. Adjust system prompt based on grade + level
            if grade.docs_relevant:
                source_instruction = (
                    "INSTRUCTION CRITIQUE : Les documents ci-dessous contiennent "
                    "des informations pertinentes pour répondre. Tu DOIS baser ta "
                    "réponse sur ces documents et citer [Source: NomFichier]. "
                    "N'utilise PAS tes connaissances générales si le cours couvre le sujet."
                )
            else:
                source_instruction = (
                    "NOTE : Les documents récupérés ne semblent pas directement "
                    "pertinents pour cette question. Tu peux utiliser tes connaissances "
                    "spécialisées (séries temporelles, économétrie financière) en "
                    "taguant [Source: Connaissances Générales]."
                )

            level_supplement = _get_level_instructions(self.user_level)

            full_system_prompt = (
                f"{ELA_BASE_INSTRUCTIONS}"
                f"{level_supplement}\n\n"
                f"{source_instruction}\n\n"
                f"CONTEXTE DU COURS (RAG) :\n{context_text}"
            )

            messages = [SystemMessage(content=full_system_prompt)]
            messages.extend(chat_history)

            # 5. Build user message (text + optional image)
            content_blocks = [{"type": "text", "text": question}]

            if image_path:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                })
                print("🖼️ Image détectée et envoyée à Groq Vision")

            messages.append(HumanMessage(content=content_blocks))

            # 6. Stream LLM response
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            yield f"❌ Erreur ELA : {str(e)}"

    # --- QUIZ ---
    async def generate_quiz_json(self, topic: str, num_questions: int = 3):
        """Generate quiz questions with domain validation.

        Args:
            topic: The quiz topic.
            num_questions: Number of questions to generate.

        Returns:
            List of quiz question dicts, or empty list on failure.
        """
        docs, grade = await self._retrieve_and_grade(topic)

        if grade.action == "refuse_domain":
            return [{"error": "domain", "message": grade.reasoning}]

        if not grade.docs_relevant:
            return []

        context_text = self._format_rag_context(docs)

        quiz_system_prompt = f"""
        Tu es ELA, un assistant expert pédagogique.
        
        TÂCHE : Créer un quiz QCM de {num_questions} questions sur le sujet : "{topic}".

        RÈGLES DE CONTENU (RAG) :
        1. Base-toi UNIQUEMENT sur le CONTEXTE DU COURS ci-dessous.
        2. Si le sujet n'est pas dans le cours, renvoie un JSON vide.

        RÈGLES DE FORMAT (CRITIQUE POUR ÉVITER LES BUGS) :
        1. **INTERDICTION TOTALE DU LATEX**. N'utilise jamais de symboles avec des backslashs.
        2. Écris les concepts mathématiques en TOUTES LETTRES ou en notation standard.
        3. La sortie doit être STRICTEMENT un objet JSON valide (RFC 8259).
        
        FORMAT ATTENDU :
        [
            {{
                "question": "Quelle est la définition du R-carré ?",
                "options": ["A) La variance...", "B) La moyenne...", "C) ..."],
                "correct_index": 0,
                "explanation": "Le R-carré représente..."
            }}
        ]

        CONTEXTE DU COURS :
        {context_text}
        """

        messages = [
            SystemMessage(content=quiz_system_prompt),
            HumanMessage(content=f"Génère le quiz sur {topic} sans LaTeX."),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            return json.loads(content)
        except Exception as e:
            print(f"❌ Erreur génération Quiz : {e}")
            return []

    # --- CODE WORKSHOP ---
    async def generate_practical_code(self, topic: str, language: str = "Python"):
        """Generate a practical code example based on course theory.

        Args:
            topic: The topic to implement.
            language: Programming language (Python or R).

        Returns:
            Generated code as a string.
        """
        context_text = self._get_rag_context(topic)
        level_supplement = _get_level_instructions(self.user_level)

        full_system_prompt = f"""{ELA_BASE_INSTRUCTIONS}
        {level_supplement}

        EXCEPTION : Pour cette tâche de programmation, tu as le droit d'utiliser tes connaissances en syntaxe {language} (librairies, fonctions), MAIS les équations et la logique théorique doivent venir strictement du CONTEXTE DU COURS.
        
        TÂCHE ACTUELLE : GÉNÉRER UN SCRIPT {language} EXÉCUTABLE
        Sujet : "{topic}"
        
        RÈGLES DE CODAGE STRICTES (PYTHON) :
        1. **Nommage des variables** : Distingue CLAIREMENT la cible (y) et les features (X). N'appelle jamais ta matrice de design 'X' si 'X' est déjà ta série temporelle brute. Utilise `y` pour la dépendante et `X_design` ou `exog` pour les explicatives.
        2. **Importations** : Importe toutes les librairies nécessaires au début.
        3. **Données** : Le code DOIT générer ses propres données synthétiques (np.random) pour être autonome.
        4. **Vérification** : Le script ne doit pas contenir d'erreur de syntaxe (comme écraser une variable utilisée ensuite).
        5. **Visualisation** : Inclus un graphique matplotlib clair si pertinent.
        
        CONTEXTE THÉORIQUE (à respecter pour la formule) :
        {context_text}
        
        FORMAT DE RÉPONSE :
        - Courte intro.
        - Bloc de code (complet, sans placeholder).
        - Courte interprétation.
        """

        messages = [
            SystemMessage(content=full_system_prompt),
            HumanMessage(content=f"Écris le script pour {topic} en {language}."),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            return f"❌ Erreur de génération de code : {str(e)}"

    # --- RELEVANCE GRADER ---
    async def _retrieve_and_grade(self, question: str) -> tuple:
        """Retrieve documents then grade their relevance.

        If grading suggests a retry, reformulates the query once.

        Args:
            question: The user's question.

        Returns:
            Tuple of (documents, grade_result).
        """
        docs = self.retriever.invoke(question)
        grade = await self.guard.grade(question, docs)

        if grade.action == "retry_query" and grade.suggested_query:
            print(f"🔄 RAG Guard: retry avec '{grade.suggested_query}'")
            docs_retry = self.retriever.invoke(grade.suggested_query)
            grade_retry = await self.guard.grade(question, docs_retry)
            if grade_retry.docs_relevant:
                return docs_retry, grade_retry

        return docs, grade

    # --- Format Context ---
    def _format_rag_context(self, docs: list) -> str:
        """Format retrieved documents into context string.

        Args:
            docs: List of LangChain Document objects.

        Returns:
            Formatted context string.
        """
        formatted = []
        for doc in docs:
            src = doc.metadata.get("source", "Inconnu")
            content = doc.page_content.replace("\n", " ")
            formatted.append(f">> [Source: {src}]\n{content}")
        return "\n\n".join(formatted)