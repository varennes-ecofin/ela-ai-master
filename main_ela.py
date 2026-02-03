# main_ela.py
import os
import json
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
LLM_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

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
            model=LLM_MODEL_NAME,
            temperature=0.2,
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
        
        # Fonction de prétraitement pour ignorer la casse
        def case_insensitive_tokenizer(text):
            return text.lower().split()

        bm25_retriever = BM25Retriever.from_documents(docs_list,preprocess_func=case_insensitive_tokenizer)
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
        # On combine la base + le format attendu par LangChain
        system_prompt_text = ELA_BASE_INSTRUCTIONS + """
        
        CONTEXTE DU COURS (Source unique) :
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
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
            
            # 2. Préparer le message Système (VERSION RENFORCÉE)
            full_system_prompt = f"""{ELA_BASE_INSTRUCTIONS}

            TÂCHE ACTUELLE :
            Réponds à la question de l'étudiant.
            
            RAPPEL CRITIQUE SUR LES SOURCES :
            - Chaque affirmation doit être sourcée.
            - Utilise `[Source: ...]` pour le RAG.
            - Utilise `[Source: Connaissances Générales]` si l'info vient de ton propre savoir.
            - Si l'info est dans le contexte RAG ci-dessous, la citation est OBLIGATOIRE.
            
            CONTEXTE DU COURS (RAG) :
            {context_text}"""

            messages = [SystemMessage(content=full_system_prompt)]
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


    # --- METHODE POUR LE QUIZ ---
    async def generate_quiz_json(self, topic: str, num_questions: int = 3):
        """
        Génère une liste de questions QCM au format JSON sans LaTeX pour éviter les bugs.
        """
        context_text = self._get_rag_context(topic)
        
        # --- CHANGEMENT MAJEUR : PROMPT DÉDIÉ SANS LATEX ---
        # On n'utilise PAS ELA_BASE_INSTRUCTIONS ici pour ne pas hériter de l'obligation LaTeX.
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
            HumanMessage(content=f"Génère le quiz sur {topic} sans LaTeX.")
        ]

        try:
            # Appel LLM avec paramètre pour forcer le JSON si le modèle le supporte (ou via prompt)
            # Pour Groq/Llama, le prompt strict fonctionne généralement bien
            response = await self.llm.ainvoke(messages)
            content = response.content.strip()
            
            # Nettoyage si le LLM ajoute du markdown ```json ... ```
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            quiz_data = json.loads(content)
            return quiz_data
            
        except Exception as e:
            print(f"❌ Erreur génération Quiz : {e}")
            # Fallback en cas d'erreur de parsing
            return []   
        

    async def generate_practical_code(self, topic: str, language: str = "Python"):
        """
        Génère un exemple de code pratique basé sur la théorie du cours.
        """
        # On récupère un peu de théorie pour guider le modèle, mais on compte surtout sur ses capacités de codage
        context_text = self._get_rag_context(topic)
        
        # ICI UNE NUANCE : On garde la base d'instruction mais on ajoute la compétence CODAGE
        full_system_prompt = f"""{ELA_BASE_INSTRUCTIONS}

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
            HumanMessage(content=f"Écris le script pour {topic} en {language}.")
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            return f"❌ Erreur de génération de code : {str(e)}"