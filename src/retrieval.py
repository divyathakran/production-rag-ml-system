from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.bm25_retrieval import BM25Retriever

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever



class HybridRetriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.bm25 = BM25Retriever()

    def retrieve(self, query, top_k=5):
        # 1. Vector results
        vector_results = self.vector_db.similarity_search(query, k=top_k)

        # 2. BM25 results
        bm25_results = self.bm25.retrieve(query, top_k=top_k)

        # 3. Combine
        combined = vector_results + bm25_results

        # 4. Remove duplicates
        unique = list(set(combined))

        return unique[:top_k]