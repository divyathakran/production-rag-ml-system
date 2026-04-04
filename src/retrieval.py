from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.bm25_retrieval import BM25Retriever
from src.reranker import CrossEncoderReranker

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
        self.reranker = CrossEncoderReranker()

    def retrieve(self, query, top_k=5):
        #step 1: get more candidates
        vector_results = self.vector_db.similarity_search(query, k=10)
        vector_texts = [doc.page_content for doc in vector_results]

        bm25_results = self.bm25.retrieve(query, top_k=10)

        #step 2: combine
        combined = list(set(vector_texts + bm25_results))

        #step 3: rerank
        reranked = self.reranker.rerank(query, combined, top_k=top_k)

        return reranked