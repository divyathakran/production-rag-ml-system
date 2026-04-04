import pickle
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunk_path="data/chunks.pkl"):
        with open(chunk_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Tokenize corpus
        self.tokenized_corpus = [chunk.split() for chunk in self.chunks]

        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [self.chunks[i] for i in top_indices]