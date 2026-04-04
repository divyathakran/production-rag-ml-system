from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self):
        #lightweight but powerful model
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, chunks,top_k=5):
        if not chunks:
            return []

        #create (query,chunk) pairs
        pairs = [(query,chunk) for chunk in chunks]

        #get relevance scores
        scores = self.model.predict(pairs)

        #sort chunks by score
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        #return top_k chunks
        return [chunk for chunk, _ in ranked[:top_k]]