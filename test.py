from src.retrieval import get_retriever

retriever = get_retriever()

query = "What is machine learning?"

docs = retriever.invoke(query)

print("\nTop Results:\n")

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n")
    print(doc.page_content[:300])