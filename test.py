from src.retrieval import get_retriever
from src.generation import generate_answer

retriever = get_retriever()

query = "What is machine learning?"

docs = retriever.invoke(query)

answer = generate_answer(query, docs)

print("\nFinal Answer:\n")
print(answer)