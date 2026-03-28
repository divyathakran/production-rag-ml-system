from src.retrieval import get_retriever
from src.generation import generate_answer

def main():
    print("🔍 RAG System Ready (Machine Learning Docs)\n")

    retriever = get_retriever()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        docs = retriever.invoke(query)

        answer = generate_answer(query, docs)

        print("\n💡 Answer:\n")
        print(answer)


if __name__ == "__main__":
    main()