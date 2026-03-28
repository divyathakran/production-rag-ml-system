from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_answer(query, docs):
    # 🔥 LIMIT context size (VERY IMPORTANT)
    context = "\n\n".join([doc.page_content[:300] for doc in docs[:3]])

    prompt = f"""
You are a helpful AI assistant.

Answer the question in a clear and concise way using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.3
    )

    answer = result[0]["generated_text"]

    return answer.split("Answer:")[-1].strip()