from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def is_answer_supported(answer, docs):
    context_text = " ".join([doc.page_content for doc in docs]).lower()
    answer = answer.lower()

    # Check overlap
    match_count = sum(
        1 for word in answer.split()
        if word in context_text
    )

    return match_count > 5


def generate_answer(query, docs):
    # 🔥 LIMIT context size (VERY IMPORTANT)
    context = "\n\n".join([doc.page_content[:300] for doc in docs[:3]])

    prompt = f"""
You are a strict question answering system.

Answer ONLY using the provided context.
Do NOT use outside knowledge.

If the context does not contain enough information, respond EXACTLY with:
"I cannot find sufficient information in the provided documents."

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

    generated_text = result[0]["generated_text"]

    #extract only the answer part
    answer = generated_text.split("Answer:")[-1].strip()

    # CITATION ENFORCEMENT
    if not is_answer_supported(answer, docs):
        return "I cannot find sufficient information in the provided documents."

    return answer