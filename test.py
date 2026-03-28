from src.ingestion import load_and_split_document, create_vector_store

chunks = load_and_split_document("data/COS324_Course_Notes_princeton.pdf")

print("Chunks created:", len(chunks))

vectorstore = create_vector_store(chunks)

print("Vector DB created successfully!")