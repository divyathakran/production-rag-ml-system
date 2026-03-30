# 🚀 Production-Grade RAG System for Machine Learning Documents

An end-to-end Retrieval-Augmented Generation (RAG) system that allows users to ask questions from Machine Learning documents and receive answers grounded in retrieved context.

This project is designed to go beyond a simple demo and evolve into a **production-ready AI system**, following best practices used in real-world applications.

---

## 📌 What is this project?

This system enables:

* 📄 Uploading and processing Machine Learning documents (PDFs)
* 🔎 Retrieving relevant information using semantic search
* 🤖 Generating answers based only on retrieved content

Unlike generic LLM responses, this system ensures that answers are **context-aware and grounded in actual documents**.

---

## 🧠 System Architecture

```id="arch001"
User Query 
   ↓
Retriever (Vector Search)
   ↓
Top-K Relevant Chunks
   ↓
LLM (Answer Generation)
   ↓
Final Response
```

---

## 🛠️ Tech Stack

* Python
* LangChain
* ChromaDB (Vector Database)
* HuggingFace Transformers
* Sentence Transformers (MiniLM)

---

## 📁 Project Structure

```id="struct001"
rag-ml-system/
│
├── src/
│   ├── ingestion.py      # Document loading & chunking
│   ├── retrieval.py      # Semantic search
│   ├── generation.py     # Answer generation
│
├── main.py               # CLI interface
├── test.py               # Pipeline testing
├── requirements.txt
└── README.md
```

---

# 🚧 Development Phases

This project is being built in multiple phases to simulate a real production AI system.

---

## ✅ Phase 1 — Core RAG Pipeline (Completed)

Implemented a complete end-to-end pipeline:

* 📄 Document ingestion (PDF)
* ✂️ Text chunking with overlap
* 🔢 Embedding generation using HuggingFace
* 🗂️ Vector storage using ChromaDB
* 🔍 Semantic retrieval (top-k chunks)
* 🤖 Basic answer generation using GPT-2
* 💻 CLI interface (`main.py`)

### ⚠️ Limitations

* Uses a basic LLM (GPT-2)
* Answers may be repetitive or low quality
* No citation enforcement
* No hybrid retrieval

---

## 🚀 Phase 2 — Production Improvements (Upcoming)

Focus: **Accuracy + Reliability**

* 🔥 Replace GPT-2 with a stronger LLM (Groq / Gemini)
* 🔍 Hybrid retrieval (BM25 + semantic search)
* 🎯 Cross-encoder re-ranking
* 📚 Citation-based answer generation (reduce hallucination)
* 🧩 Prompt management system

---

## 🔬 Phase 3 — Evaluation & Deployment (Planned)

Focus: **Quality + Scalability**

* 📊 Evaluation pipeline (RAGAS)
* 🧪 Golden dataset for benchmarking
* ⚙️ CI/CD integration for automated evaluation
* 🌐 API or web interface (FastAPI / Streamlit)

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```id="clone001"
git clone https://github.com/divyathakran/production-rag-ml-system.git
cd production-rag-ml-system
```

---

### 2. Create virtual environment

```id="venv001"
python -m venv ragvenv
ragvenv\Scripts\activate
```

---

### 3. Install dependencies

```id="install001"
pip install -r requirements.txt
```

---

### 4. Add your document

Place your PDF inside a `data/` folder.

---

### 5. Run ingestion (one-time)

```id="ingest001"
python test.py
```

---

### 6. Run the system

```id="run001"
python main.py
```

---

## 💡 Example Usage

```id="example001"
Ask a question: What is machine learning?

Answer:
Machine learning is the discipline of creating decision-making programs that improve automatically based on data.
```

---

## 🎯 Key Highlights

* Built a complete RAG system from scratch
* Modular and scalable architecture
* Designed with production practices in mind
* Clear roadmap for future improvements

---

## 👩‍💻 Author

**Divya Thakran**

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!
