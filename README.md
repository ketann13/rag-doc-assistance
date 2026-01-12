# 🧠 Offline RAG Chatbot – Document Q&A System

An end-to-end **offline Retrieval-Augmented Generation (RAG) chatbot** that allows you to ask questions from your own documents without using any paid APIs.

This project uses **FAISS for vector search**, **HuggingFace embeddings for semantic understanding**, and **Ollama for local LLM inference**.

---

## 🚀 Features

- 📄 Upload and process PDF and text documents
- ✂️ Intelligent document chunking for better retrieval
- 🔍 Semantic search using FAISS vector database
- 🧠 Offline embeddings using HuggingFace models
- 🤖 Local LLM inference using Ollama (no cloud dependency)
- 💬 Interactive CLI chatbot interface
- 📚 Source attribution for answers
- 💸 Zero API cost – fully local execution

---

## 🏗️ Project Structure

.
├── chat.py # Main CLI chatbot interface
├── document_processor.py # Document loading and chunking
├── vector_store.py # FAISS vector database management
├── rag_pipeline.py # RAG pipeline logic
├── requirements.txt # Python dependencies
├── .env.example # Environment template (optional)
├── data/ # Your documents go here
│ ├── sample.txt
│ └── notes.pdf
└── faiss_index/ # Auto-generated vector index


---

## ⚙️ Setup Instructions

### ✅ 1. Prerequisites

- Python 3.9+
- 8GB+ RAM recommended
- Ollama installed  
  👉 https://ollama.com

---

### ✅ 2. Install Dependencies

Activate virtual environment and install packages:

```bash
pip install -r requirements.txt

✅ 3. Pull Local LLM Model

Download a lightweight local model:

ollama pull tinyllama
ollama run tinyllama

✅ 4. Add Your Documents

Place your documents inside the data/ folder.

Supported formats:

.txt

.pdf

.md

▶️ Run the Chatbot
python chat.py


On first run:

Documents are loaded

Text is split into chunks

Embeddings are generated locally

FAISS vector index is created

Chat interface starts

Subsequent runs load the existing index (fast startup).

💬 Example Usage

Ask questions like:

What is supervised learning?

Summarize this document.

What are the main concepts discussed?

Explain key points from the PDF.

Exit anytime using:

exit
quit
Ctrl + C

🧩 How It Works
📄 Document Processing

Loads documents from data/

Splits text into overlapping chunks

Preserves semantic meaning

📊 Vector Store (FAISS)

Converts text chunks into embeddings

Stores vectors locally

Performs fast similarity search

🔁 RAG Pipeline

User question received

Relevant chunks retrieved from FAISS

Context injected into prompt

Local LLM generates answer

Sources returned

🧠 Local AI Stack
User Question
    ↓
FAISS Similarity Search
    ↓
Relevant Chunks
    ↓
Prompt Construction
    ↓
Ollama Local LLM
    ↓
Answer + Sources

🛠️ Tech Stack

Python

LangChain

FAISS

HuggingFace Embeddings

Ollama

PyPDF

NumPy

🎯 Why This Project Matters

✅ Demonstrates real-world RAG system design

✅ Works completely offline

✅ No dependency on paid APIs

✅ Strong ML + Systems engineering project

✅ Resume-ready production-style architecture

💡 Future Improvements

Web UI using Streamlit / React

Multi-document summarization

Chat history memory

Hybrid search (BM25 + Vector)

GPU acceleration

Model switching support

Document metadata visualization

📜 License

Open-source for educational and learning purposes.
