# Offline RAG Chatbot - Local Document Q&A System

A fully **offline** Retrieval-Augmented Generation (RAG) chatbot that answers questions from your documents. Runs fully offline after initial model downloads. No API keys or ongoing internet connectivity required. Built with Python, LangChain, FAISS, HuggingFace, and Ollama.

## Why This Project Matters

In an era where data privacy, cost, and accessibility are critical:
- **Zero API Costs**: No subscription fees or pay-per-use charges
- **Complete Privacy**: All data processing happens locally on your machine
- **No Internet Required**: Works offline once dependencies are installed
- **Production-Ready**: Demonstrates enterprise-grade RAG architecture without cloud dependencies
- **Learning Platform**: Ideal for understanding RAG systems without the barrier of API costs

This project showcases how modern AI capabilities can be democratized using open-source tools.

## Features

- 📁 Process PDF and text documents locally
- 🧠 HuggingFace embeddings for semantic understanding
- 🗄️ FAISS vector database for fast similarity search
- 🤖 Ollama integration for local LLM inference
- 💬 Interactive CLI chat interface
- 🔍 Source attribution and context tracking
- 🔒 100% offline operation
- 💰 Zero ongoing costs

## Project Structure

```
.
├── app.py                     # Streamlit web interface
├── chat.py                    # CLI chat interface
├── document_processor.py      # Document loading and chunking
├── vector_store.py           # FAISS vector database management
├── rag_pipeline.py           # RAG pipeline implementation
├── requirements.txt          # Python dependencies
├── data/                     # Directory for your documents
└── faiss_db/                 # FAISS index storage (auto-created)
    └── index.faiss
```

## Setup Instructions

### 1. Prerequisites

- **Python 3.8 or higher**
- **Ollama**: Local LLM runtime ([Download here](https://ollama.ai))
- At least 4GB RAM (8GB+ recommended for optimal performance)

### 2. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

After installation, pull the Phi model (lightweight and fast):
```bash
ollama pull phi
```

You can also try other models based on your hardware:
```bash
# For better quality (requires more RAM):
ollama pull llama2
ollama pull mistral

# List available models:
ollama list
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `langchain` and `langchain-community` for RAG framework
- `sentence-transformers` for HuggingFace embeddings
- `faiss-cpu` for vector similarity search
- `pypdf` for PDF processing
- `streamlit` for the web interface
- Additional supporting libraries

### 4. Add Your Documents

Place your PDF or text files in the `data/` directory. Supported formats:
- `.pdf` files
- `.txt` files
- `.md` files

**With Streamlit UI**: You can also upload documents directly through the web interface, which will automatically rebuild the vector database.

**With CLI**: Place files in the `data/` directory before running `chat.py`.

## Usage

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

This launches an interactive web interface where you can:
- Upload documents directly through the UI
- Automatically rebuild the vector database
- Chat with your documents in a user-friendly interface

### Option 2: CLI Chat Interface

```bash
python chat.py
```

For command-line interaction with your documents.

### First Run

On the first run, the system will:
1. Load documents from the `data/` directory
2. Split documents into semantic chunks
3. Generate embeddings using HuggingFace's local models (no API required)
4. Build a FAISS vector index for fast similarity search
5. Launch the interactive chat interface

**Note**: The first run may take a few minutes as HuggingFace downloads the embedding model (~400MB) and processes your documents. This is a one-time setup.

### Subsequent Runs

The system loads the pre-built FAISS index, making startup nearly instant.

### Chat Interfaces

#### Streamlit Web UI
- Clean, intuitive web interface
- Upload documents directly
- Auto-rebuild vector database
- Real-time chat interface
- Source attribution with interactive display

#### CLI Interface
- Type your questions and press Enter
- Ollama processes queries locally on your machine
- The bot searches your documents and provides contextual answers
- Type `quit`, `exit`, or `q` to end the conversation
- Press Ctrl+C to exit at any time

### Example Questions

Try asking questions like:
- "What is machine learning?"
- "What are the three main types of machine learning?"
- "What are Python naming conventions?"
- "How should I organize my Python imports?"
- "What is the machine learning workflow?"

## How It Works

### 1. Document Processing
- Documents are loaded from the data directory
- Text is intelligently split into chunks (default: 1000 characters with 200 overlap)
- Chunks preserve semantic context and meaning

### 2. Local Embeddings Generation
- HuggingFace Sentence Transformers convert text to vector embeddings
- Model: `all-MiniLM-L6-v2` (efficient and accurate for RAG tasks)
- All processing happens locally with no external API calls

### 3. FAISS Vector Database
- Facebook AI Similarity Search (FAISS) stores and indexes embeddings
- Enables ultra-fast similarity search across thousands of documents
- Optimized for CPU and scales to production workloads

### 4. RAG Pipeline with Ollama
When you ask a question:
- Your query is embedded using the same HuggingFace model
- FAISS performs similarity search to find relevant document chunks
- Retrieved chunks provide context to Ollama
- Ollama's local LLM generates an answer grounded in your documents
- No internet connection required after initial setup

### 5. Source Attribution
- Tracks which documents contributed to each answer
- Displays number of sources used for transparency

## Configuration

You can customize the system by modifying parameters in the code:

### Document Processing
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks for context preservation (default: 200)

### Embeddings
- `model_name`: HuggingFace model for embeddings (default: `all-MiniLM-L6-v2`)
- Other options: `all-mpnet-base-v2` (more accurate but slower)

### Retrieval
- `k`: Number of chunks to retrieve (default: 4)
- Higher values provide more context but may reduce answer focus

### LLM Generation
- `model`: Ollama model to use (default: `phi`)
- `temperature`: Response creativity (0=deterministic, 1=creative, default: 0)
- Other models: `llama2`, `mistral`, `tinyllama`

## Troubleshooting

### "No supported documents found"
Ensure you have at least one PDF or text file in the `data/` directory.

### "Ollama connection error"
- Check that Ollama is installed and running
- Verify the model is pulled: `ollama list`
- Try pulling the model again: `ollama pull phi`

### Slow first run
The first run downloads the HuggingFace embedding model (~400MB) and creates the FAISS index. Subsequent runs load the cached model and index instantly.

### Out of memory errors
- Use a smaller Ollama model: `phi` instead of `llama2`
- Reduce the `k` parameter to retrieve fewer chunks
- Close other applications to free up RAM

### FAISS index corruption
Delete the `faiss_index/` directory and restart to rebuild the index.

## Technical Stack

### Core Technologies
- **LangChain**: Framework for building RAG applications
- **Ollama**: Local LLM runtime for inference
- **HuggingFace Sentence Transformers**: Open-source embedding models
- **FAISS (Facebook AI Similarity Search)**: High-performance vector database
- **PyPDF**: PDF document parsing
- **Python 3.8+**: Core programming language

### Architecture
```
User Question
    ↓
[Local Embeddings via HuggingFace]
    ↓
FAISS Similarity Search
    ↓
Retrieve Relevant Chunks
    ↓
Construct Prompt with Context
    ↓
[Ollama Local LLM Generation]
    ↓
Return Answer + Sources
```

### Why These Technologies?

- **FAISS**: Industry-standard vector search, used by companies like Meta in production
- **HuggingFace**: Largest open-source ML community with state-of-the-art models
- **Ollama**: Simplifies local LLM deployment with optimized inference
- **LangChain**: Production-grade abstractions for RAG pipelines

## Enhancement Ideas

- 🌐 Add web scraping to load and index online documentation
- 💬 Implement conversation history for multi-turn dialogues
- 🎨 Create a web UI using Streamlit or Gradio
- 📊 Support additional formats (Word, Excel, PowerPoint)
- 🔍 Implement hybrid search (BM25 keyword + semantic)
- 📝 Add citation extraction with page numbers
- 🗂️ Multi-collection support for organized document sets
- 📈 Add query analytics and usage statistics
- 🔄 Implement document versioning and updates
- 🎯 Fine-tune embeddings on domain-specific data

## Performance & Cost Comparison

| Aspect | Cloud RAG | This Project (Offline) |
|--------|-----------|------------------------|
| **API Costs** | $0.01-0.10 per query | **$0.00** |
| **Privacy** | Data sent to cloud | **100% local** |
| **Internet** | Required | **Not required** |
| **Latency** | Network dependent | **Local speed** |
| **Setup Time** | Minutes | ~10 minutes |
| **Scaling Cost** | Linear with usage | **Zero** |

**Real-world savings**: At 1000 queries/month with GPT-3.5, you'd pay ~$30-50/month. This project costs **nothing** after setup.

## Use Cases

- **Enterprise Knowledge Bases**: Private company documents without data leakage
- **Research**: Academic papers and literature review without API costs
- **Legal/Medical**: Sensitive documents requiring air-gapped processing
- **Education**: Learning RAG systems without financial barriers
- **Development**: Prototype and test RAG applications locally
- **Personal**: Organize personal notes, books, and documents

## Performance Benchmarks

On typical consumer hardware (16GB RAM, modern CPU):
- **Embedding Speed**: ~100 documents/minute
- **Query Latency**: 2-5 seconds (including LLM generation)
- **Index Size**: ~1MB per 1000 document chunks
- **Memory Usage**: 2-4GB during operation

## License

This project is open source and available for educational and commercial use.

## Resources & Learning

- **Ollama Documentation**: [https://ollama.ai/docs](https://ollama.ai/docs)
- **LangChain RAG Guide**: [https://python.langchain.com/docs/use_cases/question_answering/](https://python.langchain.com/docs/use_cases/question_answering/)
- **FAISS Documentation**: [https://faiss.ai](https://faiss.ai)
- **HuggingFace Sentence Transformers**: [https://www.sbert.net](https://www.sbert.net)

## Contributing

Contributions are welcome! Feel free to:
- Submit bug reports and feature requests
- Improve documentation
- Add support for new document types
- Optimize performance

---

**Built with ❤️ for the open-source AI community**
