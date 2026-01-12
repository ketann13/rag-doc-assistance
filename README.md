# RAG Chatbot - Document Q&A System

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that can answer questions from your own documents. Built with Python, LangChain, OpenAI, and ChromaDB.

## Features

- Upload and process PDF and text documents
- Intelligent document chunking for better retrieval
- Vector database with embeddings for semantic search
- RAG pipeline combining retrieval with LLM generation
- Interactive CLI chat interface
- Source attribution for answers

## Project Structure

```
.
├── chat.py                    # Main chat interface
├── document_processor.py      # Document loading and chunking
├── vector_store.py           # Vector database management
├── rag_pipeline.py           # RAG pipeline implementation
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── data/                    # Directory for your documents
│   ├── machine_learning_intro.txt
│   └── python_best_practices.txt
└── chroma_db/               # Vector database (auto-created)
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- OpenAI API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

### 4. Add Your Documents

Place your PDF or text files in the `data/` directory. The system supports:
- `.pdf` files
- `.txt` files
- `.md` files

Sample documents are already included for testing.

## Usage

### Start the Chatbot

```bash
python chat.py
```

### First Run

On the first run, the system will:
1. Load documents from the `data/` directory
2. Split documents into chunks
3. Create embeddings using OpenAI's text-embedding model
4. Build a vector database (ChromaDB)
5. Display the chat interface

### Subsequent Runs

The system will load the existing vector database, making startup much faster.

### Chat Interface

- Type your questions and press Enter
- The bot will search your documents and provide answers
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
- Text is split into chunks (default: 1000 characters with 200 overlap)
- Chunks preserve context and meaning

### 2. Vector Database
- Each chunk is converted to an embedding (vector representation)
- Vectors are stored in ChromaDB
- Enables semantic search based on meaning, not just keywords

### 3. RAG Pipeline
- When you ask a question, the system:
  - Searches for relevant document chunks
  - Passes those chunks as context to the LLM
  - Generates an answer based on the retrieved context

### 4. Source Attribution
- The system tracks which documents provided the context
- Shows the number of sources used for each answer

## Configuration

You can modify parameters in the code:

### Document Processing
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

### Retrieval
- `k`: Number of chunks to retrieve (default: 4)

### Generation
- `model`: OpenAI model to use (default: gpt-3.5-turbo)
- `temperature`: Response randomness (default: 0)

## Troubleshooting

### "No supported documents found"
Make sure you have at least one PDF or text file in the `data/` directory.

### "OPENAI_API_KEY not found"
Ensure you've created the `.env` file with your API key.

### Slow first run
The first run is slower because it creates embeddings. Subsequent runs will be much faster.

### Out of context error
Try reducing the `k` parameter to retrieve fewer chunks, or adjust the chunk size.

## Technical Details

### Technologies Used
- **LangChain**: Framework for building LLM applications
- **OpenAI**: GPT model and embeddings
- **ChromaDB**: Vector database for similarity search
- **PyPDF**: PDF parsing
- **Python-dotenv**: Environment variable management

### Architecture
```
User Question
    ↓
Vector Similarity Search
    ↓
Retrieve Relevant Chunks
    ↓
Construct Prompt with Context
    ↓
Generate Answer with LLM
    ↓
Return Answer + Sources
```

## Customization Ideas

- Add web scraping to load online documents
- Implement conversation history for follow-up questions
- Create a web UI with Streamlit or Flask
- Add support for more document types (Word, Excel, etc.)
- Implement hybrid search (keyword + semantic)
- Add citation highlighting in responses

## Cost Considerations

- OpenAI API usage incurs costs
- First run requires embeddings creation (charged per token)
- Each question requires API calls for generation
- Consider using local models for cost reduction

## License

This project is open source and available for educational purposes.

## Support

For issues or questions, check:
- LangChain documentation: https://python.langchain.com/
- OpenAI API documentation: https://platform.openai.com/docs/
- ChromaDB documentation: https://docs.trychroma.com/
