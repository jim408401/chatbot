# Chatbot

A document question-answering system based on RAG technology, integrating LangChain framework and Ollama local language model. Supports file upload, vector storage and retrieval, Reranker sorting, and provides chat interface and usage record functionality.

## Features

- Document upload and content segmentation (support PDF / TXT)
- Vector storage and retrieval (using bge-m3 + ChromaDB)
- Generate natural language responses using local LLM (qwen3 / llama)
- Support Reranker to improve response relevance (bge-reranker-v2-m3)
- User/manager permission control
- Chat history, user feedback, and response time records

## Environment Requirements

- Python 3.9+
- Ollama 0.6+

## Installation & Startup

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend service
python -m src.backend.main

# Start frontend interface
streamlit run src/frontend/app.py

# Open in browser: http://localhost:8501
```

## Usage

- 1. Open the web interface or log in with Admin account
- 2. Upload PDF or TXT files
- 3. Enter your question in the chat box
- 4. The system will respond based on the uploaded documents
- 5. Admins can view all usage records and statistics

## Project Structure

```bash
dds-chatbot/
├── config.py           # Global configuration
├── requirements.txt    # Package dependencies
├── .env                # Environment variables
├── data/               # Storage for uploaded files and database
│   ├── uploads/        # Original uploaded files
│   ├── chroma_db/      # Chroma vector database
│   └── database.db     # SQLite database (qa records, etc.)
├── src/
│   ├── backend/        # FastAPI backend API
│   ├── frontend/       # Streamlit frontend (Chat / Documents / History / Admin / Login)
│   └── rag/            # RAG core module
│       ├── loader.py         # File loader
│       ├── processor.py      # Text processor
│       ├── vectorstore.py    # Vector storage
│       ├── retriever.py      # Vector retriever
│       ├── chain.py          # Question answer processing chain
│       └── models.py         # Model calling logic
```
