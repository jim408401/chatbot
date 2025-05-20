import os
from datetime import datetime
from dotenv import load_dotenv
from src.backend.logging import get_config_logger

CONFIG_MARKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "config_loaded.marker")
CONFIG_LOADED = os.path.exists(CONFIG_MARKER_FILE)

logger = get_config_logger()
load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ollama & Models Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "shmily_006/Qw3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
AVAILABLE_MODELS_STR = os.getenv("AVAILABLE_MODELS", "shmily_006/Qw3, llama3.2:3b")
AVAILABLE_MODELS = [model.strip() for model in AVAILABLE_MODELS_STR.split(",")]
DEFAULT_MODEL = LLM_MODEL if LLM_MODEL in AVAILABLE_MODELS else AVAILABLE_MODELS[0]

if not CONFIG_LOADED:
    # Reranker Model
    PARENT_DIR = os.path.dirname(PROJECT_ROOT)
    RERANKER_MODEL_PATH = os.getenv(
        "RERANKER_MODEL_PATH", 
        os.path.join(PARENT_DIR, "bge-reranker-v2-m3")
    )
    logger.info(f"Available models: {AVAILABLE_MODELS}, {EMBEDDING_MODEL}, {RERANKER_MODEL_PATH}")
    
    # Data Directory
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
    VECTORSTORE_DIR = os.path.join(DATA_DIR, "chroma_db")
    DATABASE_PATH = os.path.join(DATA_DIR, "database.db")

    for directory in [DATA_DIR, UPLOADS_DIR, VECTORSTORE_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

    logger.info(f"Database loaded: {DATABASE_PATH}")
    
    os.makedirs(os.path.dirname(CONFIG_MARKER_FILE), exist_ok=True)
    
    try:
        with open(CONFIG_MARKER_FILE, 'w') as f:
            f.write(str(datetime.now()))
    except Exception as e:
        logger.warning(f"Failed to create configuration marker file: {e}")
else:
    PARENT_DIR = os.path.dirname(PROJECT_ROOT)
    RERANKER_MODEL_PATH = os.getenv(
        "RERANKER_MODEL_PATH", 
        os.path.join(PARENT_DIR, "bge-reranker-v2-m3")
    )
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
    UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
    VECTORSTORE_DIR = os.path.join(DATA_DIR, "chroma_db")
    DATABASE_PATH = os.path.join(DATA_DIR, "database.db")

# Text Splitting
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "300"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))

# Retrieval
DEFAULT_RETRIEVER_K = int(os.getenv("DEFAULT_RETRIEVER_K", "3"))
DEFAULT_RERANKER_TOP_N = int(os.getenv("DEFAULT_RERANKER_TOP_N", "3"))

# LLM Settings
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))

# API Settings
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8080"))