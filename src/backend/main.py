import os
import sys
import uvicorn
import logging
from src.backend.api import app
from config import BACKEND_PORT, LLM_MODEL, EMBEDDING_MODEL
from src.backend.logging import get_api_logger
from src.backend.database import ensure_database, get_qa_count

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

logging.getLogger('backoff').setLevel(logging.ERROR)
for logger_name in ['uvicorn', 'uvicorn.error', 'uvicorn.access', 'fastapi']:
    logging.getLogger(logger_name).handlers.clear()
    logging.getLogger(logger_name).propagate = False

logger = get_api_logger()


def preload_models():
    """
    Preload models and vector store to avoid delay on first request
    """
    try:
        from src.rag.models import get_llm, get_embeddings, get_reranker
        from src.rag.vectorstore import load_vectorstore
        from src.backend.database import ensure_database
        qa_count = None
        try:
            conn = ensure_database() or None
            import sqlite3
            if conn is None:
                conn = sqlite3.connect('data/database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM qa_history")
            qa_count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            qa_count = None

        llm = get_llm(model_name=LLM_MODEL)
        logger.info(f"Loading LLM model: {LLM_MODEL} done")
        embeddings = get_embeddings(model_name=EMBEDDING_MODEL)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL} done")
        reranker = get_reranker()
        logger.info(f"Loading reranker model: bge-reranker-v2-m3 done")
        vectorstore = load_vectorstore()
        vec_count = vectorstore._collection.count() if vectorstore else 0
        logger.info(f"Loading vectorstore, {vec_count} chunks loaded")
        ensure_database()
        qa_count = get_qa_count()
        logger.info(f"Loading database, {qa_count if qa_count is not None else '?'} QA records loaded")
        logger.info(f"Startup complete. DDS chatbot is now running on port {BACKEND_PORT}.")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """
    Start FastAPI application
    """
    preload_models()   
    uvicorn.run(
        "src.backend.api:app",
        host="0.0.0.0",
        port=BACKEND_PORT,
        reload=False,
        log_level="warning",
        access_log=False 
    )

if __name__ == "__main__":
    main()