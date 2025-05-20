import os
import logging
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.rag.models import get_embeddings
from config import VECTORSTORE_DIR

logger = logging.getLogger("dds_chatbot_rag_vectorstore")

try:
    from src.backend.logging import get_rag_logger
    logger = get_rag_logger("vectorstore")
except ImportError:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Using fallback logger configuration")

_vectorstore_cache = None


def create_vectorstore(
    documents: List[Document], 
    persist_directory: Optional[str] = None
) -> Optional[Chroma]:
    """
    Create or update vector store
    
    Args:
        documents: List of documents
        persist_directory: Persistence directory
    
    Returns:
        Chroma: Vector store instance or None
    """
    global _vectorstore_cache
    
    if persist_directory is None:
        persist_directory = VECTORSTORE_DIR
    
    os.makedirs(persist_directory, exist_ok=True)
    
    embedding_function = get_embeddings()
    if embedding_function is None:
        logger.error("Failed to initialize embedding model, cannot create vector store")
        return None
    
    try:
        if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
            if _vectorstore_cache is not None:
                vectorstore = _vectorstore_cache
            else:
                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_function
                )
                
                _vectorstore_cache = vectorstore
            
            vectorstore.add_documents(documents)
            return vectorstore
        else:    
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                persist_directory=persist_directory
            )
            
            _vectorstore_cache = vectorstore
            return vectorstore
    except Exception as e:
        logger.error(f"Failed to create or update vector store: {e}")
        return None
    

def load_vectorstore(
    persist_directory: Optional[str] = None
) -> Optional[Chroma]:
    """
    Load vector store
    
    Args:
        persist_directory: Persistence directory
    
    Returns:
        Chroma: Vector store instance or None
    """
    global _vectorstore_cache
    
    if _vectorstore_cache is not None:
        return _vectorstore_cache
    
    if persist_directory is None:
        persist_directory = VECTORSTORE_DIR
    
    if not os.path.exists(persist_directory):
        logger.warning(f"Vector store directory {persist_directory} does not exist")
        return None
    
    if len(os.listdir(persist_directory)) == 0:
        logger.warning(f"Vector store directory {persist_directory} is empty")
        return None
    
    embedding_function = get_embeddings()
    if embedding_function is None:
        logger.error("Failed to initialize embedding model, cannot load vector store")
        return None
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
        
        _vectorstore_cache = vectorstore
        
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None