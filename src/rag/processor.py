import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from typing import List, Optional
from langchain_core.documents import Document

logger = logging.getLogger("dds_chatbot_rag_processor")

try:
    from src.backend.logging import get_rag_logger
    logger = get_rag_logger("processor")
except ImportError:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Using fallback logger configuration")


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split documents into smaller chunks
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk, defaults to value in config
        chunk_overlap: Size of overlap between chunks, defaults to value in config
        
    Returns:
        List[Document]: List of document chunks after splitting
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    
    if chunk_overlap is None:
        chunk_overlap = DEFAULT_CHUNK_OVERLAP
    
    if not documents:
        logger.warning("No documents to split")
        return []
    
    if all(doc.metadata and doc.metadata.get('source') == 'dds_qa' for doc in documents):
        return documents
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        
        doc_chunks = text_splitter.split_documents(documents)
        for chunk in doc_chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
        return doc_chunks
    except Exception as e:
        logger.error(f"Document splitting failed: {e}")
        return []