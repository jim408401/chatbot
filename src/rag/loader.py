import os
import uuid
import logging
from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
)
from fastapi import UploadFile
from config import UPLOADS_DIR

logger = logging.getLogger("dds_chatbot_rag_loader")

try:
    from src.backend.logging import get_rag_logger
    logger = get_rag_logger("loader")
except ImportError:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Using fallback logger configuration")


def save_uploaded_file(uploaded_file: UploadFile) -> Optional[str]:
    """
    Save uploaded file to disk

    Args:
        uploaded_file: The uploaded file

    Returns:
        str: Path to the saved file, or None if failed
    """
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        filename = uploaded_file.filename
        file_extension = os.path.splitext(filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)
        
        contents = uploaded_file.file.read()
        
        with open(file_path, 'wb') as f:
            f.write(contents)
            
        return file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return None
    finally:
        uploaded_file.file.close()


def load_document(file_path: str, original_filename: str = None) -> List[Document]:
    """
    Load document by selecting appropriate loader based on file type

    Args:
        file_path: Path to the file

    Returns:
        List[Document]: List of loaded documents
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
            
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return documents
            
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
            documents = loader.load()
            return documents
            
        elif file_extension in ['.txt', '.md', '.log', '.json', '.yaml', '.yml']:
            check_name = original_filename or os.path.basename(file_path)
            if check_name.startswith('dds_qa'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                qa_chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                from langchain_core.documents import Document
                documents = [Document(page_content=chunk, metadata={"source": "dds_qa"}) for chunk in qa_chunks]
                return documents
            else:
                loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                documents = loader.load()
                return documents
            
        else:
            logger.warning(f"Unknown file extension: {file_extension}, trying to load as text")
            try:
                loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                documents = loader.load()
                return documents
            except Exception as e:
                logger.error(f"Failed to load file as text: {e}")
                return []
    
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        return []