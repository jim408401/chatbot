import os
import time
import streamlit as st
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from config import (
    OLLAMA_BASE_URL, 
    LLM_MODEL, 
    EMBEDDING_MODEL, 
    RERANKER_MODEL_PATH,
    DEFAULT_TEMPERATURE
)

logger = logging.getLogger("dds_chatbot_rag_models")

try:
    from src.backend.logging import get_rag_logger
    logger = get_rag_logger("models")
except ImportError:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Using fallback logger configuration")

_cached_models = {
    "llm": {},
    "embeddings": {},
    "reranker": None
}


def is_streamlit_running():
    """Check if running in a Streamlit environment"""
    try:
        return st.runtime.exists()
    except:
        return False


def get_llm(temperature=None, retries=3, timeout=30, model_name=None):
    """
    Initialize and return LLM model with retry mechanism
    """
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    
    if model_name is None:
        model_name = LLM_MODEL
    
    cache_key = f"{model_name}_{temperature}"
    if cache_key in _cached_models["llm"]:
        return _cached_models["llm"][cache_key]
    for attempt in range(retries):
        try:
            llm = ChatOllama(
                model=model_name, 
                temperature=temperature, 
                base_url=OLLAMA_BASE_URL
            )
            _ = llm.invoke("test")
        
            _cached_models["llm"][cache_key] = llm
            return llm
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 1 * (attempt + 1)  
                logger.warning(f"Attempt {attempt+1}/{retries} to initialize LLM '{model_name}' failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to initialize LLM model '{model_name}' after {retries} attempts: {e}")
                return None


def get_embeddings(retries=3, timeout=30, model_name=None):
    """
    Initialize and return Embedding model with retry mechanism
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL
    
    if model_name in _cached_models["embeddings"]:
        return _cached_models["embeddings"][model_name]
    
    for attempt in range(retries):
        try:
            embeddings = OllamaEmbeddings(
                model=model_name, 
                base_url=OLLAMA_BASE_URL
            )
            
            _ = embeddings.embed_query("test query")
        
            _cached_models["embeddings"][model_name] = embeddings
            return embeddings
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 1 * (attempt + 1)
                logger.warning(f"Attempt {attempt+1}/{retries} to initialize embedding model '{model_name}' failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to initialize embedding model '{model_name}' after {retries} attempts: {e}")
                return None


def reranker_supported():
    """
    Check if transformers reranking is supported
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not detected, reranker performance may be affected")
            
        import transformers
        _ = transformers.__version__
        return True
    except ImportError as e:
        logger.warning(f"Reranker dependencies not available: {e}")
        return False


def get_reranker(rerank_path=None):
    """
    Initialize and return reranker model

    Returns:
        callable: Reranking function or None
    """
    if _cached_models["reranker"] is not None:
        return _cached_models["reranker"]
    
    if rerank_path is None:
        rerank_path = RERANKER_MODEL_PATH
        
    if not reranker_supported():
        logger.error("Reranker not supported in this environment")
        return None
        
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(rerank_path)
        model = AutoModelForSequenceClassification.from_pretrained(rerank_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        def rerank_fn(query, docs):
            """Rerank documents using the model"""
            with torch.no_grad():
                inputs = []
                for doc in docs:
                    text = doc.page_content
                    inputs.append([query, text])
                
                scores = []
                for q, d in inputs:
                    encoded = tokenizer(q, d, return_tensors='pt', padding=True, truncation=True).to(device)
                    outputs = model(**encoded)
                    score = outputs.logits.squeeze(-1).item()
                    scores.append(score)
                
                scored_docs = list(zip(scores, docs))
                scored_docs.sort(reverse=True, key=lambda x: x[0])
                
                return [doc for _, doc in scored_docs], [score for score, _ in scored_docs]
        
        _cached_models["reranker"] = rerank_fn
        return rerank_fn
        
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}")
        return None