import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from src.rag.models import get_reranker
from src.rag.synonyms import normalize_question
from config import DEFAULT_RETRIEVER_K, DEFAULT_RERANKER_TOP_N

logger = logging.getLogger("dds_chatbot_rag_retriever")

try:
    from src.backend.logging import get_rag_logger
    logger = get_rag_logger("retriever")
except ImportError:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Using fallback logger configuration")


def get_base_retriever(vectorstore, k=None):
    """
    Create a basic vector retriever
    
    Args:
        vectorstore: Vector store object
        k: Number of documents to retrieve, defaults to DEFAULT_RETRIEVER_K
        
    Returns:
        Vector retriever object
    """
    if k is None:
        k = DEFAULT_RETRIEVER_K
        
    if vectorstore is None:
        logger.error("Failed to create retriever: Vector store is empty")
        return None
    
    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        return retriever
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        return None


def retrieve_documents(vectorstore, query: str, return_confidence: bool = True) -> dict:
    """
    Use retriever to perform vector retrieval and calculate the maximum similarity as confidence.
    Args:
        vectorstore: Already initialized vectorstore object
        query: User query
        return_confidence: Whether to return confidence
    Returns:
        dict: {"documents": List[Document], "confidence": float}
    """
    norm_query = normalize_question(query)

    if vectorstore is None:
        logger.error("Vectorstore is None, cannot perform retrieval.")
        return {"documents": [], "confidence": 0.0}
    try:
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(norm_query, k=DEFAULT_RETRIEVER_K)

        results = []
        similarities = []
        
        RELEVANCE_THRESHOLD = 0.1
        
        for doc, score in docs_and_scores:
            if score >= RELEVANCE_THRESHOLD:
                doc.metadata["similarity"] = score
                results.append(doc)
                similarities.append(score)
            else:
                logger.info(f"Document with score {score} below threshold, skipping")

        confidence = max(similarities) if similarities else 0.0
        
        if not results:
            logger.warning("No relevant documents found above threshold")
            return {"documents": [], "confidence": 0.0}
            
        if return_confidence:
            return {
                "documents": results,
                "confidence": confidence
            }
        else:
            return {"documents": results}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"documents": [], "confidence": 0.0}


def format_docs(docs):
    """
    Format document chunks into a string
    
    Args:
        docs: List of document chunks
        
    Returns:
        str: Formatted string
    """
    if not docs:
        return "No documents available"
        
    formatted_docs = []
    
    for i, doc in enumerate(docs):
        content = doc.page_content
        metadata = doc.metadata or {}
        
        doc_id = metadata.get("document_id", "Unknown")
        source = metadata.get("source", "Unknown source")
        
        doc_text = f"[Document {i+1}] {content}\n"
        formatted_docs.append(doc_text)
    
    context = "\n".join(formatted_docs)
    return context


def rerank_documents(query, docs, reranker=None, top_n=None):
    """
    Rerank retrieval results using a Reranker
    
    Args:
        query: User query
        docs: Initially retrieved document chunks
        reranker: Reranker model, will try to load one if None
        top_n: Number of documents to return, defaults to DEFAULT_RERANKER_TOP_N
        
    Returns:
        tuple: (reranked_docs, scores) Reranked document chunks and their scores
    """
    if top_n is None:
        top_n = DEFAULT_RERANKER_TOP_N
        
    if not docs:
        logger.warning("No documents to rerank")
        return [], []
    
    logger.info(f"Starting reranking of {len(docs)} documents")
    
    if reranker is None:
        logger.info("No reranker provided, trying to load default reranker")
        reranker = get_reranker()
        if reranker is None:
            logger.warning("Failed to initialize reranker. Returning original order.")
            return docs[:top_n], []
    
    try:
        rerank_pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.compute_score(rerank_pairs)
        reranked_results = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        
        top_results = reranked_results[:top_n]
        top_docs = [doc for score, doc in top_results]
        top_scores = [score for score, doc in top_results]
        
        if len(top_scores) > 0:
            score_log = ", ".join([f"{i+1}: {score:.4f}" for i, score in enumerate(top_scores[:3])])
            logger.info(f"Top three reranking scores: {score_log}")
        
        return top_docs, top_scores
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return docs[:top_n], []