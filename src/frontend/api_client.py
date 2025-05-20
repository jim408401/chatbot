import os
import sys
import requests
import logging
import streamlit as st
from typing import Dict, List, Any, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from config import BACKEND_PORT, DEFAULT_RETRIEVER_K, DEFAULT_RERANKER_TOP_N, DEFAULT_TEMPERATURE
from src.backend.logging import get_api_logger

logger = get_api_logger()

BASE_URL = f"http://localhost:{BACKEND_PORT}"


def get_api_url(endpoint: str) -> str:
    """Combine complete API URL"""
    return f"{BASE_URL}{endpoint}"


def check_api_status() -> bool:
    """Check if API service is available"""
    try:
        response = requests.get(get_api_url("/"), timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"API service connection failed: {e}")
        logger.error(f"API service connection failed: {e}")
        return False


def register_user(username: str, password: str, is_admin: bool = False) -> Optional[Dict[str, Any]]:
    """Register new user"""
    try:
        response = requests.post(
            get_api_url("/api/users/register"),
            json={"username": username, "password": password, "is_admin": is_admin}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Registration failed: {response.json().get('detail', '')}")
            return None
    except Exception as e:
        st.error(f"Registration failed: {e}")
        return None


def login_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """User login"""
    try:
        response = requests.post(
            get_api_url("/api/users/login"),
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        logger.error(f"Login error: {e}")
        return None


def upload_document(file, user_id: int, chunk_size: int, chunk_overlap: int) -> Optional[Dict[str, Any]]:
    """Upload and process document"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/octet-stream")}
        
        data = {
            "user_id": str(user_id),
            "chunk_size": str(chunk_size),
            "chunk_overlap": str(chunk_overlap)
        }
        
        api_url = get_api_url("/api/documents/upload")
        
        response = requests.post(
            api_url,
            files=files,
            data=data,
            timeout=120
        )
        
        if response.status_code == 200:
            try:
                return response.json()
            except Exception as e:
                logger.error(f"Unable to parse API response JSON: {e}. Raw response text: {response.text[:500]}")
                st.error(f"Failed to parse API response JSON: {e}. Raw response text: {response.text[:500]}")
                return None
        else:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Document processing failed: {error_detail}")
                st.error(f"Document processing failed: {error_detail}")
            except Exception as e:
                logger.error(f"Document processing failed: Status code {response.status_code}. Unable to parse error message: {e}")
                st.error(f"Document processing failed: Status code {response.status_code}. Unable to parse error message: {e}. Raw response: {response.text[:500]}")
            return None
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        st.error(f"Document upload failed: {e}")
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None


def get_user_documents(user_id: int) -> List[Dict[str, Any]]:
    """Get user's document list"""
    try:
        response = requests.get(get_api_url(f"/api/documents/list/{user_id}"))
        if response.status_code == 200:
            return response.json().get("documents", [])
        else:
            logger.error(f"Failed to get document list: {response.json().get('detail', '')}")
            st.error(f"Failed to get document list: {response.json().get('detail', '')}")
            return []
    except Exception as e:
        logger.error(f"Failed to get document list: {e}")
        st.error(f"Failed to get document list: {e}")
        return []


def delete_document(document_id: int) -> bool:
    """Delete document"""
    try:
        response = requests.delete(get_api_url(f"/api/documents/delete/{document_id}"))
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Failed to delete document: {response.json().get('detail', '')}")
            st.error(f"Failed to delete document: {response.json().get('detail', '')}")
            return False
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        st.error(f"Failed to delete document: {e}")
        return False


def get_document_preview(document_id: int) -> Optional[str]:
    """Get document preview"""
    try:
        response = requests.get(get_api_url(f"/api/documents/preview/{document_id}"))
        if response.status_code == 200:
            return response.json().get("preview", "Unable to get document preview")
        else:
            logger.error(f"Failed to get document preview: {response.json().get('detail', '')}")
            st.error(f"Failed to get document preview: {response.json().get('detail', '')}")
            return None
    except Exception as e:
        logger.error(f"Failed to get document preview: {e}")
        st.error(f"Failed to get document preview: {e}")
        return None


def ask_question(user_id: int, question: str, retriever_k: int = DEFAULT_RETRIEVER_K, reranker_top_n: int = DEFAULT_RERANKER_TOP_N, 
               temperature: float = DEFAULT_TEMPERATURE, document_ids: List[int] = None, use_reranker: bool = True,
               llm_model: str = None, embedding_model: str = None, language: str = "English") -> Optional[Dict[str, Any]]:
    """Send question to API and get answer"""
    try:
        payload = {
            "user_id": user_id,
            "question": question,
            "retriever_k": retriever_k,
            "reranker_top_n": reranker_top_n,
            "temperature": temperature,
            "use_reranker": use_reranker,
            "language": language
        }
        
        payload["llm_model"] = llm_model
            
        if embedding_model:
            payload["embedding_model"] = embedding_model
        
        if document_ids and len(document_ids) > 0:
            payload["document_ids"] = document_ids
        
        response = requests.post(
            get_api_url("/api/qa/ask"),
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get answer: {response.json().get('detail', '')}")
            st.error(f"Failed to get answer: {response.json().get('detail', '')}")
            return None
    except Exception as e:
        logger.error(f"Failed to get answer: {e}")
        st.error(f"Failed to get answer: {e}")
        return None


def submit_feedback(record_id: int, feedback: int) -> bool:
    """Submit Q&A feedback"""
    try:
        response = requests.post(
            get_api_url("/api/qa/feedback"),
            json={"record_id": record_id, "feedback": feedback}
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        st.error(f"Failed to submit feedback: {e}")
        return False


def get_user_chat_history(user_id: int) -> List[Dict[str, Any]]:
    """Get user's Q&A history"""
    try:
        api_url = get_api_url(f"/api/qa/history/{user_id}")
        logger.debug(f"Requesting user chat history from URL: {api_url}")
        
        response = requests.get(api_url, timeout=10)
        logger.debug(f"User history response status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                if not isinstance(data, dict):
                    logger.error(f"Unexpected response format (not dictionary type): {type(data)}")
                    return []
                    
                history = data.get("history", [])
                logger.debug(f"Retrieved {len(history)} user history records")
                
                for i, item in enumerate(history):
                    if not isinstance(item, dict):
                        logger.warning(f"History record item {i} is not a dictionary type: {type(item)}")
                        continue
                        
                    required_fields = ["question", "answer", "created_at"]
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        logger.warning(f"History record item {i} missing fields: {missing_fields}")
                
                return history
            except Exception as json_err:
                logger.error(f"Failed to parse JSON response: {json_err}")
                logger.debug(f"Raw response content: {response.text[:500]}...")
                return []
        else:
            logger.error(f"API error response: {response.status_code}")
            try:
                logger.debug(f"Error content: {response.json()}")
            except:
                logger.debug(f"Raw error content: {response.text[:500]}")
            return []
    except requests.RequestException as e:
        logger.error(f"Request exception occurred while getting chat history: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error occurred while getting chat history: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def get_chat_details(record_id: int) -> Optional[Dict[str, Any]]:
    """Get details of a specific conversation"""
    try:
        api_url = get_api_url(f"/api/qa/record/{record_id}")
        logger.debug(f"Requesting chat details from URL: {api_url}")
        
        response = requests.get(api_url, timeout=10)
        logger.debug(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            details = response.json()
            logger.debug(f"Retrieved chat details for record {record_id}")
            
            if "created_at" in details and not "timestamp" in details:
                details["timestamp"] = details["created_at"]
                
            return details
        else:
            logger.error(f"API error response: {response.text}")
            error_msg = f"Failed to get chat details: Status code {response.status_code}"
            try:
                error_detail = response.json().get('detail', 'Unknown error')
                error_msg += f" - {error_detail}"
            except:
                pass
            logger.error(error_msg)
            return None
    except requests.RequestException as e:
        logger.error(f"Request exception occurred while getting chat details: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error occurred while getting chat details: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_admin_qa_history() -> List[Dict[str, Any]]:
    """Get Q&A history for all users (admin use only)"""
    try:
        api_url = get_api_url("/api/admin/qa/history")
        logger.debug(f"Requesting admin history from URL: {api_url}")
        
        response = requests.get(api_url, timeout=10)
        logger.debug(f"Admin history response status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            logger.debug(f"Raw API response: {result}")
            
            if "history" in result:
                history = result.get("history", [])
            else:
                history = result if isinstance(result, list) else []
                
            logger.debug(f"Retrieved {len(history)} admin history records")
            logger.debug(f"Sample data: {history[:2] if history and len(history) > 0 else 'No data'}")
            return history
        else:
            response_text = response.text
            logger.error(f"API error response: {response_text}")
            try:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Failed to get admin history: {error_detail}")
            except:
                st.error(f"Failed to get admin history: Status code {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Exception occurred while getting admin history: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback message: {traceback_str}")
        st.error(f"Failed to get admin history: {e}")
        return []