import os
import time
import logging
import subprocess
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag.loader import save_uploaded_file, load_document
from src.rag.processor import split_documents
from src.rag.vectorstore import create_vectorstore, load_vectorstore
from src.rag.retriever import get_base_retriever, rerank_documents, retrieve_documents
from src.rag.chain import create_rag_chain, generate_response
from src.backend.database import (
    add_user, get_user, add_qa_record, update_qa_feedback,
    get_user_qa_history, get_all_qa_history, add_document_record, get_user_documents,
    delete_document_record, get_document_by_id, get_qa_record_by_id
)
from config import (
    UPLOADS_DIR, VECTORSTORE_DIR, DEFAULT_RETRIEVER_K, 
    DEFAULT_RERANKER_TOP_N, DEFAULT_TEMPERATURE, AVAILABLE_MODELS, DEFAULT_MODEL
)

logger = logging.getLogger("dds_chatbot_api")
logger.setLevel(logging.INFO)

try:
    from src.backend.logging import get_api_logger
    logger = get_api_logger()
except ImportError:
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info("Using API fallback logger configuration")

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('backoff').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)

class User(BaseModel):
    username: str
    password: str
    is_admin: bool = False

class UserLogin(BaseModel):
    username: str
    password: str

class QARequest(BaseModel):
    user_id: int
    question: str
    retriever_k: int = DEFAULT_RETRIEVER_K
    reranker_top_n: int = DEFAULT_RERANKER_TOP_N
    temperature: float = DEFAULT_TEMPERATURE
    use_reranker: bool = True
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    document_ids: Optional[List[int]] = None
    language: str = "English"

class FeedbackRequest(BaseModel):
    record_id: int
    feedback: int

app = FastAPI(title="DDS Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"status": "ok", "message": "DDS Chatbot API is running"}


@app.post("/api/users/register")
async def register_user(user: User):
    user_id = add_user(user.username, user.password, user.is_admin)
    if user_id is None:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"user_id": user_id, "username": user.username, "is_admin": user.is_admin}


@app.post("/api/users/login")
async def login_user(user_login: UserLogin):
    user = get_user(user_login.username, user_login.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return user


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    chunk_size: int = Form(300),
    chunk_overlap: int = Form(100)
):
    file_path = save_uploaded_file(file)
    if file_path is None:
        logger.error(f"Failed to save file: {file.filename}")
        raise HTTPException(status_code=500, detail="Failed to save file")
    try:
        documents = load_document(file_path, original_filename=file.filename)
        if not documents:
            logger.error(f"Failed to load document or unsupported format: {file_path}")
            raise HTTPException(status_code=400, detail=f"Failed to load document or unsupported format: {file.filename}")
        
        docs_split = split_documents(documents, chunk_size, chunk_overlap)
        if not docs_split:
            logger.error(f"Failed to split document: {file_path}")
            raise HTTPException(status_code=500, detail="Failed to split document")
        
        file_size = os.path.getsize(file_path)
        doc_id = add_document_record(
            user_id=user_id,
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            chunk_count=len(docs_split)
        )
        
        for doc in docs_split:
            doc.metadata["document_id"] = doc_id
            
        vectorstore = create_vectorstore(docs_split)
        if vectorstore is None:
            logger.error(f"Failed to create vector store for document: {file_path}")
            raise HTTPException(status_code=500, detail="Failed to create vector store for document")
        
        result = {
            "filename": file.filename,
            "document_id": doc_id,
            "chunk_count": len(docs_split),
            "success": True
        }
        logger.info(f"Upload completed: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error during document processing: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback: {traceback_str}")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed file due to error: {file_path}")
            except:
                logger.error(f"Failed to remove file after error: {file_path}")
                pass
        raise HTTPException(status_code=500, detail=f"Error during document processing: {str(e)}")


@app.get("/api/documents/list/{user_id}")
async def list_user_documents(user_id: int):
    documents = get_user_documents(user_id)
    return {"documents": documents}


@app.delete("/api/documents/delete/{document_id}")
async def delete_document(document_id: int):
    try:
        document = get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
                
        file_path = document.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file: {file_path}")
        
        success = delete_document_record(document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document from database")
        
        return {"success": True, "message": f"Document deleted successfully: {document.get('filename')}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during document deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during document deletion: {str(e)}")


@app.get("/api/documents/preview/{document_id}")
async def get_document_preview(document_id: int):
    try:
        document = get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")
        
        file_path = document.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "document_id": document_id,
            "filename": document.get("filename"),
            "file_size": document.get("file_size"),
            "upload_date": document.get("upload_date"),
            "chunk_count": document.get("chunk_count"),
            "status": "File exists"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during document preview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during document preview: {str(e)}")


@app.post("/api/qa/ask")
async def ask_question(request: QARequest):
    start_time = time.time()
    model_name = request.llm_model or "default"
    
    if request.llm_model and request.llm_model not in AVAILABLE_MODELS:
        logger.warning(f"Requested model '{request.llm_model}' is not in the available models list. Using default model '{DEFAULT_MODEL}'.")
        model_name = DEFAULT_MODEL
    elif not request.llm_model:
        model_name = DEFAULT_MODEL
        
    reranker_status = "enabled" if request.use_reranker else "disabled"
    logger.debug(f"[QA] model={model_name}, doc_ids={request.document_ids if request.document_ids else 'all'}, reranker={reranker_status}, retriever_k={request.retriever_k}, reranker_top_n={request.reranker_top_n}")
    
    if request.document_ids and len(request.document_ids) > 0:
        existing_docs = []
        for doc_id in request.document_ids:
            doc = get_document_by_id(doc_id)
            if doc:
                existing_docs.append(doc)
        
        if not existing_docs:
            logger.warning(f"No valid documents found for specified document IDs: {request.document_ids}")
            return {
                "answer": "Sorry, we couldn't find the specified documents. Please check if the documents have been deleted.",
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
    
    vectorstore = load_vectorstore()
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="No files have been processed yet")
    
    try:
        retrieval_result = retrieve_documents(vectorstore, request.question, return_confidence=True)
        docs = retrieval_result.get("documents", [])
        confidence = retrieval_result.get("confidence", 0.0)
        if not docs:
            # 根據選擇的語言提供統一的「無相關資訊」回應
            no_info_message = "無相關資訊。系統無法找到與您的問題相關的內容。" if request.language == "中文" else "No relevant information found. The system cannot find content related to your question."
            return {
                "answer": no_info_message,
                "response_time_ms": int((time.time() - start_time) * 1000),
                "confidence": confidence
            }
        
        if request.document_ids and len(request.document_ids) > 0:
            filtered_docs = [
                doc for doc in docs 
                if 'document_id' in doc.metadata and 
                doc.metadata['document_id'] in request.document_ids
            ]
            if not filtered_docs:
                logger.warning(f"[QA] No valid documents found for doc_ids: {request.document_ids}")
            else:
                docs = filtered_docs

        top_docs = docs
        if request.use_reranker and len(docs) > 1:
            top_docs = rerank_documents(request.question, docs, top_n=request.reranker_top_n)[0]
        elif len(docs) > request.reranker_top_n:
            top_docs = docs[:request.reranker_top_n]

        rag_chain, _ = create_rag_chain(
            temperature=request.temperature,
            model_name=model_name,
            language=request.language
        )
        
        if rag_chain is None:
            raise HTTPException(status_code=500, 
                               detail=f"Unable to create RAG chain using model {model_name}. Please confirm the model is correctly installed in Ollama.")
        
        answer = generate_response(rag_chain, top_docs, request.question)
        response_time_ms = int((time.time() - start_time) * 1000)

        cpu_usage = None
        gpu_usage = None
        used_model = model_name
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                output = result.stdout
                lines = output.splitlines()
                model_lines = [line for line in lines if used_model in line]
                cpu_val = None
                gpu_val = None
                matched_line = model_lines[0] if model_lines else None
                if matched_line:
                    parts = matched_line.split()
                    if len(parts) >= 6:
                        usage = parts[4]
                        processor_type = parts[5]
                        if processor_type == "CPU/GPU" and "/" in usage:
                            cpu_part, gpu_part = usage.split("/", 1)
                            cpu_val = cpu_part.strip()
                            gpu_val = gpu_part.strip()
                        elif 'GPU' in processor_type:
                            gpu_val = usage
                        elif 'CPU' in processor_type:
                            cpu_val = usage
                def percent_to_int(val):
                    if isinstance(val, str) and '%' in val:
                        try:
                            return int(float(val.split('%')[0].strip()))
                        except Exception:
                            return 0
                    elif val is None or val == 'N/A':
                        return 0
                    try:
                        return int(float(val))
                    except Exception:
                        return 0

                if cpu_val and gpu_val:
                    cpu_usage = percent_to_int(cpu_val)
                    gpu_usage = percent_to_int(gpu_val)
                elif cpu_val:
                    cpu_usage = percent_to_int(cpu_val)
                    gpu_usage = 0
                elif gpu_val:
                    cpu_usage = 0
                    gpu_usage = percent_to_int(gpu_val)
                else:
                    cpu_usage = 0
                    gpu_usage = 0
            else:
                logger.warning(f"ollama ps failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to run ollama ps: {e}")
        
        record_id = add_qa_record(
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            response_time_ms=response_time_ms,
            confidence=confidence,
            confidence_source="retriever",
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            model_name=used_model
        )
        logger.info(f"[Q] question=\"{request.question}\", user_id={request.user_id}")
        logger.info(f"[A] answer=\"{answer[:20]}{'...' if len(answer) > 20 else ''}\",doc_ids={request.document_ids}, record_id={record_id}, language={request.language}, model={used_model}, reranker={reranker_status}, time={response_time_ms}ms, cpu={cpu_usage}%, gpu={gpu_usage}%, confidence={confidence:.4f}")
        return {
            "answer": answer,
            "record_id": record_id,
            "response_time_ms": response_time_ms,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Failed to generate answer using model {model_name}, reranker: {reranker_status}. Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "answer": f"Sorry, processing your question failed: {str(e)}",
            "response_time_ms": int((time.time() - start_time) * 1000)
        }


@app.post("/api/qa/feedback")
async def submit_feedback(request: FeedbackRequest):
    success = update_qa_feedback(request.record_id, request.feedback)
    logger.info(f"Feedback submitted for record ID {request.record_id}: {request.feedback}, result: {success}")
    return {"success": success}


@app.get("/api/qa/history/{user_id}")
async def get_qa_history(user_id: int):
    history = get_user_qa_history(user_id)
    return {"history": history}


@app.get("/api/admin/qa/history")
async def get_admin_qa_history():
    all_history = get_all_qa_history()
    return {"history": all_history}


@app.get("/api/qa/record/{record_id}")
async def get_qa_record(record_id: int):
    record = get_qa_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record not found with ID {record_id}")
    
    return {
        "record_id": record.get("id"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "timestamp": record.get("created_at"),
        "response_time_ms": record.get("response_time_ms"),
        "feedback": record.get("feedback"),
        "user_id": record.get("user_id")
    }