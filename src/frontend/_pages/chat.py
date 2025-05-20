import os
import sys
import time
import streamlit as st
from .. import api_client
from config import DEFAULT_RETRIEVER_K, DEFAULT_RERANKER_TOP_N, DEFAULT_TEMPERATURE, AVAILABLE_MODELS, DEFAULT_MODEL
from src.backend.logging import get_app_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)

logger = get_app_logger()


def add_fixed_input_css():
    with open("src/frontend/static/chat_dds.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def display_chat_message(message):
    """Display a single chat message and handle feedback buttons"""
    role = message.get("role")
    content = message.get("content")
    record_id = message.get("record_id")
    feedback_given = message.get("feedback_given", False)

    with st.chat_message(role):
        st.markdown(content)
        
        if role == "assistant" and record_id is not None and not feedback_given:
            selected = st.feedback(
                "thumbs",
                key=f"feedback_{record_id}"
            )
            
            if selected is not None:
                feedback_value = 1 if selected == 1 else -1
                feedback_text = "Thanks for your feedback!" if feedback_value == 1 else "Thanks for your feedback, we'll work to improve"
                
                if api_client.submit_feedback(record_id, feedback_value):
                    for msg in st.session_state.chat_history:
                        if msg.get("record_id") == record_id:
                            msg["feedback_given"] = True
                            break
                    st.toast(feedback_text)
                    st.rerun()
                else:
                    st.toast("Feedback submission failed")


def get_available_documents(user_id):
    """Get list of available documents for the user"""
    documents = api_client.get_user_documents(user_id)
    filtered_documents = [doc for doc in documents if doc.get("filename") != "dds_qa.txt"]
    return filtered_documents


def render_document_selector():
    """Render document selector in the sidebar"""
    user_id = 0
    if st.session_state.get("is_logged_in") and st.session_state.get("user"):
        user_id = st.session_state.user.get("id", 0)
    
    documents = get_available_documents(user_id)
    
    if not documents:
        st.sidebar.info("Upload documents on the Documents page.")
        st.session_state.selected_document_id = None
        st.session_state.selected_document_name = None
        return
    
    doc_options = {doc["filename"]: doc["id"] for doc in documents}
    
    options_list = ["- Select a document -"] + list(doc_options.keys())
    
    default_index = 0 
    
    if "selected_document_name" in st.session_state and st.session_state.selected_document_name:
        if st.session_state.selected_document_name in doc_options:
            default_index = options_list.index(st.session_state.selected_document_name)
    
    selected_doc = st.sidebar.selectbox(
        "Select document to query",
        options=options_list,
        index=default_index,
        key="doc_selector_single",
        label_visibility="collapsed"
    )
    
    previous_selection = st.session_state.get("selected_document_name")
    
    if selected_doc != "- Select a document -" and selected_doc in doc_options:
        doc_id = doc_options[selected_doc]
        st.session_state.selected_document_id = int(doc_id) if doc_id is not None else None
        st.session_state.selected_document_name = selected_doc
        
        max_length = 10
        display_name = selected_doc
        if len(selected_doc) > max_length:
            display_name = selected_doc[:max_length] + "..."
            
        st.sidebar.caption(f"Selected: {display_name} :green[success] :white_check_mark:")
        
        if previous_selection != selected_doc:
            st.rerun()
    else:
        st.session_state.selected_document_id = None
        st.session_state.selected_document_name = None
        st.sidebar.caption("No document selected :warning:")
        
        if previous_selection is not None:
            st.rerun()
    
    if "selected_documents" in st.session_state:
        del st.session_state.selected_documents


def render_chat_settings():
    """Render chat settings"""
    if "answer_language" not in st.session_state:
        st.session_state["answer_language"] = "中文"

    st.sidebar.selectbox(
        "Language Selection",
        ["中文", "English"],
        key="answer_language",
        help="Choose the language for assistant responses."
    )
    
    with st.sidebar.expander("**Model Settings**", expanded=False):
        st.checkbox("Use Reranker", key="use_reranker", 
                   help="Turn off for faster responses, but answers may be less accurate",
                   value=False)
        
        model_labels = {
            "shmily_006/Qw3": "Accuracy | Slower",
            "llama3.2:3b": "Fast | Less accurate"
        }
        
        if "llm_model" not in st.session_state:
            default_index = AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0
            if AVAILABLE_MODELS[default_index] != "shmily_006/Qw3":
                default_index = AVAILABLE_MODELS.index("shmily_006/Qw3") if "shmily_006/Qw3" in AVAILABLE_MODELS else 0
                
            selected_model = st.selectbox(
                "LLM Model", 
                AVAILABLE_MODELS, 
                index=default_index,
                format_func=lambda x: f"{x} - {model_labels.get(x, '')}",
                key="llm_model",
                help="Choose the model for the assistant:\n"
                "-Accuracy | Slower (shmily_006/Qw3)\n"
                "-Fast | Less accurate (llama3.2:3b)"
            )
        else:
            current_index = 0
            if st.session_state["llm_model"] in AVAILABLE_MODELS:
                current_index = AVAILABLE_MODELS.index(st.session_state["llm_model"])
            else:
                current_index = AVAILABLE_MODELS.index("shmily_006/Qw3") if "shmily_006/Qw3" in AVAILABLE_MODELS else 0
            
            selected_model = st.selectbox(
                "LLM Model", 
                AVAILABLE_MODELS, 
                index=current_index,
                format_func=lambda x: f"{x} - {model_labels.get(x, '')}",
                key="llm_model_selector",
                help="Choose the model for the assistant:\n"
                "-Accuracy | Slower (shmily_006/Qw3)\n"
                "-Fast | Less accurate (llama3.2:3b)"
            )
            st.session_state["llm_model"] = selected_model
    
    if st.sidebar.button("Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


def display_current_documents():
    """Display currently selected documents (no longer shown separately)"""
    pass


def handle_chat_input(chat_container, prompt):
    """Process chat input"""
    user_id = 0
    temp_id = None
    
    if st.session_state.get("is_logged_in") and st.session_state.get("user"):
        user_id = st.session_state.user.get("id", 0)
    elif "temp_user_id" in st.session_state:
        temp_id = st.session_state.temp_user_id
        user_id = abs(hash(temp_id)) % 10000000
        logger.info(f"Using hashed temp user ID: {user_id} from {temp_id}")

    user_message = {"role": "user", "content": prompt}
    st.session_state.chat_history.append(user_message)

    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            record_id = None
            confidence = None

            with st.spinner("Searching for relevant information..."):
                selected_doc_id = st.session_state.get("selected_document_id")
                
                if selected_doc_id is None:
                    answer = "Please select a document from the sidebar."
                    confidence = None
                    confidence_source = None
                else:
                    document_ids = [selected_doc_id]
                    logger.info(f"Querying document IDs: {document_ids}")
                    
                    try:
                        response = api_client.ask_question(
                            user_id=user_id,
                            question=prompt,
                            retriever_k=DEFAULT_RETRIEVER_K,
                            reranker_top_n=DEFAULT_RERANKER_TOP_N,
                            temperature=DEFAULT_TEMPERATURE,
                            use_reranker=st.session_state.get("use_reranker", True),
                            llm_model=st.session_state.get("llm_model"),
                            embedding_model=st.session_state.get("embedding_model"),
                            document_ids=document_ids,
                            language=st.session_state.get("answer_language", "English")
                        )
                        
                        if response:
                            answer = response.get("answer", "Sorry, an error occurred and I couldn't generate a response.")
                            record_id = response.get("record_id")
                            confidence = response.get("confidence")
                            confidence_source = response.get("confidence_source")
                        else:
                            answer = "Sorry, I couldn't connect to the backend service or process your request."
                    except Exception as e:
                        answer = f"An error occurred while processing your request: {e}"

            full_response = ""
            for char in answer:
                full_response += char
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)
            
            assistant_message = {
                "role": "assistant",
                "content": answer
            }
            
            if record_id is not None:
                assistant_message["record_id"] = record_id
                assistant_message["feedback_given"] = False
                
            if confidence is not None:
                assistant_message["confidence"] = confidence
                
            if confidence_source is not None:
                assistant_message["confidence_source"] = confidence_source
                
            st.session_state.chat_history.append(assistant_message)
            st.rerun()


def main():
    add_fixed_input_css()
    render_document_selector()
    render_chat_settings()

    current_doc = st.session_state.get("selected_document_name")
    st.title("Chatbot")
    
    if current_doc is not None:
        st.caption(f"Chat with {current_doc}")
    else:
        st.caption("Chat with your uploaded documents")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message)
    
    st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
    prompt = st.chat_input("Enter your question...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if prompt:
        handle_chat_input(chat_container, prompt)