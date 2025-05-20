import os
import sys
import time
import threading
import queue
import streamlit as st
from .. import api_client
from io import BytesIO
from config import DEFAULT_RETRIEVER_K, DEFAULT_RERANKER_TOP_N, DEFAULT_TEMPERATURE, AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.backend.logging import get_app_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)

logger = get_app_logger()

# Global response queue for thread communication
response_queue = queue.Queue()


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
        confidence = message.get("confidence")
        st.markdown(content)
        
        if role == "assistant" and (confidence is not None or record_id is not None):
            cols = st.columns([10, 1])
            with cols[0]:
                if confidence is not None:
                    if confidence >= 0.467:
                        explanation = "✅ **Reliable answer:** Based on available information."
                    else:
                        explanation = "❓ **Reference only:** Knowledge base might lack relevant info. Please contact ITRD for precise assistance."
                    st.markdown(f"<span style='color:gray;font-size:0.97em;'><b>{explanation}</b></span>", unsafe_allow_html=True)
            
            with cols[1]:
                if record_id is not None and not feedback_given:
                    selected = st.feedback(
                        "thumbs",
                        key=f"feedback_{record_id}"
                    )
                    if selected is not None:
                        feedback_value = 1 if selected == 1 else -1
                        feedback_text = "Thanks for the feedback!" if feedback_value == 1 else "Thank you for your feedback, we\\'ll try to improve."
                        if api_client.submit_feedback(record_id, feedback_value):
                            for i, msg_item in enumerate(st.session_state.dds_chat_history):
                                if msg_item.get("record_id") == record_id:
                                    st.session_state.dds_chat_history[i]["feedback_given"] = True
                                    break
                            st.session_state["show_feedback_msg"] = feedback_text
                            st.rerun()
                        else:
                            st.session_state["show_feedback_msg"] = "Failed to submit feedback"
                            st.rerun()


def render_chat_settings():
    """Render chat settings, ensuring shmily_006/Qw3 is the default if available."""
    if "answer_language" not in st.session_state:
        st.session_state["answer_language"] = "中文"

    st.sidebar.selectbox(
        "Language Selection",
        ["中文", "English"],
        key="answer_language",
        help="Choose the language for assistant responses."
    )

    model_keys = ["shmily_006/Qw3", "llama3.2:3b"]
    model_labels = [
        "Accuracy | Slower",
        "Fast | Less accurate"
    ]
    model_key_to_label = dict(zip(model_keys, model_labels))

    # Determine the initial index for the selectbox
    initial_select_index = 0
    desired_default_model = "shmily_006/Qw3"

    # Get current model from session state, if it exists and is valid
    current_llm_model_in_state = st.session_state.get("llm_model")

    if current_llm_model_in_state and current_llm_model_in_state in model_keys:
        initial_select_index = model_keys.index(current_llm_model_in_state)
    elif desired_default_model in model_keys:
        initial_select_index = model_keys.index(desired_default_model)
        st.session_state.llm_model = desired_default_model # Set as current model
    elif model_keys: # Fallback to the first model if shmily is not available
        initial_select_index = 0
        st.session_state.llm_model = model_keys[0]
    else: # No models available
        st.session_state.llm_model = None
        st.sidebar.warning("No models available for selection.")

    selected_model_display = st.sidebar.selectbox(
        "Model Selection",
        options=model_keys if model_keys else ["No models available"],
        format_func=lambda x: model_key_to_label.get(x, x) if model_keys else lambda x: x,
        index=initial_select_index,
        key="llm_model_selector_display", 
        help=(
            "Choose the model for the assistant:\n"
            "-Accuracy | Slower (shmily_006/Qw3)\n"
            "-Fast | Less accurate (llama3.2:3b)"
        ),
        disabled=not model_keys 
    )
    
    # Update the actual session state if the selector changes and the new selection is valid
    if model_keys and selected_model_display != st.session_state.llm_model and selected_model_display in model_keys:
        st.session_state.llm_model = selected_model_display
        st.rerun() # Rerun to reflect change if needed by other parts of the app immediately

    if "use_reranker" not in st.session_state: 
        st.session_state["use_reranker"] = False

    if st.sidebar.button("Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.dds_chat_history = []
        
        if st.session_state.get("dds_processing", False):
            st.session_state.dds_interrupt_generation = True 
            st.session_state.dds_processing = False
        
        if st.session_state.get("dds_streaming_active", False):
            st.session_state.dds_streaming_active = False
            st.session_state.dds_interrupt_stream = True 
            if "dds_pending_stream_data" in st.session_state:
                del st.session_state.dds_pending_stream_data

        if "dds_background_thread" in st.session_state and st.session_state.dds_background_thread is not None:
            st.session_state.dds_background_thread = None 
                
        if "dds_last_response" in st.session_state:
            st.session_state.dds_last_response = None

        if "dds_current_prompt" in st.session_state:
            del st.session_state.dds_current_prompt
            
        while not response_queue.empty():
            try:
                response_queue.get_nowait()
            except queue.Empty:
                break
                
        st.rerun()


def generate_response_in_background(prompt, user_id, document_ids, language, llm_model, use_reranker):
    """Generate response in background thread. Does not access session_state for interrupt."""
    response_data = {}
    try:
        response = api_client.ask_question(
            user_id=user_id,
            question=prompt,
            retriever_k=DEFAULT_RETRIEVER_K,
            reranker_top_n=DEFAULT_RERANKER_TOP_N,
            temperature=DEFAULT_TEMPERATURE,
            use_reranker=use_reranker,
            llm_model=llm_model,
            embedding_model=None, 
            document_ids=document_ids,
            language=language
        )

        if response:
            answer = response.get("answer", "Sorry, I cannot connect to the backend service or process your request.")
            record_id = response.get("record_id")
            confidence = response.get("confidence")
            
            response_data = {
                "status": "completed",
                "content": answer,
                "record_id": record_id,
                "confidence": confidence,
                "prompt": prompt 
            }
            
            short_answer = answer[:20] + ('...' if len(answer) > 20 else '')
            conf_str = f"{confidence:.3f}" if isinstance(confidence, float) else str(confidence)
            logger.info(f'[A] record_id={record_id}, answer="{short_answer}", confidence={conf_str}')
        else:
            error_msg = "Sorry, I cannot connect to the backend service or process your request."
            response_data = {"status": "error", "content": error_msg, "prompt": prompt}
    except Exception as e:
        error_msg = f"An error occurred while processing your request: {e}"
        response_data = {"status": "error", "content": error_msg, "prompt": prompt}
        logger.error(f"Error generating response: {e}", exc_info=True)
    finally:
        response_queue.put(response_data)


def handle_chat_input(chat_container, prompt): 
    """Process chat input by starting background generation."""
    if st.session_state.get("dds_processing", False) or st.session_state.get("dds_streaming_active", False) :
        st.warning("A query is already being processed or streamed, please wait...")
        return
        
    user_id = 0
    if st.session_state.get("is_logged_in") and st.session_state.get("user"):
        user_id = st.session_state.user.get("id", 0)
    elif "temp_user_id" in st.session_state:
        user_id = abs(hash(st.session_state.temp_user_id)) % 10000000

    user_message = {"role": "user", "content": prompt}
    st.session_state.dds_chat_history.append(user_message)
    
    document_ids = [st.session_state.dds_document_id]
    language = st.session_state.get("answer_language", "English")
    llm_model = st.session_state.get("llm_model") 
    use_reranker = st.session_state.get("use_reranker", False)
    
    background_thread = threading.Thread(
        target=generate_response_in_background,
        args=(prompt, user_id, document_ids, language, llm_model, use_reranker),
        daemon=True
    )
    background_thread.start()
    
    st.session_state.dds_background_thread = background_thread
    st.session_state.dds_processing = True
    st.session_state.dds_current_prompt = prompt 
    st.session_state.dds_interrupt_generation = False 
    st.session_state.dds_interrupt_stream = False 

    logger.info(f'[Q] question="{prompt[:50]}..."')
    st.rerun()


def upload_dds_qa_file():
    """Upload DDS QA file and return document ID"""
    if "dds_document_id" in st.session_state and st.session_state.dds_document_id is not None:
        return st.session_state.dds_document_id
    
    user_id = 0
    if st.session_state.get("is_logged_in") and st.session_state.get("user"):
        user_id = st.session_state.user.get("id", 0)
    
    try:
        documents = api_client.get_user_documents(user_id)
        for doc in documents:
            if doc.get("filename") == "dds_qa.txt":
                st.session_state.dds_document_id = doc.get("id")
                logger.info(f"Found existing DDS QA file for user {user_id}, ID: {st.session_state.dds_document_id}")
                return st.session_state.dds_document_id
    except Exception as e:
        logger.warning(f"Could not retrieve user documents for user {user_id}: {e}")

    dds_file_path = "data/dds_qa.txt" 
    if not os.path.exists(dds_file_path):
        logger.error(f"DDS QA file not found at path: {dds_file_path}")
        st.error(f"DDS QA file not found. Please ensure it exists at '{dds_file_path}'.")
        return None
    
    try:
        with open(dds_file_path, "rb") as f:
            file_content = f.read()
        
        file_obj = BytesIO(file_content)
        file_obj.name = "dds_qa.txt" 
        
        result = api_client.upload_document(
            file=file_obj,
            user_id=user_id, 
            chunk_size=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_overlap=st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        )
        
        if result and result.get("success"):
            st.session_state.dds_document_id = result.get("document_id")
            logger.info(f"DDS QA file uploaded successfully for user {user_id}, ID: {st.session_state.dds_document_id}")
            return st.session_state.dds_document_id
        else:
            error_detail = result.get("detail", "Upload failed with no specific detail.") if result else "Upload response was None."
            logger.error(f"DDS QA file upload failed for user {user_id}: {error_detail}")
            st.error(f"Upload DDS QA file failed: {error_detail}")
            return None
    except Exception as e:
        logger.error(f"Error processing or uploading DDS QA file for user {user_id}: {e}", exc_info=True)
        st.error(f"Error processing DDS QA file: {e}")
        return None


def check_response_queue():
    """Check for completed responses in the queue and handle them."""
    if not response_queue.empty():
        try:
            response_data_from_thread = response_queue.get_nowait() 
            
            if st.session_state.get("dds_current_prompt") != response_data_from_thread.get("prompt"):
                logger.info(f"Discarding stale response for prompt: {response_data_from_thread.get('prompt')} (current: {st.session_state.get('dds_current_prompt')})")
                return 

            if st.session_state.get("dds_interrupt_generation", False):
                logger.info("Interruption signaled. Discarding queued response from background.")
                st.session_state.dds_interrupt_generation = False 
                st.session_state.dds_processing = False 
                if "dds_current_prompt" in st.session_state:
                    del st.session_state.dds_current_prompt
                return 

            status = response_data_from_thread.get("status")

            if status == "error":
                assistant_message = {
                    "role": "assistant",
                    "content": response_data_from_thread.get("content", "An error occurred."),
                    "prompt_answered": response_data_from_thread.get("prompt")
                }
                st.session_state.dds_chat_history.append(assistant_message)
                st.session_state.dds_processing = False
                if "dds_current_prompt" in st.session_state:
                    del st.session_state.dds_current_prompt
                st.rerun()

            elif status == "completed":
                st.session_state.dds_pending_stream_data = response_data_from_thread
                st.session_state.dds_streaming_active = True
                st.session_state.dds_processing = False 
                st.rerun()
        except queue.Empty:
            pass 
        except Exception as e:
            logger.error(f"Error processing response queue: {e}", exc_info=True)
            st.session_state.dds_processing = False 
            st.session_state.dds_streaming_active = False
            if "dds_current_prompt" in st.session_state: 
                del st.session_state.dds_current_prompt


def main():
    add_fixed_input_css()
    
    default_states = {
        "dds_processing": False,
        "dds_streaming_active": False,
        "dds_pending_stream_data": None,
        "dds_interrupt_generation": False,
        "dds_interrupt_stream": False,
        "dds_background_thread": None,
        "dds_chat_history": [],
        "dds_current_prompt": None,
        "show_feedback_msg": None,
        "answer_language": "中文",
        "dds_document_id": None,
        "use_reranker": False,
        "llm_model": "shmily_006/Qw3"
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    render_chat_settings()

    if st.session_state.show_feedback_msg:
        st.toast(st.session_state.show_feedback_msg, icon=":material/thumb_up:")
        st.session_state.show_feedback_msg = None 

    st.title("DDS Chatbot")
    st.caption("Ask me anything about the DDS!")
    
    if st.session_state.dds_document_id is None: 
        with st.spinner("Loading DDS document..."): 
            st.session_state.dds_document_id = upload_dds_qa_file()
            if st.session_state.dds_document_id is None:
                return 

    check_response_queue() 

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.dds_chat_history:
            display_chat_message(message)

        if st.session_state.get("dds_streaming_active", False):
            data_to_stream = st.session_state.get("dds_pending_stream_data")
            if data_to_stream and data_to_stream.get("content"):
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response_streamed = ""
                    answer_content = data_to_stream['content']
                    
                    for char_code in answer_content:
                        if st.session_state.get("dds_interrupt_stream", False):
                            message_placeholder.markdown(full_response_streamed + " (Streaming interrupted by user)")
                            break
                        full_response_streamed += char_code
                        message_placeholder.markdown(full_response_streamed + "▌")
                        time.sleep(0.05) 
                    
                    if not st.session_state.get("dds_interrupt_stream", False):
                        message_placeholder.markdown(full_response_streamed)
                    else: 
                         message_placeholder.markdown(full_response_streamed + " (Streaming interrupted by user)")

                    final_content = full_response_streamed
                    if st.session_state.get("dds_interrupt_stream", False):
                         if not final_content.endswith("(Streaming interrupted by user)"):
                            final_content += " (Streaming interrupted by user)"
                    
                    assistant_message_final = {
                        "role": "assistant",
                        "content": final_content,
                        "record_id": data_to_stream.get("record_id"),
                        "confidence": data_to_stream.get("confidence"),
                        "feedback_given": False,
                        "prompt_answered": data_to_stream.get("prompt") 
                    }
                    if not st.session_state.dds_chat_history or \
                       st.session_state.dds_chat_history[-1].get("content") != final_content or \
                       st.session_state.dds_chat_history[-1].get("prompt_answered") != data_to_stream.get("prompt"):
                        st.session_state.dds_chat_history.append(assistant_message_final)
                    
                    st.session_state.dds_streaming_active = False
                    st.session_state.dds_interrupt_stream = False 
                    if "dds_pending_stream_data" in st.session_state:
                        del st.session_state.dds_pending_stream_data
                    if "dds_current_prompt" in st.session_state and \
                       st.session_state.dds_current_prompt == data_to_stream.get("prompt"): 
                         del st.session_state.dds_current_prompt
                    st.rerun()

        elif st.session_state.get("dds_processing", False):
            with st.chat_message("assistant"):
                cols = st.columns([15, 1])
                with cols[0]:
                    with st.spinner("Searching for relevant information..."):
                        st.markdown("Searching for relevant information...")
                with cols[1]:
                    if st.button("⬜", key="interrupt_button_processing_active", help="Stop response generation"):
                        st.session_state.dds_interrupt_generation = True 
                        st.session_state.dds_processing = False 
                        
                        interrupted_msg_content = "Response generation has been interrupted by the user."
                        already_added = any(
                            msg.get("role") == "assistant" and
                            msg.get("content") == interrupted_msg_content and
                            msg.get("prompt_answered") == st.session_state.get("dds_current_prompt")
                            for msg in st.session_state.dds_chat_history
                        )
                        if not already_added:
                            interrupted_msg = {
                                "role": "assistant", 
                                "content": interrupted_msg_content,
                                "prompt_answered": st.session_state.get("dds_current_prompt")
                            }
                            st.session_state.dds_chat_history.append(interrupted_msg)
                        
                        if "dds_current_prompt" in st.session_state:
                            del st.session_state.dds_current_prompt
                        st.rerun()
            
            if st.session_state.get("dds_processing", False):
                time.sleep(0.2) 
                st.rerun() 
    
    prompt_disabled = st.session_state.get("dds_processing", False) or \
                      st.session_state.get("dds_streaming_active", False)
    prompt_text = "Processing, please wait..." if prompt_disabled else "Enter your question..."
    
    prompt_input_key = "chat_input_dds" 
    prompt = st.chat_input(prompt_text, disabled=prompt_disabled, key=prompt_input_key)
    
    if prompt and not prompt_disabled:
        handle_chat_input(chat_container, prompt)