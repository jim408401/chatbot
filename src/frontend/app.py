import os
import sys
import uuid
import time
import logging
import streamlit as st
from pathlib import Path

try:
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
except NameError:
    current_dir = Path.cwd()
root_dir = current_dir.parent.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))

for logger_name in ['streamlit', 'streamlit.runtime']:
    logging.getLogger(logger_name).handlers.clear()
    logging.getLogger(logger_name).propagate = False

from src.backend.logging import get_app_logger

logger = get_app_logger()

# --- Global Settings ---
PAGE_CONFIG = {
    "page_title": "ITRD Chatbot",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
st.set_page_config(**PAGE_CONFIG)

# --- Custom CSS ---
with open("src/frontend/static/app.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Initialize Session State ---
def init_session_state():
    defaults = {
        "user": None, "is_logged_in": False, "current_page": "dds",
        "chat_history": [], "selected_documents": [], "documents_list": [],
        "last_doc_fetch_time": 0, "chunk_size": 300, "chunk_overlap": 100,
        "temperature": 0.0, "retriever_k": 3, "reranker_top_n": 3,
        "use_reranker": False, "llm_model": "shmily_006/Qw3", "embedding_model": "bge-m3", "language": "中文"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if "temp_user_id" not in st.session_state:
        st.session_state.temp_user_id = f"anon-{int(time.time())}-{str(uuid.uuid4())[:8]}"
        logger.info(f"Generated temporary user ID: {st.session_state.temp_user_id}")

# --- User Info UI ---
def render_user_info():
    # Login/Logout section
    if st.session_state.get("is_logged_in") and st.session_state.user:
        username = st.session_state.user.get("username", "Unknown User")
        is_admin = st.session_state.user.get("is_admin", False)
        role_text = "Administrator" if is_admin else "Regular User"
        
        avatar_letter = username[0].upper() if username else "U"
        
        st.sidebar.markdown(f"""
            <div class="user-info">
                <div class="user-avatar">{avatar_letter}</div>
                <div class="user-details">
                    <div class="username">{username}</div>
                    <div class="user-role">{role_text}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Logout", type="secondary", use_container_width=True):
            st.session_state.is_logged_in = False
            st.session_state.user = None
            st.rerun()
    return False

# --- Login/Logout Page ---
def login_page():
    from src.frontend._pages import login
    if st.session_state.get("is_logged_in") and st.session_state.user:
        if st.button("Logout", type="primary", use_container_width=True):
            st.session_state.is_logged_in = False
            st.session_state.user = None
            st.rerun()
    else:
        login.main()

# --- Main Function ---
def main():
    init_session_state()

    try:
        from src.frontend.api_client import check_api_status
        from src.frontend._pages import chat, documents, history, admin, login, dds
        
        def dds_page():
            return dds.main()
            
        def chat_page():
            return chat.main()
            
        def documents_page():
            return documents.main()
            
        def history_page():
            return history.main()
            
        def admin_page():
            return admin.main()
            
    except ImportError as e:
        st.error(f"Error loading modules: {e}")
        st.info("Please check `sys.path` setting and ensure modules exist in `src/frontend/_pages` or `src/frontend/api_client`.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dependencies: {e}")
        st.stop()

    # --- API Check ---
    if 'api_checked' not in st.session_state:
        st.session_state.api_available = check_api_status()
        st.session_state.api_checked = True
    if not st.session_state.api_available:
        st.error("Backend API service is not running...")
        st.info("Please run: `python -m src.backend.main`")
        st.stop()
    
    is_admin = st.session_state.get("is_logged_in") and st.session_state.user and st.session_state.user.get("is_admin")

    pages = [
        st.Page(dds_page, title="DDS Chatbot", icon="💬"),
        st.Page(history_page, title="Chat History", icon="🕘"),
    ]

    if is_admin:
        pages.insert(1, st.Page(chat_page, title="New Chat", icon="➕"))
        pages.insert(2, st.Page(documents_page, title="Documents", icon="📁"))
        pages.append(st.Page(admin_page, title="Dashboard", icon="📈"))

    if not st.session_state.get("is_logged_in"):
        pages.append(st.Page(login_page, title="Login", icon="🔓"))

    render_user_info()

    current_page = st.navigation(pages, expanded=True)
    current_page.run()


if __name__ == "__main__":
    main()