import os
import sys
import time
import streamlit as st
from .. import api_client
from src.backend.logging import get_app_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)

logger = get_app_logger()

try:
    from src.frontend.api_client import login_user
except ImportError:
    logger.error("Cannot import api_client function")
    st.error("Cannot import api_client function")
    st.stop()


def main():
    with open("src/frontend/static/login.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if st.session_state.get("is_logged_in"):
        st.info("You are already logged in")
        if st.button("Go to Chat", type="primary"):
            st.session_state.current_page = "chat"
            st.rerun()
        st.stop()

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<div style='height: 8vh;'></div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>Welcome Back</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: gray;'>Enter your credentials to sign in</p>", unsafe_allow_html=True)
            
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user", placeholder="ad_account")
                password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
                
                st.write("")
                login_submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            login_error = st.empty()
            
            if "login_error_message" in st.session_state:
                login_error.error(st.session_state.login_error_message)
                st.session_state.pop("login_error_message") 

            if login_submit:
                if not username or not password:
                    logger.warning(f"Login attempt failed: username or password is empty")
                    st.session_state.login_error_message = "Please enter username and password"
                    st.rerun()
                else:
                    with st.spinner("Signing in..."):
                        logger.info(f"User attempting to login: {username}")
                        user = login_user(username, password)
                        if user:
                            logger.info(f"User {username} login successful")
                            st.session_state.user = user
                            st.session_state.is_logged_in = True
                            st.success("Login successful!")
                            st.session_state.current_page = "chat"
                            st.rerun()
                        else:
                            logger.warning(f"User {username} login failed")
                            st.session_state.login_error_message = "Invalid username or password"
                            st.rerun()