import os
import sys
import time
import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
from .. import api_client
from src.backend.logging import get_app_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)

logger = get_app_logger()


def add_custom_css():
    with open("src/frontend/static/history.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    from src.frontend.api_client import get_user_chat_history, get_chat_details
except ImportError:
    st.error("Cannot import api_client function")
    st.stop()


def format_timestamp(ts_str):
    """Format ISO timestamp to readable format"""
    if not ts_str:
        return "N/A"
    
    try:
        dt_obj = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        
        taiwan_tz = timezone(timedelta(hours=8))
        dt_obj = dt_obj.astimezone(taiwan_tz)
        
        return dt_obj.strftime("%b %d, %Y %I:%M %p")
    except Exception as e:
        logger.error(f"Time conversion error: {e}, original string: '{ts_str}'")
        return f"{ts_str} (format error)"


def format_feedback(feedback):
    """Format feedback value to readable text"""
    if feedback is None:
        return "No Feedback"
    elif feedback > 0:
        return "👍 Positive"
    elif feedback < 0:
        return "👎 Negative"
    else:
        return "Neutral"


def main():
    add_custom_css()
    
    st.title("Chat History")

    search_query = st.text_input(
        "Search",
        placeholder="Search questions and responses...",
        key="history_search",
        label_visibility="collapsed"
    )

    @st.cache_data(ttl=30)
    def load_user_history(_user_id):
        try:
            history = get_user_chat_history(_user_id)
            
            for item in history:
                if "created_at" in item and not "timestamp" in item:
                    item["timestamp"] = item["created_at"]
            
            if history:
                history.sort(key=lambda x: x.get('timestamp', x.get('created_at', '1970-01-01T00:00:00Z')), reverse=True)
            return history
        except Exception as e:
            st.error(f"Error fetching history: {e}")
            return []

    if not st.session_state.get("is_logged_in") or "user" not in st.session_state:
        if "temp_user_id" in st.session_state:
            temp_id = st.session_state.temp_user_id
            user_id = abs(hash(temp_id)) % 10000000
        else:
            user_id = 0
    else:
        user_id = st.session_state.user["id"]

    all_history = load_user_history(user_id)
    
    if not all_history:
        st.info("No chat history found.")
    else:
        filtered_history = all_history
        if search_query:
            filtered_history = []
            for item in all_history:
                question = item.get("question", "").lower()
                answer = item.get("answer", "").lower()
                if search_query.lower() in question or search_query.lower() in answer:
                    filtered_history.append(item)
            
            if not filtered_history:
                st.warning(f"No results found for '{search_query}'")
            else:
                st.success(f"Found {len(filtered_history)} results for '{search_query}'")
        
        df_data = []
        for item in filtered_history:
            record_id = item.get("id")
            
            question = item.get("question", "N/A")
            answer = item.get("answer", "N/A")
            
            display_q = question
            display_a = answer
            
            if search_query:
                search_term = search_query.lower()
                if search_term in display_q.lower():
                    highlighted_q = display_q.replace(
                        search_query, 
                        search_query
                    )
                    display_q = highlighted_q
                
                if search_term in display_a.lower():
                    highlighted_a = display_a.replace(
                        search_query, 
                        search_query
                    )
                    display_a = highlighted_a
            
            timestamp = item.get("timestamp", item.get("created_at", ""))
            date_str = format_timestamp(timestamp)
            
            response_time_ms = item.get("response_time_ms")
            resp_time_str = f"{response_time_ms / 1000:.1f}s" if response_time_ms is not None else "N/A"
            
            feedback = item.get("feedback")
            feedback_str = format_feedback(feedback)
            
            confidence = item.get("confidence")
            confidence_source = item.get("confidence_source", "")
            
            if isinstance(confidence, float):
                confidence = abs(confidence) if confidence < 0 else confidence
                confidence = min(confidence, 1.0)
                percent = confidence * 100
                if confidence_source == "reranker":
                    confidence_str = f"{percent:.1f}% (reranker)"
                else:
                    confidence_str = f"{percent:.1f}%" 
            else:
                confidence_str = "N/A"
            
            cpu_usage = item.get("cpu_usage", "N/A")
            gpu_usage = item.get("gpu_usage", "N/A")
            model_name = item.get("model_name", "N/A")
            df_data.append({
                "ID": record_id,
                "Question": display_q,
                "Response": display_a,
                "Confidence Score": confidence_str,
                "User Feedback": feedback_str,
                "Response Time": resp_time_str,
                "Timestamp": date_str,
                "GPU (%)": gpu_usage,
                "CPU (%)": cpu_usage,
                "Model Name": model_name
            })
        
        df = pd.DataFrame(df_data)
        
        st.dataframe(
            df,
            column_config={
                "ID": None,
                "Question": st.column_config.TextColumn("Question", width="medium"),
                "Response": st.column_config.TextColumn("Response", width="medium"),
                "Confidence Score": st.column_config.TextColumn("Confidence Score"),
                "User Feedback": st.column_config.TextColumn("User Feedback"),
                "Response Time": st.column_config.TextColumn("Response Time"),
                "Timestamp": st.column_config.TextColumn("Timestamp"),
                "GPU (%)": st.column_config.TextColumn("GPU (%)"),
                "CPU (%)": st.column_config.TextColumn("CPU (%)"),
                "Model Name": st.column_config.TextColumn("Model Name"),
            },
            hide_index=True,
            column_order=["Question", "Response", "Confidence Score", "User Feedback", "Response Time", "Timestamp", "GPU (%)", "CPU (%)", "Model Name"],
            use_container_width=True
        )