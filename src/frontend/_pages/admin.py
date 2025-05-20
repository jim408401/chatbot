import os
import sys
import traceback
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timezone, timedelta, date
from .. import api_client
from src.backend.logging import get_app_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)

logger = get_app_logger()


def add_custom_css():
    with open("src/frontend/static/admin.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    from src.frontend.api_client import get_admin_qa_history
except ImportError:
    st.error("Cannot import api_client.")
    st.stop()


def format_timestamp(ts_str):
    """Format timestamp"""
    if not ts_str: 
        return "N/A"
    
    try:
        if ts_str.endswith('Z'):
            ts_str = ts_str.replace("Z", "+00:00")
        elif '+' not in ts_str and '-' not in ts_str[10:]:
            ts_str = ts_str + "+00:00"
            
        try:
            dt_obj = datetime.fromisoformat(ts_str)
        except ValueError:
            from dateutil import parser
            dt_obj = parser.parse(ts_str)
        
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        
        taiwan_tz = timezone(timedelta(hours=8))
        dt_obj = dt_obj.astimezone(taiwan_tz)
        
        return dt_obj.strftime("%b %d, %Y %I:%M %p")
    except Exception as e:
        logger.error(f"Time conversion error: {e}, original string: '{ts_str}'")
        return f"{ts_str} (format error)"


def parse_datetime(ts_str):
    """Parse time string"""
    if not ts_str:
        return None
    
    try:
        if ts_str.endswith('Z'):
            ts_str = ts_str.replace("Z", "+00:00")
        elif '+' not in ts_str and '-' not in ts_str[10:]:
            ts_str = ts_str + "+00:00"
            
        try:
            dt_obj = datetime.fromisoformat(ts_str)
        except ValueError:
            from dateutil import parser
            dt_obj = parser.parse(ts_str)
        
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        
        taiwan_tz = timezone(timedelta(hours=8))
        dt_obj = dt_obj.astimezone(taiwan_tz)
        
        return dt_obj
    except Exception as e:
        logger.error(f"Time parsing error: {e}, original string: '{ts_str}'")
        return None


def format_feedback(feedback):
    """Format feedback value"""
    if feedback is None:
        return "No Feedback"
    elif feedback > 0:
        return "👍 Positive"
    elif feedback < 0:
        return "👎 Negative"
    else:
        return "Neutral"


def create_usage_chart(history_data):
    """Create line chart showing Q&A usage over time"""
    if not history_data:
        return None
    
    date_counts = {}
    for item in history_data:
        ts_str = item.get("timestamp", item.get("created_at", ""))
        dt_obj = parse_datetime(ts_str)
        
        if dt_obj:
            date_only = dt_obj.date()
            if date_only in date_counts:
                date_counts[date_only] += 1
            else:
                date_counts[date_only] = 1
    
    chart_data = pd.DataFrame([
        {"Date": date, "QA Count": count} 
        for date, count in date_counts.items()
    ])
    
    if not chart_data.empty:
        chart_data = chart_data.sort_values("Date")
        
        chart = alt.Chart(chart_data).mark_line(
            point=alt.OverlayMarkDef(color="blue", filled=True),
            color="blue"
        ).encode(
            x=alt.X('Date:T', title='Time', axis=alt.Axis(format='%b %d, %Y')),
            y=alt.Y('QA Count:Q', title='Number of Q&A', 
                   scale=alt.Scale(zero=True),
                   axis=alt.Axis(tickMinStep=1)),
            tooltip=['Date:T', 'QA Count:Q']
        ).properties(
            width='container',
            height=300
        )
        
        return chart
    
    return None


def main():
    add_custom_css()
    
    st.title("Dashboard")

    if not st.session_state.get("is_logged_in") or not st.session_state.user.get("is_admin"):
        st.warning("🚫 Access denied. You must be logged in as an admin.")
        st.stop()

    @st.cache_data(ttl=60)
    def load_admin_history():
        try:
            with st.spinner("Loading history..."):
                logger.debug("Calling get_admin_qa_history()...")
                api_result = get_admin_qa_history()
                
                if isinstance(api_result, dict) and "history" in api_result:
                    history = api_result["history"]
                elif isinstance(api_result, list):
                    history = api_result
                else:
                    history = api_result
                
                logger.debug(f"Received data type: {type(api_result)}")
                logger.debug(f"get_admin_qa_history() return result: {type(history)}, length: {len(history) if history else 0}")
                logger.debug(f"Sample data: {history[:2] if history and len(history) > 0 else 'No data'}")
            
            for item in history:
                if isinstance(item, dict):
                    if "created_at" in item and not "timestamp" in item:
                        item["timestamp"] = item["created_at"]
            
            if history:
                history.sort(key=lambda x: x.get('timestamp', x.get('created_at', '1970-01-01T00:00:00Z')), reverse=True)
                logger.debug(f"Sorted history records, first record timestamp: {history[0].get('timestamp', 'N/A') if history else 'N/A'}")
            return history
        except Exception as e:
            logger.error(f"load_admin_history exception occurred: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(f"Error fetching admin history: {e}")
            return []

    all_history = load_admin_history()

    def parse_percent(val):
        if isinstance(val, str) and val.strip().endswith('%'):
            try:
                return float(val.strip().replace('%','').replace('CPU','').replace('GPU','').strip())
            except Exception:
                return 0.0
        elif val is None or val == 'N/A':
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    if all_history:
        st.subheader("Usage Analytics")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        usage_chart = create_usage_chart(all_history)
        if usage_chart:
            st.altair_chart(usage_chart, use_container_width=True)
        else:
            st.info("No data available for usage chart.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.subheader("GPU/CPU Usage Over Time")
        df_usage = pd.DataFrame([
            {
                'Timestamp': parse_datetime(item.get('timestamp', item.get('created_at', ''))),
                'GPU (%)': parse_percent(item.get('gpu_usage', 0)),
                'CPU (%)': parse_percent(item.get('cpu_usage', 0)),
                'Model Name': item.get('model_name', 'N/A'),
            }
            for item in all_history
        ])
        df_usage = df_usage.dropna(subset=['Timestamp'])
        df_usage = df_usage.sort_values('Timestamp')
        models = df_usage['Model Name'].unique().tolist()
        tab_objs = st.tabs(models)
        for idx, model in enumerate(models):
            with tab_objs[idx]:
                df_model = df_usage[df_usage['Model Name'] == model]
                if df_model.empty:
                    st.info(f"No data for model {model}")
                else:
                    df_melt = df_model.melt(id_vars=['Timestamp'], value_vars=['GPU (%)','CPU (%)'], var_name='Type', value_name='Usage')
                    chart = alt.Chart(df_melt).mark_line(point=True).encode(
                        x=alt.X('Timestamp:T', title='Time', axis=alt.Axis(format='%m-%d %H:%M', labelAngle=-30)),
                        y=alt.Y('Usage:Q', title='Usage (%)', scale=alt.Scale(domain=[0,100])),
                        color=alt.Color('Type:N', title='Type', scale=alt.Scale(domain=['GPU (%)','CPU (%)'], range=['#ff7f0e','#1f77b4'])),
                        tooltip=['Timestamp:T', 'Type:N', 'Usage:Q']
                    ).properties(width='container', height=300)
                    st.altair_chart(chart, use_container_width=True)

    st.subheader("Chat History Records")

    search_query = st.text_input(
    "Search",
    placeholder="Search questions and responses...",
    key="admin_search",
    label_visibility="collapsed"
)
    
    if not all_history:
        st.info("No Q&A records found for all users.")
    else:
        st.write(f"Found {len(all_history)} records")
        
        filtered_history = all_history
        if search_query:
            filtered_history = []
            for item in all_history:
                question = item.get("question", "").lower()
                answer = item.get("answer", "").lower()
                username = item.get("username", "").lower()
                if (search_query.lower() in question or 
                    search_query.lower() in answer or 
                    search_query.lower() in username):
                    filtered_history.append(item)
            
            if not filtered_history:
                st.warning(f"No results found for '{search_query}'")
            else:
                st.success(f"Found {len(filtered_history)} results for '{search_query}'")
        
        df_data = []
        for item in filtered_history:
            record_id = item.get("id")
            username = item.get("username", "Unknown user")
            user_id = item.get("user_id", "N/A")
            
            question = item.get("question", "N/A")
            answer = item.get("answer", "N/A")
            
            # 使用完整內容而非截斷版本
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
            
            timestamp = item.get("timestamp", "")
            date_str = format_timestamp(timestamp)
            
            response_time_ms = item.get("response_time_ms")
            resp_time_str = f"{response_time_ms / 1000:.1f}s" if response_time_ms is not None else "N/A"
            
            feedback = item.get("feedback")
            feedback_str = format_feedback(feedback)
            
            confidence = item.get("confidence")
            confidence_source = item.get("confidence_source", "")
            
            if isinstance(confidence, float):
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
                "User": username,
                "User ID": user_id,
                "Question": display_q,
                "Response": display_a,
                "Confidence": confidence_str,
                "Feedback": feedback_str,
                "Response Time": resp_time_str,
                "Timestamp": date_str,
                "CPU (%)": cpu_usage,
                "GPU (%)": gpu_usage,
                "Model Name": model_name
            })
        
        df = pd.DataFrame(df_data)
        
        st.dataframe(
            df,
            column_config={
                "ID": None,
                "User": st.column_config.TextColumn("User", width="small"),
                "User ID": None,
                "Question": st.column_config.TextColumn("Question", width="medium"),
                "Response": st.column_config.TextColumn("Response", width="medium"),
                "Confidence": st.column_config.TextColumn("Confidence"),
                "Feedback": st.column_config.TextColumn("User Feedback"),
                "Response Time": st.column_config.TextColumn("Response Time"),
                "Timestamp": st.column_config.TextColumn("Timestamp"),
                "GPU (%)": st.column_config.TextColumn("GPU (%)"),
                "CPU (%)": st.column_config.TextColumn("CPU (%)"),
                "Model Name": st.column_config.TextColumn("Model Name"),
            },
            hide_index=True,
            column_order=["User", "Question", "Response", "Confidence", "Feedback", "Response Time", "Timestamp", "GPU (%)", "CPU (%)", "Model Name"],
            use_container_width=True
        )