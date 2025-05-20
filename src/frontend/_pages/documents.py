import os
import sys
import time
import streamlit as st
import pandas as pd
from datetime import datetime
from .. import api_client
from src.backend.logging import get_app_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)

logger = get_app_logger()


def add_custom_css():
    with open("src/frontend/static/documents.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def format_bytes(size):
    """Convert Bytes to KB, MB, GB"""
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size/1024:.1f} KB"
    elif size < 1024**3:
        return f"{size/1024**2:.1f} MB"
    else:
        return f"{size/1024**3:.1f} GB"


def format_timestamp(ts_str):
    """Format ISO timestamp"""
    if not ts_str: return "N/A"
    try:
        dt_obj = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt_obj.strftime("%b %d, %Y")
    except:
        return ts_str
    

def get_file_type(filename):
    """Get file type from filename"""
    if not filename:
        return "Unknown"
    
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return "PDF"
    elif ext == ".txt":
        return "TXT"
    else:
        return "Unknown"


def main():
    add_custom_css()
    
    st.title("My Documents")
    
    if "selected_doc_id" not in st.session_state:
        st.session_state.selected_doc_id = None

    if "selected_filename" not in st.session_state:
        st.session_state.selected_filename = None

    if "clear_uploader" not in st.session_state:
        st.session_state.clear_uploader = False
    
    if not st.session_state.get("is_logged_in") or "user" not in st.session_state:
        user_id = 0
    else:
        user_id = st.session_state.user["id"]
        
    with st.expander("☁️ **Upload Documents**", expanded=False):
        st.markdown('<div class="upload-btn-wrapper">', unsafe_allow_html=True)
        with st.form("upload_form", clear_on_submit=True):
            key_suffix = ""
            uploaded_files = st.file_uploader(
                "Select PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                key=f"doc_uploader{key_suffix}",
                label_visibility="collapsed"
            )

            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                chunk_size = st.number_input("Chunk Size", min_value=50, max_value=1000, value=st.session_state.get("chunk_size", 300), step=100, key=f"doc_chunk_size{key_suffix}")
            with col2:
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=300, value=st.session_state.get("chunk_overlap", 100), step=100, key=f"doc_chunk_overlap{key_suffix}")
            with col3:
                st.markdown("<div style='height: 26px'></div>", unsafe_allow_html=True)
                submit_upload = st.form_submit_button("Upload Documents", type="primary", use_container_width=True)
            with open("src/frontend/static/documents.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if 'submit_upload' in locals() and submit_upload and uploaded_files:
        total_files = len(uploaded_files)
        successful_uploads = 0
        logger.info(f"Starting upload of {total_files} files")
        with st.status(f"Processing {total_files} documents...", expanded=True) as status:
            for i, file in enumerate(uploaded_files):
                status_msg = f"Processing file {i+1}/{total_files}: {file.name}"
                st.write(status_msg)
                logger.info(f"Processing file {i+1}/{total_files}: {file.name}")
                try:
                    result = api_client.upload_document(
                        file=file,
                        user_id=user_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    if result and result.get("success"):
                        successful_uploads += 1
                        st.write(f"{file.name} uploaded successfully!")
                    else:
                        st.write(f"{file.name} upload failed!")
                except Exception as e:
                    st.write(f"{file.name} upload failed: {e}")
            status.update(
                label=f"All {successful_uploads} files uploaded successfully!",
                state="complete",
                expanded=False
            )
        st.success(f"{successful_uploads}/{total_files} files uploaded successfully!")
        st.session_state.last_doc_fetch_time = 0
        st.rerun()

    search_query = st.text_input("Search documents...", key="doc_search", placeholder="Search documents...", label_visibility="collapsed")

    fetch_interval = 60
    now = time.time()
    if now - st.session_state.get("last_doc_fetch_time", 0) > fetch_interval or not st.session_state.get("documents_list"):
        with st.spinner("Fetching document list..."):
            st.session_state.documents_list = api_client.get_user_documents(user_id)
            st.session_state.last_doc_fetch_time = now

    documents = st.session_state.documents_list

    if search_query:
        documents = [doc for doc in documents if search_query.lower() in doc.get("filename", "").lower()]
    
    if not documents:
        st.info("No documents found.")
    else:
        dds_docs = []
        regular_docs = []
        
        for doc in documents:
            filename = doc.get("filename", "Unknown File")
            if filename == "dds_qa.txt":
                dds_docs.append(doc)
            else:
                regular_docs.append(doc)
        
        if regular_docs:
            st.subheader("Your Documents")
            doc_data = []
            for doc in regular_docs:
                doc_id = doc.get("id")
                filename = doc.get("filename", "Unknown File")
                file_size = doc.get("file_size", 0)
                upload_date = doc.get("upload_date", "")
                file_type = get_file_type(filename)
                
                doc_data.append({
                    "ID": doc_id,
                    "Filename": filename,
                    "Type": file_type,
                    "Size": format_bytes(file_size),
                    "Upload Date": format_timestamp(upload_date)
                })
            
            df = pd.DataFrame(doc_data)
            
            modified_df = df.copy()
            modified_df["Delete"] = False
            
            edited_df = st.data_editor(
                modified_df,
                column_config={
                    "ID": None,
                    "Filename": st.column_config.TextColumn("Filename", disabled=True),
                    "Type": st.column_config.TextColumn("Type", disabled=True),
                    "Size": st.column_config.TextColumn("Size", disabled=True),
                    "Upload Date": st.column_config.TextColumn("Upload Date", disabled=True),
                    "Delete": st.column_config.CheckboxColumn("Delete")
                },
                hide_index=True,
                column_order=["Filename", "Type", "Size", "Upload Date", "Delete"],
                key="doc_table",
                use_container_width=True
            )
            
            if "doc_table" in st.session_state:
                rows_to_delete = edited_df[edited_df["Delete"] == True]
                
                if not rows_to_delete.empty:
                    if st.button("Delete Selected", type="primary", key="delete_btn", use_container_width=True):
                        successful_deletes = 0
                        total_to_delete = len(rows_to_delete)
                        deleted_files = []
                        
                        with st.status(f"Deleting {total_to_delete} documents...", expanded=True) as status:
                            for _, row in rows_to_delete.iterrows():
                                doc_id = row["ID"]
                                filename = row["Filename"]
                                if api_client.delete_document(doc_id):
                                    successful_deletes += 1
                                    deleted_files.append(filename)
                                    st.session_state.documents_list = [d for d in st.session_state.documents_list if d.get("id") != doc_id]
                                else:
                                    logger.error(f"Failed to delete file: {filename}")
                            
                            if successful_deletes == total_to_delete:
                                if total_to_delete == 1:
                                    logger.info(f"Successfully deleted file: {deleted_files[0]}")
                                else:
                                    logger.info(f"Successfully deleted all {successful_deletes} files")
                                status.update(
                                    label=f"Successfully deleted {successful_deletes} files",
                                    state="complete",
                                    expanded=False
                                )
                            else:
                                logger.warning(f"Failed to delete {total_to_delete - successful_deletes} files")
                                status.update(
                                    label=f"Successfully deleted {successful_deletes}/{total_to_delete} files",
                                    state="complete",
                                    expanded=False
                                )
                                    
                        st.rerun()
        
        if dds_docs:
            st.subheader("System Documents")
            dds_data = []
            for doc in dds_docs:
                doc_id = doc.get("id")
                filename = doc.get("filename", "Unknown File")
                file_size = doc.get("file_size", 0)
                upload_date = doc.get("upload_date", "")
                file_type = get_file_type(filename)
                
                dds_data.append({
                    "ID": doc_id,
                    "Filename": filename,
                    "Type": file_type,
                    "Size": format_bytes(file_size),
                    "Upload Date": format_timestamp(upload_date)
                })
            
            dds_df = pd.DataFrame(dds_data)
            st.dataframe(
                dds_df,
                column_config={
                    "ID": None,
                    "Filename": st.column_config.TextColumn("Filename", disabled=True),
                    "Type": st.column_config.TextColumn("Type", disabled=True),
                    "Size": st.column_config.TextColumn("Size", disabled=True),
                    "Upload Date": st.column_config.TextColumn("Upload Date", disabled=True)
                },
                hide_index=True,
                use_container_width=True
            )