import os
import sqlite3
import datetime
import shutil
import sys
from typing import List, Dict, Any, Optional
from config import DATABASE_PATH
from src.backend.logging import get_db_logger

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logger = get_db_logger()


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_database():
    """Ensure database exists and create necessary tables"""
    existing_records = False
    if os.path.exists(DATABASE_PATH):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT COUNT(*) FROM qa_history")
                qa_count = cursor.fetchone()[0]
                if qa_count > 0:
                    existing_records = True
            except:
                pass
                
            conn.close()
        except Exception as e:
            logger.warning(f"Unable to check existing records: {e}")
    
    db_dir = os.path.dirname(DATABASE_PATH)
    os.makedirs(db_dir, exist_ok=True)
    
    if os.path.exists(DATABASE_PATH):
        if existing_records:
            return True
            
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM sqlite_master")
            conn.close()

            return True
        except Exception as e:
            logger.warning(f"Existing database file is corrupted: {e}")
            logger.info(f"Creating backup of corrupted database")
            backup_path = f"{DATABASE_PATH}.backup.{int(datetime.datetime.now().timestamp())}"
            try:
                shutil.copy2(DATABASE_PATH, backup_path)
                logger.info(f"Backup created at: {backup_path}")
                os.remove(DATABASE_PATH)
                logger.info(f"Corrupted database file has been removed")
            except Exception as e:
                logger.error(f"Failed to backup/remove corrupted database: {e}")
                return False
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            feedback INTEGER DEFAULT NULL,
            response_time_ms INTEGER,
            confidence REAL DEFAULT NULL,
            confidence_source TEXT DEFAULT NULL,
            cpu_usage TEXT DEFAULT NULL,
            gpu_usage TEXT DEFAULT NULL,
            model_name TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            chunk_count INTEGER,
            file_type TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'active',
            embedding_model TEXT,
            num_chunks INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        
        def ensure_column_exists(cursor, table, column, coltype):
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if column not in columns:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        ensure_column_exists(cursor, "qa_history", "cpu_usage", "TEXT")
        ensure_column_exists(cursor, "qa_history", "gpu_usage", "TEXT")
        ensure_column_exists(cursor, "qa_history", "model_name", "TEXT")
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                ("admin", "admin123", 1)
            )
        
        conn.commit()
        conn.close()

        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def add_user(username: str, password: str, is_admin: bool = False) -> Optional[int]:
    """Add a new user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            (username, password, is_admin)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        logger.error(f"User {username} already exists")
        return None
    except Exception as e:
        logger.error(f"Failed to add user: {e}")
        return None


def get_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify user credentials and return user information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, is_admin FROM users WHERE username = ? AND password = ?",
            (username, password)
        )
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return dict(user)
        return None
    except Exception as e:
        logger.error(f"Failed to get user information: {e}")
        return None


def add_qa_record(user_id: int, question: str, answer: str, response_time_ms: int, confidence: float = None, confidence_source: str = None, cpu_usage: str = None, gpu_usage: str = None, model_name: str = None) -> Optional[int]:
    """Add a new Q&A record"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO qa_history (user_id, question, answer, response_time_ms, confidence, confidence_source, cpu_usage, gpu_usage, model_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, question, answer, response_time_ms, confidence, confidence_source, cpu_usage, gpu_usage, model_name)
        )
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id
    except Exception as e:
        logger.error(f"Failed to add Q&A record: {e}")
        return None


def update_qa_feedback(record_id: int, feedback: int) -> bool:
    """Update feedback score for a Q&A record"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE qa_history SET feedback = ? WHERE id = ?",
            (feedback, record_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to update Q&A feedback: {e}")
        return False


def get_user_qa_history(user_id: int) -> List[Dict[str, Any]]:
    """Get Q&A history for a specific user"""
    try:
        logger.debug(f"Retrieving Q&A history for user ID={user_id}")

        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(qa_history)")
        qa_history_columns = cursor.fetchall()
        logger.debug(f"QA history table columns: {[col[1] for col in qa_history_columns]}")
        
        cursor.execute("SELECT COUNT(*) FROM qa_history")
        total_records = cursor.fetchone()[0]
        logger.debug(f"Total number of records in QA history table: {total_records}")
        
        cursor.execute(
            "SELECT * FROM qa_history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        records = cursor.fetchall()
        
        result = []
        for record in records:
            record_dict = dict(record)
            if "created_at" in record_dict and record_dict["created_at"]:
                try:
                    if isinstance(record_dict["created_at"], str):
                        pass
                    else:
                        record_dict["created_at"] = str(record_dict["created_at"])
                except:
                    record_dict["created_at"] = str(record_dict["created_at"])
            result.append(record_dict)
        
        logger.debug(f"Found {len(result)} Q&A records for user {user_id}")
        if result:
            logger.debug(f"First record: {str(result[0])[:100]}...")
            
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Failed to get user Q&A history: {e}")
        return []


def get_all_qa_history() -> List[Dict[str, Any]]:
    """Get Q&A history for all users (admin use)"""
    try:
        logger.debug("Getting all Q&A history")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT q.*, u.username
            FROM qa_history q
            LEFT JOIN users u ON q.user_id = u.id
            ORDER BY q.created_at DESC
        """)
        records = cursor.fetchall()
        
        result = []
        for record in records:
            record_dict = dict(record)
            if "created_at" in record_dict and record_dict["created_at"]:
                try:
                    if isinstance(record_dict["created_at"], str):
                        pass
                    else:
                        record_dict["created_at"] = str(record_dict["created_at"])
                except:
                    record_dict["created_at"] = str(record_dict["created_at"])
            
            if "username" not in record_dict and "user_id" in record_dict:
                record_dict["username"] = f"User-{record_dict['user_id']}"
                
            result.append(record_dict)
        
        logger.debug(f"Found a total of {len(result)} Q&A records")
        if result:
            logger.debug(f"First record: {str(result[0])[:100]}...")
            
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Failed to get all Q&A history: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def add_document_record(user_id: int, filename: str, file_path: str, file_size: int, chunk_count: int) -> Optional[int]:
    """Add a document record"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (user_id, filename, file_path, file_size, chunk_count) VALUES (?, ?, ?, ?, ?)",
            (user_id, filename, file_path, file_size, chunk_count)
        )
        document_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return document_id
    except Exception as e:
        logger.error(f"Failed to add document record: {e}")
        return None


def get_user_documents(user_id: int) -> List[Dict[str, Any]]:
    """Get document list for a specific user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY upload_date DESC",
            (user_id,)
        )
        documents = cursor.fetchall()
        conn.close()
        
        result = [dict(doc) for doc in documents]
        logger.debug(f"Found {len(result)} documents for user {user_id}")
        return result
    except Exception as e:
        logger.error(f"Failed to get user documents: {e}")
        return []


def get_document_by_id(document_id: int) -> Optional[Dict[str, Any]]:
    """Get document information by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,)
        )
        document = cursor.fetchone()
        conn.close()
        
        if document:
            logger.debug(f"Found document with ID {document_id}: {document['filename']}")
            return dict(document)
        else:
            logger.warning(f"Document with ID {document_id} not found")
            return None
    except Exception as e:
        logger.error(f"Failed to get document by ID {document_id}: {e}")
        return None


def delete_document_record(document_id: int) -> bool:
    """Delete a document record"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,)
        )
        conn.commit()
        conn.close()
        logger.info(f"Document record with ID {document_id} has been deleted")
        return True
    except Exception as e:
        logger.error(f"Failed to delete document record with ID {document_id}: {e}")
        return False


def get_qa_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Get Q&A record by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM qa_history WHERE id = ?",
            (record_id,)
        )
        record = cursor.fetchone()
        conn.close()
        
        if record:
            logger.debug(f"Found Q&A record with ID {record_id}")
            return dict(record)
        else:
            logger.warning(f"Q&A record with ID {record_id} not found")
            return None
    except Exception as e:
        logger.error(f"Failed to get Q&A record by ID {record_id}: {e}")
        return None

def get_qa_count():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM qa_history")
        qa_count = cursor.fetchone()[0]
        conn.close()
        return qa_count
    except Exception as e:
        logger.warning(f"Unable to get QA count: {e}")
        return None

ensure_database()