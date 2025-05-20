import os
import sys
import logging
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_error_handler():
    error_handler = logging.FileHandler(os.path.join(LOGS_DIR, "error.log"))
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return error_handler


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger
    
    Args:
        name: Logger name
        log_file: Log file path, if None then only output to console
        level: Logging level
        
    Returns:
        logging.Logger: Setup logger
    """
    logger = logging.getLogger(name)
    
    if logger.level == level and logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    try:
        console_handler.encoding = 'utf-8'
    except Exception:
        pass
    logger.addHandler(console_handler)
    
    if log_file:
        log_file_path = os.path.join(LOGS_DIR, log_file)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        error_logger = logging.getLogger("error_logger")
        if not error_logger.hasHandlers():
            error_logger.addHandler(get_error_handler())
    
    return logger


def get_config_logger():
    """
    Get configuration logger (record to app.log)
    
    Returns:
        logging.Logger: Configuration logger
    """
    return setup_logger("dds_chatbot_config", "app.log")


def get_db_logger():
    """
    Get database logger (record to app.log)
    
    Returns:
        logging.Logger: Database logger
    """
    return setup_logger("dds_chatbot_db", "app.log")


def get_chat_logger():
    """
    Get chat logger (record to app.log)
    
    Returns:
        logging.Logger: Chat logger
    """
    return setup_logger("dds_chatbot_chat", "app.log")


def get_api_logger():
    """
    Get API logger (record to api.log)
    
    Returns:
        logging.Logger: API logger
    """
    return setup_logger("dds_chatbot_api", "api.log")


def get_app_logger():
    """
    Get application logger (record to app.log)
    
    Returns:
        logging.Logger: Application logger
    """
    return setup_logger("dds_chatbot_app", "app.log")


def get_rag_logger(module_name="general"):
    """
    Get RAG related logger (record to app.log)
    
    Args:
        module_name: RAG module name
        
    Returns:
        logging.Logger: RAG logger
    """
    return setup_logger(f"dds_chatbot_rag_{module_name}", "app.log")