import os
from langchain_community.chat_message_histories import SQLChatMessageHistory

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "chat_history.db")

def get_message_history(session_id: str) -> SQLChatMessageHistory:
    """
    Returns an SQLChatMessageHistory object backed by a local SQLite database.
    This provides fast local caching of context, keeping response times low.
    """
    connection_string = f"sqlite:///{DB_PATH}"
    
    chat_message_history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=connection_string
    )
    
    return chat_message_history
