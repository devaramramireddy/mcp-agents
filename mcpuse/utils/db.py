import os
import json
import sqlite3

def get_connection(path="chat_data.db"):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    session_id TEXT,
    user TEXT,
    role TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
    conn.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    session_id TEXT,
    score INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
    return conn

def save_message(session_id, role, content, username, path="chathistory.json"):
    history = []
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                history = json.load(f)
            except Exception:
                pass
    history.append({
        "session_id": session_id,
        "role": role,
        "content": content,
        "username": username
    })
    with open(path, "w") as f:
        json.dump(history, f)

def save_feedback(session_id, score):
    conn = get_connection()
    conn.execute("INSERT INTO feedback (session_id, score) VALUES (?, ?)", (session_id, score))
    conn.commit()
