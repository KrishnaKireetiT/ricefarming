"""
Database module for user authentication and query history.
Uses SQLite for local development, can be swapped for PostgreSQL in production.
"""

import sqlite3
import json
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager

import bcrypt

import config


def get_db_path() -> Path:
    """Get the database file path."""
    return config.DB_PATH


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize database tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                theme TEXT DEFAULT 'light'
            )
        """)
        
        # Batch runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT,
                pipeline_version TEXT,
                question_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Query history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                batch_run_id INTEGER,
                question TEXT NOT NULL,
                farmer_answer TEXT,
                trace_id TEXT,
                trace_url TEXT,
                raw_entities TEXT,
                aligned_entities TEXT,
                graph_facts TEXT,
                vector_context TEXT,
                keyword_results TEXT,
                visual_refs TEXT,
                pipeline_version TEXT,
                execution_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (batch_run_id) REFERENCES batch_runs (id)
            )
        """)
        
        # Query comments table - for user annotations on result components
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                component_type TEXT NOT NULL,
                comment_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history (id) ON DELETE CASCADE,
                UNIQUE(query_id, component_type)
            )
        """)
        
        conn.commit()


# ================================================================
# User Authentication
# ================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_user(username: str, email: str, password: str) -> Optional[int]:
    """
    Create a new user.
    Returns user ID on success, None on failure.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            password_hash = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            conn.commit()
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user.
    Returns user dict on success, None on failure.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, password_hash, theme FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        
        if row and verify_password(password, row["password_hash"]):
            # Update last login
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now(), row["id"])
            )
            conn.commit()
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "theme": row["theme"]
            }
    return None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, theme FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def update_user_theme(user_id: int, theme: str) -> bool:
    """Update user's theme preference."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET theme = ? WHERE id = ?",
            (theme, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def user_exists(username: str = None, email: str = None) -> bool:
    """Check if a user exists."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if username:
            cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        elif email:
            cursor.execute("SELECT 1 FROM users WHERE email = ?", (email,))
        else:
            return False
        return cursor.fetchone() is not None


# ================================================================
# Query History
# ================================================================

def save_query_result(
    user_id: int,
    question: str,
    farmer_answer: str,
    trace_id: str,
    trace_url: str,
    raw_entities: List[str],
    aligned_entities: List[Dict],
    graph_facts: List[Dict],
    vector_context: List[Dict],
    keyword_results: List[Dict],
    pipeline_version: str,
    execution_time: float,
    batch_run_id: int = None
) -> int:
    """Save a query result to history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_history (
                user_id, batch_run_id, question, farmer_answer, trace_id, trace_url,
                raw_entities, aligned_entities, graph_facts, vector_context,
                keyword_results, pipeline_version, execution_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            batch_run_id,
            question,
            farmer_answer,
            trace_id,
            trace_url,
            json.dumps(raw_entities),
            json.dumps(aligned_entities),
            json.dumps(graph_facts),
            json.dumps(vector_context),
            json.dumps(keyword_results),
            pipeline_version,
            execution_time
        ))
        conn.commit()
        return cursor.lastrowid


def get_user_history(user_id: int, limit: int = 50, offset: int = 0) -> List[Dict]:
    """Get user's query history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, question, farmer_answer, trace_id, trace_url,
                   pipeline_version, execution_time, created_at, batch_run_id
            FROM query_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))
        
        return [dict(row) for row in cursor.fetchall()]


def get_query_detail(query_id: int, user_id: int) -> Optional[Dict]:
    """Get full details of a specific query."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM query_history
            WHERE id = ? AND user_id = ?
        """, (query_id, user_id))
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            # Parse JSON fields
            for field in ["raw_entities", "aligned_entities", "graph_facts", 
                          "vector_context", "keyword_results"]:
                if result[field]:
                    result[field] = json.loads(result[field])
            return result
    return None


def delete_query(query_id: int, user_id: int) -> bool:
    """Delete a query from history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM query_history WHERE id = ? AND user_id = ?",
            (query_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


# ================================================================
# Batch Runs
# ================================================================

def create_batch_run(user_id: int, name: str, pipeline_version: str, question_count: int) -> int:
    """Create a new batch run."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO batch_runs (user_id, name, pipeline_version, question_count)
            VALUES (?, ?, ?, ?)
        """, (user_id, name, pipeline_version, question_count))
        conn.commit()
        return cursor.lastrowid


def get_user_batch_runs(user_id: int, limit: int = 20) -> List[Dict]:
    """Get user's batch runs."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, pipeline_version, question_count, created_at
            FROM batch_runs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]


def get_batch_run_results(batch_run_id: int, user_id: int) -> List[Dict]:
    """Get all results from a batch run."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM query_history
            WHERE batch_run_id = ? AND user_id = ?
            ORDER BY id ASC
        """, (batch_run_id, user_id))
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            for field in ["raw_entities", "aligned_entities", "graph_facts", 
                          "vector_context", "keyword_results"]:
                if result[field]:
                    result[field] = json.loads(result[field])
            results.append(result)
        return results


# ================================================================
# Query Comments
# ================================================================

def save_comment(query_id: int, component_type: str, comment_text: str) -> int:
    """
    Save or update a comment for a query component.
    Uses INSERT OR REPLACE for upsert behavior.
    Returns the comment ID.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO query_comments (query_id, component_type, comment_text, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (query_id, component_type, comment_text))
        conn.commit()
        return cursor.lastrowid


def get_comments_for_query(query_id: int) -> Dict[str, str]:
    """
    Get all comments for a query, keyed by component_type.
    Returns dict like {"answer": "comment text", "graph_facts": "another comment"}
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT component_type, comment_text
            FROM query_comments
            WHERE query_id = ?
        """, (query_id,))
        
        return {row["component_type"]: row["comment_text"] for row in cursor.fetchall()}


def delete_comment(query_id: int, component_type: str) -> bool:
    """Delete a specific comment."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM query_comments WHERE query_id = ? AND component_type = ?",
            (query_id, component_type)
        )
        conn.commit()
        return cursor.rowcount > 0


# Initialize database on import
init_db()

# ================================================================
# Session Tokens for Persistent Login
# ================================================================

def save_session_token(user_id: int, token: str) -> bool:
    """Save a session token for a user."""
    from datetime import datetime, timedelta
    with get_connection() as conn:
        cursor = conn.cursor()
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        # Insert token (expires in 30 days)
        expires_at = datetime.now() + timedelta(days=30)
        cursor.execute(
            "INSERT OR REPLACE INTO session_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at)
        )
        conn.commit()
        return True

def validate_session_token(user_id: int, token: str) -> bool:
    """Validate a session token for a user."""
    from datetime import datetime
    with get_connection() as conn:
        cursor = conn.cursor()
        # Check if table exists first
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='session_tokens'
        """)
        if not cursor.fetchone():
            return False
        
        cursor.execute("""
            SELECT expires_at FROM session_tokens 
            WHERE user_id = ? AND token = ?
        """, (user_id, token))
        row = cursor.fetchone()
        if row:
            expires_at = datetime.fromisoformat(row["expires_at"])
            return expires_at > datetime.now()
        return False

def clear_user_sessions(user_id: int) -> bool:
    """Clear all session tokens for a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM session_tokens WHERE user_id = ?", (user_id,))
        conn.commit()
        return True
