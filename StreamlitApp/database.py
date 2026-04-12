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


def _migrate_evaluations_table(cursor):
    """Add new scoring columns to existing evaluations table."""
    new_columns = [
        ("relevance_v7", "INTEGER"),
        ("relevance_aditi", "INTEGER"),
        ("harmfulness_v7", "INTEGER"),
        ("harmfulness_aditi", "INTEGER"),
        ("ground_truth", "TEXT"),
        ("failure_modes_v7", "TEXT"),
        ("failure_modes_aditi", "TEXT"),
    ]
    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE evaluations ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists


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
                trace_data TEXT,
                raw_entities TEXT,
                aligned_entities TEXT,
                graph_facts TEXT,
                vector_context TEXT,
                keyword_results TEXT,
                final_context TEXT,
                visual_refs TEXT,
                pipeline_version TEXT,
                execution_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (batch_run_id) REFERENCES batch_runs (id)
            )
        """)
        
        # Add trace_data column to existing tables (migration)
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN trace_data TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Add final_context column to existing tables (migration)
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN final_context TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Add llm_context column to existing tables (migration)
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN llm_context TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Add rrf_fused_results column (migration)
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN rrf_fused_results TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Evaluations table (Extension Officer mode)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                batch_eval_id INTEGER,
                question TEXT NOT NULL,
                v7_answer TEXT,
                aditi_answer TEXT,
                v7_execution_time REAL,
                aditi_execution_time REAL,
                v7_trace_id TEXT,
                aditi_trace_id TEXT,
                preference TEXT,
                correctness_v7 INTEGER,
                correctness_aditi INTEGER,
                completeness_v7 INTEGER,
                completeness_aditi INTEGER,
                specificity_v7 INTEGER,
                specificity_aditi INTEGER,
                clarity_v7 INTEGER,
                clarity_aditi INTEGER,
                relevance_v7 INTEGER,
                relevance_aditi INTEGER,
                harmfulness_v7 INTEGER,
                harmfulness_aditi INTEGER,
                ground_truth TEXT,
                failure_modes_v7 TEXT,
                failure_modes_aditi TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Migrate existing evaluations tables — add new columns if missing
        _migrate_evaluations_table(cursor)

        # Batch evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT,
                question_count INTEGER DEFAULT 0,
                evaluated_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                comment_type TEXT NOT NULL,
                comment_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
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


def create_session(user_id: int) -> str:
    """Create a persistent session token for the given user. Returns the token."""
    token = secrets.token_urlsafe(32)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "INSERT INTO sessions (token, user_id) VALUES (?, ?)",
            (token, user_id),
        )
        conn.commit()
    return token


def get_user_by_session(token: str) -> Optional[Dict]:
    """Look up the user associated with a session token."""
    if not token:
        return None
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT u.id, u.username, u.email, u.theme
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ?
                """,
                (token,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        except sqlite3.OperationalError:
            # sessions table doesn't exist yet
            return None
    return None


def delete_session(token: str) -> None:
    """Delete a session token (used on logout)."""
    if not token:
        return
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
        except sqlite3.OperationalError:
            pass


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
    batch_run_id: int = None,
    trace_data: Dict[str, Any] = None,
    rrf_fused_results: List[Dict] = None,
    llm_context: Dict[str, str] = None
) -> int:
    """Save a query result to history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_history (
                user_id, batch_run_id, question, farmer_answer, trace_id, trace_url,
                trace_data, raw_entities, aligned_entities, graph_facts, vector_context,
                keyword_results, rrf_fused_results, llm_context, pipeline_version, execution_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            batch_run_id,
            question,
            farmer_answer,
            trace_id,
            trace_url,
            json.dumps(trace_data) if trace_data else None,
            json.dumps(raw_entities),
            json.dumps(aligned_entities),
            json.dumps(graph_facts),
            json.dumps(vector_context),
            json.dumps(keyword_results),
            json.dumps(rrf_fused_results) if rrf_fused_results else None,
            json.dumps(llm_context) if llm_context else None,
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
                   pipeline_version, execution_time, created_at, batch_run_id,
                   graph_facts, vector_context, keyword_results, trace_data
            FROM query_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))
        
        rows = []
        for row in cursor.fetchall():
            entry = dict(row)
            # Parse JSON fields for counts and timeline
            for field in ["graph_facts", "vector_context", "keyword_results", "trace_data"]:
                if entry.get(field):
                    try:
                        entry[field] = json.loads(entry[field])
                    except (json.JSONDecodeError, TypeError):
                        entry[field] = [] if field != "trace_data" else None
                else:
                    entry[field] = [] if field != "trace_data" else None
            rows.append(entry)
        return rows


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
                          "vector_context", "keyword_results", "trace_data", "rrf_fused_results",
                          "llm_context"]:
                if result.get(field):
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
# Comments
# ================================================================

def save_comment(query_id: int, user_id: int, comment_type: str, comment_text: str) -> int:
    """
    Save a comment for a query.
    
    Args:
        query_id: ID of the query to comment on
        user_id: ID of the user making the comment
        comment_type: Type of comment ('answer', 'graph_facts', 'semantic', 'keyword', 'final_context')
        comment_text: The comment text
        
    Returns:
        Comment ID
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO comments (query_id, user_id, comment_type, comment_text)
            VALUES (?, ?, ?, ?)
        """, (query_id, user_id, comment_type, comment_text))
        conn.commit()
        return cursor.lastrowid


def get_comments_for_query(query_id: int) -> Dict[str, List[Dict]]:
    """
    Get all comments for a query, grouped by comment type.
    
    Args:
        query_id: ID of the query
        
    Returns:
        Dictionary mapping comment_type to list of comments
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, u.username
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.query_id = ?
            ORDER BY c.created_at DESC
        """, (query_id,))
        
        comments_by_type = {}
        for row in cursor.fetchall():
            comment = dict(row)
            comment_type = comment['comment_type']
            if comment_type not in comments_by_type:
                comments_by_type[comment_type] = []
            comments_by_type[comment_type].append(comment)
        
        return comments_by_type


def delete_comment(comment_id: int, user_id: int) -> bool:
    """
    Delete a comment (only if it belongs to the user).
    
    Args:
        comment_id: ID of the comment to delete
        user_id: ID of the user attempting to delete
        
    Returns:
        True if deleted, False otherwise
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM comments
            WHERE id = ? AND user_id = ?
        """, (comment_id, user_id))
        conn.commit()
        return cursor.rowcount > 0


# ================================================================
# Evaluations (Extension Officer Mode)
# ================================================================

def save_evaluation(
    user_id: int,
    question: str,
    v7_answer: str,
    aditi_answer: str,
    v7_execution_time: float,
    aditi_execution_time: float,
    v7_trace_id: str = None,
    aditi_trace_id: str = None,
    preference: str = None,
    correctness_v7: int = None,
    correctness_aditi: int = None,
    completeness_v7: int = None,
    completeness_aditi: int = None,
    specificity_v7: int = None,
    specificity_aditi: int = None,
    clarity_v7: int = None,
    clarity_aditi: int = None,
    relevance_v7: int = None,
    relevance_aditi: int = None,
    harmfulness_v7: int = None,
    harmfulness_aditi: int = None,
    ground_truth: str = None,
    failure_modes_v7: str = None,
    failure_modes_aditi: str = None,
    notes: str = None,
    batch_eval_id: int = None
) -> int:
    """Save an evaluation from Extension Officer mode."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluations (
                user_id, batch_eval_id, question,
                v7_answer, aditi_answer,
                v7_execution_time, aditi_execution_time,
                v7_trace_id, aditi_trace_id,
                preference,
                correctness_v7, correctness_aditi,
                completeness_v7, completeness_aditi,
                specificity_v7, specificity_aditi,
                clarity_v7, clarity_aditi,
                relevance_v7, relevance_aditi,
                harmfulness_v7, harmfulness_aditi,
                ground_truth,
                failure_modes_v7, failure_modes_aditi,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, batch_eval_id, question,
            v7_answer, aditi_answer,
            v7_execution_time, aditi_execution_time,
            v7_trace_id, aditi_trace_id,
            preference,
            correctness_v7, correctness_aditi,
            completeness_v7, completeness_aditi,
            specificity_v7, specificity_aditi,
            clarity_v7, clarity_aditi,
            relevance_v7, relevance_aditi,
            harmfulness_v7, harmfulness_aditi,
            ground_truth,
            failure_modes_v7, failure_modes_aditi,
            notes
        ))
        conn.commit()
        return cursor.lastrowid


def update_evaluation(eval_id: int, user_id: int, **kwargs) -> bool:
    """Update an existing evaluation with new scores/preference."""
    allowed_fields = {
        'preference', 'correctness_v7', 'correctness_aditi',
        'completeness_v7', 'completeness_aditi',
        'specificity_v7', 'specificity_aditi',
        'clarity_v7', 'clarity_aditi',
        'relevance_v7', 'relevance_aditi',
        'harmfulness_v7', 'harmfulness_aditi',
        'ground_truth', 'failure_modes_v7', 'failure_modes_aditi',
        'notes'
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return False
    
    set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [eval_id, user_id]
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE evaluations SET {set_clause} WHERE id = ? AND user_id = ?",
            values
        )
        conn.commit()
        return cursor.rowcount > 0


def get_evaluation(eval_id: int, user_id: int) -> Optional[Dict]:
    """Get a single evaluation."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM evaluations WHERE id = ? AND user_id = ?",
            (eval_id, user_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_evaluations(user_id: int, limit: int = 50) -> List[Dict]:
    """Get user's evaluation history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, question, preference,
                   correctness_v7, correctness_aditi,
                   completeness_v7, completeness_aditi,
                   relevance_v7, relevance_aditi,
                   specificity_v7, specificity_aditi,
                   clarity_v7, clarity_aditi,
                   harmfulness_v7, harmfulness_aditi,
                   ground_truth,
                   failure_modes_v7, failure_modes_aditi,
                   v7_execution_time, aditi_execution_time,
                   created_at, batch_eval_id
            FROM evaluations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]


def get_evaluation_summary(user_id: int) -> Dict:
    """Get summary statistics of evaluations."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN preference = 'V7' THEN 1 ELSE 0 END) as v7_preferred,
                SUM(CASE WHEN preference = 'ADITI' THEN 1 ELSE 0 END) as aditi_preferred,
                SUM(CASE WHEN preference = 'TIE' THEN 1 ELSE 0 END) as tie,
                SUM(CASE WHEN preference IS NULL THEN 1 ELSE 0 END) as unevaluated,
                AVG(correctness_v7) as avg_correctness_v7,
                AVG(correctness_aditi) as avg_correctness_aditi,
                AVG(completeness_v7) as avg_completeness_v7,
                AVG(completeness_aditi) as avg_completeness_aditi,
                AVG(specificity_v7) as avg_specificity_v7,
                AVG(specificity_aditi) as avg_specificity_aditi,
                AVG(clarity_v7) as avg_clarity_v7,
                AVG(clarity_aditi) as avg_clarity_aditi,
                AVG(relevance_v7) as avg_relevance_v7,
                AVG(relevance_aditi) as avg_relevance_aditi,
                AVG(harmfulness_v7) as avg_harmfulness_v7,
                AVG(harmfulness_aditi) as avg_harmfulness_aditi,
                AVG(v7_execution_time) as avg_v7_time,
                AVG(aditi_execution_time) as avg_aditi_time,
                SUM(CASE WHEN ground_truth IS NOT NULL AND ground_truth != '' THEN 1 ELSE 0 END) as ground_truth_count
            FROM evaluations
            WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        result = dict(row) if row else {}

        # Aggregate failure modes across all evaluations
        cursor.execute("""
            SELECT failure_modes_v7, failure_modes_aditi
            FROM evaluations
            WHERE user_id = ? AND (failure_modes_v7 IS NOT NULL OR failure_modes_aditi IS NOT NULL)
        """, (user_id,))
        v7_fm_counts = {}
        aditi_fm_counts = {}
        for r in cursor.fetchall():
            for fm in (r["failure_modes_v7"] or "").split(","):
                fm = fm.strip()
                if fm:
                    v7_fm_counts[fm] = v7_fm_counts.get(fm, 0) + 1
            for fm in (r["failure_modes_aditi"] or "").split(","):
                fm = fm.strip()
                if fm:
                    aditi_fm_counts[fm] = aditi_fm_counts.get(fm, 0) + 1
        result["failure_modes_v7"] = v7_fm_counts
        result["failure_modes_aditi"] = aditi_fm_counts
        return result


def create_batch_evaluation(user_id: int, name: str, question_count: int) -> int:
    """Create a new batch evaluation run."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO batch_evaluations (user_id, name, question_count)
            VALUES (?, ?, ?)
        """, (user_id, name, question_count))
        conn.commit()
        return cursor.lastrowid


def delete_evaluation(eval_id: int, user_id: int) -> bool:
    """Delete an evaluation."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM evaluations WHERE id = ? AND user_id = ?",
            (eval_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


# Initialize database on import
init_db()
