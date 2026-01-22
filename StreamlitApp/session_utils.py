"""
Session persistence utilities for cookie-based login.
"""

import time
import secrets
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional
import config
import database as db

try:
    import extra_streamlit_components as stx
    COOKIES_AVAILABLE = True
except ImportError:
    COOKIES_AVAILABLE = False


def get_cookie_manager():
    """Get cookie manager instance."""
    if COOKIES_AVAILABLE:
        return stx.CookieManager()
    return None

def generate_session_token(user_id: int) -> str:
    """Generate secure session token for user."""
    random_data = secrets.token_urlsafe(32)
    data = f"{user_id}:{random_data}:{config.APP_SECRET_KEY}"
    return hashlib.sha256(data.encode()).hexdigest()

def save_session_cookie(user_id: int, username: str, cookie_manager):
    """Save session cookie for automatic login."""
    if cookie_manager:
        session_token = generate_session_token(user_id)
        # Save to database
        db.save_session_token(user_id, session_token)
        # Set cookie (expires in 30 days) - combine into single JSON to avoid duplicate key
        expires = datetime.now() + timedelta(days=30)
        session_data = json.dumps({"token": session_token, "user_id": user_id})
        cookie_manager.set("session_data", session_data, expires_at=expires)

def restore_session_from_cookie(cookie_manager) -> Optional[dict]:
    """Restore user session from cookie."""
    if not cookie_manager:
        return None
    
    try:
        cookies = cookie_manager.get_all()
        if cookies and "session_data" in cookies:
            session_data = json.loads(cookies["session_data"])
            user_id = session_data["user_id"]
            session_token = session_data["token"]
            
            # Validate session token
            if db.validate_session_token(user_id, session_token):
                user = db.get_user_by_id(user_id)
                return user
    except:
        pass
    
    return None

def clear_session_cookie(cookie_manager):
    """Clear session cookie on logout."""
    if cookie_manager:
        cookie_manager.delete("session_data")
