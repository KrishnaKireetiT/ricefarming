"""
Session persistence utilities for cookie-based login.
"""

import time
import secrets
import hashlib
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
        # Set cookie (expires in 30 days)
        cookie_manager.set("session_token", session_token, expires_at=time.time() + 30*24*60*60)
        cookie_manager.set("user_id", str(user_id), expires_at=time.time() + 30*24*60*60)

def restore_session_from_cookie(cookie_manager) -> Optional[dict]:
    """Restore user session from cookie."""
    if not cookie_manager:
        return None
    
    try:
        cookies = cookie_manager.get_all()
        if cookies and "session_token" in cookies and "user_id" in cookies:
            user_id = int(cookies["user_id"])
            session_token = cookies["session_token"]
            
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
        cookie_manager.delete("session_token")
        cookie_manager.delete("user_id")
