"""
Component for displaying and managing comments on query results.
Allows users to add, view, and delete comments on different sections.
"""

import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime
import database as db


def display_comment_section(
    query_id: Optional[int],
    user_id: int,
    comment_type: str,
    section_title: str,
    existing_comments: List[Dict] = None
):
    """
    Display a comment section for a specific part of the query result.
    
    Args:
        query_id: ID of the query (None for unsaved queries)
        user_id: ID of the current user
        comment_type: Type of comment ('answer', 'graph_facts', 'semantic', 'keyword', 'final_context')
        section_title: Display title for the section
        existing_comments: List of existing comments for this section
    """
    if existing_comments is None:
        existing_comments = []
    
    # Only show comment section if query is saved
    if query_id is None:
        st.caption("💬 _Save this query to add comments_")
        return
    
    with st.expander(f"💬 Comments ({len(existing_comments)})", expanded=False):
        # Display existing comments
        if existing_comments:
            for comment in existing_comments:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{comment['username']}** · {_format_timestamp(comment['created_at'])}")
                    st.markdown(comment['comment_text'])
                with col2:
                    # Allow user to delete their own comments
                    if comment['user_id'] == user_id:
                        if st.button("🗑️", key=f"del_{comment['id']}", help="Delete comment"):
                            if db.delete_comment(comment['id'], user_id):
                                st.success("Comment deleted")
                                st.rerun()
                            else:
                                st.error("Failed to delete comment")
                st.divider()
        
        # Add new comment
        st.markdown("**Add a comment:**")
        comment_key = f"comment_{comment_type}_{query_id}"
        new_comment = st.text_area(
            "Your comment",
            key=comment_key,
            placeholder=f"Add your thoughts about {section_title.lower()}...",
            label_visibility="collapsed",
            height=100
        )
        
        if st.button("💾 Save Comment", key=f"save_{comment_type}_{query_id}"):
            if new_comment and new_comment.strip():
                comment_id = db.save_comment(query_id, user_id, comment_type, new_comment.strip())
                if comment_id:
                    st.success("Comment saved!")
                    st.rerun()
                else:
                    st.error("Failed to save comment")
            else:
                st.warning("Please enter a comment")


def _format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "just now"
    except:
        return timestamp_str
