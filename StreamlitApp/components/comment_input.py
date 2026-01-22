"""
Component for displaying and editing comments on query result sections.
Provides reusable comment input/display for each component type.
"""

import streamlit as st
from typing import Optional, Callable


def display_comment_section(
    query_id: int,
    component_type: str,
    existing_comment: Optional[str] = None,
    on_save: Optional[Callable[[int, str, str], None]] = None,
    editable: bool = True
):
    """
    Display a comment section for a specific component.
    
    Args:
        query_id: The ID of the query this comment belongs to
        component_type: Type of component (e.g., "answer", "graph_facts")
        existing_comment: Existing comment text to display
        on_save: Callback function(query_id, component_type, comment_text) to save comment
        editable: Whether the comment can be edited (True for live view, False for history)
    """
    import uuid
    unique_suffix = uuid.uuid4().hex[:8]
    key_prefix = f"comment_{query_id}_{component_type}"
    
    labels = {
        "answer": "💬 Comment on Answer",
        "entities": "💬 Comment on Entities",
        "graph_facts": "💬 Comment on Graph Facts",
        "semantic_search": "💬 Comment on Semantic Search",
        "keyword_search": "💬 Comment on Keyword Search"
    }
    label = labels.get(component_type, f"💬 Comment on {component_type}")
    
    placeholders = {
        "answer": "Is the answer accurate? Note any incorrect info or missing details...",
        "entities": "Are the extracted entities relevant? Note any missed terms...",
        "graph_facts": "Are the graph relationships relevant? Note any issues...",
        "semantic_search": "Are the retrieved chunks relevant? Note any issues...",
        "keyword_search": "Do the keyword results match? Note any issues..."
    }
    placeholder = placeholders.get(component_type, "Add your feedback...")
    
    if editable:
        st.markdown(f"**{label}**")
        
        with st.form(key=f"{key_prefix}_form"):
            comment_text = st.text_area(
                "Add your comment:",
                value=existing_comment or "",
                height=80,
                placeholder=placeholder,
                label_visibility="collapsed"
            )
            
            submitted = st.form_submit_button("💾 Save", use_container_width=False)
            
            if submitted:
                if on_save and comment_text.strip():
                    on_save(query_id, component_type, comment_text.strip())
                elif not comment_text.strip():
                    st.warning("Please enter a comment")
    else:
        if existing_comment:
            st.markdown(f"**{label}**")
            st.info(existing_comment)


def display_all_comments_summary(comments: dict):
    """Display a summary of all comments for a query."""
    if not comments:
        return
    
    st.markdown("### 📝 Comments Summary")
    
    labels = {
        "answer": "Answer",
        "entities": "Entities", 
        "graph_facts": "Graph Facts",
        "semantic_search": "Semantic Search",
        "keyword_search": "Keyword Search"
    }
    
    for comp_type, comment in comments.items():
        if comment:
            label = labels.get(comp_type, comp_type)
            st.markdown(f"**{label}:** {comment}")
