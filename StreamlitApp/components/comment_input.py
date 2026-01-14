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
    # Create a unique key for this comment section
    key_prefix = f"comment_{query_id}_{component_type}"
    
    # Component type to display label mapping
    labels = {
        "answer": "💬 Comment on Answer",
        "entities": "💬 Comment on Entities",
        "graph_facts": "💬 Comment on Graph Facts",
        "semantic_search": "💬 Comment on Semantic Search",
        "keyword_search": "💬 Comment on Keyword Search"
    }
    label = labels.get(component_type, f"💬 Comment on {component_type}")
    
    with st.container():
        if editable:
            # Editable mode: show text area and save button
            with st.expander(label, expanded=bool(existing_comment)):
                comment_text = st.text_area(
                    "Add your comment:",
                    value=existing_comment or "",
                    height=80,
                    key=f"{key_prefix}_input",
                    placeholder="Add notes, observations, or feedback about this section...",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("💾 Save", key=f"{key_prefix}_save", use_container_width=True):
                        if on_save and comment_text.strip():
                            on_save(query_id, component_type, comment_text.strip())
                            st.success("Comment saved!")
                            st.rerun()
                        elif not comment_text.strip():
                            st.warning("Please enter a comment")
        else:
            # Read-only mode: just display the comment if it exists
            if existing_comment:
                with st.expander(label, expanded=True):
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
                            padding: 12px 16px;
                            border-radius: 8px;
                            border-left: 4px solid #ffc107;
                            color: #333;
                        ">
                            {existing_comment}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


def display_all_comments_summary(comments: dict):
    """
    Display a summary of all comments for a query.
    
    Args:
        comments: Dict of component_type -> comment_text
    """
    if not comments:
        return
    
    st.markdown("### 💬 Your Comments")
    
    labels = {
        "answer": "📝 Answer",
        "entities": "🔍 Entities",
        "graph_facts": "📊 Graph Facts",
        "semantic_search": "🔎 Semantic Search",
        "keyword_search": "🔤 Keyword Search"
    }
    
    for component_type, comment in comments.items():
        label = labels.get(component_type, component_type)
        st.markdown(f"**{label}:**")
        st.caption(comment)
