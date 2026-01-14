"""
Component for displaying user query history.
Now loads and displays comments when viewing history details.
"""

import streamlit as st
from typing import List, Dict, Any, Callable
from datetime import datetime

import database as db


def display_history_sidebar(
    history: List[Dict],
    on_select: Callable[[int], None] = None
):
    """
    Display query history in sidebar.
    
    Args:
        history: List of history entries
        on_select: Callback when a history item is selected
    """
    st.sidebar.markdown("### 📜 Recent Queries")
    
    if not history:
        st.sidebar.caption("No query history yet")
        return
    
    for entry in history[:10]:
        question = entry.get("question", "")[:50]
        created_at = entry.get("created_at", "")
        query_id = entry.get("id")
        
        # Format timestamp
        if isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at)
                time_str = dt.strftime("%m/%d %H:%M")
            except:
                time_str = created_at[:10] if created_at else ""
        else:
            time_str = ""
        
        # Create clickable item
        if st.sidebar.button(
            f"📝 {question}...",
            key=f"history_{query_id}",
            help=f"{time_str}",
            use_container_width=True
        ):
            if on_select:
                on_select(query_id)


def display_history_table(
    history: List[Dict],
    on_view: Callable[[int], None] = None,
    on_delete: Callable[[int], None] = None
):
    """
    Display query history in a table format.
    
    Args:
        history: List of history entries
        on_view: Callback when view button is clicked
        on_delete: Callback when delete button is clicked
    """
    st.markdown("### 📜 Query History")
    
    if not history:
        st.info("No query history yet. Start by asking a question!")
        return
    
    # Header
    cols = st.columns([4, 2, 1, 1, 1])
    cols[0].markdown("**Question**")
    cols[1].markdown("**Pipeline**")
    cols[2].markdown("**Time**")
    cols[3].markdown("**View**")
    cols[4].markdown("**Del**")
    
    st.divider()
    
    for entry in history:
        query_id = entry.get("id")
        question = entry.get("question", "")
        pipeline = entry.get("pipeline_version", "v4")
        exec_time = entry.get("execution_time", 0)
        batch_id = entry.get("batch_run_id")
        
        cols = st.columns([4, 2, 1, 1, 1])
        
        # Question (truncated)
        display_question = question[:60] + "..." if len(question) > 60 else question
        cols[0].markdown(f"{'📦 ' if batch_id else ''}{display_question}")
        
        # Pipeline version
        cols[1].caption(pipeline)
        
        # Execution time
        cols[2].caption(f"{exec_time:.1f}s" if exec_time else "-")
        
        # View button
        if cols[3].button("👁️", key=f"view_{query_id}", help="View details"):
            if on_view:
                on_view(query_id)
        
        # Delete button
        if cols[4].button("🗑️", key=f"del_{query_id}", help="Delete"):
            if on_delete:
                on_delete(query_id)


def display_history_detail(entry: Dict, editable: bool = True):
    """
    Display full details of a history entry with comments.
    
    Args:
        entry: Full history entry with all fields
        editable: Whether comments can be edited in this view
    """
    from components.answer_display import display_answer
    from components.context_display import display_all_context
    from components.trace_display import display_trace_link
    from components.agent_graph_display import display_agent_graph
    
    query_id = entry.get("id")
    question = entry.get("question", "")
    farmer_answer = entry.get("farmer_answer", "")
    
    # Load existing comments for this query
    comments = db.get_comments_for_query(query_id)
    
    # Callback for saving comments
    def on_comment_save(qid: int, component_type: str, comment_text: str):
        db.save_comment(qid, component_type, comment_text)
    
    # Display answer with comment
    display_answer(
        farmer_answer, 
        question,
        query_id=query_id,
        comments=comments,
        on_comment_save=on_comment_save,
        editable=editable
    )
    
    st.divider()
    
    # Display context with comments
    display_all_context(
        raw_entities=entry.get("raw_entities", []),
        aligned_entities=entry.get("aligned_entities", []),
        graph_facts=entry.get("graph_facts", []),
        vector_context=entry.get("vector_context", []),
        keyword_results=entry.get("keyword_results", []),
        query_id=query_id,
        comments=comments,
        on_comment_save=on_comment_save,
        editable=editable
    )
    
    st.divider()
    
    # Display trace
    trace_id = entry.get("trace_id", "")
    trace_url = entry.get("trace_url", "")
    if trace_id:
        display_trace_link(trace_id, trace_url)
    
    # Display agent graph
    display_agent_graph()
