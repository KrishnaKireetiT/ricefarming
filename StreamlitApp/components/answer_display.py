"""
Component for displaying the farmer answer in a prominent, user-friendly format.
Now includes optional comment section.
"""

import streamlit as st
from typing import Optional, Callable, Dict

from components.comment_input import display_comment_section


def display_answer(
    farmer_answer: str, 
    question: str = None,
    query_id: int = None,
    comments: Dict[str, str] = None,
    on_comment_save: Callable = None,
    editable: bool = True
):
    """
    Display the farmer answer prominently.
    
    Args:
        farmer_answer: The generated answer text
        question: Optional question to display above the answer
        query_id: Query ID for comment association
        comments: Existing comments dict
        on_comment_save: Callback to save comments
        editable: Whether comments can be edited
    """
    if question:
        st.markdown(f"### ❓ Question")
        st.info(question)
    
    st.markdown("### 🌾 Farmer Advisory")
    
    # Display answer in a styled container
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4caf50;
            margin: 10px 0;
        ">
            {farmer_answer}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Comment section for answer
    if query_id:
        display_comment_section(
            query_id=query_id,
            component_type="answer",
            existing_comment=comments.get("answer") if comments else None,
            on_save=on_comment_save,
            editable=editable
        )


def display_answer_card(
    farmer_answer: str,
    question: str,
    execution_time: float = None,
    trace_url: str = None,
    expanded: bool = True
):
    """
    Display answer in an expandable card format (for batch results).
    
    Args:
        farmer_answer: The generated answer text
        question: The question asked
        execution_time: Time taken to generate answer
        trace_url: Link to Langfuse trace
        expanded: Whether the card should be expanded by default
    """
    # Create header with stats
    header_parts = [f"**Q:** {question[:80]}{'...' if len(question) > 80 else ''}"]
    if execution_time:
        header_parts.append(f"⏱ {execution_time:.2f}s")
    
    header = " | ".join(header_parts)
    
    with st.expander(header, expanded=expanded):
        st.markdown("#### Answer")
        st.markdown(farmer_answer)
        
        if trace_url:
            st.markdown(f"[🔗 View Langfuse Trace]({trace_url})")
