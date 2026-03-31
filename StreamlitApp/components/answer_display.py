"""
Component for displaying the farmer answer in a prominent, user-friendly format.
"""

import streamlit as st
from components.i18n import t


def display_answer(farmer_answer: str, question: str = None, query_id: int = None, user_id: int = None, comments: dict = None):
    """
    Display the farmer answer prominently.
    
    Args:
        farmer_answer: The generated answer text
        question: Optional question to display above the answer
        query_id: ID of the query (for comments)
        user_id: ID of the current user (for comments)
        comments: Dictionary of comments by type
    """
    if question:
        st.markdown(
            f'<div class="section-header">{t("question_label")}</div>',
            unsafe_allow_html=True
        )
        st.info(question)
    
    st.markdown(
        f'<div class="section-header">{t("farmer_advisory")}</div>',
        unsafe_allow_html=True
    )
    
    # Display answer in a styled container
    st.markdown(
        f'<div class="answer-box">{farmer_answer}</div>',
        unsafe_allow_html=True
    )
    
    # Add comment section
    if query_id and user_id:
        from components.comment_display import display_comment_section
        answer_comments = comments.get('answer', []) if comments else []
        display_comment_section(query_id, user_id, 'answer', 'Answer', answer_comments)


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
