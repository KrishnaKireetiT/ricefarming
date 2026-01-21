"""
Component for displaying the farmer answer in a prominent, user-friendly format.
"""

import streamlit as st


def display_answer(farmer_answer: str, question: str = None):
    """
    Display the farmer answer prominently.
    
    Args:
        farmer_answer: The generated answer text
        question: Optional question to display above the answer
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
