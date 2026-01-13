"""
Component for batch testing input and output display.
"""

import streamlit as st
import pandas as pd
import json
import io
from typing import List, Dict, Any, Tuple, Optional


def parse_batch_input(
    uploaded_file = None,
    text_input: str = None
) -> Tuple[List[str], Optional[str]]:
    """
    Parse batch input from file or text.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        text_input: Text pasted by user
        
    Returns:
        Tuple of (list of questions, error message or None)
    """
    questions = []
    error = None
    
    # Parse from file
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        content = uploaded_file.read()
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
                # Look for question column
                for col in ['question', 'Question', 'QUESTION', 'q', 'Q', 'query']:
                    if col in df.columns:
                        questions = df[col].dropna().tolist()
                        break
                if not questions and len(df.columns) >= 1:
                    # Use first column
                    questions = df.iloc[:, 0].dropna().tolist()
                    
            elif filename.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            questions.append(item)
                        elif isinstance(item, dict):
                            for key in ['question', 'Question', 'q', 'query']:
                                if key in item:
                                    questions.append(item[key])
                                    break
                                    
            elif filename.endswith('.txt'):
                text = content.decode('utf-8')
                questions = [line.strip() for line in text.split('\n') if line.strip()]
                
            else:
                error = f"Unsupported file format: {filename}"
                
        except Exception as e:
            error = f"Error parsing file: {str(e)}"
    
    # Parse from text input
    elif text_input and text_input.strip():
        lines = text_input.strip().split('\n')
        questions = [line.strip() for line in lines if line.strip()]
    
    # Filter empty questions
    questions = [q for q in questions if q and len(q) > 3]
    
    return questions, error


def display_batch_input() -> List[str]:
    """
    Display batch input interface.
    
    Returns:
        List of questions to process
    """
    st.markdown("### 📥 Batch Input")
    
    input_method = st.radio(
        "Input method:",
        ["Upload File", "Paste Text"],
        horizontal=True
    )
    
    questions = []
    error = None
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload questions file",
            type=["csv", "json", "txt"],
            help="CSV: needs 'question' column. JSON: array of strings or objects with 'question' key. TXT: one question per line."
        )
        if uploaded_file:
            questions, error = parse_batch_input(uploaded_file=uploaded_file)
            
    else:  # Paste Text
        text_input = st.text_area(
            "Paste questions (one per line):",
            height=150,
            placeholder="Khi nào nên bón phân đạm cho lúa?\nLàm thế nào xử lý rầy nâu?\n..."
        )
        if text_input:
            questions, error = parse_batch_input(text_input=text_input)
    
    if error:
        st.error(error)
    elif questions:
        st.success(f"✅ {len(questions)} questions loaded")
        
        # Preview
        with st.expander("Preview questions"):
            for i, q in enumerate(questions[:10]):
                st.markdown(f"{i+1}. {q}")
            if len(questions) > 10:
                st.caption(f"... and {len(questions) - 10} more")
    
    return questions


def display_batch_results(
    results: List[Dict],
    view_mode: str = "cards"
):
    """
    Display batch processing results.
    
    Args:
        results: List of pipeline results
        view_mode: 'cards' for expandable cards, 'detailed' for full report
    """
    if not results:
        st.info("No results to display")
        return
    
    # Stats summary
    total = len(results)
    success = sum(1 for r in results if r.get("farmer_answer"))
    total_time = sum(r.get("execution_time", 0) for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    st.markdown(f"""
    **Summary:** {success}/{total} successful | 
    Total time: {total_time:.1f}s | 
    Avg: {avg_time:.2f}s/question
    """)
    
    st.divider()
    
    if view_mode == "cards":
        _display_cards_view(results)
    else:
        _display_detailed_view(results)


def _display_cards_view(results: List[Dict]):
    """Display results as expandable cards."""
    for i, result in enumerate(results):
        question = result.get("question", f"Question {i+1}")
        answer = result.get("farmer_answer", "No answer generated")
        exec_time = result.get("execution_time", 0)
        trace_url = result.get("trace_url", "")
        
        # Context counts
        graph_count = len(result.get("graph_facts", []))
        vector_count = len(result.get("vector_context", []))
        keyword_count = len(result.get("keyword_results", []))
        
        header = f"**Q{i+1}:** {question[:70]}{'...' if len(question) > 70 else ''}"
        
        with st.expander(header, expanded=(i == 0)):
            # Answer
            st.markdown("#### 📝 Answer")
            st.markdown(answer)
            
            st.divider()
            
            # Context summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Graph Facts", graph_count)
            col2.metric("Semantic", vector_count)
            col3.metric("Keyword", keyword_count)
            col4.metric("Time", f"{exec_time:.1f}s")
            
            # Links
            col1, col2, col3 = st.columns(3)
            if trace_url:
                col1.link_button("🔗 Langfuse Trace", trace_url)
            
            # Show context details button
            if col2.button("📊 Show Context", key=f"ctx_{i}"):
                st.session_state[f"show_context_{i}"] = True
            
            # Context details (if expanded)
            if st.session_state.get(f"show_context_{i}"):
                from components.context_display import display_all_context
                display_all_context(
                    raw_entities=result.get("raw_entities", []),
                    aligned_entities=result.get("aligned_entities", []),
                    graph_facts=result.get("graph_facts", []),
                    vector_context=result.get("vector_context", []),
                    keyword_results=result.get("keyword_results", [])
                )


def _display_detailed_view(results: List[Dict]):
    """Display results with full details for each."""
    from components.answer_display import display_answer
    from components.context_display import display_all_context
    from components.trace_display import display_trace_embedded
    from components.agent_graph_display import display_agent_graph
    
    for i, result in enumerate(results):
        st.markdown(f"---")
        st.markdown(f"## Result {i+1} of {len(results)}")
        
        question = result.get("question", "")
        farmer_answer = result.get("farmer_answer", "")
        
        # Answer
        display_answer(farmer_answer, question)
        
        # Context
        display_all_context(
            raw_entities=result.get("raw_entities", []),
            aligned_entities=result.get("aligned_entities", []),
            graph_facts=result.get("graph_facts", []),
            vector_context=result.get("vector_context", []),
            keyword_results=result.get("keyword_results", [])
        )
        
        # Trace
        trace_id = result.get("trace_id", "")
        trace_url = result.get("trace_url", "")
        if trace_id:
            display_trace_embedded(trace_id, trace_url, height=300)
        
        # Agent graph (smaller for batch)
        display_agent_graph(height=250)


def display_batch_export(results: List[Dict]):
    """Display export options for batch results."""
    st.markdown("### 📤 Export Results")
    
    col1, col2 = st.columns(2)
    
    # CSV export
    with col1:
        csv_data = _results_to_csv(results)
        st.download_button(
            "📊 Download CSV",
            csv_data,
            file_name="batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # JSON export
    with col2:
        json_data = json.dumps([r for r in results], indent=2, ensure_ascii=False)
        st.download_button(
            "📄 Download JSON",
            json_data,
            file_name="batch_results.json",
            mime="application/json",
            use_container_width=True
        )


def _results_to_csv(results: List[Dict]) -> str:
    """Convert results to CSV string."""
    rows = []
    for r in results:
        rows.append({
            "question": r.get("question", ""),
            "answer": r.get("farmer_answer", ""),
            "execution_time": r.get("execution_time", 0),
            "trace_url": r.get("trace_url", ""),
            "graph_facts_count": len(r.get("graph_facts", [])),
            "vector_results_count": len(r.get("vector_context", [])),
            "keyword_results_count": len(r.get("keyword_results", []))
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
