"""
Component for displaying user query history.
Redesigned with expandable cards, answer previews, retrieval badges,
click-to-open, and right-click delete.
"""

import streamlit as st
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
from components.i18n import t, get_lang


def _format_time_ago(created_at: str) -> str:
    """Format timestamp as relative time (e.g., '2h ago', '3d ago')."""
    if not created_at:
        return ""
    try:
        if isinstance(created_at, str):
            dt = datetime.fromisoformat(created_at)
        else:
            dt = created_at
        
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 30:
            return dt.strftime("%b %d, %Y")
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "just now"
    except:
        return str(created_at)[:10] if created_at else ""


def _get_retrieval_badges(entry: Dict) -> str:
    """Generate HTML badges for retrieval sources with counts."""
    badges = []
    
    graph_facts = entry.get("graph_facts", [])
    vector_context = entry.get("vector_context", [])
    keyword_results = entry.get("keyword_results", [])
    
    kg_count = len(graph_facts) if isinstance(graph_facts, list) else 0
    sem_count = len(vector_context) if isinstance(vector_context, list) else 0
    kw_count = len(keyword_results) if isinstance(keyword_results, list) else 0
    
    if kg_count > 0:
        badges.append(f'<span style="background:#2e7d32;color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;margin-right:4px;">📊 KG {kg_count}</span>')
    if sem_count > 0:
        badges.append(f'<span style="background:#1565c0;color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;margin-right:4px;">🔎 Semantic {sem_count}</span>')
    if kw_count > 0:
        badges.append(f'<span style="background:#e65100;color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;margin-right:4px;">🔤 Keyword {kw_count}</span>')
    
    if not badges:
        badges.append('<span style="background:#616161;color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;">No retrieval data</span>')
    
    return " ".join(badges)


def _get_pipeline_badge(version: str) -> str:
    """Generate a pipeline version badge."""
    label_map = {"aditi": t("pipeline_b"), "1.0": "K-GraphRAG"}
    color_map = {"aditi": "#00838f", "1.0": "#7b1fa2"}
    label = label_map.get(version, version.upper() if version else "Unknown")
    color = color_map.get(version, "#616161")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;">{label}</span>'


def _generate_execution_timeline(entry: Dict) -> str:
    """
    Generate a visual execution timeline showing time breakdown by step.
    Extracts timing from trace_data if available, otherwise estimates.
    """
    total_time = entry.get("execution_time", 0)
    if not total_time or total_time <= 0:
        return ""
    
    # Try to extract step timings from trace_data
    trace_data = entry.get("trace_data")
    step_timings = {}
    
    if trace_data and isinstance(trace_data, dict):
        # Parse observations from trace data
        observations = trace_data.get("observations", [])
        for obs in observations:
            name = obs.get("name", "")
            start = obs.get("startTime")
            end = obs.get("endTime")
            
            if start and end:
                try:
                    from datetime import datetime
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                    
                    # Map observation names to display names
                    if "Extract" in name or "extract" in name:
                        step_timings["Extract"] = step_timings.get("Extract", 0) + duration
                    elif "KG" in name or "Graph" in name or "Traversal" in name:
                        step_timings["KG"] = step_timings.get("KG", 0) + duration
                    elif "Vector" in name or "Semantic" in name:
                        step_timings["Vector"] = step_timings.get("Vector", 0) + duration
                    elif "Keyword" in name or "BM25" in name:
                        step_timings["Keyword"] = step_timings.get("Keyword", 0) + duration
                    elif "RRF" in name or "Aggregate" in name:
                        step_timings["RRF"] = step_timings.get("RRF", 0) + duration
                    elif "Generate" in name or "Answer" in name or "LLM" in name:
                        step_timings["LLM"] = step_timings.get("LLM", 0) + duration
                except:
                    pass
    
    # If we couldn't extract timings, use estimates
    if not step_timings:
        # Rough estimates based on typical pipeline behavior
        step_timings = {
            "Extract": total_time * 0.15,
            "KG": total_time * 0.25,
            "Vector": total_time * 0.15,
            "RRF": total_time * 0.05,
            "LLM": total_time * 0.40
        }
    
    # Generate HTML timeline
    colors = {
        "Extract": "#9c27b0",
        "KG": "#2e7d32",
        "Vector": "#1565c0",
        "Keyword": "#e65100",
        "RRF": "#f57c00",
        "LLM": "#d32f2f"
    }
    
    # Generate timeline as a single container
    segments = ''
    for step, duration in step_timings.items():
        if duration > 0:
            percentage = (duration / total_time) * 100
            color = colors.get(step, "#757575")
            segments += f'<div style="background:{color};width:{percentage}%;display:flex;align-items:center;justify-content:center;color:white;font-size:0.65em;font-weight:600;padding:0 2px;min-width:0;" title="{step}: {duration:.2f}s">{step}</div>'
    
    timeline_html = f'<div style="display:flex;align-items:center;gap:6px;flex:1 1 auto;min-width:0;"><span style="font-size:0.75em;color:#666;white-space:nowrap;">⏱️ {total_time:.1f}s</span><div style="display:flex;height:18px;border-radius:4px;overflow:hidden;background:#eee;flex:1 1 auto;min-width:80px;">{segments}</div></div>'
    return timeline_html


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
        
        time_str = _format_time_ago(created_at)
        
        # Create clickable item
        if st.sidebar.button(
            f"📝 {question}...",
            key=f"history_{query_id}",
            help=f"{time_str}",
            use_container_width=True
        ):
            if on_select:
                on_select(query_id)


def display_history_cards(
    history: List[Dict],
    on_view: Callable[[int], None] = None,
    on_delete: Callable[[int], None] = None
):
    """
    Display query history as expandable cards with answer previews and retrieval badges.
    Click to open, right-click context menu to delete.
    
    Args:
        history: List of history entries
        on_view: Callback when a card is clicked to view details
        on_delete: Callback when delete is triggered
    """
    
    if not history:
        st.info("No query history yet. Start by asking a question!")
        return
    
    # Inject CSS + JS for right-click context menu
    st.markdown("""
    <style>
        .history-card {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
            transition: all 0.2s ease;
        }
        .history-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-color: #90caf9;
        }
        .card-question {
            font-size: 1.05em;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        .card-answer-preview {
            color: #555;
            font-size: 0.9em;
            line-height: 1.5;
            margin-bottom: 10px;
            padding: 8px 12px;
            background: rgba(76, 175, 80, 0.08);
            border-left: 3px solid #4caf50;
            border-radius: 4px;
        }
        .card-meta {
            display: flex;
            align-items: center;
            flex-wrap: nowrap;
            gap: 6px;
            margin-top: 6px;
        }
        
        /* Context menu styles */
        .context-menu-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: 9998;
        }
        .context-menu {
            display: none;
            position: fixed;
            z-index: 9999;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            padding: 4px 0;
            min-width: 160px;
        }
        .context-menu-item {
            padding: 8px 16px;
            cursor: pointer;
            font-size: 0.9em;
            color: #d32f2f;
            transition: background 0.15s;
        }
        .context-menu-item:hover {
            background: #ffebee;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display summary stats
    total = len(history)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", total)
    
    pipeline_counts = {}
    for e in history:
        pv = e.get("pipeline_version", "unknown")
        pipeline_counts[pv] = pipeline_counts.get(pv, 0) + 1
    most_used = max(pipeline_counts, key=pipeline_counts.get) if pipeline_counts else "-"
    col2.metric("Most Used Pipeline", most_used.upper())
    
    avg_time = sum(e.get("execution_time", 0) or 0 for e in history) / max(total, 1)
    col3.metric("Avg Response Time", f"{avg_time:.1f}s")
    
    st.divider()
    
    # Render each history entry as a card
    for entry in history:
        query_id = entry.get("id")
        question = entry.get("question", "")
        farmer_answer = entry.get("farmer_answer", "")
        pipeline_version = entry.get("pipeline_version", "unknown")
        exec_time = entry.get("execution_time", 0)
        created_at = entry.get("created_at", "")
        batch_id = entry.get("batch_run_id")
        
        # Prepare display values
        answer_preview = farmer_answer[:150] + "..." if len(farmer_answer) > 150 else farmer_answer
        time_ago = _format_time_ago(created_at)
        exec_str = f"{exec_time:.1f}s" if exec_time else ""
        retrieval_badges = _get_retrieval_badges(entry)
        pipeline_badge = _get_pipeline_badge(pipeline_version)
        batch_icon = "📦 " if batch_id else ""
        
        # Use st.expander for the expandable card with timestamp in header
        expander_label = f"{question[:70]}{'...' if len(question) > 70 else ''}"
        if batch_id:
            expander_label = f"📦 {expander_label}"
        # Add timestamp to the end
        expander_label += f"  •  {time_ago}"
        
        with st.expander(expander_label, expanded=False):
            # Answer preview
            st.markdown(f"""
            <div class="card-answer-preview">
                🌾 {answer_preview}
            </div>
            """, unsafe_allow_html=True)
            
            # Retrieval badges + inline timeline on one row
            timeline_html = _generate_execution_timeline(entry)
            st.markdown(f"""
            <div class="card-meta">
                {retrieval_badges}
                {pipeline_badge}
                {timeline_html if timeline_html else ''}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")  # spacing
            
            # Action buttons
            btn_col1, btn_col2 = st.columns([4, 1])
            
            with btn_col1:
                if st.button("🔍 View Full Details", key=f"view_{query_id}", use_container_width=True):
                    if on_view:
                        on_view(query_id)
            
            with btn_col2:
                if st.button("🗑️", key=f"del_{query_id}", help="Delete this query", use_container_width=True):
                    if on_delete:
                        on_delete(query_id)


def display_history_detail(entry: Dict, user_id: int):
    """
    Display full details of a history entry.
    
    Args:
        entry: Full history entry with all fields
        user_id: ID of the current user
    """
    import database as db
    from components.answer_display import display_answer
    from components.context_display import display_all_context
    from components.trace_display import display_trace_link
    from components.agent_graph_display import display_agent_graph
    
    question = entry.get("question", "")
    farmer_answer = entry.get("farmer_answer", "")
    query_id = entry.get("id")
    
    # Load comments for this query
    comments = db.get_comments_for_query(query_id) if query_id else {}
    
    # Display answer with comments
    display_answer(farmer_answer, question, query_id=query_id, user_id=user_id, comments=comments)
    
    st.divider()
    
    # Display context with comments
    display_all_context(
        raw_entities=entry.get("raw_entities", []),
        aligned_entities=entry.get("aligned_entities", []),
        graph_facts=entry.get("graph_facts", []),
        vector_context=entry.get("vector_context", []),
        keyword_results=entry.get("keyword_results", []),
        rrf_fused_results=entry.get("rrf_fused_results", []),
        llm_context=entry.get("llm_context"),
        query_id=query_id,
        user_id=user_id,
        comments=comments
    )
    
    st.divider()
    
    # Display trace with cached data
    trace_id = entry.get("trace_id", "")
    trace_url = entry.get("trace_url", "")
    trace_data = entry.get("trace_data")
    
    if trace_id:
        from components.trace_display import display_trace_embedded
        display_trace_embedded(trace_id, trace_url, height=800, cached_trace_data=trace_data)
    
    # Display agent graph from the pipeline that was used
    pipeline_version = entry.get("pipeline_version", "unknown")
    if pipeline_version in ("1.0", "2.0"):
        from pipelines.k_graph_rag_pipeline import KGraphRAGPipeline
        pipeline_used = KGraphRAGPipeline()
    else:
        from pipelines.aditi_pipeline import AditiPipeline
        pipeline_used = AditiPipeline()
    graph_data = pipeline_used.get_agent_graph()
    display_agent_graph(
        graph_data=graph_data,
        height=600,
        theme=st.session_state.get("theme", "light")
    )
