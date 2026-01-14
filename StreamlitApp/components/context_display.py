"""
Component for displaying context information with expandable sections.
Shows graph facts, semantic search results, and keyword search results.
Now includes comment sections for each component.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable

from components.comment_input import display_comment_section


def display_entities(
    raw_entities: List[str], 
    aligned_entities: List[Dict],
    query_id: int = None,
    comments: Dict[str, str] = None,
    on_comment_save: Callable = None,
    editable: bool = True
):
    """Display extracted and aligned entities with optional comment section."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Extracted Entities")
        if raw_entities:
            for entity in raw_entities:
                st.markdown(f"- `{entity}`")
        else:
            st.caption("No entities extracted")
    
    with col2:
        st.markdown("#### 🎯 Aligned to KG")
        if aligned_entities:
            for ent in aligned_entities:
                score = ent.get("score", 0)
                st.markdown(f"- `{ent.get('raw', '')}` → **{ent.get('name', '')}** ({score:.2f})")
        else:
            st.caption("No entities aligned")
    
    # Comment section for entities
    if query_id:
        display_comment_section(
            query_id=query_id,
            component_type="entities",
            existing_comment=comments.get("entities") if comments else None,
            on_save=on_comment_save,
            editable=editable
        )


def display_graph_facts(
    graph_facts: List[Dict], 
    expanded: bool = True,
    query_id: int = None,
    comments: Dict[str, str] = None,
    on_comment_save: Callable = None,
    editable: bool = True
):
    """
    Display graph traversal facts.
    
    Args:
        graph_facts: List of graph facts with source, relation, target
        expanded: Whether the section is expanded by default
        query_id: Query ID for comment association
        comments: Existing comments dict
        on_comment_save: Callback to save comments
        editable: Whether comments can be edited
    """
    with st.expander(f"📊 Graph Facts ({len(graph_facts)})", expanded=expanded):
        if not graph_facts:
            st.caption("No graph facts found")
        else:
            # Display as a table-like structure
            for fact in graph_facts:
                tier_emoji = "🔵" if fact.get("tier") == "core" else "⚪"
                source = fact.get("source", "")
                relation = fact.get("relation", "")
                target = fact.get("target", "")
                score = fact.get("score", 0)
                
                st.markdown(
                    f"{tier_emoji} `{source}` —[**{relation}**]→ `{target}` "
                    f"<span style='color: gray; font-size: 0.8em;'>({score:.2f})</span>",
                    unsafe_allow_html=True
                )
    
    # Comment section for graph facts
    if query_id:
        display_comment_section(
            query_id=query_id,
            component_type="graph_facts",
            existing_comment=comments.get("graph_facts") if comments else None,
            on_save=on_comment_save,
            editable=editable
        )


def display_semantic_search(
    vector_context: List[Dict], 
    expanded: bool = False,
    query_id: int = None,
    comments: Dict[str, str] = None,
    on_comment_save: Callable = None,
    editable: bool = True
):
    """
    Display semantic (vector) search results.
    
    Args:
        vector_context: List of search results with text, score, type
        expanded: Whether the section is expanded by default
        query_id: Query ID for comment association
        comments: Existing comments dict
        on_comment_save: Callback to save comments
        editable: Whether comments can be edited
    """
    chunks = [r for r in vector_context if r.get("type") == "Chunk"]
    
    with st.expander(f"🔎 Semantic Search ({len(chunks)} chunks)", expanded=expanded):
        if not vector_context:
            st.caption("No semantic search results")
        elif chunks:
            # Display chunks
            st.markdown("##### 📄 Text Chunks")
            for i, chunk in enumerate(chunks[:5]):
                score = chunk.get("score", 0)
                title = chunk.get("title", "Untitled")
                text = chunk.get("text", "")[:300]
                
                st.markdown(f"**[{i+1}] {title}** (score: {score:.3f})")
                st.markdown(f"> {text}...")
                st.divider()
    
    # Comment section for semantic search
    if query_id:
        display_comment_section(
            query_id=query_id,
            component_type="semantic_search",
            existing_comment=comments.get("semantic_search") if comments else None,
            on_save=on_comment_save,
            editable=editable
        )


def display_keyword_search(
    keyword_results: List[Dict], 
    expanded: bool = False,
    query_id: int = None,
    comments: Dict[str, str] = None,
    on_comment_save: Callable = None,
    editable: bool = True
):
    """
    Display keyword (fulltext) search results.
    
    Args:
        keyword_results: List of keyword search results
        expanded: Whether the section is expanded by default
        query_id: Query ID for comment association
        comments: Existing comments dict
        on_comment_save: Callback to save comments
        editable: Whether comments can be edited
    """
    with st.expander(f"🔤 Keyword Search ({len(keyword_results)})", expanded=expanded):
        if not keyword_results:
            st.caption("No keyword search results")
        else:
            for i, result in enumerate(keyword_results[:5]):
                score = result.get("score", 0)
                text = result.get("text", "")[:300]
                chunk_id = result.get("id", "")
                
                st.markdown(f"**[{i+1}]** (score: {score:.2f}) `{chunk_id}`")
                st.markdown(f"> {text}...")
                st.divider()
    
    # Comment section for keyword search
    if query_id:
        display_comment_section(
            query_id=query_id,
            component_type="keyword_search",
            existing_comment=comments.get("keyword_search") if comments else None,
            on_save=on_comment_save,
            editable=editable
        )


def display_all_context(
    raw_entities: List[str],
    aligned_entities: List[Dict],
    graph_facts: List[Dict],
    vector_context: List[Dict],
    keyword_results: List[Dict],
    query_id: int = None,
    comments: Dict[str, str] = None,
    on_comment_save: Callable = None,
    editable: bool = True
):
    """
    Display all context information in a structured layout.
    
    Args:
        raw_entities: List of extracted entity strings
        aligned_entities: List of aligned entity dicts
        graph_facts: List of graph traversal facts
        vector_context: List of semantic search results
        keyword_results: List of keyword search results
        query_id: Query ID for comment association
        comments: Existing comments dict keyed by component_type
        on_comment_save: Callback function(query_id, component_type, text) to save comments
        editable: Whether comments can be edited
    """
    
    st.markdown("### 📋 Context Details")
    
    # Entities section
    display_entities(
        raw_entities, 
        aligned_entities,
        query_id=query_id,
        comments=comments,
        on_comment_save=on_comment_save,
        editable=editable
    )
    
    st.divider()
    
    # Three main context sections
    display_graph_facts(
        graph_facts, 
        expanded=True,
        query_id=query_id,
        comments=comments,
        on_comment_save=on_comment_save,
        editable=editable
    )
    display_semantic_search(
        vector_context, 
        expanded=False,
        query_id=query_id,
        comments=comments,
        on_comment_save=on_comment_save,
        editable=editable
    )
    display_keyword_search(
        keyword_results, 
        expanded=False,
        query_id=query_id,
        comments=comments,
        on_comment_save=on_comment_save,
        editable=editable
    )
