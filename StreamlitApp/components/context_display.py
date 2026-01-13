"""
Component for displaying context information with expandable sections.
Shows graph facts, semantic search results, and keyword search results.
"""

import streamlit as st
from typing import List, Dict, Any


def display_entities(raw_entities: List[str], aligned_entities: List[Dict]):
    """Display extracted and aligned entities."""
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


def display_graph_facts(graph_facts: List[Dict], expanded: bool = True):
    """
    Display graph traversal facts.
    
    Args:
        graph_facts: List of graph facts with source, relation, target
        expanded: Whether the section is expanded by default
    """
    with st.expander(f"📊 Graph Facts ({len(graph_facts)})", expanded=expanded):
        if not graph_facts:
            st.caption("No graph facts found")
            return
        
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


def display_semantic_search(vector_context: List[Dict], expanded: bool = False):
    """
    Display semantic (vector) search results.
    
    Args:
        vector_context: List of search results with text, score, type
        expanded: Whether the section is expanded by default
    """
    chunks = [r for r in vector_context if r.get("type") == "Chunk"]
    
    with st.expander(f"🔎 Semantic Search ({len(chunks)} chunks)", expanded=expanded):
        if not vector_context:
            st.caption("No semantic search results")
            return
        
        # Display chunks
        if chunks:
            st.markdown("##### 📄 Text Chunks")
            for i, chunk in enumerate(chunks[:5]):
                score = chunk.get("score", 0)
                title = chunk.get("title", "Untitled")
                text = chunk.get("text", "")[:300]
                
                st.markdown(f"**[{i+1}] {title}** (score: {score:.3f})")
                st.markdown(f"> {text}...")
                st.divider()


def display_keyword_search(keyword_results: List[Dict], expanded: bool = False):
    """
    Display keyword (fulltext) search results.
    
    Args:
        keyword_results: List of keyword search results
        expanded: Whether the section is expanded by default
    """
    with st.expander(f"🔤 Keyword Search ({len(keyword_results)})", expanded=expanded):
        if not keyword_results:
            st.caption("No keyword search results")
            return
        
        for i, result in enumerate(keyword_results[:5]):
            score = result.get("score", 0)
            text = result.get("text", "")[:300]
            chunk_id = result.get("id", "")
            
            st.markdown(f"**[{i+1}]** (score: {score:.2f}) `{chunk_id}`")
            st.markdown(f"> {text}...")
            st.divider()





def display_all_context(
    raw_entities: List[str],
    aligned_entities: List[Dict],
    graph_facts: List[Dict],
    vector_context: List[Dict],
    keyword_results: List[Dict]
):
    """Display all context information in a structured layout."""
    
    st.markdown("### 📋 Context Details")
    
    # Entities section
    display_entities(raw_entities, aligned_entities)
    
    st.divider()
    
    # Three main context sections
    display_graph_facts(graph_facts, expanded=True)
    display_semantic_search(vector_context, expanded=False)
    display_keyword_search(keyword_results, expanded=False)
