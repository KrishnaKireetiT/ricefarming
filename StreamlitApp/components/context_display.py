"""
Component for displaying context information with expandable sections.
Shows graph facts, semantic search results, keyword search results,
and fused results with scoring explanations.
"""

import streamlit as st
from typing import List, Dict, Any


# ================================================================
# Tooltip / Help Texts
# ================================================================

SEMANTIC_HELP = (
    "**Cosine Similarity (0-1)**\n\n"
    "Measures how semantically close the query is to each chunk using embedding vectors. "
    "A score of 1.0 means identical meaning; 0.0 means completely unrelated.\n\n"
    "score = dot(query_embedding, chunk_embedding)\n\n"
    "Embeddings are L2-normalized, so this equals cosine similarity."
)

KEYWORD_HELP = (
    "**BM25 Score (normalized 0-1)**\n\n"
    "BM25 (Best Matching 25) scores how well each chunk matches the query keywords. "
    "It considers term frequency, inverse document frequency, and document length.\n\n"
    "Raw BM25 scores are unbounded, so we apply min-max normalization within the "
    "current result set: normalized = (score - min) / (max - min).\n\n"
    "The raw BM25 score is shown in parentheses for transparency."
)

RRF_HELP = (
    "**Reciprocal Rank Fusion (RRF)**\n\n"
    "Combines results from semantic and keyword search by rank position, not by score, "
    "so different score scales don't matter.\n\n"
    "RRF(d) = sum of 1 / (k + rank)\n\n"
    "where k=60 (constant) and rank is the document's position in each result list. "
    "Documents appearing in both lists get higher fused scores.\n\n"
    "These are the final chunks sent to the LLM for answer generation."
)

MERGE_HELP = (
    "**Merged Results (Dedup Merge)**\n\n"
    "Semantic results are taken first (in score order), then keyword results "
    "are appended if they are not already present.\n\n"
    "These are the final chunks sent to the LLM for answer generation."
)


def display_entities(raw_entities: List[str], aligned_entities: List[Dict]):
    """Display extracted and aligned entities."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Extracted Entities</div>',
            unsafe_allow_html=True
        )
        if raw_entities:
            chips_html = "".join(
                f'<span class="entity-chip">{entity}</span>'
                for entity in raw_entities
            )
            st.markdown(f'<div style="margin: 4px 0;">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("No entities extracted")

    with col2:
        st.markdown(
            '<div class="section-header">Aligned to KG</div>',
            unsafe_allow_html=True
        )
        if aligned_entities:
            for ent in aligned_entities:
                score = ent.get("score", 0)
                raw = ent.get('raw', '')
                name = ent.get('name', '')
                score_class = "score-badge-high" if score >= 0.85 else ("score-badge-medium" if score >= 0.6 else "score-badge-low")
                st.markdown(
                    f'<span class="entity-chip">{raw}</span> '
                    f'<span style="color:#94a3b8;margin:0 4px;">&#8594;</span> '
                    f'<span class="entity-chip-aligned">{name}</span> '
                    f'<span class="score-badge {score_class}">{score:.2f}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No entities aligned")


def display_graph_facts(graph_facts: List[Dict], expanded: bool = True, query_id: int = None, user_id: int = None, comments: dict = None):
    """
    Display graph traversal facts.

    Args:
        graph_facts: List of graph facts with source, relation, target
        expanded: Whether the section is expanded by default
        query_id: ID of the query (for comments)
        user_id: ID of the current user (for comments)
        comments: Dictionary of comments by type
    """
    with st.expander(f"Graph Facts ({len(graph_facts)})", expanded=expanded):
        if not graph_facts:
            st.caption("No graph facts found")
            return

        for fact in graph_facts:
            tier = fact.get("tier", "ctx")
            tier_badge = '<span class="retrieval-badge retrieval-badge-kg">core</span>' if tier == "core" else '<span class="retrieval-badge" style="background:#94a3b8;">ctx</span>'
            source = fact.get("source", "")
            relation = fact.get("relation", "")
            target = fact.get("target", "")
            score = fact.get("score", 0)
            score_class = "score-badge-high" if score >= 0.7 else ("score-badge-medium" if score >= 0.4 else "score-badge-low")

            st.markdown(
                f'<div class="fact-row">'
                f'{tier_badge} '
                f'<span class="entity-chip">{source}</span> '
                f'<span style="color:#64748b;">&#8594; <strong>{relation}</strong> &#8594;</span> '
                f'<span class="entity-chip-aligned">{target}</span> '
                f'<span class="score-badge {score_class}">{score:.2f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Add comment section
        if query_id and user_id:
            from components.comment_display import display_comment_section
            graph_comments = comments.get('graph_facts', []) if comments else []
            display_comment_section(query_id, user_id, 'graph_facts', 'Graph Facts', graph_comments)


def display_semantic_search(vector_context: List[Dict], expanded: bool = False, query_id: int = None, user_id: int = None, comments: dict = None):
    """
    Display semantic (vector) search results with cosine similarity scores.
    """
    chunks = vector_context if vector_context else []

    with st.expander(f"Semantic Search ({len(chunks)} results)", expanded=expanded):
        if not chunks:
            st.caption("No semantic search results")
            return

        st.info(SEMANTIC_HELP)
        st.markdown("##### Text Chunks")
        for i, chunk in enumerate(chunks[:5]):
            score = chunk.get("score", 0)
            title = chunk.get("title", "Untitled")
            text = chunk.get("text", "")
            chapter = chunk.get("chapter_num", "")
            section = chunk.get("section_num", "")
            section_title = chunk.get("section_title", "")

            # Build source reference
            source_ref = ""
            if chapter and section:
                source_ref = f" | 📖 Chapter {chapter}, Section {section}"
            elif section:
                source_ref = f" | 📖 Section {section}"

            st.markdown(f"**[{i+1}] {title}** (score: {score:.3f}){source_ref}")
            st.text(text)
            st.divider()

        # Add comment section
        if query_id and user_id:
            from components.comment_display import display_comment_section
            semantic_comments = comments.get('semantic', []) if comments else []
            display_comment_section(query_id, user_id, 'semantic', 'Semantic Search', semantic_comments)


def display_keyword_search(keyword_results: List[Dict], expanded: bool = False, query_id: int = None, user_id: int = None, comments: dict = None):
    """
    Display keyword (fulltext) search results with normalized BM25 scores.
    """
    with st.expander(f"Keyword Search ({len(keyword_results)})", expanded=expanded):
        if not keyword_results:
            st.caption("No keyword search results")
            return

        st.info(KEYWORD_HELP)
        # Sort by score descending for proper display order
        sorted_results = sorted(keyword_results, key=lambda x: x.get("score", 0), reverse=True)

        for i, result in enumerate(sorted_results[:5]):
            score = result.get("score", 0)
            raw_bm25 = result.get("raw_bm25_score")
            text = result.get("text", "")
            chunk_id = result.get("id", "")
            chapter = result.get("chapter_num", "")
            section = result.get("section_num", "")
            section_title = result.get("section_title", "")

            # Show normalized score + raw BM25 in parentheses
            score_display = f"score: {score:.2f}"
            if raw_bm25 is not None:
                score_display += f" (raw BM25: {raw_bm25:.2f})"

            # Build source reference
            source_ref = ""
            if chapter and section:
                source_ref = f" | 📖 Chapter {chapter}, Section {section}"
            elif section:
                source_ref = f" | 📖 Section {section}"

            st.markdown(f"**[{i+1}]** ({score_display}) `{chunk_id}`{source_ref}")
            st.text(text)
            st.divider()

        # Add comment section
        if query_id and user_id:
            from components.comment_display import display_comment_section
            keyword_comments = comments.get('keyword', []) if comments else []
            display_comment_section(query_id, user_id, 'keyword', 'Keyword Search', keyword_comments)


def display_fused_results(rrf_fused_results: List[Dict], expanded: bool = False, is_rrf: bool = True):
    """
    Display fused results (RRF or dedup-merge) sent to the LLM.

    Args:
        rrf_fused_results: List of fused result dicts with retrieval_source tags
        expanded: Whether the section is expanded by default
        is_rrf: True for V7 (RRF), False for V6 (dedup merge)
    """
    help_text = RRF_HELP if is_rrf else MERGE_HELP
    label = "RRF Fused Results" if is_rrf else "Merged Results"

    with st.expander(f"{label} ({len(rrf_fused_results)} chunks -> LLM)", expanded=expanded):
        if not rrf_fused_results:
            st.caption("No fused results")
            return

        st.info(help_text)

        for i, chunk in enumerate(rrf_fused_results[:6]):
            score = chunk.get("score", 0)
            text = chunk.get("text", "")[:300]
            source = chunk.get("retrieval_source", "unknown")
            chunk_id = chunk.get("id", "")
            title = chunk.get("title", "")
            chapter = chunk.get("chapter_num", "")
            section = chunk.get("section_num", "")

            # Source badge
            if source == "semantic":
                badge = "Semantic"
                badge_color = "#1565c0"
            elif source == "keyword":
                badge = "Keyword"
                badge_color = "#e65100"
            else:
                badge = "Both"
                badge_color = "#6a1b9a"

            source_html = f'<span style="background:{badge_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;">{badge}</span>'

            # Score label
            if is_rrf:
                score_label = f"RRF: {score:.4f}"
            else:
                score_label = f"score: {score:.3f}"

            # Source reference
            source_ref = ""
            if chapter and section:
                source_ref = f" 📖 Ch.{chapter}, Sec.{section}"
            elif section:
                source_ref = f" 📖 Sec.{section}"

            header = f"**[{i+1}]** {source_html} ({score_label})"
            if title:
                header += f" -- {title}"
            elif chunk_id:
                header += f" `{chunk_id}`"
            if source_ref:
                header += source_ref

            st.markdown(header, unsafe_allow_html=True)
            st.markdown(f"> {text}...")
            st.divider()


def display_all_context(
    raw_entities: List[str],
    aligned_entities: List[Dict],
    graph_facts: List[Dict],
    vector_context: List[Dict],
    keyword_results: List[Dict],
    rrf_fused_results: List[Dict] = None,
    llm_context: Dict = None,
    query_id: int = None,
    user_id: int = None,
    comments: dict = None
):
    """Display all context information in a structured layout."""

    st.markdown("### Context Details")

    # Entities section
    display_entities(raw_entities, aligned_entities)

    st.divider()

    # Three main context sections
    display_graph_facts(graph_facts, expanded=True, query_id=query_id, user_id=user_id, comments=comments)
    display_semantic_search(vector_context, expanded=False, query_id=query_id, user_id=user_id, comments=comments)
    display_keyword_search(keyword_results, expanded=False, query_id=query_id, user_id=user_id, comments=comments)

    # Fused results section
    if rrf_fused_results is None:
        rrf_fused_results = []

    # Detect RRF vs merge by checking for rrf_score key
    is_rrf = any(chunk.get("rrf_score") is not None for chunk in rrf_fused_results) if rrf_fused_results else False
    display_fused_results(rrf_fused_results, expanded=False, is_rrf=is_rrf)

    # Context sent to LLM section
    if llm_context:
        st.divider()
        display_llm_context(llm_context, query_id=query_id, user_id=user_id, comments=comments)


def display_llm_context(llm_context: Dict, query_id: int = None, user_id: int = None, comments: dict = None):
    """
    Display the exact context sent to the LLM for answer generation.
    Shows system prompt, question, and full context.

    Args:
        llm_context: Dictionary with 'system_prompt', 'question', and 'full_context' keys
        query_id: ID of the query (for comments)
        user_id: ID of the current user (for comments)
        comments: Dictionary of comments by type
    """
    if not llm_context:
        return

    st.markdown("### Context Sent to LLM")
    st.caption(
        "This section shows the exact inputs sent to the LLM for answer generation. "
        "The system prompt defines the LLM's role, the question is the user's query, "
        "and the full context is the combined KG facts + retrieved text chunks."
    )

    system_prompt = llm_context.get("system_prompt", "")
    question = llm_context.get("question", "")
    full_context = llm_context.get("full_context", "")

    with st.expander("System Prompt", expanded=False):
        st.code(system_prompt, language=None)

    with st.expander("Question", expanded=False):
        st.markdown(f"> {question}")

    with st.expander(f"Full Context ({len(full_context):,} chars)", expanded=False):
        st.text(full_context)

    # Add comment section
    if query_id and user_id:
        from components.comment_display import display_comment_section
        llm_comments = comments.get('llm_context', []) if comments else []
        display_comment_section(query_id, user_id, 'llm_context', 'Context Sent to LLM', llm_comments)
