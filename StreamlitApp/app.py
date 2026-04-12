"""
Pipeline Testing App - Main Streamlit Application

A comprehensive testing app for the multi-modal rice farming KG pipeline.
Features:
- User authentication (signup/login)
- Single query testing with full result display
- Batch testing with CSV/JSON/TXT upload or text paste
- Query history per user
- Light/dark mode toggle
- Langfuse trace and agent graph visualization
"""

import streamlit as st
import time
from typing import Optional

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="Pipeline Testing App",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
import config
import database as db
from pipelines.aditi_pipeline import AditiPipeline
from pipelines.aditi_trial_pipeline import AditiTrialPipeline
from pipelines.k_graph_rag_pipeline import KGraphRAGPipeline
from pipelines.base import PipelineResult

# Import components
from components.answer_display import display_answer, display_answer_card
from components.context_display import display_all_context, display_entities, display_llm_context
from components.trace_display import display_trace_embedded, display_trace_link
from components.agent_graph_display import display_agent_graph
from components.history_display import display_history_cards, display_history_detail
from components.batch_display import display_batch_input, display_batch_results, display_batch_export
from components.evaluation_display import (
    display_evaluation_question_input, display_side_by_side_answers,
    display_evaluation_results, display_batch_evaluation_input,
    display_batch_evaluation_results
)
from components.styles import GLOBAL_CSS, DARK_THEME_CSS
from components.i18n import t, get_lang


# ================================================================
# Session State Initialization
# ================================================================

def init_session_state():
    """Initialize session state variables."""
    if "user" not in st.session_state:
        st.session_state.user = None
    # Restore login from persistent session token in the URL (survives back-button & reload)
    if st.session_state.user is None:
        token = st.query_params.get("session")
        if token:
            user = db.get_user_by_session(token)
            if user:
                st.session_state.user = user
                st.session_state.session_token = token
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    if "pipeline_cache" not in st.session_state:
        st.session_state.pipeline_cache = {}
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None  # lazily loaded via pipeline_cache
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None
    if "batch_run_id" not in st.session_state:
        st.session_state.batch_run_id = None
    if "view_history_id" not in st.session_state:
        st.session_state.view_history_id = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Evaluate"
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "vi"
    if "selected_pipeline" not in st.session_state:
        st.session_state.selected_pipeline = None
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Extension Officer"
    if "eval_kgraphrag_result" not in st.session_state:
        st.session_state.eval_kgraphrag_result = None
    if "eval_aditi_result" not in st.session_state:
        st.session_state.eval_aditi_result = None
    if "eval_batch_results" not in st.session_state:
        st.session_state.eval_batch_results = []
    if "current_eval_id" not in st.session_state:
        st.session_state.current_eval_id = None
    if "pipeline_kgraphrag" not in st.session_state:
        st.session_state.pipeline_kgraphrag = None
    if "pipeline_aditi" not in st.session_state:
        st.session_state.pipeline_aditi = None
    if "editing_eval_id" not in st.session_state:
        st.session_state.editing_eval_id = None


# ================================================================
# Authentication UI
# ================================================================

def display_login_page():
    """Display login/signup page."""
    # Center the login form
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        # Language toggle: ON = Vietnamese, OFF = English
        vi_on = st.toggle("VI / EN", value=(get_lang() == "vi"), key="login_lang_toggle")
        new_lang = "vi" if vi_on else "en"
        if new_lang != st.session_state.ui_lang:
            st.session_state.ui_lang = new_lang
            st.rerun()

        st.markdown(
            '<div style="text-align:center;">'
            '<span style="font-size: 3.5rem;">🌾</span>'
            f'<h1 style="margin: 0.5rem 0 0.25rem 0; font-weight: 700;">{t("app_title")}</h1>'
            f'<p style="color: #64748b; font-size: 1.05rem; margin-bottom: 2rem;">'
            f'{t("app_subtitle")}</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        tab1, tab2 = st.tabs([t("login"), t("sign_up")])
        
        with tab1:
            with st.form("login_form"):
                st.text_input(t("username"), key="login_username", placeholder=t("enter_username"))
                st.text_input(t("password"), type="password", key="login_password", placeholder=t("enter_password"))
                submitted = st.form_submit_button(t("sign_in"), use_container_width=True, type="primary")
                
                if submitted:
                    username = st.session_state.login_username
                    password = st.session_state.login_password
                    if username and password:
                        user = db.authenticate_user(username, password)
                        if user:
                            st.session_state.user = user
                            st.session_state.theme = user.get("theme", "light")
                            # Persist login across reloads / back-button via URL token
                            token = db.create_session(user["id"])
                            st.session_state.session_token = token
                            st.query_params["session"] = token
                            st.success(t("login_successful"))
                            st.rerun()
                        else:
                            st.error(t("invalid_credentials"))
                    else:
                        st.warning(t("enter_both"))
        
        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input(t("choose_username"), placeholder="username")
                new_email = st.text_input(t("email_address"), placeholder="you@example.com")
                new_password = st.text_input(t("choose_password"), type="password", placeholder=t("min_6_chars"))
                confirm_password = st.text_input(t("confirm_password"), type="password", placeholder=t("repeat_password"))
                submitted = st.form_submit_button(t("create_account"), use_container_width=True, type="primary")
                
                if submitted:
                    if not all([new_username, new_email, new_password, confirm_password]):
                        st.warning(t("fill_all_fields"))
                    elif new_password != confirm_password:
                        st.error(t("passwords_no_match"))
                    elif len(new_password) < 6:
                        st.error(t("password_too_short"))
                    elif db.user_exists(username=new_username):
                        st.error(t("username_taken"))
                    elif db.user_exists(email=new_email):
                        st.error(t("email_registered"))
                    else:
                        user_id = db.create_user(new_username, new_email, new_password)
                        if user_id:
                            st.success(t("account_created"))
                        else:
                            st.error(t("account_failed"))


# ================================================================
# Sidebar
# ================================================================

def _init_both_pipelines():
    """Initialize K-GraphRAG and ADITI pipelines for Extension Officer mode."""
    cache = st.session_state.pipeline_cache
    if st.session_state.pipeline_kgraphrag is None:
        cached = cache.get("K-GraphRAG Pipeline (ColBERT + Qwen + RRF)")
        if cached is None:
            with st.sidebar.status(t("loading_pipeline_a"), expanded=True):
                cached = KGraphRAGPipeline()
                cached.initialize()
                cache["K-GraphRAG Pipeline (ColBERT + Qwen + RRF)"] = cached
            st.sidebar.success(t("pipeline_a_ready"))
        st.session_state.pipeline_kgraphrag = cached
    if st.session_state.pipeline_aditi is None:
        cached = cache.get("ADITI Triple-Hybrid Pipeline")
        if cached is None:
            with st.sidebar.status(t("loading_pipeline_b"), expanded=True):
                cached = AditiPipeline()
                cached.initialize()
                cache["ADITI Triple-Hybrid Pipeline"] = cached
            st.sidebar.success(t("pipeline_b_ready"))
        st.session_state.pipeline_aditi = cached


def display_sidebar():
    """Display sidebar with user info, settings, and history."""
    user = st.session_state.user
    
    # App branding
    st.sidebar.markdown(
        '<div style="text-align:center; padding: 8px 0 4px 0;">'
        '<span style="font-size:1.8rem;">🌾</span>'
        f'<div style="font-size:1.1rem; font-weight:700; margin-top:2px;">{t("pipeline_tester")}</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # User badge
    st.sidebar.markdown(
        f'<div style="text-align:center; padding:6px 0 12px 0; color:#64748b; font-size:0.85rem;">'
        f'👤 {user["username"]}</div>',
        unsafe_allow_html=True
    )
    
    st.sidebar.divider()
    
    # Language toggle: ON = Vietnamese, OFF = English
    vi_on = st.sidebar.toggle("VI / EN", value=(get_lang() == "vi"), key="sidebar_lang_toggle")
    new_lang = "vi" if vi_on else "en"
    if new_lang != st.session_state.ui_lang:
        st.session_state.ui_lang = new_lang
        st.rerun()
    
    st.sidebar.divider()
    
    # Technical Mode toggle (Extension Officer is default, toggle ON = Technical)
    is_technical = st.sidebar.toggle(
        t("technical_mode"),
        value=(st.session_state.app_mode == "Technical"),
        key="mode_toggle"
    )
    new_mode = "Technical" if is_technical else "Extension Officer"
    if new_mode != st.session_state.app_mode:
        st.session_state.app_mode = new_mode
        if new_mode == "Technical":
            st.session_state.active_tab = "Single Query"
        else:
            st.session_state.active_tab = "Evaluate"
        st.rerun()
    
    if st.session_state.app_mode == "Technical":
        _display_sidebar_technical(user)
    else:
        _display_sidebar_extension_officer(user)
    
    # Logout at bottom
    st.sidebar.divider()
    if st.sidebar.button(t("logout"), use_container_width=True):
        token = st.session_state.get("session_token")
        if token:
            db.delete_session(token)
        st.session_state.user = None
        st.session_state.session_token = None
        st.session_state.pipeline = None
        st.session_state.pipeline_cache = {}
        try:
            del st.query_params["session"]
        except KeyError:
            pass
        st.rerun()


def _display_sidebar_technical(user):
    """Sidebar content for Technical mode."""
    # Pipeline selection
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Pipeline")
    
    AVAILABLE_PIPELINES = {
        "ADITI Triple-Hybrid Pipeline": AditiPipeline,
        "ADITI Trial — Updated Handbook 2025": AditiTrialPipeline,
        "K-GraphRAG Pipeline (ColBERT + Qwen + RRF)": KGraphRAGPipeline,
    }

    selected_pipeline_name = st.sidebar.selectbox(
        "Select pipeline:",
        list(AVAILABLE_PIPELINES.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    if st.session_state.selected_pipeline != selected_pipeline_name:
        st.session_state.selected_pipeline = selected_pipeline_name
        st.session_state.current_result = None
        st.session_state.batch_results = []

    # Reuse a previously-initialized pipeline instead of re-loading models on every switch
    cached = st.session_state.pipeline_cache.get(selected_pipeline_name)
    if cached is None:
        with st.sidebar.status("Loading pipeline...", expanded=True):
            PipelineClass = AVAILABLE_PIPELINES[selected_pipeline_name]
            cached = PipelineClass()
            cached.initialize()
            st.session_state.pipeline_cache[selected_pipeline_name] = cached
        st.sidebar.success(f"✅ {selected_pipeline_name} ready!")
    st.session_state.pipeline = cached
    
    # Database status check
    if st.sidebar.button("🔄 Check DB Status", use_container_width=True):
        with st.sidebar.status("Checking database...", expanded=True):
            status = st.session_state.pipeline.verify_database()
            st.session_state.db_status = status
    
    if "db_status" in st.session_state:
        status = st.session_state.db_status
        if status.get("ready"):
            st.sidebar.success(f"✅ DB Ready: {status['chunk_count']} chunks, {status.get('entity_count', 0)} entities")
        else:
            st.sidebar.warning("⚠️ Database Issues:")
            if status.get("missing_indexes"):
                st.sidebar.caption(f"Missing indexes: {', '.join(status['missing_indexes'])}")
                st.sidebar.caption("Run the original notebook to create indexes.")
            if status.get("chunk_count", 0) == 0:
                st.sidebar.caption("No chunks found in database.")
            if status.get("indexes"):
                st.sidebar.caption(f"Found indexes: {', '.join(status['indexes'])}")
    
    # Recent history
    st.sidebar.divider()
    st.sidebar.markdown("### 📜 Recent Queries")
    
    history = db.get_user_history(user["id"], limit=5)
    if history:
        for entry in history:
            question = entry.get("question", "")[:40]
            if st.sidebar.button(
                f"📝 {question}...",
                key=f"hist_{entry['id']}",
                use_container_width=True
            ):
                st.session_state.view_history_id = entry["id"]
                st.session_state.active_tab = "History"
                st.rerun()
    else:
        st.sidebar.caption("No queries yet")


def _display_sidebar_extension_officer(user):
    """Sidebar content for Extension Officer mode."""
    st.sidebar.divider()
    st.sidebar.markdown(f"### ⚙️ {t('pipelines')}")
    st.sidebar.caption(t("both_pipelines_caption"))
    
    _init_both_pipelines()
    
    # Evaluation summary
    st.sidebar.divider()
    st.sidebar.markdown(f"### 📊 {t('evaluation_progress')}")
    summary = db.get_evaluation_summary(user["id"])
    total = summary.get("total", 0)
    if total > 0:
        v7_pref = summary.get("v7_preferred", 0)
        aditi_pref = summary.get("aditi_preferred", 0)
        tie = summary.get("tie", 0)
        st.sidebar.markdown(f"**{total}** {t('questions_evaluated')}")
        st.sidebar.markdown(f"{t('pipeline_a')}: **{v7_pref}** | {t('pipeline_b')}: **{aditi_pref}** | {t('tie')}: **{tie}**")
    else:
        st.sidebar.caption(t("no_evaluations_yet"))


# ================================================================
# Single Query Tab
# ================================================================

def display_single_query_tab():
    """Display single query interface."""
    st.markdown(
        '<div class="section-header" style="font-size:1.5rem; margin-bottom:4px;">Single Query</div>',
        unsafe_allow_html=True
    )
    st.caption("Ask a question and inspect the full pipeline response with retrieval details.")
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Khi nào nên bón phân đạm cho lúa? / When should I apply nitrogen fertilizer to rice?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Run Query", use_container_width=True, type="primary")
    
    if run_button and question:
        with st.spinner("Running pipeline..."):
            pipeline = st.session_state.pipeline
            result = pipeline.run_query(question, language=get_lang())
            st.session_state.current_result = result
            
            # Save to history and get query_id
            user_id = st.session_state.user["id"]
            query_id = db.save_query_result(
                user_id=user_id,
                question=result.question,
                farmer_answer=result.farmer_answer,
                trace_id=result.trace_id,
                trace_url=result.trace_url,
                raw_entities=result.raw_entities,
                aligned_entities=result.aligned_entities,
                graph_facts=result.graph_facts,
                vector_context=result.vector_context,
                keyword_results=result.keyword_results,
                pipeline_version=pipeline.get_version(),
                execution_time=result.execution_time,
                trace_data=result.trace_data,
                rrf_fused_results=result.rrf_fused_results,
                llm_context=result.llm_context
            )
            st.session_state.current_query_id = query_id
    
    # Display result
    if st.session_state.current_result:
        result = st.session_state.current_result
        user_id = st.session_state.user["id"]
        query_id = st.session_state.get("current_query_id")
        
        # Load comments for this query
        comments = db.get_comments_for_query(query_id) if query_id else {}
        
        st.divider()
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("⏱ Time", f"{result.execution_time:.2f}s")
        col2.metric("🏷 Topic", result.query_topic.replace('_', ' ').title())
        col3.metric("📊 Graph Facts", len(result.graph_facts))
        col4.metric("🔎 Semantic", len(result.vector_context))
        col5.metric("🔤 Keyword", len(result.keyword_results))
        
        st.divider()
        
        # Answer with comments
        display_answer(result.farmer_answer, result.question, query_id=query_id, user_id=user_id, comments=comments)
        
        st.divider()
        
        # Context with comments
        display_all_context(
            raw_entities=result.raw_entities,
            aligned_entities=result.aligned_entities,
            graph_facts=result.graph_facts,
            vector_context=result.vector_context,
            keyword_results=result.keyword_results,
            rrf_fused_results=result.rrf_fused_results,
            llm_context=result.llm_context,
            query_id=query_id,
            user_id=user_id,
            comments=comments
        )
        
        
        st.divider()
        
        # Trace and Agent Graph in columns
        col1, col2 = st.columns(2)
        
        with col1:
            display_trace_embedded(result.trace_id, result.trace_url, height=400)
        
        with col2:
            # Get the pipeline that was actually used for this query
            pipeline_used = st.session_state.pipeline  # This is the one we just used
            graph_data = pipeline_used.get_agent_graph()
            
            display_agent_graph(
                graph_data=graph_data, 
                height=600,
                theme=st.session_state.theme
            )


# ================================================================
# Batch Testing Tab
# ================================================================

def display_batch_testing_tab():
    """Display batch testing interface."""
    st.markdown(
        '<div class="section-header" style="font-size:1.5rem; margin-bottom:4px;">Batch Testing</div>',
        unsafe_allow_html=True
    )
    st.caption("Upload or paste a list of questions to test in bulk.")
    
    # Input section
    questions = display_batch_input()
    
    col1, col2 = st.columns([1, 4])
    with col1:
        can_run = len(questions) > 0 and st.session_state.pipeline is not None
        run_batch = st.button(
            "🚀 Run Batch",
            use_container_width=True,
            type="primary",
            disabled=not can_run
        )
    
    if run_batch and questions:
        pipeline = st.session_state.pipeline
        user_id = st.session_state.user["id"]
        
        # Create batch run
        batch_id = db.create_batch_run(
            user_id=user_id,
            name=f"Batch {len(questions)} questions",
            pipeline_version=pipeline.get_version(),
            question_count=len(questions)
        )
        
        results = []
        progress_bar = st.progress(0, text="Processing...")
        
        for i, question in enumerate(questions):
            progress_bar.progress(
                (i + 1) / len(questions),
                text=f"Processing question {i+1}/{len(questions)}..."
            )
            
            result = pipeline.run_query(question, language=get_lang())
            results.append(result.to_dict())
            
            # Save to history
            db.save_query_result(
                user_id=user_id,
                question=result.question,
                farmer_answer=result.farmer_answer,
                trace_id=result.trace_id,
                trace_url=result.trace_url,
                raw_entities=result.raw_entities,
                aligned_entities=result.aligned_entities,
                graph_facts=result.graph_facts,
                vector_context=result.vector_context,
                keyword_results=result.keyword_results,
                pipeline_version=pipeline.get_version(),
                execution_time=result.execution_time,
                batch_run_id=batch_id,
                trace_data=result.trace_data,
                rrf_fused_results=result.rrf_fused_results,
                llm_context=result.llm_context
            )
        
        progress_bar.empty()
        st.session_state.batch_results = results
        st.success(f"✅ Completed {len(results)} questions!")
    
    # Display results
    if st.session_state.batch_results:
        st.divider()
        
        # View mode toggle
        col1, col2 = st.columns([3, 1])
        with col2:
            view_mode = st.radio(
                "View mode:",
                ["Cards", "Detailed"],
                horizontal=True,
                label_visibility="collapsed"
            )
        
        # Export options
        display_batch_export(st.session_state.batch_results)
        
        st.divider()
        
        # Results
        display_batch_results(
            st.session_state.batch_results,
            view_mode="cards" if view_mode == "Cards" else "detailed"
        )


# ================================================================
# History Tab
# ================================================================

def display_history_tab():
    """Display query history."""
    st.markdown(
        '<div class="section-header" style="font-size:1.5rem; margin-bottom:4px;">Query History</div>',
        unsafe_allow_html=True
    )
    
    user_id = st.session_state.user["id"]
    
    # Check if viewing a specific entry
    if st.session_state.view_history_id:
        entry = db.get_query_detail(st.session_state.view_history_id, user_id)
        
        if entry:
            # Back button
            if st.button("← Back to History"):
                st.session_state.view_history_id = None
                st.rerun()
            
            st.divider()
            display_history_detail(entry, user_id)
        else:
            st.error("Entry not found")
            st.session_state.view_history_id = None
        return
    
    # Display history cards
    history = db.get_user_history(user_id, limit=50)
    
    def on_view(query_id):
        st.session_state.view_history_id = query_id
        st.rerun()
    
    def on_delete(query_id):
        db.delete_query(query_id, user_id)
        st.rerun()
    
    display_history_cards(history, on_view=on_view, on_delete=on_delete)


# ================================================================
# Extension Officer Mode Tabs
# ================================================================

def display_evaluate_tab():
    """Extension Officer: Single question evaluation with side-by-side answers."""
    st.markdown(
        f'<div class="section-header" style="font-size:1.5rem; margin-bottom:4px;">{t("evaluate_pipelines")}</div>',
        unsafe_allow_html=True
    )
    st.caption(t("evaluate_caption"))

    question = display_evaluation_question_input()

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button(t("run_both_pipelines"), use_container_width=True, type="primary")

    if run_button and question:
        kgraphrag = st.session_state.pipeline_kgraphrag
        aditi = st.session_state.pipeline_aditi

        col_a, col_b = st.columns(2)
        with col_a:
            with st.spinner(t("pipeline_a_thinking")):
                kgraphrag_result = kgraphrag.run_query(question, language=get_lang())
        with col_b:
            with st.spinner(t("pipeline_b_thinking")):
                aditi_result = aditi.run_query(question, language=get_lang())

        st.session_state.eval_kgraphrag_result = kgraphrag_result
        st.session_state.eval_aditi_result = aditi_result
        st.session_state.current_eval_id = None

    # Display results if available
    if st.session_state.eval_kgraphrag_result and st.session_state.eval_aditi_result:
        st.divider()
        display_side_by_side_answers(
            st.session_state.eval_kgraphrag_result,
            st.session_state.eval_aditi_result,
            eval_id=st.session_state.current_eval_id,
            user_id=st.session_state.user["id"]
        )


def display_batch_evaluate_tab():
    """Extension Officer: Batch evaluation — run many questions through both pipelines."""
    st.markdown(
        f'<div class="section-header" style="font-size:1.5rem; margin-bottom:4px;">{t("batch_evaluation")}</div>',
        unsafe_allow_html=True
    )
    st.caption(t("batch_eval_caption"))

    questions = display_batch_evaluation_input()

    col1, col2 = st.columns([1, 4])
    with col1:
        can_run = len(questions) > 0
        run_batch = st.button(
            t("run_batch"),
            use_container_width=True,
            type="primary",
            disabled=not can_run
        )

    if run_batch and questions:
        kgraphrag = st.session_state.pipeline_kgraphrag
        aditi = st.session_state.pipeline_aditi
        user_id = st.session_state.user["id"]

        batch_eval_id = db.create_batch_evaluation(
            user_id=user_id,
            name=f"Batch {len(questions)} questions",
            question_count=len(questions)
        )

        results = []
        progress_bar = st.progress(0, text="Processing...")

        for i, question in enumerate(questions):
            progress_bar.progress(
                (i + 1) / len(questions),
                text=f"Question {i+1}/{len(questions)}: running both pipelines..."
            )

            kgraphrag_result = kgraphrag.run_query(question, language=get_lang())
            aditi_result = aditi.run_query(question, language=get_lang())

            eval_id = db.save_evaluation(
                user_id=user_id,
                question=question,
                v7_answer=kgraphrag_result.farmer_answer,
                aditi_answer=aditi_result.farmer_answer,
                v7_execution_time=kgraphrag_result.execution_time,
                aditi_execution_time=aditi_result.execution_time,
                v7_trace_id=kgraphrag_result.trace_id,
                aditi_trace_id=aditi_result.trace_id,
                batch_eval_id=batch_eval_id
            )

            results.append({
                "eval_id": eval_id,
                "question": question,
                "v7_answer": kgraphrag_result.farmer_answer,
                "aditi_answer": aditi_result.farmer_answer,
                "v7_time": kgraphrag_result.execution_time,
                "aditi_time": aditi_result.execution_time,
            })

        progress_bar.empty()
        st.session_state.eval_batch_results = results
        st.success(t("batch_completed", n=len(results)))

    # Show batch results for review
    if st.session_state.eval_batch_results:
        st.divider()
        display_batch_evaluation_results(st.session_state.eval_batch_results)


def display_results_tab():
    """Extension Officer: View evaluation results and summary."""
    st.markdown(
        f'<div class="section-header" style="font-size:1.5rem; margin-bottom:4px;">{t("evaluation_results")}</div>',
        unsafe_allow_html=True
    )
    display_evaluation_results(st.session_state.user["id"])


# ================================================================
# Main App
# ================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Apply global styles (always)
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # Apply dark theme overrides
    if st.session_state.theme == "dark":
        st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    
    # Check authentication
    if not st.session_state.user:
        display_login_page()
        return
    
    # Display sidebar
    display_sidebar()
    
    # Top bar: tabs on left, theme toggle on right
    if st.session_state.app_mode == "Technical":
        tab_options = [t("tab_single_query"), t("tab_batch_testing"), t("tab_history")]
        tab_names = ["Single Query", "Batch Testing", "History"]
    else:
        tab_options = [t("tab_evaluate"), t("tab_batch_evaluate"), t("tab_results")]
        tab_names = ["Evaluate", "Batch Evaluate", "Results"]
    
    # Find index of active tab
    try:
        active_index = tab_names.index(st.session_state.active_tab)
    except ValueError:
        active_index = 0
    
    # Layout: tabs + theme button
    col_tabs, col_spacer, col_theme = st.columns([6, 2, 1])
    
    with col_tabs:
        selected_tab = st.radio(
            "Select view:",
            tab_options,
            index=active_index,
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col_theme:
        # Sun/moon theme toggle
        theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
        if st.button(theme_icon, key="theme_btn", help="Toggle dark/light mode"):
            new_theme = "dark" if st.session_state.theme == "light" else "light"
            st.session_state.theme = new_theme
            db.update_user_theme(st.session_state.user["id"], new_theme)
            st.rerun()
    
    # Update active tab if user manually changed it
    selected_tab_name = tab_names[tab_options.index(selected_tab)]
    if selected_tab_name != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab_name
        st.rerun()
    
    st.divider()
    
    # Display content based on active tab
    if st.session_state.app_mode == "Technical":
        if st.session_state.active_tab == "Single Query":
            display_single_query_tab()
        elif st.session_state.active_tab == "Batch Testing":
            display_batch_testing_tab()
        elif st.session_state.active_tab == "History":
            display_history_tab()
    else:
        if st.session_state.active_tab == "Evaluate":
            display_evaluate_tab()
        elif st.session_state.active_tab == "Batch Evaluate":
            display_batch_evaluate_tab()
        elif st.session_state.active_tab == "Results":
            display_results_tab()


if __name__ == "__main__":
    main()
