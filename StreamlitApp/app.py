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
from pipelines.pipeline_v6 import PipelineV6
from pipelines.base import PipelineResult

# Import components
from components.answer_display import display_answer, display_answer_card
from components.context_display import display_all_context, display_entities
from components.trace_display import display_trace_embedded, display_trace_link
from components.agent_graph_display import display_agent_graph
from components.history_display import display_history_table, display_history_detail
from components.batch_display import display_batch_input, display_batch_results, display_batch_export


# ================================================================
# Session State Initialization
# ================================================================

def init_session_state():
    """Initialize session state variables."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "view_history_id" not in st.session_state:
        st.session_state.view_history_id = None
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


# ================================================================
# Authentication UI
# ================================================================

def display_login_page():
    """Display login/signup page."""
    st.title("🌾 Pipeline Testing App")
    st.markdown("Welcome! Please sign in or create an account to continue.")
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    user = db.authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.theme = user.get("theme", "light")
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email address")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.warning("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif db.user_exists(username=new_username):
                    st.error("Username already taken")
                elif db.user_exists(email=new_email):
                    st.error("Email already registered")
                else:
                    user_id = db.create_user(new_username, new_email, new_password)
                    if user_id:
                        st.success("Account created! Please login.")
                    else:
                        st.error("Failed to create account")


# ================================================================
# Sidebar
# ================================================================

def display_sidebar():
    """Display sidebar with user info, settings, and history."""
    user = st.session_state.user
    
    st.sidebar.title("🌾 Pipeline Tester")
    
    # User info
    st.sidebar.markdown(f"**👤 {user['username']}**")
    
    # Theme toggle
    st.sidebar.divider()
    theme = st.sidebar.radio(
        "🎨 Theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True
    )
    new_theme = theme.lower()
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        db.update_user_theme(user["id"], new_theme)
        st.rerun()
    
    # Pipeline selection
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Pipeline")
    
    # V6 Pipeline - No Visual Nodes (image info embedded in chunks)
    pipeline_options = ["KG Pipeline V6 (No Visuals)"]
    selected_pipeline = st.sidebar.selectbox(
        "Select pipeline:",
        pipeline_options,
        label_visibility="collapsed"
    )
    
    # Initialize pipeline if needed
    if st.session_state.pipeline is None:
        with st.sidebar.status("Loading pipeline...", expanded=True):
            st.session_state.pipeline = PipelineV6()
            st.session_state.pipeline.initialize()
        st.sidebar.success("Pipeline ready!")
    
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
                st.rerun()
    else:
        st.sidebar.caption("No queries yet")
    
    # Logout button
    st.sidebar.divider()
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.pipeline = None
        st.rerun()


# ================================================================
# Single Query Tab
# ================================================================

def display_single_query_tab():
    """Display single query interface."""
    st.header("🔍 Single Query")
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Khi nào nên bón phân đạm cho lúa? / When should I apply nitrogen fertilizer to rice?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("🚀 Run Query", use_container_width=True, type="primary")
    
    if run_button and question:
        with st.spinner("Running pipeline..."):
            pipeline = st.session_state.pipeline
            result = pipeline.run_query(question)
            st.session_state.current_result = result
            
            # Save to history
            user_id = st.session_state.user["id"]
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
                execution_time=result.execution_time
            )
    
    # Display results
    result = st.session_state.current_result
    if result:
        st.divider()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⏱ Time", f"{result.execution_time:.2f}s")
        col2.metric("📊 Graph Facts", len(result.graph_facts))
        col3.metric("🔎 Semantic", len(result.vector_context))
        col4.metric("🔤 Keyword", len(result.keyword_results))
        
        st.divider()
        
        # Answer
        display_answer(result.farmer_answer, result.question)
        
        st.divider()
        
        # Context
        display_all_context(
            raw_entities=result.raw_entities,
            aligned_entities=result.aligned_entities,
            graph_facts=result.graph_facts,
            vector_context=result.vector_context,
            keyword_results=result.keyword_results,

        )
        
        st.divider()
        
        # Trace and Agent Graph in columns
        col1, col2 = st.columns(2)
        
        with col1:
            display_trace_embedded(result.trace_id, result.trace_url, height=400)
        
        with col2:
            display_agent_graph(height=400)


# ================================================================
# Batch Testing Tab
# ================================================================

def display_batch_testing_tab():
    """Display batch testing interface."""
    st.header("📦 Batch Testing")
    
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
            
            result = pipeline.run_query(question)
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
                batch_run_id=batch_id
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
    st.header("📜 Query History")
    
    user_id = st.session_state.user["id"]
    
    # Check if viewing a specific entry
    if st.session_state.view_history_id:
        entry = db.get_query_detail(st.session_state.view_history_id, user_id)
        
        if entry:
            if st.button("← Back to History"):
                st.session_state.view_history_id = None
                st.rerun()
            
            st.divider()
            display_history_detail(entry)
        else:
            st.error("Entry not found")
            st.session_state.view_history_id = None
        return
    
    # Display history table
    history = db.get_user_history(user_id, limit=50)
    
    def on_view(query_id):
        st.session_state.view_history_id = query_id
        st.rerun()
    
    def on_delete(query_id):
        db.delete_query(query_id, user_id)
        st.rerun()
    
    display_history_table(history, on_view=on_view, on_delete=on_delete)


# ================================================================
# Main App
# ================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Apply theme
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
            /* ============================================ */
            /* Dark Theme - Comprehensive Styling          */
            /* ============================================ */
            
            /* Base colors */
            :root {
                --bg-primary: #0e1117;
                --bg-secondary: #1a1d24;
                --bg-tertiary: #262730;
                --text-primary: #fafafa;
                --text-secondary: #b0b8c4;
                --text-muted: #7a8599;
                --accent-primary: #4da6ff;
                --accent-success: #00d084;
                --accent-warning: #ffc107;
                --border-color: #3d4654;
            }
            
            /* Main app background */
            .stApp {
                background-color: var(--bg-primary);
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: var(--bg-secondary);
                border-right: 1px solid var(--border-color);
            }
            [data-testid="stSidebar"] .stMarkdown {
                color: var(--text-primary);
            }
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
            }
            
            /* Regular text */
            .stMarkdown, p, span, label {
                color: var(--text-secondary);
            }
            
            /* ===== METRICS FIX - Most important ===== */
            [data-testid="stMetric"] {
                background-color: var(--bg-tertiary);
                padding: 12px 16px;
                border-radius: 8px;
                border: 1px solid var(--border-color);
            }
            [data-testid="stMetricLabel"] {
                color: var(--text-secondary) !important;
            }
            [data-testid="stMetricLabel"] > div {
                color: var(--text-secondary) !important;
            }
            [data-testid="stMetricLabel"] p {
                color: var(--text-secondary) !important;
                font-size: 0.9rem;
            }
            [data-testid="stMetricValue"] {
                color: var(--accent-primary) !important;
            }
            [data-testid="stMetricValue"] > div {
                color: var(--accent-primary) !important;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: var(--bg-secondary);
                border-radius: 8px;
                padding: 4px;
                gap: 4px;
            }
            .stTabs [data-baseweb="tab"] {
                color: var(--text-secondary);
                background-color: transparent;
                border-radius: 6px;
            }
            .stTabs [aria-selected="true"] {
                background-color: var(--bg-tertiary);
                color: var(--text-primary) !important;
            }
            
            /* Expanders */
            .streamlit-expanderHeader {
                background-color: var(--bg-secondary);
                color: var(--text-primary) !important;
                border-radius: 8px;
            }
            .streamlit-expanderContent {
                background-color: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-top: none;
                border-radius: 0 0 8px 8px;
            }
            
            /* Text inputs and text areas */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                background-color: var(--bg-tertiary);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
            }
            .stTextInput > div > div > input:focus,
            .stTextArea > div > div > textarea:focus {
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 1px var(--accent-primary);
            }
            
            /* Buttons */
            .stButton > button {
                background-color: var(--bg-tertiary);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
            }
            .stButton > button:hover {
                background-color: var(--border-color);
                border-color: var(--accent-primary);
            }
            .stButton > button[kind="primary"] {
                background-color: var(--accent-success);
                color: #000;
                border: none;
            }
            
            /* Select boxes / dropdowns */
            .stSelectbox > div > div {
                background-color: var(--bg-tertiary);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
            }
            
            /* Radio buttons */
            .stRadio > div {
                color: var(--text-secondary);
            }
            
            /* Dividers */
            hr {
                border-color: var(--border-color);
            }
            
            /* Alerts and info boxes */
            .stAlert {
                background-color: var(--bg-tertiary);
                border: 1px solid var(--border-color);
            }
            
            /* Code blocks */
            code {
                background-color: var(--bg-tertiary);
                color: var(--accent-primary);
                padding: 2px 6px;
                border-radius: 4px;
            }
            
            /* Tables */
            .stDataFrame {
                background-color: var(--bg-secondary);
            }
            
            /* Captions */
            .stCaption {
                color: var(--text-muted) !important;
            }
            
            /* Question display box styling */
            .question-box {
                background-color: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 12px 16px;
                color: var(--text-primary);
            }
            
            /* Blockquotes */
            blockquote {
                border-left-color: var(--accent-primary);
                background-color: var(--bg-tertiary);
                color: var(--text-secondary);
                padding: 8px 16px;
                border-radius: 0 8px 8px 0;
            }
            
            /* Links */
            a {
                color: var(--accent-primary);
            }
            a:hover {
                color: #7ac0ff;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Check authentication
    if not st.session_state.user:
        display_login_page()
        return
    
    # Display sidebar
    display_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Query",
        "📦 Batch Testing",
        "📜 History"
    ])
    
    with tab1:
        display_single_query_tab()
    
    with tab2:
        display_batch_testing_tab()
    
    with tab3:
        display_history_tab()


if __name__ == "__main__":
    main()
