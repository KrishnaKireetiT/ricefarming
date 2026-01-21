"""
Centralized configuration management for the Streamlit Pipeline Testing App.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ========================================
# Paths
# ========================================
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# ========================================
# Qwen LLM Configuration
# ========================================
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://hanoi2.ucd.ie/v1")
LLM_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")

# ========================================
# Neo4j Database
# ========================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ========================================
# Langfuse Tracing
# ========================================
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_TIMEOUT = int(os.getenv("LANGFUSE_TIMEOUT", "10"))

# ========================================
# Embedding Model
# ========================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
EMBEDDING_NODE_PROPERTY = "embedding_qwen"
VECTOR_INDEX_NAME = "multimodal_kg_index"

# ========================================
# Search Configuration
# ========================================
ENTITY_ALIGNMENT_THRESHOLD = float(os.getenv("ENTITY_ALIGNMENT_THRESHOLD", "0.6"))
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "5"))
KEYWORD_SEARCH_K = int(os.getenv("KEYWORD_SEARCH_K", "3"))
GRAPH_TRAVERSAL_TOP_K = int(os.getenv("GRAPH_TRAVERSAL_TOP_K", "20"))

# V7 Pipeline Configuration
RRF_K = int(os.getenv("RRF_K", "60"))  # Reciprocal Rank Fusion constant
VECTOR_SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.8"))  # Semantic search threshold
KEYWORD_SCORE_THRESHOLD = float(os.getenv("KEYWORD_SCORE_THRESHOLD", "0.2"))  # Keyword search threshold

# ========================================
# Relation Weights for Graph Traversal
# ========================================
RELATION_WEIGHTS = {"REL": 1.0, "MENTIONS": 0.25}
DEFAULT_REL_WEIGHT = 0.1

# ========================================
# App Settings
# ========================================
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "change-this-to-a-random-secret")
APP_NAME = "Pipeline Testing App"
APP_VERSION = "1.0.0"


def validate_config():
    """Validate required configuration is present."""
    errors = []
    
    if not QWEN_API_KEY:
        errors.append("QWEN_API_KEY is required")
    
    if not NEO4J_PASSWORD or NEO4J_PASSWORD == "neo4j":
        errors.append("NEO4J_PASSWORD should be set (not default)")
    
    return errors
