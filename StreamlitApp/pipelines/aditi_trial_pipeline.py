"""
ADITI Trial Pipeline — Updated Handbook (2025)

Identical to the production ADITI Triple-Hybrid Pipeline but connects to the
trial Neo4j instance (port 7689) which contains the updated handbook data.
Production Neo4j on port 7687 is not touched.
"""

import logging
from neo4j import GraphDatabase
from pipelines.aditi_pipeline import AditiPipeline
import config

logger = logging.getLogger(__name__)

# Trial Neo4j connection settings
TRIAL_NEO4J_URI = "bolt://172.17.0.1:7689"
TRIAL_NEO4J_USER = "neo4j"
TRIAL_NEO4J_PASSWORD = "Shanty@18"
TRIAL_NEO4J_DATABASE = "neo4j"


class AditiTrialPipeline(AditiPipeline):
    """ADITI pipeline pointing at the trial Neo4j (updated handbook)."""

    def get_name(self) -> str:
        return "ADITI Trial — Updated Handbook"

    def get_version(self) -> str:
        return "2.0-trial"

    def get_description(self) -> str:
        return (
            "Triple-hybrid retrieval (structured + BM25 + vector) against the "
            "updated 2025 handbook in the trial Neo4j instance (port 7689)"
        )

    def initialize(self) -> None:
        """Initialize with trial Neo4j instead of production."""
        if self._initialized:
            return

        logger.info("Initializing ADITI Trial Pipeline (updated handbook)...")

        # Initialize LLM (same as production)
        from pipelines.aditi_pipeline import QwenRemoteLLM
        self.llm = QwenRemoteLLM()
        logger.info("✓ LLM initialized")

        # Connect to TRIAL Neo4j (port 7689)
        self.driver = GraphDatabase.driver(
            TRIAL_NEO4J_URI,
            auth=(TRIAL_NEO4J_USER, TRIAL_NEO4J_PASSWORD),
            max_connection_pool_size=10,
            connection_acquisition_timeout=5.0,
            connection_timeout=5.0,
        )
        self.driver.verify_connectivity()
        logger.info(f"✓ Connected to TRIAL Neo4j at {TRIAL_NEO4J_URI}")

        # Load embedding models (same as production — both use multilingual-e5-large)
        from sentence_transformers import SentenceTransformer
        embedding_device = getattr(config, "EMBEDDING_DEVICE", "cpu")

        self.entity_model_name = "intfloat/multilingual-e5-large"
        self.entity_embedding_model = SentenceTransformer(
            self.entity_model_name, device=embedding_device
        )
        logger.info("✓ Entity model loaded (1024-dim, multilingual)")

        self.doc_model_name = config.EMBEDDING_MODEL_NAME
        self.embedding_model = SentenceTransformer(
            self.doc_model_name, device=embedding_device
        )
        logger.info(f"✓ Document model loaded ({self.doc_model_name})")

        # Topic centroids
        self._topic_centroids = self._compute_topic_centroids()
        logger.info(f"✓ Topic centroids computed for {len(self._topic_centroids)} topics")

        # Langfuse
        from langfuse import Langfuse
        self.langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
            timeout=config.LANGFUSE_TIMEOUT,
        )
        logger.info("✓ Langfuse tracing initialized")

        self._initialized = True
        logger.info("✅ ADITI Trial Pipeline ready (trial Neo4j on port 7689)")

    def verify_database(self):
        """Verify trial database status."""
        try:
            with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
                # Count nodes
                result = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt")
                chunk_count = result.single()["cnt"]

                result = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
                entity_count = result.single()["cnt"]

                result = session.run("MATCH (s:Section) RETURN count(s) AS cnt")
                section_count = result.single()["cnt"]

                # Check vector index
                result = session.run(
                    "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name"
                )
                indexes = [r["name"] for r in result]

                return {
                    "ready": chunk_count > 0,
                    "chunk_count": chunk_count,
                    "entity_count": entity_count,
                    "section_count": section_count,
                    "vector_indexes": indexes,
                    "neo4j_uri": TRIAL_NEO4J_URI,
                    "note": "Trial instance — updated handbook 2025",
                }
        except Exception as e:
            return {"ready": False, "error": str(e)}
