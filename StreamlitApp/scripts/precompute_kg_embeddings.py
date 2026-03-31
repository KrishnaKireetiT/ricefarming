"""
Precompute and store ADITI multilingual embeddings in Neo4j for fast alignment.
Run this for ADITI pipeline - uses separate embedding_multilingual property.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precompute_embeddings():
    """Compute and store ADITI multilingual embeddings for all KG entities."""
    
    # Initialize
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    # ADITI uses multilingual-e5-large for Vietnamese + English support
    aditi_model_name = "intfloat/multilingual-e5-large"
    embedding_model = SentenceTransformer(
        aditi_model_name,
        device=getattr(config, 'EMBEDDING_DEVICE', 'cpu')
    )
    
    logger.info(f"Using ADITI multilingual embedding model: {aditi_model_name}")
    logger.info("This will NOT affect V6/V7 pipelines (they use separate embeddings)")
    
    with driver.session(database=config.NEO4J_DATABASE) as session:
        # Get all entities that need embeddings
        result = session.run("""
            MATCH (n) 
            WHERE n.name IS NOT NULL 
              AND NOT n:Chunk AND NOT n:Section AND NOT n:Chapter AND NOT n:Document
            RETURN id(n) as node_id, n.name AS name, labels(n) AS labels
            LIMIT 10000
        """)
        
        entities = list(result)
        logger.info(f"Found {len(entities)} entities to embed")
        
        # Batch process embeddings
        batch_size = 100
        total_updated = 0
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            names = [e["name"] for e in batch]
            
            # Compute embeddings
            embeddings = embedding_model.encode(names, show_progress_bar=False)
            
            # Store in Neo4j (separate property for ADITI)
            for j, entity in enumerate(batch):
                session.run("""
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n.embedding_multilingual = $embedding
                """, node_id=entity["node_id"], embedding=embeddings[j].tolist())
                
                total_updated += 1
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Processed {total_updated} / {len(entities)} entities")
        
        logger.info(f"✓ Successfully stored embeddings for {total_updated} entities")
        
        # Create vector index for ADITI fast similarity search
        logger.info("Creating ADITI multilingual vector index...")
        try:
            # Note: multilingual-e5-large produces 1024-dimensional embeddings
            session.run("""
                CREATE VECTOR INDEX aditi_multilingual_index IF NOT EXISTS
                FOR (n:Entity)
                ON n.embedding_multilingual
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            logger.info("✓ ADITI vector index created (1024 dimensions, separate from V6/V7)")
        except Exception as e:
            logger.warning(f"Vector index creation skipped (may already exist): {e}")
    
    driver.close()
    logger.info("✓ Embedding precomputation complete!")


if __name__ == "__main__":
    precompute_embeddings()
