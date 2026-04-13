"""
ADITI Trial Pipeline — Updated Handbook (2025), Clean Schema v2

Subclasses AditiPipeline and overrides all methods that contain
schema-specific Cypher queries to match the v2 schema:

  v1 → v2 changes:
    embedding_qwen  → embedding
    multimodal_kg_index → chunk_vector_index
    aditi_multilingual_index → entity_vector_index
    chunk_text_index → chunk_fulltext_index
    PART_OF (Chunk→Section) → HAS_CHUNK (Section→Chunk)  [reversed]
    IN_CHAPTER (Section→Chapter) → HAS_SECTION (Chapter→Section) [reversed]
    CONTAINS_DATA → HAS_DATA
    REL → RELATED_TO (with r.type property)
    NutrientRequirement → NutrientReq
    Dual labels (:Entity:FarmingActivity) → single :Entity with e.type property
    section_num property → section_id property
"""

import re
import json
import logging
import time
import numpy as np
from typing import List, Dict, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import GraphDatabase
from langchain_core.messages import HumanMessage

from pipelines.aditi_pipeline import AditiPipeline, QwenRemoteLLM
from pipelines.base import PipelineResult
import config

logger = logging.getLogger(__name__)

# Trial Neo4j connection settings
TRIAL_NEO4J_URI = "bolt://172.17.0.1:7689"
TRIAL_NEO4J_USER = "neo4j"
TRIAL_NEO4J_PASSWORD = "Shanty@18"
TRIAL_NEO4J_DATABASE = "neo4j"

# v2 schema index names
V2_CHUNK_VECTOR_INDEX = "chunk_vector_index"
V2_ENTITY_VECTOR_INDEX = "entity_vector_index"
V2_FULLTEXT_INDEX = "chunk_fulltext_index"


class AditiTrialPipeline(AditiPipeline):
    """ADITI pipeline pointing at the trial Neo4j with clean schema v2."""

    def get_name(self) -> str:
        return "ADITI Trial — Updated Handbook"

    def get_version(self) -> str:
        return "2.0-trial-v2"

    def get_description(self) -> str:
        return (
            "Triple-hybrid retrieval (structured + BM25 + vector) against the "
            "updated 2025 handbook in the trial Neo4j instance (port 7689, clean schema v2)"
        )

    def initialize(self) -> None:
        if self._initialized:
            return

        logger.info("Initializing ADITI Trial Pipeline (v2 schema)...")

        self.llm = QwenRemoteLLM()
        logger.info("✓ LLM initialized")

        self.driver = GraphDatabase.driver(
            TRIAL_NEO4J_URI,
            auth=(TRIAL_NEO4J_USER, TRIAL_NEO4J_PASSWORD),
            max_connection_pool_size=10,
            connection_acquisition_timeout=5.0,
            connection_timeout=5.0,
        )
        self.driver.verify_connectivity()
        logger.info(f"✓ Connected to TRIAL Neo4j at {TRIAL_NEO4J_URI}")

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

        self._topic_centroids = self._compute_topic_centroids()
        logger.info(f"✓ Topic centroids computed for {len(self._topic_centroids)} topics")

        try:
            from langfuse import Langfuse
            self.langfuse = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
                timeout=config.LANGFUSE_TIMEOUT,
            )
            logger.info("✓ Langfuse tracing initialized")
        except Exception as e:
            logger.warning(f"Langfuse init failed (non-fatal): {e}")
            self.langfuse = None

        self._initialized = True
        logger.info("✅ ADITI Trial Pipeline ready (v2 schema, port 7689)")

    # ------------------------------------------------------------------
    # TOPIC CENTROIDS — v2 uses `embedding` not `embedding_qwen`
    # ------------------------------------------------------------------
    def _compute_topic_centroids(self):
        centroids = {}
        target_topics = [
            'water_management', 'fertilizer_management', 'pest_management',
            'seed_preparation', 'land_preparation', 'harvest',
            'straw_management', 'general',
        ]
        try:
            with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
                for topic in target_topics:
                    result = session.run("""
                        MATCH (c:Chunk)
                        WHERE c.topic_tag = $topic AND c.embedding IS NOT NULL
                        RETURN c.embedding AS emb
                    """, topic=topic)
                    embeddings = [record['emb'] for record in result]
                    if embeddings:
                        centroid = np.mean(embeddings, axis=0)
                        norm = np.linalg.norm(centroid)
                        if norm > 0:
                            centroid = centroid / norm
                        centroids[topic] = centroid
            logger.info(f"Computed centroids for topics: {list(centroids.keys())}")
        except Exception as e:
            logger.warning(f"Failed to compute topic centroids: {e}")
        return centroids

    # ------------------------------------------------------------------
    # ENTITY ALIGNMENT — v2 index name + single :Entity label
    # ------------------------------------------------------------------
    def _align_entity_with_vector_index(self, entity_text: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.entity_embedding_model.encode(entity_text).tolist()

        with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes($idx, $top_k, $query_emb)
                    YIELD node, score
                    WHERE node:Entity
                    RETURN node.name AS name, node.type AS type, score AS similarity
                    ORDER BY similarity DESC
                """, idx=V2_ENTITY_VECTOR_INDEX, query_emb=query_embedding,
                     top_k=top_k * 3)

                matches = []
                for record in result:
                    matches.append({
                        "kg_name": record["name"],
                        "kg_labels": ["Entity", record["type"]],
                        "score": record["similarity"]
                    })
                return matches

            except Exception as e:
                logger.warning(f"Entity vector search failed, falling back: {e}")
                result = session.run("""
                    MATCH (n:Entity)
                    WHERE toLower(n.name) CONTAINS toLower($entity)
                    RETURN n.name AS name, n.type AS type, 1.0 AS similarity
                    LIMIT $top_k
                """, entity=entity_text, top_k=top_k)
                return [{
                    "kg_name": r["name"],
                    "kg_labels": ["Entity", r["type"]],
                    "score": r["similarity"],
                } for r in result]

    # ------------------------------------------------------------------
    # KG TRAVERSAL — v2 uses RELATED_TO instead of REL
    # This is the method called by run_query (not kg_graph_traversal)
    # ------------------------------------------------------------------
    def traverse_kg_graph(self, aligned_entities: List[Dict], query: str = "", top_k: int = 10) -> List[Dict]:
        if not aligned_entities:
            return []

        facts = []

        with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
            for entity in aligned_entities:
                entity_name = entity.get("name") or entity.get("matched")
                if not entity_name:
                    continue

                search_names = [entity_name]
                acronyms = re.findall(r'\(([A-Z]{2,})\)', entity_name)
                search_names.extend(acronyms)

                result = session.run("""
                    MATCH (n:Entity)-[r:RELATED_TO]-(m:Entity)
                    WHERE n.name IN $names
                    RETURN n.name AS source,
                           'RELATED_TO' AS rel_type,
                           r.type AS rel_detail,
                           m.name AS target,
                           m.type AS target_label,
                           elementId(m) AS target_id
                    LIMIT 15
                """, names=search_names)

                for record in result:
                    rel_detail = record["rel_detail"] or "RELATED_TO"
                    weight_map = {
                        "REQUIRES": 1.0,
                        "APPLIED_AT": 0.9,
                        "PREVENTS": 0.9,
                        "REDUCES": 0.9,
                        "ENABLES": 0.9,
                        "RISK_DURING": 0.85,
                        "ACHIEVES": 0.85,
                        "USED_FOR": 0.85,
                        "CAUSES": 0.8,
                        "FOLLOWS": 0.7,
                        "PART_OF": 0.6,
                    }
                    weight = weight_map.get(rel_detail, 0.5)

                    facts.append({
                        "source": record["source"],
                        "relation": rel_detail,
                        "target": record["target"],
                        "target_label": record["target_label"],
                        "target_id": record["target_id"],
                        "tier": "core",
                        "score": entity.get("score", 1.0) * weight
                    })

        # Deduplicate
        unique_facts = {}
        for fact in facts:
            key = (fact["source"], fact["relation"], fact["target"])
            if key not in unique_facts or fact["score"] > unique_facts[key]["score"]:
                unique_facts[key] = fact

        all_facts = list(unique_facts.values())

        # Apply relevance filtering + reranking from parent
        filtered_facts = self._filter_relevant_facts(all_facts, aligned_entities, query)
        top_facts = self._rerank_facts(filtered_facts, query, top_k=top_k)
        logger.info(f"KG traversal v2: {len(all_facts)} total → {len(top_facts)} after filter+rerank")
        return top_facts

    # ------------------------------------------------------------------
    # STRUCTURED DATA SEARCH — v2 uses NutrientReq + HAS_DATA
    # ------------------------------------------------------------------
    def structured_data_search(self, intent: Dict, language: str = 'vi') -> List[Dict]:
        results = []

        with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
            if intent['type'] == 'nutrient_query' and intent.get('soil_type'):
                season = intent.get('season') or 'Winter-Spring'
                result = session.run("""
                    MATCH (n:NutrientReq)
                    WHERE toLower(n.soil_type) CONTAINS $soil_type
                      AND toLower(n.season) CONTAINS toLower($season)
                    OPTIONAL MATCH (c:Chunk)-[:HAS_DATA]->(n)
                    RETURN n.soil_type AS soil_type,
                           n.nutrient AS nutrient,
                           n.min_kg_ha AS min_kg_ha,
                           n.max_kg_ha AS max_kg_ha,
                           n.season AS season,
                           c.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN c.text_en ELSE c.text_vi END, c.text) AS chunk_text,
                           c.title AS chunk_title
                    LIMIT 10
                """, soil_type=intent['soil_type'], season=season, lang=language)

                for record in result:
                    results.append({
                        'source': 'structured_data',
                        'type': 'nutrient_requirement',
                        'data': {
                            'soil_type': record['soil_type'],
                            'nutrient': record['nutrient'],
                            'min_kg_ha': record['min_kg_ha'],
                            'max_kg_ha': record['max_kg_ha'],
                            'season': record['season'],
                        },
                        'chunk_id': record['chunk_id'],
                        'chunk_text': record['chunk_text'],
                        'chunk_title': record['chunk_title'],
                        'score': 1.0,
                    })

            # Other intent types (lime, AWD, seed, harvest, organic):
            # v2 schema doesn't have those structured nodes — chunk text handles them.

        return results

    # ------------------------------------------------------------------
    # BM25 SEARCH — v2 fulltext index name + reversed traversal
    # Also: broader topic mapping so water_management queries find AWD
    # content tagged seed_preparation
    # ------------------------------------------------------------------
    def bm25_search(self, query: str, top_k: int = 10, original_query: str = None,
                    query_topic: str = None, language: str = 'vi') -> List[Dict]:
        sanitized = re.sub(r'[\/\(\)\[\]\{\}\^\~\*\?\:\"\'\']', ' ', query)
        keywords = [w for w in sanitized.split() if len(w) >= 3]

        if query_topic and query_topic != 'general':
            expansion_terms = {
                'fertilizer_management': ['fertilizer', 'nitrogen', 'phosphorus', 'potassium', 'NPK', 'urea', 'lime', 'organic'],
                'water_management': ['water', 'irrigation', 'AWD', 'flooding', 'drainage', 'wetting', 'drying', 'alternate'],
                'pest_management': ['pest', 'disease', 'insect', 'weed', 'IPM', 'snail', 'blast', 'planthopper', 'control'],
                'seed_preparation': ['seed', 'variety', 'germination', 'sowing'],
                'land_preparation': ['soil', 'tillage', 'plowing', 'leveling'],
                'harvest': ['harvest', 'drying', 'storage', 'moisture'],
                'straw_management': ['straw', 'residue', 'compost'],
            }
            keywords.extend(expansion_terms.get(query_topic, [])[:5])

        search_query = ' '.join(keywords[:15])
        logger.info(f"BM25 search: topic={query_topic}, query='{search_query}'")

        if not search_query:
            return []

        # BROADER topic mapping: allow cross-topic retrieval
        # AWD content is tagged seed_preparation but relevant for water_management
        topic_filter_clause = ""
        params = {'search_query': search_query, 'top_k': top_k * 2, 'lang': language}

        if query_topic and query_topic != 'general':
            topic_map = {
                'direct_seeding': 'seed_preparation',
                'organic_fertilizer': 'fertilizer_management',
                'straw_use': 'straw_management',
            }
            canonical = topic_map.get(query_topic, query_topic)
            allowed = list(set([query_topic, canonical]))

            # Cross-topic expansion: topics that share content
            cross_topic = {
                'fertilizer_management': ['organic_fertilizer', 'straw_management'],
                'water_management': ['seed_preparation', 'land_preparation'],  # AWD in seed_preparation
                'seed_preparation': ['water_management', 'direct_seeding', 'land_preparation'],
                'pest_management': ['seed_preparation'],  # seed treatment for pest control
                'harvest': ['straw_management'],
            }
            allowed.extend(cross_topic.get(query_topic, []))
            allowed = list(set(allowed))

            topic_filter_clause = " WHERE node.topic_tag IN $allowed_topics"
            params['allowed_topics'] = allowed
            logger.info(f"BM25: filtering by topics {allowed}")

        with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
            try:
                cypher = f"""
                    CALL db.index.fulltext.queryNodes('{V2_FULLTEXT_INDEX}', $search_query)
                    YIELD node, score
                    {topic_filter_clause}
                    OPTIONAL MATCH (s:Section)-[:HAS_CHUNK]->(node)
                    OPTIONAL MATCH (ch:Chapter)-[:HAS_SECTION]->(s)
                    RETURN node.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN node.text_en ELSE node.text_vi END, node.text) AS text,
                           node.title AS title,
                           node.topic_tag AS topic_tag,
                           node.pages AS pages,
                           s.title AS section_title,
                           s.section_id AS section_num,
                           ch.chapter_num AS chapter_num,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
                result = session.run(cypher, **params)

                all_results = [{
                    'source': 'bm25',
                    'chunk_id': r['chunk_id'],
                    'text': r['text'],
                    'title': r['title'],
                    'topic_tag': r['topic_tag'] or 'general',
                    'pages': r['pages'] or [],
                    'section_title': r['section_title'] or r['title'],
                    'section_num': r['section_num'] or '',
                    'chapter_num': r['chapter_num'] or '',
                    'score': r['score'],
                } for r in result]

                print(f"[BM25] Raw results: {len(all_results)} chunks", flush=True)

                rerank_query = original_query or query
                all_results = self._llm_relevance_rerank(all_results, rerank_query)

                filtered = [r for r in all_results
                            if r.get('llm_relevance', -1) == -1 or r.get('llm_relevance', 0) >= 3]

                filtered.sort(key=lambda x: x['score'], reverse=True)
                return filtered[:top_k]
            except Exception as e:
                logger.warning(f"BM25 search error: {e}")
                return []

    # ------------------------------------------------------------------
    # VECTOR SEARCH — v2 chunk_vector_index + reversed traversal
    # Also: broader topic mapping (same as BM25)
    # ------------------------------------------------------------------
    def vector_search(self, query: str, top_k: int = 10, use_hyde: bool = True,
                      query_topic: str = None, language: str = 'vi') -> List[Dict]:
        search_query = self._generate_hyde_answer(query) if use_hyde else query
        query_embedding = list(self._embed_query_cached(search_query))

        allowed_topics = None
        if query_topic and query_topic != 'general':
            topic_map = {
                'direct_seeding': 'seed_preparation',
                'organic_fertilizer': 'fertilizer_management',
                'straw_use': 'straw_management',
            }
            canonical = topic_map.get(query_topic, query_topic)
            allowed_topics = list(set([query_topic, canonical]))
            cross_topic = {
                'fertilizer_management': ['organic_fertilizer', 'straw_management'],
                'water_management': ['seed_preparation', 'land_preparation'],
                'seed_preparation': ['water_management', 'direct_seeding', 'land_preparation'],
                'pest_management': ['seed_preparation'],
                'harvest': ['straw_management'],
            }
            allowed_topics.extend(cross_topic.get(query_topic, []))
            allowed_topics = list(set(allowed_topics))
            logger.info(f"Vector search: filtering by topics {allowed_topics}")

        with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
            try:
                fetch_k = top_k * 3 if allowed_topics else top_k * 2
                topic_where = "WHERE node.topic_tag IN $allowed_topics" if allowed_topics else ""

                cypher = f"""
                    CALL db.index.vector.queryNodes($idx, $top_k, $query_embedding)
                    YIELD node, score
                    {topic_where}
                    OPTIONAL MATCH (s:Section)-[:HAS_CHUNK]->(node)
                    OPTIONAL MATCH (ch:Chapter)-[:HAS_SECTION]->(s)
                    RETURN node.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN node.text_en ELSE node.text_vi END, node.text) AS text,
                           node.title AS title,
                           node.topic_tag AS topic_tag,
                           node.pages AS pages,
                           s.title AS section_title,
                           s.section_id AS section_num,
                           ch.chapter_num AS chapter_num,
                           score
                    ORDER BY score DESC
                """
                params = {
                    'idx': V2_CHUNK_VECTOR_INDEX,
                    'top_k': fetch_k,
                    'query_embedding': query_embedding,
                    'lang': language,
                }
                if allowed_topics:
                    params['allowed_topics'] = allowed_topics

                result = session.run(cypher, **params)

                all_results = [{
                    'source': 'vector',
                    'chunk_id': r['chunk_id'],
                    'text': r['text'],
                    'title': r['title'],
                    'topic_tag': r['topic_tag'] or 'general',
                    'pages': r['pages'] or [],
                    'section_title': r['section_title'] or r['title'],
                    'section_num': r['section_num'] or '',
                    'chapter_num': r['chapter_num'] or '',
                    'score': r['score'],
                } for r in result]

                filtered = self._filter_generic_chunks(all_results)
                return filtered[:top_k]
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                return []

    # ------------------------------------------------------------------
    # SIBLING CHUNK INJECTION — v2 reversed direction
    # ------------------------------------------------------------------
    def _inject_sibling_chunks(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results

        existing_ids = {r.get('chunk_id') for r in results}
        injected = []
        seen_prefixes = set()

        with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
            for r in results:
                chunk_id = r.get('chunk_id')
                if not chunk_id:
                    continue
                try:
                    sec_result = session.run("""
                        MATCH (s:Section)-[:HAS_CHUNK]->(c:Chunk {chunk_id: $cid})
                        RETURN s.section_id AS section_id
                    """, cid=chunk_id)
                    sec_rec = sec_result.single()
                    if not sec_rec:
                        continue

                    section_id = sec_rec['section_id']
                    parts = section_id.rsplit('.', 1)
                    if len(parts) < 2:
                        continue
                    parent_prefix = parts[0]

                    if parent_prefix in seen_prefixes:
                        continue
                    seen_prefixes.add(parent_prefix)

                    siblings = session.run("""
                        MATCH (sib_sec:Section)-[:HAS_CHUNK]->(sib_chunk:Chunk)
                        WHERE sib_sec.section_id STARTS WITH $prefix
                          AND sib_sec.section_id <> $sid
                          AND NOT sib_chunk.chunk_id IN $existing
                        OPTIONAL MATCH (ch:Chapter)-[:HAS_SECTION]->(sib_sec)
                        RETURN sib_chunk.chunk_id AS chunk_id,
                               sib_chunk.text AS text,
                               sib_chunk.title AS title,
                               sib_sec.title AS section_title,
                               sib_sec.section_id AS section_num,
                               ch.chapter_num AS chapter_num
                        LIMIT 3
                    """, prefix=parent_prefix, sid=section_id, existing=list(existing_ids))

                    for sib in siblings:
                        sib_id = sib['chunk_id']
                        if sib_id not in existing_ids:
                            injected.append({
                                'chunk_id': sib_id,
                                'text': sib['text'],
                                'title': sib['title'],
                                'section_title': sib['section_title'] or sib['title'],
                                'section_num': sib['section_num'] or '',
                                'chapter_num': sib['chapter_num'] or '',
                                'structured_score': 0,
                                'bm25_score': 0,
                                'vector_score': 0,
                                'hybrid_score': r.get('hybrid_score', 0) * 0.8,
                                'source': 'sibling_injection',
                            })
                            existing_ids.add(sib_id)
                except Exception as e:
                    logger.warning(f"Sibling injection failed for {chunk_id}: {e}")

        if injected:
            logger.info(f"Sibling injection: added {len(injected)} sibling chunks")
            results.extend(injected[:3])
        return results

    # ------------------------------------------------------------------
    # VERIFY DATABASE — v2 schema
    # ------------------------------------------------------------------
    def verify_database(self):
        try:
            with self.driver.session(database=TRIAL_NEO4J_DATABASE) as session:
                chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt").single()["cnt"]
                entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS cnt").single()["cnt"]
                section_count = session.run("MATCH (s:Section) RETURN count(s) AS cnt").single()["cnt"]

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
                    "schema": "v2-clean",
                    "note": "Trial instance — updated handbook 2025, clean schema v2",
                }
        except Exception as e:
            return {"ready": False, "error": str(e)}
