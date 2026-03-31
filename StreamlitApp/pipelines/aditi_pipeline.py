"""
ADITI Triple-Hybrid Pipeline
Based on /mnt/node5_tpu_data_code_1/aditi/ADITI_PIPELINE/triple_hybrid_retrieval.py

Combines:
1. Structured data lookup (highest priority for factual queries)
2. BM25 full-text search (good for keyword matching)
3. Vector similarity (good for semantic understanding)
"""

import os
import time
import json
import re
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.prompts import ChatPromptTemplate

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langfuse import Langfuse

from pipelines.base import BasePipeline, PipelineResult
import config

logger = logging.getLogger(__name__)


# ================================================================
# Custom LLM Wrapper
# ================================================================

class QwenRemoteLLM(BaseChatModel):
    """Remote Qwen LLM wrapper using OpenAI-compatible API."""
    
    base_url: str = Field(default_factory=lambda: config.QWEN_BASE_URL)
    api_key: str = Field(default_factory=lambda: config.QWEN_API_KEY)
    model_name: str = Field(default_factory=lambda: config.LLM_MODEL_NAME)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    )
    def _call_api(self, url: str, headers: dict, payload: dict) -> dict:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        openai_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage): role = "system"
            elif isinstance(m, AIMessage): role = "assistant"
            openai_messages.append({"role": role, "content": m.content})

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "messages": openai_messages, "temperature": 0.0}

        try:
            data = self._call_api(url, headers, payload)
            content = data["choices"][0]["message"]["content"]
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        except Exception as e:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(e)))])

    @property
    def _llm_type(self):
        return "qwen-remote"


class AditiPipeline(BasePipeline):
    """
    ADITI Triple-Hybrid Pipeline
    
    Features:
    - Structured data extraction for factual queries (nutrients, lime, etc.)
    - BM25 full-text search with Lucene
    - Vector similarity search
    - Hybrid score fusion (50% structured, 30% BM25, 20% vector)
    """
    
    _translation_cache: Dict[str, str] = {}  # module-level cache: vi_term -> en_term

    def __init__(self):
        self._initialized = False
        self.driver = None
        self.embedding_model = None
        self.llm = None
        self.langfuse = None
        self.top_k = 5
    
    def get_name(self) -> str:
        return "ADITI Triple-Hybrid Pipeline"
    
    def get_version(self) -> str:
        return "1.0"
    
    def get_description(self) -> str:
        return "Triple-hybrid retrieval combining structured data lookup, BM25 search, and vector similarity"
    
    def initialize(self) -> None:
        """Initialize Neo4j connection, embedding model, and Langfuse."""
        if self._initialized:
            return
        
        logger.info("Initializing ADITI Triple-Hybrid Pipeline...")
        
        # Initialize LLM
        self.llm = QwenRemoteLLM()
        logger.info("✓ LLM initialized")
        
        # Connect to Neo4j with connection pool + fast-fail timeout
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_pool_size=10,
            connection_acquisition_timeout=5.0,
            connection_timeout=5.0,
        )
        self.driver.verify_connectivity()
        logger.info("✓ Connected to Neo4j")
        
        # Load ADITI's multilingual embedding model for entity alignment
        self.entity_model_name = "intfloat/multilingual-e5-large"
        logger.info(f"Loading ADITI entity alignment model: {self.entity_model_name}")
        embedding_device = getattr(config, 'EMBEDDING_DEVICE', 'cpu')
        self.entity_embedding_model = SentenceTransformer(
            self.entity_model_name,
            device=embedding_device
        )
        logger.info("✓ Entity model loaded (1024-dim, multilingual)")
        
        # Load document retrieval model (must match V6/V7 document embeddings)
        self.doc_model_name = config.EMBEDDING_MODEL_NAME
        logger.info(f"Loading document retrieval model: {self.doc_model_name}")
        self.embedding_model = SentenceTransformer(
            self.doc_model_name,
            device=embedding_device
        )
        logger.info("✓ Document model loaded (768-dim, matches Neo4j document embeddings)")
        
        # Compute topic centroid embeddings for semantic topic detection
        self._topic_centroids = self._compute_topic_centroids()
        logger.info(f"✓ Topic centroids computed for {len(self._topic_centroids)} topics")
        
        # Initialize Langfuse
        self.langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST
        )
        logger.info("✓ Langfuse initialized")
        
        self._initialized = True
        logger.info("✅ ADITI Pipeline ready")
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def _compute_topic_centroids(self) -> Dict[str, Any]:
        """
        Compute mean embedding vectors per topic_tag from Neo4j chunk embeddings.
        These centroids are used for semantic topic detection at query time.
        """
        import numpy as np
        centroids = {}
        target_topics = [
            'water_management', 'fertilizer_management', 'pest_management',
            'seed_preparation', 'land_preparation', 'harvest',
            'straw_management', 'direct_seeding', 'organic_fertilizer',
            'straw_use'
        ]
        try:
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                for topic in target_topics:
                    result = session.run("""
                        MATCH (c:Chunk)
                        WHERE c.topic_tag = $topic AND c.embedding_qwen IS NOT NULL
                        RETURN c.embedding_qwen AS emb
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
    
    def _cosine_sim(self, a, b) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        return float(np.dot(a, b))
    
    def _align_entity_with_vector_index(self, entity_text: str, top_k: int = 5) -> List[Dict]:
        """Use Neo4j vector index for fast entity alignment (requires precomputed embeddings)."""
        query_embedding = self.entity_embedding_model.encode(entity_text).tolist()
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            # Use aditi_multilingual_index for fast vector search (instead of brute-force Cypher)
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('aditi_multilingual_index', $top_k, $query_emb)
                    YIELD node, score
                    WHERE NOT node:Chunk AND NOT node:Section AND NOT node:Chapter AND NOT node:Document
                    RETURN node.name AS name, labels(node) AS labels, score AS similarity
                    ORDER BY similarity DESC
                """, query_emb=query_embedding, top_k=top_k * 3)
                
                matches = []
                for record in result:
                    matches.append({
                        "kg_name": record["name"],
                        "kg_labels": record["labels"],
                        "score": record["similarity"]
                    })
                
                return matches
                
            except Exception as e:
                logger.warning(f"Vector index search failed, falling back to exact match: {e}")
                # Fallback to exact name match
                result = session.run("""
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS toLower($entity)
                      AND NOT n:Chunk AND NOT n:Section AND NOT n:Chapter AND NOT n:Document
                    RETURN n.name AS name, labels(n) AS labels, 1.0 AS similarity
                    LIMIT $top_k
                """, entity=entity_text, top_k=top_k)
                
                matches = []
                for record in result:
                    matches.append({
                        "kg_name": record["name"],
                        "kg_labels": record["labels"],
                        "score": record["similarity"]
                    })
                
                return matches
    
    def _convert_units(self, value: float, unit: str) -> tuple:
        """Convert Vietnamese and other agricultural units to standard SI units"""
        unit_lower = unit.lower()
        
        # Area conversions
        if 'công' in unit_lower:
            # 1 công = 1,300 m² = 0.13 hectare
            return value * 0.13, 'hectare'
        elif 'ha' in unit_lower or 'hectare' in unit_lower:
            return value, 'hectare'
        elif 'm²' in unit_lower or 'square meter' in unit_lower:
            return value / 10000, 'hectare'
        
        # Keep original if no conversion needed
        return value, unit
    
    def extract_entities_and_translate(self, question: str) -> Dict:
        """
        OPTIMIZED: Combined entity extraction + translation in a single LLM call.
        Returns: {"entities": [...], "english_query": "...", "is_vietnamese": bool}
        """
        try:
            combined_prompt = f"""Extract agricultural entities from this query AND translate to English if Vietnamese.

Query: {question}

Return JSON with:
{{
  "entities": ["entity1", "entity2", ...],
  "english_query": "translated query in English",
  "is_vietnamese": true/false
}}

Entities to extract: soil types, nutrients (N/P/K), seasons, farming activities, measurements."""
            
            result = self.llm.invoke([HumanMessage(content=combined_prompt)])
            response = result.content.strip()
            
            # Parse JSON response
            if '```json' in response:
                response = response.split('```json', 1)[1].split('```', 1)[0].strip()
            elif '```' in response:
                response = response.split('```', 1)[1].split('```', 1)[0].strip()
            
            import json as _json
            data = _json.loads(response)
            
            logger.info(f"[COMBINED] Extracted {len(data.get('entities', []))} entities, is_vietnamese={data.get('is_vietnamese')}")
            return data
        except Exception as e:
            logger.warning(f"Combined extraction failed: {e}, falling back to separate calls")
            return {"entities": [], "english_query": question, "is_vietnamese": False}
    
    def extract_entities_multistage(self, question: str) -> Dict[str, Any]:
        """
        Multi-stage entity extraction with domain-specific classification.
        
        Returns dict with:
        - raw_entities: List of raw entity strings
        - typed_entities: Dict of entities by type
        - values: Extracted numeric values with units
        - relationships: Detected entity relationships
        """
        from langchain_core.output_parsers import JsonOutputParser
        
        # Stage 1: Extract entities using simple list format (supports Vietnamese and English)
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract key agricultural entities from the question in ANY language (Vietnamese or English). Focus on:
- Soil types: phù sa/alluvial, chua/acidic, phèn/acid sulfate
- Nutrients: đạm/nitrogen, lân/phosphorus, kali/potassium, vôi/lime
- Seasons: Đông-Xuân/Winter-Spring, Hè-Thu/Summer-Autumn, Thu-Đông/Autumn-Winter
- Farming activities: bón phân/fertilization, gieo sạ/sowing, tưới/irrigation, AWD/tưới ngập khô xen kẽ
- Water management: tưới/irrigation, tiêu nước/drainage, ngập khô xen kẽ/alternate wetting drying

Return JSON array with entity names in ORIGINAL language: ["entity1", "entity2", ...]"""),
            ("user", "Question: {q}")
        ])
        
        extracted_list = []
        try:
            chain = prompt | self.llm | parser
            result = chain.invoke({"q": question})
            if isinstance(result, list):
                extracted_list = result
        except Exception as e:
            # Fallback: extract from error message if JSON parsing failed
            error_msg = str(e)
            if "Invalid json output:" in error_msg:
                try:
                    import ast
                    raw = error_msg.split("Invalid json output:")[1].strip()
                    extracted_list = list(ast.literal_eval(raw))
                except:
                    pass
            logger.warning(f"Entity extraction failed: {e}")
        
        # Classify extracted entities by type (simple keyword matching)
        typed_entities = {
            "soil_types": [],
            "nutrients": [],
            "seasons": [],
            "activities": [],
            "measurements": [],
            "other": []
        }
        
        for entity in extracted_list:
            entity_lower = entity.lower()
            
            # Classify by keywords
            if any(soil in entity_lower for soil in ['alluvial', 'acidic', 'acid sulfate', 'soil']):
                typed_entities["soil_types"].append(entity)
            elif any(nut in entity_lower for nut in ['nitrogen', 'phosphorus', 'potassium', 'lime', ' n ', ' p ', ' k ']):
                typed_entities["nutrients"].append(entity)
            elif any(season in entity_lower for season in ['winter', 'spring', 'summer', 'autumn']):
                typed_entities["seasons"].append(entity)
            elif any(act in entity_lower for act in ['fertilization', 'sowing', 'irrigation', 'harvest', 'land preparation']):
                typed_entities["activities"].append(entity)
            else:
                typed_entities["other"].append(entity)
        
        # Stage 2: Extract numeric values with units (regex fallback)
        import re
        
        # Detect công units (Vietnamese)
        cong_pattern = r'(\d+(?:\.\d+)?)\s*công'
        cong_matches = re.findall(cong_pattern, question, re.IGNORECASE)
        for match in cong_matches:
            value = float(match)
            converted_value, converted_unit = self._convert_units(value, 'công')
            typed_entities["measurements"].append({
                "value": value,
                "unit": "công",
                "converted_value": converted_value,
                "converted_unit": converted_unit
            })
        
        # Detect hectares
        ha_pattern = r'(\d+(?:\.\d+)?)\s*(?:ha|hectare)s?'
        ha_matches = re.findall(ha_pattern, question, re.IGNORECASE)
        for match in ha_matches:
            value = float(match)
            typed_entities["measurements"].append({
                "value": value,
                "unit": "hectare",
                "converted_value": value,
                "converted_unit": "hectare"
            })
        
        # Detect kg/ha rates
        rate_pattern = r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*kg/ha'
        rate_matches = re.findall(rate_pattern, question, re.IGNORECASE)
        for match in rate_matches:
            if match[1]:  # Range
                typed_entities["measurements"].append({
                    "min_value": float(match[0]),
                    "max_value": float(match[1]),
                    "unit": "kg/ha"
                })
            else:  # Single value
                typed_entities["measurements"].append({
                    "value": float(match[0]),
                    "unit": "kg/ha"
                })
        
        # Stage 3: Detect entity relationships
        relationships = []
        
        # Soil + Nutrient relationship
        if typed_entities["soil_types"] and typed_entities["nutrients"]:
            for soil in typed_entities["soil_types"]:
                for nutrient in typed_entities["nutrients"]:
                    relationships.append({
                        "type": "nutrient_requirement",
                        "entities": [soil, nutrient],
                        "description": f"{nutrient} requirement for {soil} soil"
                    })
        
        # Soil + Lime relationship
        if typed_entities["soil_types"] and "lime" in [n.lower() for n in typed_entities["nutrients"]]:
            for soil in typed_entities["soil_types"]:
                relationships.append({
                    "type": "lime_requirement",
                    "entities": [soil, "lime"],
                    "description": f"Lime requirement for {soil} soil"
                })
        
        # Activity + Season relationship
        if typed_entities["activities"] and typed_entities["seasons"]:
            for activity in typed_entities["activities"]:
                for season in typed_entities["seasons"]:
                    relationships.append({
                        "type": "seasonal_activity",
                        "entities": [activity, season],
                        "description": f"{activity} during {season}"
                    })
        
        # Build raw entity list for backward compatibility
        raw_entities = []
        for entity_type, entities in typed_entities.items():
            if entity_type != "measurements" and isinstance(entities, list):
                raw_entities.extend(entities)
        
        return {
            "raw_entities": raw_entities,
            "typed_entities": typed_entities,
            "relationships": relationships
        }
    
    def extract_entities(self, question: str) -> List[str]:
        """Extract entities - now uses multi-stage extraction"""
        result = self.extract_entities_multistage(question)
        return result["raw_entities"]
    
    def align_entities_to_kg(self, extracted_entities: List[str], min_sim: float = 0.6) -> List[Dict]:
        """
        Align extracted entities to KG entities using precomputed embeddings and vector index.
        
        Confidence Tiers:
        - High (>0.85): Direct confident match
        - Medium (0.6-0.85): Probable match, show alternatives
        - Low (<0.6): No good match found
        """
        if not extracted_entities:
            return []
        
        aligned = []
        
        # Translate entities to English for bilingual alignment (cached)
        english_entities = {}
        uncached = [e for e in extracted_entities if e not in AditiPipeline._translation_cache]
        if uncached:
            try:
                entities_text = ", ".join(uncached)
                translate_result = self.llm.invoke([HumanMessage(
                    content=f"Translate these Vietnamese agricultural terms to English. Return ONLY a JSON object mapping each term to its English translation. Terms: {entities_text}"
                )])
                import json as _json
                response = translate_result.content.strip()
                if response.startswith('```'):
                    response = response.split('\n', 1)[-1]
                    if response.endswith('```'):
                        response = response[:-3].strip()
                new_translations = _json.loads(response)
                AditiPipeline._translation_cache.update(new_translations)
                print(f"[ENTITY ALIGN] Translations (new): {new_translations}", flush=True)
            except Exception as e:
                print(f"[ENTITY ALIGN] Translation failed: {e}", flush=True)
        else:
            print(f"[ENTITY ALIGN] All translations served from cache", flush=True)
        english_entities = {e: AditiPipeline._translation_cache.get(e, e) for e in extracted_entities}
        
        # Date patterns to skip — dates don't have meaningful KG entities
        date_pattern = re.compile(
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december'
            r'|tháng\s*\d|ngày\s*\d|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})'
            r'|^\d{1,2}\s+\w+\s+\d{4}$',
            re.IGNORECASE
        )
        
        for raw_entity in extracted_entities:
            # Skip date-like entities
            if date_pattern.search(raw_entity):
                logger.info(f"Skipping date entity: '{raw_entity}'")
                continue
            
            # Try Vietnamese entity first
            scored_vi = self._align_entity_with_vector_index(raw_entity, top_k=5)
            
            # Also try English translation if available
            english_name = english_entities.get(raw_entity, "")
            scored_en = []
            if english_name and english_name.lower() != raw_entity.lower():
                scored_en = self._align_entity_with_vector_index(english_name, top_k=5)
                en_best_name = scored_en[0]['kg_name'] if scored_en else 'none'
                en_best_score = scored_en[0]['score'] if scored_en else 0
                print(f"[ENTITY ALIGN] '{raw_entity}' -> EN '{english_name}': best={en_best_name} ({en_best_score:.3f})", flush=True)
            
            # Pick the best match across both languages
            best_vi = scored_vi[0] if scored_vi else None
            best_en = scored_en[0] if scored_en else None
            
            if best_en and best_vi:
                if best_en["score"] > best_vi["score"]:
                    scored = scored_en
                    print(f"[ENTITY ALIGN] Using English match for '{raw_entity}': '{best_en['kg_name']}' ({best_en['score']:.3f}) > '{best_vi['kg_name']}' ({best_vi['score']:.3f})", flush=True)
                else:
                    scored = scored_vi
            elif best_en:
                scored = scored_en
            else:
                scored = scored_vi
            
            if not scored:
                logger.info(f"No alignment for '{raw_entity}' (no matches found)")
                continue
            
            best = scored[0]
            
            # Determine confidence tier (0.92 threshold for Vietnamese agricultural terms
            # which share vocabulary across topics, e.g. "tưới ngập" vs "xới ướt")
            if best["score"] >= 0.92:
                confidence = "high"
                tier_desc = "Direct match"
            elif best["score"] >= 0.65:
                confidence = "medium"
                tier_desc = "Probable match"
            else:
                confidence = "low"
                tier_desc = "No good match"
            
            # Only align if score meets minimum threshold
            if best["score"] >= min_sim:
                alignment = {
                    "raw": raw_entity,
                    "matched": best["kg_name"],
                    "name": best["kg_name"],
                    "labels": best.get("kg_labels", []),
                    "score": best["score"],
                    "confidence": confidence,
                    "tier_description": tier_desc
                }
                
                # For medium confidence, include alternatives
                if confidence == "medium" and len(scored) > 1:
                    alignment["alternatives"] = [
                        {"name": s["kg_name"], "score": s["score"]}
                        for s in scored[1:4]
                    ]
                
                aligned.append(alignment)
                logger.info(f"Aligned '{raw_entity}' -> '{best['kg_name']}' (score: {best['score']:.3f}, confidence: {confidence})")
            else:
                logger.info(f"No alignment for '{raw_entity}' (best score: {best['score']:.3f})")
        
        return aligned
    
    def _filter_relevant_facts(self, facts: List[Dict], aligned_entities: List[Dict], query: str) -> List[Dict]:
        """Filter facts to keep only those relevant to aligned entities or query intent."""
        if not facts:
            return []
        
        # Get aligned entity names (both high and medium confidence)
        aligned_names = set()
        for entity in aligned_entities:
            if entity.get("confidence") in ("high", "medium"):
                aligned_names.add(entity["name"].lower())
        
        # Extract query keywords for relevance check
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        filtered = []
        for fact in facts:
            source_lower = fact["source"].lower()
            target_lower = fact["target"].lower()
            
            # Keep if involves an aligned entity (or alias/substring match)
            if any(name in source_lower or source_lower in name 
                   for name in aligned_names if len(name) >= 3):
                filtered.append(fact)
                continue
            if any(name in target_lower or target_lower in name 
                   for name in aligned_names if len(name) >= 3):
                filtered.append(fact)
                continue
            
            # Keep if matches query keywords
            fact_text = f"{source_lower} {fact['relation']} {target_lower}".lower()
            if any(keyword in fact_text for keyword in query_keywords if len(keyword) > 3):
                filtered.append(fact)
                continue
        
        return filtered if filtered else facts
    
    def _rerank_facts(self, facts: List[Dict], query: str, top_k: int = 10) -> List[Dict]:
        """Rerank facts based on query relevance using English translation for matching."""
        if not facts:
            return []
        
        # Use English translation for keyword matching
        english_query = self._translate_to_english_if_needed(query)
        query_lower = english_query.lower()
        query_terms = set(query_lower.split())
        
        for fact in facts:
            source_lower = fact["source"].lower()
            target_lower = fact["target"].lower()
            fact_text = f"{source_lower} {fact['relation']} {target_lower}".lower()
            
            # Base: entity alignment score (already 0-1 from cosine similarity)
            base = fact["score"]
            
            # Relevance boost: word overlap between query and fact target/source
            target_words = set(target_lower.split())
            overlap = len(query_terms & target_words)
            
            # Also check substring containment for multi-word matches
            substr_boost = 0.0
            for term in query_terms:
                if len(term) > 2 and term in target_lower:
                    substr_boost += 0.05
            
            boost = overlap * 0.05 + substr_boost
            if fact.get("tier") == "core":
                boost += 0.02
            
            fact["rerank_score"] = base + boost
        
        # Sort by reranked score
        reranked = sorted(facts, key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        reranked = reranked[:top_k]
        
        # Display score: clamp to 0-1 range using entity alignment score + small relevance delta
        max_rerank = max(f["rerank_score"] for f in reranked) if reranked else 1.0
        for i, f in enumerate(reranked):
            # Rank-based decay: top fact gets entity score, others slightly less
            rank_decay = i * 0.02
            f["score"] = round(max(0.1, min(1.0, f["rerank_score"] / max_rerank - rank_decay)), 2)
        
        return reranked
    
    def traverse_kg_graph(self, aligned_entities: List[Dict], query: str = "", top_k: int = 10) -> List[Dict]:
        """
        Traverse KG to find related facts with relevance filtering and reranking.
        Returns top 6-10 most relevant facts instead of 20.
        """
        if not aligned_entities:
            return []
        
        facts = []
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for entity in aligned_entities:
                entity_name = entity.get("name") or entity.get("matched")
                if not entity_name:
                    continue
                
                # Build list of names to search: exact name + parenthetical acronyms
                # e.g., "Alternate Wetting and Drying (AWD)" -> also search "AWD"
                search_names = [entity_name]
                acronyms = re.findall(r'\(([A-Z]{2,})\)', entity_name)
                search_names.extend(acronyms)
                
                result = session.run("""
                    MATCH (n)-[r]-(m)
                    WHERE n.name IN $names
                    AND NOT m:Chunk AND NOT m:Section AND NOT m:Chapter AND NOT m:Document
                    RETURN n.name AS source, 
                           type(r) AS rel_type, 
                           r.type AS rel_detail,
                           m.name AS target, 
                           labels(m)[0] AS target_label,
                           elementId(m) AS target_id
                    LIMIT 15
                """, {"names": search_names})
                
                for record in result:
                    rel_type = record["rel_type"]
                    
                    # Weight relationships
                    weight_map = {
                        "REL": 1.0,
                        "MENTIONS": 0.3,
                        "CONTAINS": 0.5,
                        "PART_OF": 0.7
                    }
                    weight = weight_map.get(rel_type, 0.2)
                    
                    facts.append({
                        "source": record["source"],
                        "relation": record["rel_detail"] or rel_type,
                        "target": record["target"],
                        "target_label": record["target_label"],
                        "target_id": record["target_id"],
                        "tier": "core" if rel_type == "REL" else "context",
                        "score": entity.get("score", 1.0) * weight
                    })
        
        # Deduplicate
        unique_facts = {}
        for fact in facts:
            key = (fact["source"], fact["relation"], fact["target"])
            if key not in unique_facts or fact["score"] > unique_facts[key]["score"]:
                unique_facts[key] = fact
        
        all_facts = list(unique_facts.values())
        
        # Apply relevance filtering
        filtered_facts = self._filter_relevant_facts(all_facts, aligned_entities, query)
        logger.info(f"KG facts: {len(all_facts)} total → {len(filtered_facts)} after relevance filter")
        
        # Rerank and select top-k
        top_facts = self._rerank_facts(filtered_facts, query, top_k=top_k)
        logger.info(f"Selected top {len(top_facts)} most relevant facts")
        
        return top_facts
    
    def extract_query_intent(self, query: str) -> Dict:
        """Detect query type and extract parameters (bilingual: Vietnamese + English)"""
        query_lower = query.lower()
        
        intent = {
            'type': 'general',
            'soil_type': None,
            'season': None,
            'nutrient': None,
            'hectares': None
        }
        
        # Detect soil type (Vietnamese + English)
        if any(t in query_lower for t in ['alluvial', 'phù sa']):
            intent['soil_type'] = 'alluvial'
        elif any(t in query_lower for t in ['clay', 'đất sét', 'sét']):
            intent['soil_type'] = 'alluvial'  # clay == alluvial in Mekong Delta context
        elif any(t in query_lower for t in ['light acid', 'nhẹ', 'phèn nhẹ']):
            intent['soil_type'] = 'light acid sulfate'
        elif any(t in query_lower for t in ['medium acid', 'trung bình', 'phèn trung']):
            intent['soil_type'] = 'medium acid sulfate'
        elif any(t in query_lower for t in ['acidic', 'acid sulfate', 'phèn', 'chua']):
            intent['soil_type'] = 'light acid sulfate'  # default acidic → light
        
        # Detect season (Vietnamese + English + month-based inference)
        # Months 11,12,1,2,3 → Winter-Spring (Đông Xuân)
        # Months 4,5,6,7,8 → Summer-Autumn (Hè Thu)
        # Months 9,10 → Autumn-Winter (Thu Đông)
        WS_months = ['november', 'december', 'january', 'february', 'march',
                     'tháng 11', 'tháng 12', 'tháng 1', 'tháng 2', 'tháng 3',
                     'tháng một', 'tháng hai', 'tháng ba']
        SA_months = ['april', 'may', 'june', 'july', 'august',
                     'tháng 4', 'tháng 5', 'tháng 6', 'tháng 7', 'tháng 8',
                     'tháng tư', 'tháng năm', 'tháng sáu', 'tháng bảy', 'tháng tám']
        AW_months = ['september', 'october', 'tháng 9', 'tháng 10',
                     'tháng chín', 'tháng mười']
        if any(t in query_lower for t in ['winter', 'spring', 'đông xuân', 'đông-xuân']):
            intent['season'] = 'Winter-Spring'
        elif any(t in query_lower for t in ['summer', 'autumn', 'hè thu', 'hè-thu']):
            intent['season'] = 'Summer-Autumn'
        elif any(t in query_lower for t in ['thu đông', 'thu-đông']):
            intent['season'] = 'Autumn-Winter'
        elif any(t in query_lower for t in WS_months):
            intent['season'] = 'Winter-Spring'
        elif any(t in query_lower for t in SA_months):
            intent['season'] = 'Summer-Autumn'
        elif any(t in query_lower for t in AW_months):
            intent['season'] = 'Autumn-Winter'
        else:
            intent['season'] = 'Winter-Spring'  # default: most common season
        
        # Detect nutrient (Vietnamese + English)
        if any(t in query_lower for t in ['nitrogen', 'đạm', ' n ', 'n/ha']):
            intent['nutrient'] = 'nitrogen'
            intent['type'] = 'nutrient_query'
        elif any(t in query_lower for t in ['phosphorus', 'lân']):
            intent['nutrient'] = 'phosphorus'
            intent['type'] = 'nutrient_query'
        elif any(t in query_lower for t in ['potassium', 'kali']):
            intent['nutrient'] = 'potassium'
            intent['type'] = 'nutrient_query'
        elif any(t in query_lower for t in ['lime', 'vôi']):
            intent['type'] = 'lime_query'
        
        # General fertilizer question → nutrient_query, but ONLY if the question
        # focus is actually about fertilizer. We use compound keywords to avoid
        # false positives like "vùi phân" (fertilizer incorporation — sowing context)
        # or "phân hữu cơ" mentioned in passing.
        fertilizer_focus_keywords = [
            'bón phân', 'phân bón', 'lượng phân', 'loại phân', 'liều phân',
            'fertiliz', 'nutrient', 'dinh dưỡng', 'nên bón', 'cách bón',
            'bao nhiêu phân', 'phân gì'
        ]
        is_fertilizer_focus = any(t in query_lower for t in fertilizer_focus_keywords)
        
        if intent['type'] == 'general' and intent['soil_type'] and is_fertilizer_focus:
            intent['type'] = 'nutrient_query'
            intent['nutrient'] = 'nitrogen'  # default to N (primary nutrient)
        
        # Detect AWD / water management query
        if intent['type'] == 'general' and any(t in query_lower for t in [
            'awd', 'alternate wetting', 'wetting and drying', 'ướt khô xen kẽ',
            'tưới', 'rút nước', 'quản lý nước', 'water manag', 'irrigation',
            'drain', 'flood', 'ngập', 'mực nước'
        ]):
            intent['type'] = 'awd_query'
        
        # Detect seed rate / seed preparation query
        if intent['type'] == 'general' and any(t in query_lower for t in [
            'seed rate', 'seeding rate', 'lượng giống', 'gieo sạ', 'giống gieo',
            'kg/ha seed', 'how much seed', 'bao nhiêu giống', 'soak', 'ngâm',
            'germination', 'nảy mầm', 'ủ giống'
        ]):
            intent['type'] = 'seed_query'
        
        # Detect harvest / drying query
        if intent['type'] == 'general' and any(t in query_lower for t in [
            'harvest', 'thu hoạch', 'moisture', 'độ ẩm', 'drying', 'sấy',
            'ripeness', 'chín', 'storage', 'bảo quản'
        ]):
            intent['type'] = 'harvest_query'
        
        # Detect organic fertilizer query
        if intent['type'] == 'general' and any(t in query_lower for t in [
            'organic fertil', 'phân hữu cơ', 'compost', 'hữu cơ'
        ]):
            intent['type'] = 'organic_query'
        
        # Extract hectares / công
        ha_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ha|hectares?)', query_lower)
        if ha_match:
            intent['hectares'] = float(ha_match.group(1))
        cong_match = re.search(r'(\d+(?:\.\d+)?)\s*công', query_lower)
        if cong_match:
            intent['hectares'] = float(cong_match.group(1)) * 0.13
        
        return intent
    
    def structured_data_search(self, intent: Dict, language: str = 'vi') -> List[Dict]:
        """Query structured data nodes for factual lookups"""
        results = []
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            # Nutrient requirement query
            if intent['type'] == 'nutrient_query' and intent['soil_type']:
                season = intent.get('season') or 'Winter-Spring'
                result = session.run("""
                    MATCH (n:NutrientRequirement)
                    WHERE toLower(n.soil_type) CONTAINS $soil_type
                      AND toLower(n.season) CONTAINS toLower($season)
                    OPTIONAL MATCH (c:Chunk)-[:CONTAINS_DATA]->(n)
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
                            'season': record['season']
                        },
                        'chunk_id': record['chunk_id'],
                        'chunk_text': record['chunk_text'],
                        'chunk_title': record['chunk_title'],
                        'score': 1.0
                    })
            
            # Lime requirement query
            elif intent['type'] == 'lime_query':
                result = session.run("""
                    MATCH (l:LimeRequirement)
                    OPTIONAL MATCH (c:Chunk)-[:CONTAINS_DATA]->(l)
                    RETURN l.ph_range AS ph_range,
                           l.min_kg_ha AS min_kg_ha,
                           l.max_kg_ha AS max_kg_ha,
                           c.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN c.text_en ELSE c.text_vi END, c.text) AS chunk_text,
                           c.title AS chunk_title
                    ORDER BY l.min_kg_ha DESC
                    LIMIT 3
                """, lang=language)
                
                for record in result:
                    results.append({
                        'source': 'structured_data',
                        'type': 'lime_requirement',
                        'data': {
                            'ph_range': record['ph_range'],
                            'min_kg_ha': record['min_kg_ha'],
                            'max_kg_ha': record['max_kg_ha']
                        },
                        'chunk_id': record['chunk_id'],
                        'chunk_text': record['chunk_text'],
                        'chunk_title': record['chunk_title'],
                        'score': 1.0
                    })
            
            # AWD water management schedule query
            elif intent['type'] == 'awd_query':
                result = session.run("""
                    MATCH (a:AWDSchedule)
                    OPTIONAL MATCH (c:Chunk)-[:CONTAINS_DATA]->(a)
                    RETURN a.phase AS phase,
                           a.phase_name_en AS phase_name_en,
                           a.phase_name_vi AS phase_name_vi,
                           a.days_after_sowing AS days_after_sowing,
                           a.action_en AS action_en,
                           a.action_vi AS action_vi,
                           a.max_water_depth_cm AS max_water_depth_cm,
                           c.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN c.text_en ELSE c.text_vi END, c.text) AS chunk_text,
                           c.title AS chunk_title
                    ORDER BY a.phase
                """, lang=language)
                
                for record in result:
                    phase_name = record['phase_name_en'] if language == 'en' else record['phase_name_vi']
                    action = record['action_en'] if language == 'en' else record['action_vi']
                    results.append({
                        'source': 'structured_data',
                        'type': 'awd_schedule',
                        'data': {
                            'phase': record['phase'],
                            'phase_name': phase_name,
                            'days_after_sowing': record['days_after_sowing'],
                            'action': action,
                            'max_water_depth_cm': record['max_water_depth_cm']
                        },
                        'chunk_id': record['chunk_id'],
                        'chunk_text': record['chunk_text'],
                        'chunk_title': record['chunk_title'],
                        'score': 1.0
                    })
            
            # Seed rate query
            elif intent['type'] == 'seed_query':
                result = session.run("""
                    MATCH (sr:SeedRate)
                    OPTIONAL MATCH (c:Chunk)-[:CONTAINS_DATA]->(sr)
                    RETURN sr.max_kg_ha AS max_kg_ha,
                           sr.description_en AS desc_en,
                           sr.description_vi AS desc_vi,
                           sr.soak_hours AS soak_hours,
                           sr.incubation_hours AS incubation_hours,
                           c.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN c.text_en ELSE c.text_vi END, c.text) AS chunk_text,
                           c.title AS chunk_title
                """, lang=language)
                
                for record in result:
                    results.append({
                        'source': 'structured_data',
                        'type': 'seed_rate',
                        'data': {
                            'max_kg_ha': record['max_kg_ha'],
                            'description': record['desc_en'] if language == 'en' else record['desc_vi'],
                            'soak_hours': record['soak_hours'],
                            'incubation_hours': record['incubation_hours']
                        },
                        'chunk_id': record['chunk_id'],
                        'chunk_text': record['chunk_text'],
                        'chunk_title': record['chunk_title'],
                        'score': 1.0
                    })
            
            # Harvest / drying moisture query
            elif intent['type'] == 'harvest_query':
                result = session.run("""
                    MATCH (h:HarvestMoisture)
                    OPTIONAL MATCH (c:Chunk)-[:CONTAINS_DATA]->(h)
                    RETURN h.type AS type,
                           h.ripeness_percent AS ripeness,
                           h.target_moisture_percent AS moisture,
                           h.description_en AS desc_en,
                           h.description_vi AS desc_vi,
                           c.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN c.text_en ELSE c.text_vi END, c.text) AS chunk_text,
                           c.title AS chunk_title
                """, lang=language)
                
                for record in result:
                    results.append({
                        'source': 'structured_data',
                        'type': 'harvest_moisture',
                        'data': {
                            'harvest_type': record['type'],
                            'ripeness_percent': record['ripeness'],
                            'target_moisture_percent': record['moisture'],
                            'description': record['desc_en'] if language == 'en' else record['desc_vi']
                        },
                        'chunk_id': record['chunk_id'],
                        'chunk_text': record['chunk_text'],
                        'chunk_title': record['chunk_title'],
                        'score': 1.0
                    })
            
            # Organic fertilizer query
            elif intent['type'] == 'organic_query':
                result = session.run("""
                    MATCH (o:OrganicFertilizer)
                    OPTIONAL MATCH (c:Chunk)-[:CONTAINS_DATA]->(o)
                    RETURN o.min_tons_ha AS min_tons,
                           o.max_tons_ha AS max_tons,
                           o.description_en AS desc_en,
                           o.description_vi AS desc_vi,
                           c.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN c.text_en ELSE c.text_vi END, c.text) AS chunk_text,
                           c.title AS chunk_title
                """, lang=language)
                
                for record in result:
                    results.append({
                        'source': 'structured_data',
                        'type': 'organic_fertilizer',
                        'data': {
                            'min_tons_ha': record['min_tons'],
                            'max_tons_ha': record['max_tons'],
                            'description': record['desc_en'] if language == 'en' else record['desc_vi']
                        },
                        'chunk_id': record['chunk_id'],
                        'chunk_text': record['chunk_text'],
                        'chunk_title': record['chunk_title'],
                        'score': 1.0
                    })
        
        return results
    
    def bm25_search(self, query: str, top_k: int = 10, original_query: str = None, query_topic: str = None, language: str = 'vi') -> List[Dict]:
        """Full-text search using BM25 with LLM-based relevance filtering.
        query: search query (English for BM25 index matching)
        original_query: user's original query for LLM relevance scoring (may be Vietnamese)
        """
        sanitized = re.sub(r'[\/\(\)\[\]\{\}\^\~\*\?\:\"\'\']', ' ', query)
        # Lower threshold to 3 for Vietnamese (many words like 'khô', 'xen', 'tác', 'lúa' are 3 chars)
        keywords = [w for w in sanitized.split() if len(w) >= 3]
        
        # Topic-based query expansion: add domain-specific keywords
        if query_topic and query_topic != 'general':
            expansion_terms = {
                'fertilizer_management': ['fertilizer', 'nitrogen', 'phosphorus', 'potassium', 'NPK', 'urea', 'lime', 'organic', 'phân', 'đạm', 'lân', 'kali'],
                'water_management': ['water', 'irrigation', 'AWD', 'flooding', 'drainage', 'nước', 'tưới'],
                'pest_management': ['pest', 'disease', 'insect', 'weed', 'IPM', 'sâu', 'bệnh', 'cỏ'],
                'seed_preparation': ['seed', 'variety', 'germination', 'sowing', 'giống', 'gieo'],
                'land_preparation': ['soil', 'tillage', 'plowing', 'leveling', 'đất', 'cày'],
                'harvest': ['harvest', 'drying', 'storage', 'moisture', 'thu hoạch', 'sấy'],
                'straw_management': ['straw', 'residue', 'compost', 'rơm', 'rạ']
            }
            topic_keywords = expansion_terms.get(query_topic, [])
            # Add 3-5 most relevant topic keywords to boost recall
            keywords.extend(topic_keywords[:5])
        
        search_query = ' '.join(keywords[:15])  # Increased from 10 to 15 to accommodate expansion
        
        logger.info(f"BM25 search: topic={query_topic}, keywords={keywords[:15]}, final_query='{search_query}' (from: '{query}')")
        
        if not search_query:
            logger.warning("BM25 search query is empty after sanitization")
            return []
        
        # Build topic filter for BM25
        topic_filter_clause = ""
        params = {'search_query': search_query, 'top_k': top_k * 2}
        
        if query_topic and query_topic != 'general':
            topic_map = {
                'direct_seeding': 'seed_preparation',
                'organic_fertilizer': 'fertilizer_management',
                'straw_use': 'straw_management'
            }
            canonical_topic = topic_map.get(query_topic, query_topic)
            allowed_topics = list(set([query_topic, canonical_topic]))
            
            if query_topic == 'fertilizer_management':
                allowed_topics.append('organic_fertilizer')
            elif query_topic == 'water_management':
                allowed_topics.append('land_preparation')
            elif query_topic == 'seed_preparation':
                allowed_topics.append('direct_seeding')
            
            topic_filter_clause = " WHERE node.topic_tag IN $allowed_topics"
            params['allowed_topics'] = allowed_topics
            logger.info(f"BM25 search: filtering by topics {allowed_topics}")
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            try:
                # Get more results for filtering
                cypher_query = f"""
                    CALL db.index.fulltext.queryNodes('chunk_text_index', $search_query)
                    YIELD node, score
                    {topic_filter_clause}
                    OPTIONAL MATCH (node)-[:PART_OF]->(s:Section)
                    OPTIONAL MATCH (s)-[:IN_CHAPTER]->(ch:Chapter)
                    RETURN node.chunk_id AS chunk_id,
                           COALESCE(CASE WHEN $lang = 'en' THEN node.text_en ELSE node.text_vi END, node.text) AS text,
                           node.title AS title,
                           node.topic_tag AS topic_tag,
                           s.title AS section_title,
                           s.section_num AS section_num,
                           s.chapter_num AS chapter_num,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """
                params['lang'] = language
                result = session.run(cypher_query, **params)
                
                all_results = [{
                    'source': 'bm25',
                    'chunk_id': r['chunk_id'],
                    'text': r['text'],
                    'title': r['title'],
                    'topic_tag': r['topic_tag'] or 'general',
                    'section_title': r['section_title'] or r['title'],
                    'section_num': r['section_num'] or '',
                    'chapter_num': r['chapter_num'] or '',
                    'score': r['score']
                } for r in result]
                
                print(f"[BM25] Raw results from Neo4j: {len(all_results)} chunks", flush=True)
                for i, r in enumerate(all_results):
                    print(f"  [BM25 raw {i}] score={r['score']:.2f} title='{r.get('title', '')[:60]}'", flush=True)
                
                # Use LLM to score relevance using original query (may be Vietnamese)
                rerank_query = original_query or query
                all_results = self._llm_relevance_rerank(all_results, rerank_query)
                
                # Filter: keep only results where LLM scored >= 3/10 (moderately relevant)
                # If LLM scoring failed (llm_relevance == -1), keep the result as fallback
                filtered_results = []
                for r in all_results:
                    rel = r.get('llm_relevance', -1)
                    if rel == -1:  # LLM scoring failed, keep as fallback
                        filtered_results.append(r)
                        print(f"  [BM25 KEPT fallback] llm_relevance=-1 title='{r.get('title', '')[:60]}'", flush=True)
                    elif rel >= 3:  # LLM scored as relevant (3+ on 0-10 scale)
                        filtered_results.append(r)
                        print(f"  [BM25 KEPT] llm_relevance={rel}/10 title='{r.get('title', '')[:60]}'", flush=True)
                    else:
                        print(f"  [BM25 FILTERED] llm_relevance={rel}/10 title='{r.get('title', '')[:60]}'", flush=True)
                
                print(f"[BM25] After LLM filter: {len(all_results)} → {len(filtered_results)}", flush=True)
                
                # Sort by boosted score and return top-k
                filtered_results.sort(key=lambda x: x['score'], reverse=True)
                return filtered_results[:top_k]
            except Exception as e:
                logger.warning(f"BM25 search error: {e}")
                return []
    
    def _llm_relevance_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Use LLM to score chunk relevance to query. Works for ANY topic in ANY language.
        Replaces hardcoded topic taxonomy with universal relevance scoring.
        Returns results with updated scores based on LLM relevance judgment.
        """
        if not results:
            return results
        
        num_docs = len(results)
        
        # Build compact document list for LLM (title + first 150 chars of text)
        doc_summaries = []
        for i, r in enumerate(results):
            title = r.get('title', 'Untitled') or 'Untitled'
            text_preview = (r.get('text', '') or '')[:150].replace('\n', ' ')
            doc_summaries.append(f"[{i}] {title}: {text_preview}")
        
        docs_text = '\n'.join(doc_summaries)
        
        try:
            relevance_prompt = f"""Rate how relevant each document is to the query. Score 0-10.
0 = completely unrelated topic
5 = somewhat related but not directly answering
10 = directly about the query topic

Query: {query}

There are exactly {num_docs} documents. Return exactly {num_docs} scores.
Documents:
{docs_text}

Return ONLY a JSON array of {num_docs} integers. Example for 4 docs: [8, 2, 5, 1]"""
            
            result = self.llm.invoke([HumanMessage(content=relevance_prompt)])
            response = result.content.strip()
            print(f"[LLM RELEVANCE] Raw response: {response[:300]}", flush=True)
            
            # Parse scores - handle markdown code blocks
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('\n', 1)[-1]  # Remove first line
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3].strip()
            scores = json.loads(clean_response)
            print(f"[LLM RELEVANCE] Parsed {len(scores)} scores: {scores}", flush=True)
            
            if isinstance(scores, list) and len(scores) >= 1:
                # Handle count mismatch: pad with 5 (neutral) or truncate
                if len(scores) < num_docs:
                    logger.info(f"LLM returned {len(scores)} scores for {num_docs} docs, padding with 5")
                    scores.extend([5] * (num_docs - len(scores)))
                scores = scores[:num_docs]
                
                for i, r in enumerate(results):
                    r['original_bm25_score'] = r['score']
                    relevance = max(0, min(10, scores[i]))
                    # Convert 0-10 score to multiplier: 0→0.05, 5→0.5, 10→2.0
                    if relevance >= 7:
                        boost = 1.0 + (relevance - 7) * 0.33  # 7→1.0, 10→2.0
                    elif relevance >= 4:
                        boost = 0.3 + (relevance - 4) * 0.23  # 4→0.3, 7→1.0
                    else:
                        boost = max(0.05, relevance * 0.075)   # 0→0.05, 4→0.3
                    
                    r['score'] = r['score'] * boost
                    r['context_boost'] = boost
                    r['llm_relevance'] = relevance
                    logger.info(f"LLM relevance: [{i}] rel={relevance}/10 boost={boost:.2f}x title='{r.get('title', '')[:50]}'")
            else:
                logger.warning(f"LLM relevance returned non-list: {response[:100]}")
                for r in results:
                    r['original_bm25_score'] = r['score']
                    r['context_boost'] = 1.0
                    r['llm_relevance'] = -1
                    
        except Exception as e:
            logger.warning(f"LLM relevance scoring failed: {e}")
            for r in results:
                r['original_bm25_score'] = r['score']
                r['context_boost'] = 1.0
                r['llm_relevance'] = -1
        
        return results
    
    def _generate_hyde_answer(self, query: str) -> str:
        """Generate hypothetical answer for better retrieval (HyDE technique)."""
        try:
            hyde_prompt = f"""Write a concise 2-3 sentence answer to this question about rice farming. Be specific and technical:

Question: {query}

Answer:"""
            result = self.llm.invoke([HumanMessage(content=hyde_prompt)])
            hyde_answer = result.content.strip()
            logger.info(f"HyDE answer generated: {hyde_answer[:100]}...")
            return hyde_answer
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query
    
    @lru_cache(maxsize=512)
    def _embed_query_cached(self, query_text: str) -> tuple:
        """Cache query embeddings (returns tuple for hashability)."""
        emb = self.embedding_model.encode(query_text, convert_to_numpy=True)
        return tuple(emb.tolist())
    
    def _filter_generic_chunks(self, results: List[Dict], min_results: int = 3) -> List[Dict]:
        """Filter out generic overview/summary chunks, but preserve minimum results."""
        generic_keywords = ['overview', 'introduction', 'summary', 'general']
        
        filtered = []
        generic_chunks = []
        
        for r in results:
            title = (r.get('title') or '').lower()
            # Separate generic from specific chunks
            if any(kw in title for kw in generic_keywords):
                generic_chunks.append(r)
                logger.info(f"PRE-FILTERED generic chunk: {r.get('title')}")
            else:
                filtered.append(r)
        
        # Safety check: if filtering removed too many results, add back some generic ones
        if len(filtered) < min_results and generic_chunks:
            needed = min(min_results - len(filtered), len(generic_chunks))
            filtered.extend(generic_chunks[:needed])
            logger.info(f"Added back {needed} generic chunks to meet minimum threshold")
        
        return filtered if filtered else results[:min_results]  # Ultimate fallback
    
    def vector_search(self, query: str, top_k: int = 10, use_hyde: bool = True, query_topic: str = None, language: str = 'vi') -> List[Dict]:
        """Vector similarity search using vector index with optional HyDE."""
        # Use HyDE: generate hypothetical answer and search for that instead
        search_query = self._generate_hyde_answer(query) if use_hyde else query
        # Use cached embedding
        query_embedding = list(self._embed_query_cached(search_query))
        
        # Build topic filter clause if topic is specified
        topic_filter = ""
        if query_topic and query_topic != 'general':
            # Map sub-topics to canonical topics
            topic_map = {
                'direct_seeding': 'seed_preparation',
                'organic_fertilizer': 'fertilizer_management',
                'straw_use': 'straw_management'
            }
            canonical_topic = topic_map.get(query_topic, query_topic)
            # Allow both the query topic and related topics
            allowed_topics = [query_topic, canonical_topic]
            if query_topic == 'fertilizer_management':
                allowed_topics.extend(['organic_fertilizer'])
            elif query_topic == 'water_management':
                allowed_topics.extend(['land_preparation'])  # Land prep affects water management
            elif query_topic == 'seed_preparation':
                allowed_topics.extend(['direct_seeding'])
            
            topic_filter = f" AND c.topic_tag IN {allowed_topics}"
            logger.info(f"Vector search: filtering by topics {allowed_topics}")
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            try:
                # Build Cypher query with optional topic filter
                if topic_filter:
                    cypher_query = f"""
                        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                        YIELD node, score
                        WHERE node.topic_tag IN $allowed_topics
                        OPTIONAL MATCH (node)-[:PART_OF]->(s:Section)
                        OPTIONAL MATCH (s)-[:IN_CHAPTER]->(ch:Chapter)
                        RETURN node.chunk_id AS chunk_id,
                               COALESCE(CASE WHEN $lang = 'en' THEN node.text_en ELSE node.text_vi END, node.text) AS text,
                               node.title AS title,
                               node.topic_tag AS topic_tag,
                               s.title AS section_title,
                               s.section_num AS section_num,
                               s.chapter_num AS chapter_num,
                               score
                        ORDER BY score DESC
                    """
                    result = session.run(cypher_query, index_name=config.VECTOR_INDEX_NAME, 
                                       query_embedding=query_embedding, top_k=top_k * 3,
                                       allowed_topics=allowed_topics, lang=language)
                else:
                    # No topic filter
                    result = session.run("""
                        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                        YIELD node, score
                        OPTIONAL MATCH (node)-[:PART_OF]->(s:Section)
                        OPTIONAL MATCH (s)-[:IN_CHAPTER]->(ch:Chapter)
                        RETURN node.chunk_id AS chunk_id,
                               COALESCE(CASE WHEN $lang = 'en' THEN node.text_en ELSE node.text_vi END, node.text) AS text,
                               node.title AS title,
                               node.topic_tag AS topic_tag,
                               s.title AS section_title,
                               s.section_num AS section_num,
                               s.chapter_num AS chapter_num,
                               score
                        ORDER BY score DESC
                    """, index_name=config.VECTOR_INDEX_NAME, query_embedding=query_embedding, top_k=top_k * 2, lang=language)
                
                all_results = [{
                    'source': 'vector',
                    'chunk_id': r['chunk_id'],
                    'text': r['text'],
                    'title': r['title'],
                    'topic_tag': r['topic_tag'] or 'general',
                    'section_title': r['section_title'] or r['title'],
                    'section_num': r['section_num'] or '',
                    'chapter_num': r['chapter_num'] or '',
                    'score': r['score']
                } for r in result]
                
                # Apply aggressive pre-filtering
                filtered_results = self._filter_generic_chunks(all_results)
                
                return filtered_results[:top_k]
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                return []
    
    
    def extract_structured_data_from_text(self, text: str, intent: Dict) -> Dict:
        """Extract structured nutrient/lime data from chunk text"""
        import re
        
        structured_data = {}
        
        if intent['type'] == 'nutrient_query' and intent['soil_type']:
            # Look for patterns like "Alluvial soil: 90-100 kg/ha of Nitrogen"
            pattern = rf"{intent['soil_type']}[^:]*:\s*(\d+)[-–](\d+)\s*kg/ha.*[Nn]itrogen.*\(N\)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                structured_data = {
                    'soil_type': intent['soil_type'],
                    'nutrient': 'Nitrogen (N)',
                    'min_kg_ha': int(match.group(1)),
                    'max_kg_ha': int(match.group(2)),
                    'season': intent.get('season', 'Winter-Spring')
                }
        
        return structured_data
    
    def generate_answer(self, question: str, retrieved_chunks: List[Dict], measurements: List[Dict] = None) -> tuple:
        """
        Generate a farmer-friendly answer from retrieved chunks.
        
        Args:
            question: The user's question
            retrieved_chunks: Retrieved context chunks
            measurements: Pre-computed unit conversions from entity extraction
        
        Returns:
            Tuple of (answer_string, llm_context_dict)
        """
        # Format context from retrieved chunks with source metadata
        context_text = ""
        
        if retrieved_chunks:
            for i, chunk in enumerate(retrieved_chunks, 1):
                text = chunk.get('text', '')
                chapter = chunk.get('chapter_num', '')
                section = chunk.get('section_num', '')
                section_title = chunk.get('section_title', chunk.get('title', ''))
                
                # Build source reference
                source_ref = ""
                if chapter and section:
                    source_ref = f"[Chapter {chapter}, Section {section}: {section_title}]"
                elif section:
                    source_ref = f"[Section {section}: {section_title}]"
                elif section_title:
                    source_ref = f"[{section_title}]"
                
                if source_ref:
                    context_text += f"Source {i} {source_ref}:\n{text}\n\n"
                else:
                    context_text += f"Source {i}:\n{text}\n\n"
        
        # Inject structured data (NutrientRequirement, LimeRequirement) as authoritative facts
        structured_lines = []
        for chunk in (retrieved_chunks or []):
            struct_data = chunk.get('structured_data')
            if not struct_data:
                continue
            data_list = struct_data if isinstance(struct_data, list) else [struct_data]
            for sd in data_list:
                if not isinstance(sd, dict):
                    continue
                soil = sd.get('soil_type', '')
                nutrient = sd.get('nutrient', '')
                min_val = sd.get('min_kg_ha', '')
                max_val = sd.get('max_kg_ha', '')
                season = sd.get('season', '')
                if min_val and max_val and nutrient:
                    structured_lines.append(f"  {nutrient}: {min_val}–{max_val} kg/ha (Soil: {soil}, Season: {season})")
        if structured_lines:
            context_text += (
                "⚠️ AUTHORITATIVE STRUCTURED DATA — use these EXACT values for the farmer's soil type. "
                "Do NOT pick different rows from tables:\n"
                + "\n".join(structured_lines) + "\n\n"
            )
        
        # Inject pre-computed measurements into context
        if measurements:
            conversion_lines = []
            for m in measurements:
                if 'converted_value' in m and m.get('unit') != m.get('converted_unit'):
                    conversion_lines.append(
                        f"- {m['value']} {m['unit']} = {m['converted_value']:.4g} {m['converted_unit']}"
                    )
            if conversion_lines:
                context_text += "Pre-computed Unit Conversions (USE THESE, do NOT guess):\n"
                context_text += "\n".join(conversion_lines) + "\n\n"
        
        system_prompt = """You are a friendly rice farming advisor. Answer using ONLY the context provided.
- Use simple language and be direct
- Give specific numbers when available
- Cite the source (chapter and section) when referencing specific information
- If you don't know, say so honestly
- SOIL TYPES: The handbook only classifies soils as: alluvial (đất phù sa), light acid sulfate (đất phèn nhẹ), or medium acid sulfate (đất phèn trung bình). If the user mentions a soil type not in this list (e.g. 'clay soil', 'sandy soil'), do NOT silently assume which type it is. Instead, state explicitly: "The handbook classifies soils as alluvial, light acid sulfate, or medium acid sulfate. Clay soil in the Mekong Delta is often alluvial, but you should verify your soil type before applying these rates."
- ORGANIC FERTILIZER + INORGANIC NPK: These are SEPARATE recommendations used TOGETHER. Organic fertilizer (1.5–3 tons/ha) is a base soil amendment applied before sowing. The inorganic NPK rates (N, P₂O₅, K₂O kg/ha) are the required topdressing schedule. Do NOT reduce the inorganic NPK rates because organic fertilizer is applied — the handbook specifies both to be used together, and the NPK rates already account for this combined approach.
- UNIT CONVERSIONS (MANDATORY — use these exact values):
  • 1 công = 1,300 m² = 0.13 hectare (NOT 0.1 ha)
  • 1 hectare = 10,000 m²
  If the question uses "công", ALWAYS convert to hectares first using 1 công = 0.13 ha.
- MATH: For ANY calculations, you MUST:
  1. Show step-by-step work with each arithmetic step explicit
  2. ALWAYS complete the FULL calculation to the FINAL answer the question asks for
     — Do NOT stop at per-hectare values if the question asks about a specific total area
     — Example: If asked about 2 ha and rate is 90 kg/ha → must compute 90 × 2 = 180 kg
  3. RANGES — CRITICAL: When the handbook gives a RANGE for fertilizer rates (e.g., N: 90–100 kg/ha, P₂O₅: 30–40 kg/ha, organic: 1.5–3.0 t/ha), you MUST show the FULL RANGE in your answer, NOT just a single middle value. Calculate both the minimum and maximum amounts for the user's field size, then apply split percentages to BOTH the min and max values. Present final amounts as ranges in tables.
  4. Double-check your multiplication and division before presenting the final number
  5. FERTILIZER SPLIT PERCENTAGES — CRITICAL RULE: When breaking down fertilizer applications by stage (e.g., Lần 1: 40% N, Lần 2: 40% N + 40% K₂O, Lần 3: 20% N + 60% K₂O), ALWAYS apply the ORIGINAL PERCENTAGE (40%, 60%, etc.) to the TOTAL amount for the user's field. NEVER use a per-hectare quantity as if it were a percentage. The percentages (40%, 40%, 20% for N; 40%, 60% for K₂O) are the SPLIT ratios, not the per-hectare amounts."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """Question: {question}

Context:
{context}""")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({"question": question, "context": context_text})
        answer = result.content
        
        # Verify math calculations in the answer using Python eval
        answer = self._verify_math_in_answer(answer, question)
        
        # Capture the exact context sent to the LLM
        llm_context = {
            "system_prompt": system_prompt,
            "question": question,
            "full_context": context_text
        }
        
        return answer, llm_context
    
    def _verify_math_in_answer(self, answer: str, question: str) -> str:
        """
        Detect arithmetic expressions in the LLM answer, verify them with Python eval,
        and ask the LLM to correct any errors found.
        """
        import re
        
        # Pattern: number × number = number (various formats)
        math_patterns = [
            # "0.2 × 70 = 1.4" or "0.2 * 70 = 1.4" or "0.2 x 70 = 1.4"
            r'([\d.,]+)\s*[×x\*]\s*([\d.,]+)\s*=\s*([\d.,]+)',
            # "70 ÷ 5 = 14" or "70 / 5 = 14"
            r'([\d.,]+)\s*[÷/]\s*([\d.,]+)\s*=\s*([\d.,]+)',
            # "70 + 30 = 100"
            r'([\d.,]+)\s*\+\s*([\d.,]+)\s*=\s*([\d.,]+)',
            # "70 - 30 = 40" or "70 – 30 = 40"
            r'([\d.,]+)\s*[–\-]\s*([\d.,]+)\s*=\s*([\d.,]+)',
        ]
        operators = ['*', '/', '+', '-']
        
        errors_found = []
        
        for pattern, op in zip(math_patterns, operators):
            for match in re.finditer(pattern, answer):
                try:
                    # Parse numbers (handle comma as thousand separator)
                    a_str = match.group(1).replace(',', '')
                    b_str = match.group(2).replace(',', '')
                    claimed_str = match.group(3).replace(',', '')
                    
                    a = float(a_str)
                    b = float(b_str)
                    claimed = float(claimed_str)
                    
                    # Compute correct result
                    if op == '*':
                        correct = a * b
                    elif op == '/':
                        correct = a / b if b != 0 else float('inf')
                    elif op == '+':
                        correct = a + b
                    else:
                        correct = a - b
                    
                    # Check if the claimed result is wrong (allow 1% tolerance for rounding)
                    if correct != 0 and abs(claimed - correct) / abs(correct) > 0.01:
                        errors_found.append({
                            'expression': match.group(0),
                            'a': a, 'b': b, 'op': op,
                            'claimed': claimed,
                            'correct': correct
                        })
                        logger.warning(f"Math error detected: {match.group(0)} → correct answer is {correct}")
                except (ValueError, ZeroDivisionError):
                    continue
        
        if not errors_found:
            return answer
        
        # Build correction prompt
        error_descriptions = []
        for err in errors_found:
            op_symbol = {'*': '×', '/': '÷', '+': '+', '-': '-'}[err['op']]
            # Format correct result: use int if it's a whole number
            correct_val = int(err['correct']) if err['correct'] == int(err['correct']) else err['correct']
            error_descriptions.append(
                f"- \"{err['expression']}\" is WRONG. {err['a']} {op_symbol} {err['b']} = {correct_val} (not {err['claimed']})"
            )
        
        correction_prompt = f"""The following answer contains arithmetic errors. Fix ONLY the wrong numbers, keep everything else exactly the same.

Errors found (verified by calculator):
{chr(10).join(error_descriptions)}

Original answer:
{answer}

Return the corrected answer with the right numbers. Do not add any explanation about the corrections."""
        
        try:
            corrected = self.llm.invoke([HumanMessage(content=correction_prompt)])
            logger.info(f"Math verification: corrected {len(errors_found)} error(s)")
            return corrected.content
        except Exception as e:
            logger.warning(f"Math correction failed: {e}, returning original answer")
            return answer
    
    def _translate_to_english_if_needed(self, query: str) -> str:
        """Translate Vietnamese query to English for document retrieval."""
        # Simple language detection - if contains Vietnamese characters
        vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ')
        if any(c in vietnamese_chars for c in query.lower()):
            try:
                # Use LLM to translate
                translation_prompt = f"Translate this Vietnamese question to English. Only output the English translation, nothing else:\n\n{query}"
                result = self.llm.invoke([HumanMessage(content=translation_prompt)])
                english_query = result.content.strip()
                logger.info(f"Translated Vietnamese query to English: '{query}' -> '{english_query}'")
                return english_query
            except Exception as e:
                logger.warning(f"Translation failed: {e}, using original query")
                return query
        return query
    
    def triple_hybrid_retrieve(self, query: str, aligned_entities: List[Dict] = None, language: str = 'vi',
                                 english_query: str = None, query_topic: str = None) -> List[Dict]:
        """
        Triple-hybrid retrieval combining structured data, BM25, and vector search.
        Weights: 40% structured, 15% BM25, 45% vector
        
        OPTIMIZED: Runs all 3 retrievals in parallel using ThreadPoolExecutor
        Pass english_query and query_topic to avoid redundant LLM calls.
        """
        intent = self.extract_query_intent(query)
        
        # Use pre-computed values if available, otherwise compute
        if english_query is None:
            english_query = self._translate_to_english_if_needed(query)
        if query_topic is None:
            query_topic = self._detect_query_topic(english_query)
        logger.info(f"Detected query topic: {query_topic}")
        
        # PARALLEL RETRIEVAL: Run all 3 searches concurrently
        structured_results = []
        bm25_results = []
        vector_results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all 3 tasks
            future_structured = executor.submit(self.structured_data_search, intent, language)
            future_bm25 = executor.submit(self.bm25_search, english_query, 10, query, query_topic, language)
            future_vector = executor.submit(self.vector_search, english_query, 10, True, query_topic, language)
            
            # Collect results as they complete
            for future in as_completed([future_structured, future_bm25, future_vector]):
                try:
                    if future == future_structured:
                        structured_results = future.result()
                        logger.info(f"[PARALLEL] Structured search completed: {len(structured_results)} results")
                    elif future == future_bm25:
                        bm25_results = future.result()
                        logger.info(f"[PARALLEL] BM25 search completed: {len(bm25_results)} results")
                    elif future == future_vector:
                        vector_results = future.result()
                        logger.info(f"[PARALLEL] Vector search completed: {len(vector_results)} results")
                except Exception as e:
                    logger.error(f"[PARALLEL] Retrieval task failed: {e}")
        
        # Merge and rank results
        chunk_scores = {}
        
        # Add structured results (highest weight)
        # Multiple nutrient results may share the same chunk_id — accumulate all structured data
        for result in structured_results:
            chunk_id = result['chunk_id']
            if chunk_id in chunk_scores:
                # Append additional structured data to existing chunk
                existing = chunk_scores[chunk_id].get('structured_data')
                new_data = result.get('data')
                if existing and new_data:
                    if isinstance(existing, list):
                        existing.append(new_data)
                    else:
                        chunk_scores[chunk_id]['structured_data'] = [existing, new_data]
                chunk_scores[chunk_id]['structured_score'] = 1.0
            else:
                chunk_scores[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': result['chunk_text'],
                    'title': result['chunk_title'],
                    'section_title': result.get('section_title', ''),
                    'section_num': result.get('section_num', ''),
                    'chapter_num': result.get('chapter_num', ''),
                    'structured_score': 1.0,
                    'bm25_score': 0,
                    'vector_score': 0,
                    'structured_data': result.get('data')
                }
        
        # Add BM25 results
        for result in bm25_results:
            chunk_id = result['chunk_id']
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]['bm25_score'] = result['score']
                # Update metadata if not already set
                if not chunk_scores[chunk_id].get('section_num'):
                    chunk_scores[chunk_id]['section_title'] = result.get('section_title', '')
                    chunk_scores[chunk_id]['section_num'] = result.get('section_num', '')
                    chunk_scores[chunk_id]['chapter_num'] = result.get('chapter_num', '')
            else:
                chunk_scores[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': result['text'],
                    'title': result['title'],
                    'topic_tag': result.get('topic_tag', 'general'),
                    'section_title': result.get('section_title', ''),
                    'section_num': result.get('section_num', ''),
                    'chapter_num': result.get('chapter_num', ''),
                    'structured_score': 0,
                    'bm25_score': result['score'],
                    'vector_score': 0
                }
        
        # Add vector results with topic-aware pre-boost
        # This ensures topic-matching chunks from vector search compete well with BM25
        for result in vector_results:
            chunk_id = result['chunk_id']
            chunk_topic = result.get('topic_tag', 'general')
            
            # Pre-boost vector score if topic matches query topic
            vector_score = result['score']
            if query_topic != 'general' and chunk_topic == query_topic:
                vector_score = min(vector_score * 1.15, 1.0)  # 15% boost, capped at 1.0
                logger.info(f"Pre-boosted vector score for topic-matching chunk: {result['title'][:50]} ({chunk_topic})")
            
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]['vector_score'] = vector_score
                if not chunk_scores[chunk_id].get('topic_tag'):
                    chunk_scores[chunk_id]['topic_tag'] = chunk_topic
                # Update metadata if not already set
                if not chunk_scores[chunk_id].get('section_num'):
                    chunk_scores[chunk_id]['section_title'] = result.get('section_title', '')
                    chunk_scores[chunk_id]['section_num'] = result.get('section_num', '')
                    chunk_scores[chunk_id]['chapter_num'] = result.get('chapter_num', '')
            else:
                chunk_scores[chunk_id] = {
                    'chunk_id': chunk_id,
                    'text': result['text'],
                    'title': result['title'],
                    'topic_tag': chunk_topic,
                    'section_title': result.get('section_title', ''),
                    'section_num': result.get('section_num', ''),
                    'chapter_num': result.get('chapter_num', ''),
                    'structured_score': 0,
                    'bm25_score': 0,
                    'vector_score': vector_score
                }
        
        # Calculate hybrid score with intelligent reranking
        for chunk_id, scores in chunk_scores.items():
            bm25_normalized = min(scores['bm25_score'] / 5.0, 1.0)
            base_score = (
                0.4 * scores['structured_score'] +
                0.15 * bm25_normalized +
                0.45 * scores['vector_score']
            )
            
            # Apply reranking boosts/penalties (including topic-aware boost)
            rerank_multiplier = self._calculate_rerank_multiplier(
                scores.get('title', ''),
                scores.get('parent_title', ''),
                query,
                chunk_topic=scores.get('topic_tag', 'general'),
                query_topic=query_topic
            )
            
            scores['hybrid_score'] = base_score * rerank_multiplier
            scores['rerank_multiplier'] = rerank_multiplier
        
        # Sort and return top-k
        ranked = sorted(chunk_scores.values(), key=lambda x: x['hybrid_score'], reverse=True)
        
        # HARD FILTER: Remove off-topic chunks entirely (not just penalize)
        off_topic_map = {
            'water_management':      {'fertilizer_management', 'pest_management', 'straw_management', 'straw_use', 'organic_fertilizer', 'direct_seeding'},
            'fertilizer_management': {'water_management', 'pest_management', 'straw_management', 'straw_use', 'seed_preparation', 'land_preparation', 'harvest', 'direct_seeding'},
            'pest_management':       {'water_management', 'fertilizer_management', 'straw_management', 'straw_use', 'organic_fertilizer', 'direct_seeding'},
            'seed_preparation':      {'straw_management', 'straw_use', 'organic_fertilizer', 'harvest'},
            'land_preparation':      {'straw_management', 'straw_use', 'organic_fertilizer', 'pest_management'},
            'harvest':               {'straw_management', 'straw_use', 'seed_preparation', 'pest_management'},
            'straw_management':      {'water_management', 'pest_management', 'seed_preparation'},
        }
        if query_topic != 'general':
            off_topics = off_topic_map.get(query_topic, set())
            on_topic = [r for r in ranked if r.get('topic_tag', 'general') not in off_topics]
            off_topic = [r for r in ranked if r.get('topic_tag', 'general') in off_topics]
            # Keep all on-topic; only add off-topic as fallback if fewer than 2 on-topic
            if len(on_topic) >= 2:
                top_results = on_topic[:self.top_k]
                logger.info(f"Hard filter: kept {len(top_results)} on-topic chunks, dropped {len(off_topic)} off-topic for topic='{query_topic}'")
            else:
                top_results = (on_topic + off_topic)[:self.top_k]
                logger.info(f"Hard filter: only {len(on_topic)} on-topic, keeping {len(top_results)} total as fallback")
        else:
            top_results = ranked[:self.top_k]
        
        # CRITICAL: Pre-filter to remove wrong fertilizer protocol chunk
        top_results = self._filter_fertilizer_protocol(top_results, query)
        
        # Sibling chunk injection: for each retrieved chunk, pull in sibling chunks
        # from the same parent section to provide fuller context
        top_results = self._inject_sibling_chunks(top_results)
        
        return top_results
    
    def _filter_fertilizer_protocol(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Filter out wrong fertilizer protocol chunk based on query keywords.
        If query mentions 'without combined seeder', keep only 3-stage protocol.
        If query mentions 'with combined seeder', keep only 2-stage protocol.
        """
        query_lower = query.lower()
        
        # Check for WITHOUT keywords (must check 'không' negation first)
        has_without = (
            'không kết hợp vùi phân' in query_lower or
            'without combined' in query_lower or
            'does not incorporate fertilizer' in query_lower or
            'not incorporate fertiliser' in query_lower or
            'no fertilizer burial' in query_lower
        )
        
        # Check for WITH keywords (only if WITHOUT is not present)
        # Avoid substring match: "không kết hợp vùi phân" contains "kết hợp vùi phân"
        has_with = (
            not has_without and (
                'kết hợp vùi phân' in query_lower or
                'with combined' in query_lower or
                'incorporate fertilizer' in query_lower or
                'incorporate fertiliser' in query_lower or
                'fertilizer burial' in query_lower
            )
        )
        
        # If query clearly specifies the seeder type, filter accordingly
        if has_without and not has_with:
            # Keep only 3-stage protocol, remove 2-stage
            filtered = []
            for r in results:
                title = r.get('title', '').lower()
                chunk_id = r.get('chunk_id', '')
                # Remove 2-stage protocol chunk
                if '2-stage' in title or 'sub-2.5b' in chunk_id:
                    logger.info(f"Filtered out 2-stage protocol (query specifies WITHOUT combined seeder): {r.get('title', '')[:60]}")
                    continue
                filtered.append(r)
            return filtered
        
        elif has_with and not has_without:
            # Keep only 2-stage protocol, remove 3-stage
            filtered = []
            for r in results:
                title = r.get('title', '').lower()
                chunk_id = r.get('chunk_id', '')
                # Remove 3-stage protocol chunk
                if '3-stage' in title or 'sub-2.5a' in chunk_id:
                    logger.info(f"Filtered out 3-stage protocol (query specifies WITH combined seeder): {r.get('title', '')[:60]}")
                    continue
                filtered.append(r)
            return filtered
        
        # If ambiguous or no fertilizer query, return all results
        return results
    
    def _inject_sibling_chunks(self, results: List[Dict]) -> List[Dict]:
        """
        For each retrieved chunk, find sibling chunks from the same parent section
        using section_id prefix matching (e.g., sub-2.5.c → siblings sub-2.5.a, sub-2.5.b, sub-2.5.d).
        This ensures that when sub-2.5.c (timing) is retrieved, sub-2.5.b (quantities) comes along.
        """
        if not results:
            return results
        
        existing_ids = {r.get('chunk_id') for r in results}
        injected = []
        seen_prefixes = set()
        
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for r in results:
                chunk_id = r.get('chunk_id')
                if not chunk_id:
                    continue
                
                try:
                    # Get the section_id for this chunk
                    sec_result = session.run("""
                        MATCH (c:Chunk {chunk_id: $cid})-[:PART_OF]->(s:Section)
                        RETURN s.section_id AS section_id
                    """, cid=chunk_id)
                    sec_rec = sec_result.single()
                    if not sec_rec:
                        continue
                    
                    section_id = sec_rec['section_id']
                    # Extract parent prefix: sub-2.5.c → sub-2.5, sec-2.1 → sec-2
                    parts = section_id.rsplit('.', 1)
                    if len(parts) < 2:
                        continue
                    parent_prefix = parts[0]
                    
                    if parent_prefix in seen_prefixes:
                        continue
                    seen_prefixes.add(parent_prefix)
                    
                    # Find sibling sections with same prefix
                    siblings = session.run("""
                        MATCH (sib_sec:Section)
                        WHERE sib_sec.section_id STARTS WITH $prefix
                          AND sib_sec.section_id <> $sid
                        MATCH (sib_chunk:Chunk)-[:PART_OF]->(sib_sec)
                        WHERE NOT sib_chunk.chunk_id IN $existing
                        OPTIONAL MATCH (sib_sec)-[:IN_CHAPTER]->(ch:Chapter)
                        RETURN sib_chunk.chunk_id AS chunk_id,
                               sib_chunk.text AS text,
                               sib_chunk.title AS title,
                               sib_sec.title AS section_title,
                               sib_sec.section_num AS section_num,
                               sib_sec.chapter_num AS chapter_num
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
                                'source': 'sibling_injection'
                            })
                            existing_ids.add(sib_id)
                except Exception as e:
                    logger.warning(f"Sibling injection failed for {chunk_id}: {e}")
        
        if injected:
            logger.info(f"Sibling injection: added {len(injected)} sibling chunks")
            results.extend(injected[:3])
        
        return results
    
    def _detect_query_topic(self, query: str) -> str:
        """
        LLM-based topic detection. Asks the LLM to classify the query into one of the
        canonical rice farming topics. Returns one of the canonical topics or 'general'.
        """
        valid_topics = [
            'water_management', 'fertilizer_management', 'pest_management',
            'seed_preparation', 'land_preparation', 'straw_management',
            'harvest', 'general'
        ]
        
        try:
            prompt = f"""Classify this rice farming query into exactly ONE topic.

Query: {query}

Valid topics:
- water_management: irrigation, AWD, flooding, drainage, water level
- fertilizer_management: fertilizer, nutrients, nitrogen, phosphorus, potassium, NPK, urea, lime
- pest_management: pests, diseases, insects, weeds, rats, snails, fungus, herbicide, IPM
- seed_preparation: seed selection, seed soaking, germination, sowing, nursery, seed treatment
- land_preparation: tillage, plowing, leveling, laser leveling, soil preparation
- straw_management: rice straw, composting, straw by-products, mushroom cultivation
- harvest: harvesting, post-harvest, grain drying, milling, storage
- general: if the query does not clearly fit any topic above

Return ONLY the topic name, nothing else."""

            result = self.llm.invoke([HumanMessage(content=prompt)])
            topic = result.content.strip().lower().replace(' ', '_')
            
            # Normalize sub-topic aliases
            topic_map = {
                'direct_seeding': 'seed_preparation',
                'organic_fertilizer': 'fertilizer_management',
                'straw_use': 'straw_management',
            }
            topic = topic_map.get(topic, topic)
            
            if topic not in valid_topics:
                logger.warning(f"LLM returned invalid topic '{topic}', defaulting to 'general'")
                topic = 'general'
            
            logger.info(f"Topic detection (LLM): {topic} for query: '{query[:80]}'")
            return topic
            
        except Exception as e:
            logger.warning(f"LLM topic detection failed: {e}, defaulting to 'general'")
            return 'general'
    
    def _calculate_rerank_multiplier(self, title: str, parent_title: str, query: str,
                                      chunk_topic: str = 'general', query_topic: str = 'general') -> float:
        """
        Reranking multiplier combining title match and topic-awareness.
        Topic match: strong boost (+40%) for exact match, strong penalty (-35%) for off-topic.
        Returns multiplier between 0.55 and 1.5.
        """
        multiplier = 1.0
        title_lower = (title or "").lower()
        
        # --- Topic-aware boost/penalty ---
        off_topic_map = {
            'water_management':    {'fertilizer_management', 'pest_management', 'straw_management', 'straw_use', 'organic_fertilizer'},
            'fertilizer_management': {'water_management', 'pest_management', 'straw_management', 'straw_use', 'seed_preparation', 'land_preparation', 'harvest', 'direct_seeding'},
            'pest_management':     {'water_management', 'fertilizer_management', 'straw_management', 'straw_use', 'organic_fertilizer'},
            'seed_preparation':    {'straw_management', 'straw_use', 'organic_fertilizer', 'harvest'},
            'land_preparation':    {'straw_management', 'straw_use', 'organic_fertilizer', 'pest_management'},
            'harvest':             {'straw_management', 'straw_use', 'seed_preparation', 'pest_management'},
            'straw_management':    {'water_management', 'pest_management', 'seed_preparation'},
        }
        
        if query_topic != 'general':
            if chunk_topic == query_topic:
                multiplier *= 1.4  # Strong boost: chunk matches query topic
            elif chunk_topic in off_topic_map.get(query_topic, set()):
                multiplier *= 0.55  # Strong penalty: chunk is clearly off-topic
        
        # --- Mild generic-title penalty ---
        generic_terms = ['overview', 'introduction', 'summary', 'general']
        if any(term in title_lower for term in generic_terms):
            multiplier *= 0.85
        
        # --- Mild title-term match boost ---
        query_terms = [w.lower() for w in query.lower().split()
                       if len(w) > 3 and w not in {'what', 'how', 'why', 'when', 'where', 'the', 'and', 'for',
                                                     'giải', 'thích', 'phương', 'pháp', 'trong', 'canh'}]
        title_match_count = sum(1 for term in query_terms if term in title_lower)
        if title_match_count > 0:
            multiplier *= min(1.0 + (0.08 * title_match_count), 1.2)
        
        return max(0.55, min(1.5, multiplier))
    
    def _align_with_tracing(self, raw_entities: List[str]) -> List[Dict[str, Any]]:
        """Wrapper for align_entities_to_kg with Langfuse tracing for parallel execution."""
        with self.langfuse.start_as_current_observation(
            as_type="chain",
            name="align_entities_to_kg",
            input={"raw_entities": raw_entities}
        ) as align_span:
            aligned_entities = self.align_entities_to_kg(raw_entities)
            align_span.update(output={"aligned_count": len(aligned_entities)})
            return aligned_entities
    
    def _detect_topic_with_tracing(self, query: str) -> str:
        """Wrapper for _detect_query_topic with Langfuse tracing for parallel execution."""
        with self.langfuse.start_as_current_observation(
            as_type="generation",
            name="detect_query_topic",
            input={"query": query}
        ) as topic_span:
            topic = self._detect_query_topic(query)
            topic_span.update(output={"topic": topic})
            return topic
    
    def run_query(self, question: str, language: str = 'vi', skip_answer: bool = False, skip_trace_fetch: bool = False) -> PipelineResult:
        """
        Run a single query through the ADITI triple-hybrid pipeline.
        
        Args:
            question: The user's question
            skip_answer: If True, skip LLM answer generation (caller generates its own)
            skip_trace_fetch: If True, skip Langfuse trace fetch + 0.5s sleep
            
        Returns:
            PipelineResult with all pipeline outputs
        """
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Create Langfuse trace using context manager
        with self.langfuse.start_as_current_observation(
            as_type="agent",
            name="ADITI_Triple_Hybrid_Pipeline",
            input={"question": question}
        ) as agent:
            
            # Step 1: Multi-stage entity extraction
            t_step = time.time()
            with self.langfuse.start_as_current_observation(
                as_type="generation",
                name="extract_entities_multistage",
                input={"question": question}
            ) as extract_span:
                extraction_result = self.extract_entities_multistage(question)
                raw_entities = extraction_result["raw_entities"]
                typed_entities = extraction_result["typed_entities"]
                relationships = extraction_result["relationships"]
                
                extract_span.update(output={
                    "raw_entities": raw_entities,
                    "typed_entities": typed_entities,
                    "relationships": relationships,
                    "measurements": typed_entities.get("measurements", [])
                })
            logger.info(f"[TIMING] Step 1 entity_extraction: {time.time()-t_step:.2f}s | entities={raw_entities}")
            
            # Step 2 & 3: Parallelize entity alignment and topic detection (independent operations)
            t_step = time.time()
            english_q = self._translate_to_english_if_needed(question)
            logger.info(f"[TIMING] Step 2a translate: {time.time()-t_step:.2f}s | english_q='{english_q[:80]}'")
            
            t_step = time.time()
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks concurrently
                align_future = executor.submit(self._align_with_tracing, raw_entities)
                topic_future = executor.submit(self._detect_topic_with_tracing, english_q)
                
                # Wait for both to complete
                aligned_entities = align_future.result()
                query_topic = topic_future.result()
            logger.info(f"[TIMING] Step 2b align+topic (parallel): {time.time()-t_step:.2f}s | aligned={len(aligned_entities)}, topic='{query_topic}'")
            t_step = time.time()
            with self.langfuse.start_as_current_observation(
                as_type="chain",
                name="traverse_kg_graph",
                input={"aligned_entities": [e["name"] for e in aligned_entities]}
            ) as kg_span:
                graph_facts = self.traverse_kg_graph(aligned_entities, query=question, top_k=10)
                
                # Filter facts by topic relevance
                if query_topic != 'general' and graph_facts:
                    topic_keywords = {
                        'water_management': {'water', 'irrigation', 'awd', 'flood', 'drain', 'wet', 'dry', 'moisture', 'tưới', 'nước', '15 cm', '3-5 cm', 'ngập', 'rút'},
                        'fertilizer_management': {'kg/ha', 'n ', 'p2o5', 'k2o', 'nitrogen', 'phosph', 'potassium', 'fertiliz', 'phân', 'đạm', 'lân', 'kali', 'urea', 'ssnm', 'nutrient', 'npk', 'organic fertilizer', 'lime', 'vôi', 'bón'},
                        'pest_management': {'pest', 'disease', 'snail', 'rat', 'weed', 'insect', 'fungus', 'sâu', 'bệnh', 'ốc', 'chuột', 'cỏ', 'ipm'},
                        'seed_preparation': {'seed', 'germination', 'sowing', 'giống', 'nảy mầm', 'gieo', 'variety'},
                        'land_preparation': {'soil', 'plow', 'till', 'level', 'đất', 'cày', 'xới', 'san', 'laser'},
                        'harvest': {'harvest', 'dry', 'storage', 'thu hoạch', 'sấy', 'bảo quản', 'moisture'},
                        'straw_management': {'straw', 'stubble', 'residue', 'rơm', 'rạ', 'mulch'},
                    }
                    
                    relevant_keywords = topic_keywords.get(query_topic, set())
                    
                    if relevant_keywords:
                        filtered = []
                        for fact in graph_facts:
                            source_lower = str(fact.get('source', '')).lower()
                            target_lower = str(fact.get('target', '')).lower()
                            relation_lower = str(fact.get('relation', '')).lower()
                            fact_text = f"{source_lower} {relation_lower} {target_lower}"
                            
                            # Keep fact if it contains topic-relevant keywords
                            is_relevant = any(kw in fact_text for kw in relevant_keywords)
                            
                            if is_relevant:
                                filtered.append(fact)
                        
                        if len(filtered) < len(graph_facts):
                            logger.info(f"Topic filter: kept {len(filtered)}/{len(graph_facts)} graph facts for topic='{query_topic}'")
                        graph_facts = filtered if filtered else graph_facts[:3]  # Keep at least 3 facts as fallback
                
                kg_span.update(output={"graph_facts_count": len(graph_facts), "query_topic": query_topic})
            logger.info(f"[TIMING] Step 3 graph_traversal: {time.time()-t_step:.2f}s | facts={len(graph_facts)}")
            
            # Step 4: Triple-hybrid retrieval (structured data + BM25 + vector)
            t_step = time.time()
            with self.langfuse.start_as_current_observation(
                as_type="chain",
                name="triple_hybrid_retrieval",
                input={"question": question}
            ) as retrieval_span:
                retrieved_chunks = self.triple_hybrid_retrieve(question, aligned_entities=aligned_entities, language=language, english_query=english_q, query_topic=query_topic)
                retrieval_span.update(output={"num_chunks": len(retrieved_chunks)})
            logger.info(f"[TIMING] Step 4 triple_hybrid_retrieve: {time.time()-t_step:.2f}s | chunks={len(retrieved_chunks)}")
        
            # Step 5: Extract structured data from retrieved chunks
            t_step = time.time()
            intent = self.extract_query_intent(question)
            for chunk in retrieved_chunks:
                structured_data = self.extract_structured_data_from_text(chunk.get('text', ''), intent)
                if structured_data:
                    # Add as additional graph fact
                    graph_facts.append({
                        'source': 'Extracted Data',
                        'relation': 'contains_info',
                        'target': f"{structured_data['nutrient']}: {structured_data['min_kg_ha']}-{structured_data['max_kg_ha']} kg/ha ({structured_data['soil_type']} soil)",
                        'structured_data': structured_data,
                        'score': 1.0,
                        'tier': 'structured'
                    })
            
            logger.info(f"[TIMING] Step 5 structured_data_extract: {time.time()-t_step:.2f}s")
            
            # Step 6: Generate farmer-friendly answer using LLM
            farmer_answer = ""
            llm_context = {}
            if not skip_answer:
                with self.langfuse.start_as_current_observation(
                    as_type="generation",
                    name="generate_farmer_answer",
                    input=question  # Plain string for evaluators
                ) as gen_span:
                    farmer_answer, llm_context = self.generate_answer(
                        question, retrieved_chunks,
                        measurements=typed_entities.get('measurements', [])
                    )
                    gen_span.update(
                        output=farmer_answer,  # Plain string for evaluators
                        metadata={"context": llm_context}  # Context for LLM-as-a-Judge evaluators
                    )
            else:
                logger.info("[FAST] Skipping answer generation (skip_answer=True)")
            
            # Keyword results from BM25
            keyword_results = [
                {
                    'chunk_id': c['chunk_id'],
                    'text': c['text'],
                    'title': c.get('title', ''),
                    'section_title': c.get('section_title', ''),
                    'section_num': c.get('section_num', ''),
                    'chapter_num': c.get('chapter_num', ''),
                    'score': c.get('bm25_score', 0)
                }
                for c in retrieved_chunks if c.get('bm25_score', 0) > 0
            ]
            
            # Vector results
            vector_context = [
                {
                    'chunk_id': c['chunk_id'],
                    'text': c['text'],
                    'title': c.get('title', ''),
                    'section_title': c.get('section_title', ''),
                    'section_num': c.get('section_num', ''),
                    'chapter_num': c.get('chapter_num', ''),
                    'score': c.get('vector_score', 0)
                }
                for c in retrieved_chunks if c.get('vector_score', 0) > 0
            ]
            
            # RRF fused results (hybrid scores)
            rrf_fused_results = [
                {
                    'chunk_id': c['chunk_id'],
                    'text': c['text'],
                    'title': c.get('title', ''),
                    'section_title': c.get('section_title', ''),
                    'section_num': c.get('section_num', ''),
                    'chapter_num': c.get('chapter_num', ''),
                    'hybrid_score': c['hybrid_score'],
                    'structured_score': c['structured_score'],
                    'bm25_score': c['bm25_score'],
                    'vector_score': c['vector_score'],
                    'structured_data': c.get('structured_data')
                }
                for c in retrieved_chunks
            ]
            
            trace_id = agent.trace_id
        
        # Flush to Langfuse
        self.langfuse.flush()
        
        execution_time = time.time() - start_time

        # Structured retrieval metrics log
        logger.info(
            "[METRICS] query_done "
            f"latency={execution_time:.2f}s "
            f"entities={len(raw_entities)} "
            f"aligned={len(aligned_entities)} "
            f"graph_facts={len(graph_facts)} "
            f"chunks_retrieved={len(retrieved_chunks)} "
            f"chunks_fused={len(rrf_fused_results)} "
            f"language={language} "
            f"topic={query_topic if 'query_topic' in dir() else 'unknown'}"
        )

        # Build trace URL
        trace_url = f"{config.LANGFUSE_HOST}/trace/{trace_id}"
        
        # Fetch trace data for full UI visualization (skip for fast/voice paths)
        trace_data = None
        if not skip_trace_fetch:
            try:
                import time as time_module
                time_module.sleep(0.5)  # Brief delay to ensure trace is available
                fetched_trace = self.langfuse.api.trace.get(trace_id)
                if fetched_trace:
                    # Convert trace to serializable dict
                    trace_data = {
                        "id": trace_id,
                        "name": getattr(fetched_trace, 'name', None),
                        "timestamp": getattr(fetched_trace, 'timestamp', None).isoformat() if hasattr(fetched_trace, 'timestamp') and fetched_trace.timestamp else None,
                        "observations": []
                    }
                    # Store observations if available
                    if hasattr(fetched_trace, 'observations') and fetched_trace.observations:
                        for obs in fetched_trace.observations:
                            obs_dict = {
                                "id": getattr(obs, 'id', None),
                                "name": getattr(obs, 'name', None),
                                "type": getattr(obs, 'type', None),
                                "start_time": getattr(obs, 'start_time', None).isoformat() if hasattr(obs, 'start_time') and obs.start_time else None,
                                "end_time": getattr(obs, 'end_time', None).isoformat() if hasattr(obs, 'end_time') and obs.end_time else None,
                                "input": getattr(obs, 'input', None),
                                "output": getattr(obs, 'output', None),
                                "parent_observation_id": getattr(obs, 'parent_observation_id', None)
                            }
                            trace_data["observations"].append(obs_dict)
            except Exception as e:
                logger.warning(f"Failed to fetch trace data for persistence: {e}")
        else:
            logger.info("[FAST] Skipping trace fetch (skip_trace_fetch=True)")
        
        return PipelineResult(
            question=question,
            farmer_answer=farmer_answer,
            raw_entities=raw_entities,
            aligned_entities=aligned_entities,
            graph_facts=graph_facts,
            vector_context=vector_context,
            keyword_results=keyword_results,
            rrf_fused_results=rrf_fused_results,
            llm_context=llm_context,
            trace_id=trace_id,
            trace_url=trace_url,
            trace_data=trace_data,
            execution_time=execution_time,
            query_topic=query_topic
        )
    
    def verify_database(self) -> dict:
        """
        Verify database setup and return status of required components.
        Call this to check if the KG and indexes are properly set up.
        """
        status = {
            "connected": False,
            "chunk_count": 0,
            "entity_count": 0,
            "structured_data_count": 0,
            "vector_indexes": [],
            "fulltext_indexes": [],
            "missing_indexes": [],
            "ready": False
        }
        
        try:
            # Check connection
            with self.driver.session(database=config.NEO4J_DATABASE) as session:
                session.run("RETURN 1")
                status["connected"] = True
                
                # Check node counts
                chunk_result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
                status["chunk_count"] = chunk_result.single()["count"] if chunk_result else 0
                
                entity_result = session.run("""
                    MATCH (n) WHERE n.name IS NOT NULL
                    AND NOT n:Chunk AND NOT n:Section AND NOT n:Chapter AND NOT n:Document
                    RETURN count(n) as count
                """)
                status["entity_count"] = entity_result.single()["count"] if entity_result else 0
                
                # Check structured data nodes
                structured_result = session.run("""
                    MATCH (n)
                    WHERE n:NutrientRequirement OR n:LimeRequirement
                    RETURN count(n) as count
                """)
                status["structured_data_count"] = structured_result.single()["count"] if structured_result else 0
                
                # Check indexes
                indexes_result = session.run("SHOW INDEXES YIELD name, type RETURN name, type")
                for record in indexes_result:
                    if record["type"] == "VECTOR":
                        status["vector_indexes"].append(record["name"])
                    elif record["type"] == "FULLTEXT":
                        status["fulltext_indexes"].append(record["name"])
                
                # Check required indexes
                required_fulltext = ["chunk_text_index"]
                
                missing = []
                for idx in required_fulltext:
                    if idx not in status["fulltext_indexes"]:
                        missing.append(f"fulltext:{idx}")
                
                status["missing_indexes"] = missing
                status["indexes"] = status["vector_indexes"] + status["fulltext_indexes"]
                
                # Overall ready status
                status["ready"] = (
                    status["chunk_count"] > 0 and 
                    len(status["missing_indexes"]) == 0
                )
                
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def get_agent_graph(self) -> Dict[str, Any]:
        """Return the agent graph structure for visualization."""
        return {
            "nodes": [
                {"id": "start", "label": "User Query", "type": "start"},
                {"id": "extract", "label": "Entity Extraction (LLM)", "type": "generation"},
                {"id": "translate", "label": "Translate to English", "type": "process"},
                {"id": "align", "label": "Entity Alignment\n(Vector Index)", "type": "retrieval"},
                {"id": "topic", "label": "Topic Detection\n(LLM)", "type": "generation"},
                {"id": "kg_traverse", "label": "KG Graph Traversal", "type": "tool"},
                {"id": "bm25", "label": "BM25 Keyword Search", "type": "retrieval"},
                {"id": "vector", "label": "Vector Search", "type": "retrieval"},
                {"id": "structured", "label": "Structured Data\nExtraction", "type": "tool"},
                {"id": "fusion", "label": "Triple-Hybrid Fusion", "type": "process"},
                {"id": "generate", "label": "Generate Answer (LLM)", "type": "generation"},
                {"id": "end", "label": "Pipeline Result", "type": "end"},
            ],
            "edges": [
                ("start", "extract"),
                ("extract", "translate"),
                ("translate", "align"),
                ("translate", "topic"),
                ("align", "kg_traverse"),
                ("topic", "kg_traverse"),
                ("kg_traverse", "bm25"),
                ("kg_traverse", "vector"),
                ("kg_traverse", "structured"),
                ("bm25", "fusion"),
                ("vector", "fusion"),
                ("structured", "fusion"),
                ("fusion", "generate"),
                ("generate", "end"),
            ],
        }
