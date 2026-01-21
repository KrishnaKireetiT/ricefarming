"""
Rice Farming KG Pipeline V7 - BM25 + RRF Hybrid Search
Based on kg_pipeline_v7.ipynb

Major Upgrades from V6:
1. Neo4j Full-Text Search (BM25): Replaced manual CONTAINS with Lucene-based BM25 ranking
2. Reciprocal Rank Fusion (RRF): Proper hybrid aggregation of KG, Vector, and Keyword results
3. Entity-Driven Query Expansion: Aligned entities expand keyword queries automatically
4. Graph Neighbor Retrieval: NEXT_CHUNK relationships for context continuity
"""

import os
import time
import json
import ast
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_neo4j import Neo4jGraph
from sentence_transformers import SentenceTransformer
from langfuse import Langfuse

from pipelines.base import BasePipeline, PipelineResult
import config

logger = logging.getLogger(__name__)


# ================================================================
# Custom Model Wrappers
# ================================================================

class LocalHuggingFaceEmbeddings(Embeddings):
    """Local HuggingFace embedding model wrapper."""
    
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512):
        logger.info(f"Loading Embedding Model {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()


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


# ================================================================
# Entity Extraction Models
# ================================================================

class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: Optional[str] = None


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str


class GraphExtraction(BaseModel):
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]


# ================================================================
# Pipeline V7 Implementation - BM25 + RRF
# ================================================================

class PipelineV7(BasePipeline):
    """
    Rice Farming KG Pipeline V7 - BM25 + RRF
    
    Features:
    - L1/L2 Knowledge Graph structure
    - BM25 keyword search with entity-driven query expansion
    - Reciprocal Rank Fusion (RRF) for hybrid result aggregation
    - Graph neighbor retrieval via NEXT_CHUNK relationships
    - Dynamic score-based filtering for semantic search
    - Langfuse agent graph tracing
    """
    
    def __init__(self):
        self._initialized = False
        self.llm = None
        self.embeddings = None
        self.graph = None
        self.langfuse = None
        self._entity_cache = None
    
    def get_name(self) -> str:
        return "KG Pipeline V7 (BM25 + RRF)"
    
    def get_version(self) -> str:
        return "7.0.0"
    
    def get_description(self) -> str:
        return "Rice farming pipeline with BM25 keyword search, RRF hybrid aggregation, entity expansion, and graph neighbor retrieval. Langfuse tracing enabled."
    
    def initialize(self) -> None:
        """Initialize models and connections."""
        if self._initialized:
            return
        
        logger.info("Initializing Pipeline V7...")
        
        # Initialize LLM
        self.llm = QwenRemoteLLM()
        
        # Initialize embeddings
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = LocalHuggingFaceEmbeddings(
            config.EMBEDDING_MODEL_NAME, 
            device=device, 
            max_length=512
        )
        
        # Initialize Neo4j connection
        self.graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE,
            driver_config={"max_connection_pool_size": 10},
        )
        
        # Initialize Langfuse
        self.langfuse = Langfuse(timeout=config.LANGFUSE_TIMEOUT)
        
        self._initialized = True
        logger.info("Pipeline V7 initialized successfully.")
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def get_agent_graph(self) -> Dict[str, Any]:
        """Return the agent graph structure for V7 pipeline visualization."""
        return {
            "nodes": [
                {"id": "start", "label": "__start__", "type": "start"},
                {"id": "agent", "label": "Rice_Farming_Advisor_V7", "type": "agent"},
                {"id": "extract", "label": "Extract_Question_Entities", "type": "generation"},
                {"id": "align", "label": "Align_Entities_To_KG", "type": "chain"},
                {"id": "retrieval", "label": "Multi_Source_Evidence_Retrieval", "type": "chain"},
                {"id": "kg_tool", "label": "KG_Graph_Traversal", "type": "tool"},
                {"id": "vector_tool", "label": "Vector_Semantic_Search", "type": "tool"},
                {"id": "keyword_tool", "label": "Keyword_BM25_Search", "type": "tool"},
                {"id": "aggregate", "label": "Aggregate_Evidence_RRF", "type": "chain"},
                {"id": "generate", "label": "Generate_Farmer_Answer", "type": "generation"},
                {"id": "end", "label": "__end__", "type": "end"}
            ],
            "edges": [
                ("start", "agent"),
                ("agent", "extract"),
                ("extract", "align"),
                ("align", "retrieval"),
                ("retrieval", "kg_tool"),
                ("retrieval", "vector_tool"),
                ("retrieval", "keyword_tool"),
                ("kg_tool", "aggregate"),
                ("vector_tool", "aggregate"),
                ("keyword_tool", "aggregate"),
                ("aggregate", "generate"),
                ("generate", "end")
            ]
        }
    
    # ============================================================
    # Core Search Functions
    # ============================================================
    
    def _get_cached_entity_embeddings(self) -> dict:
        """Get cached KG entity embeddings for alignment."""
        if self._entity_cache is None:
            kg_entities = self.graph.query("""
                MATCH (n) WHERE n.name IS NOT NULL
                  AND NOT n:Chunk AND NOT n:Section AND NOT n:Chapter AND NOT n:Document
                RETURN DISTINCT n.name AS name, labels(n) AS labels
            """)
            kg_names = [e["name"] for e in kg_entities]
            kg_vectors = self.embeddings.embed_documents(kg_names) if kg_names else []
            self._entity_cache = {"entities": kg_entities, "names": kg_names, "vectors": kg_vectors}
        return self._entity_cache
    
    def _cosine_sim(self, a, b) -> float:
        return float(np.dot(a, b))
    
    def _extract_raw_entities(self, question: str) -> List[str]:
        """Extract key entities from question using LLM."""
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract key entities. Return JSON array: [\"entity1\", \"entity2\"]"),
            ("user", "Question: {q}")
        ])
        try:
            chain = prompt | self.llm | parser
            entities = chain.invoke({"q": question})
            return entities if isinstance(entities, list) else []
        except Exception as e:
            error_msg = str(e)
            if "Invalid json output:" in error_msg:
                try:
                    raw = error_msg.split("Invalid json output:")[1].strip()
                    return list(ast.literal_eval(raw))
                except:
                    pass
            return []
    
    def _rank_and_align_entities(self, extracted_entities: List[str], min_sim: float = None) -> dict:
        """Align extracted entities to KG entities."""
        if min_sim is None:
            min_sim = config.ENTITY_ALIGNMENT_THRESHOLD
        
        cache = self._get_cached_entity_embeddings()
        aligned = []
        
        for raw in extracted_entities:
            q_emb = self.embeddings.embed_query(raw)
            scored = [{"kg_name": cache["entities"][i]["name"], "score": self._cosine_sim(q_emb, cache["vectors"][i])}
                      for i in range(len(cache["entities"]))]
            scored.sort(key=lambda x: x["score"], reverse=True)
            best = scored[0] if scored else {"kg_name": "", "score": 0}
            
            if best["score"] >= min_sim:
                aligned.append({"raw": raw, "matched": best["kg_name"], "name": best["kg_name"], "score": best["score"]})
        
        return {"aligned_entities": aligned}
    
    def _graph_traversal_search(self, aligned_entities: List[Dict], top_k: int = None) -> List[Dict]:
        """Traverse KG for related facts."""
        if top_k is None:
            top_k = config.GRAPH_TRAVERSAL_TOP_K
        
        facts = []
        for item in aligned_entities:
            entity_name = item.get("name") or item.get("matched")
            if not entity_name:
                continue
            
            res = self.graph.query("""
                MATCH (n {name: $name})-[r:REL|MENTIONS]-(m)
                RETURN n.name AS source, type(r) AS rel_type, r.type AS rel_detail,
                       m.name AS target, labels(m)[0] AS target_label, elementId(m) AS target_id
            """, {"name": entity_name})
            
            for row in res:
                rel_type = row["rel_type"]
                weight = config.RELATION_WEIGHTS.get(rel_type, config.DEFAULT_REL_WEIGHT)
                facts.append({
                    "source": row["source"], "relation": row["rel_detail"] or rel_type,
                    "target": row["target"], "target_label": row["target_label"],
                    "target_id": row["target_id"], "tier": "core" if rel_type == "REL" else "context",
                    "score": item.get("score", 1.0) * weight
                })
        
        unique = {}
        for f in facts:
            key = (f["source"], f["relation"], f["target"])
            if key not in unique or f["score"] > unique[key]["score"]:
                unique[key] = f
        
        return sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    def verify_database(self) -> dict:
        """
        Verify database setup and return status of required components.
        Call this to check if the KG and indexes are properly set up.
        """
        status = {
            "connected": False,
            "chunk_count": 0,
            "entity_count": 0,
            "vector_indexes": [],
            "fulltext_indexes": [],
            "missing_indexes": [],
            "ready": False
        }
        
        try:
            # Check connection
            self.graph.query("RETURN 1")
            status["connected"] = True
            
            # Check node counts
            chunk_result = self.graph.query("MATCH (c:Chunk) RETURN count(c) as count")
            status["chunk_count"] = chunk_result[0]["count"] if chunk_result else 0
            
            entity_result = self.graph.query("""
                MATCH (n) WHERE n.name IS NOT NULL
                AND NOT n:Chunk AND NOT n:Section AND NOT n:Chapter AND NOT n:Document
                RETURN count(n) as count
            """)
            status["entity_count"] = entity_result[0]["count"] if entity_result else 0
            
            # Check vector indexes
            vector_result = self.graph.query("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name")
            status["vector_indexes"] = [r["name"] for r in vector_result]
            
            # Check fulltext indexes
            fulltext_result = self.graph.query("SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' RETURN name")
            status["fulltext_indexes"] = [r["name"] for r in fulltext_result]
            
            # Check required indexes
            required_vector = [f"{config.VECTOR_INDEX_NAME}"]
            required_fulltext = ["chunk_text_index"]
            
            missing = []
            for idx in required_vector:
                if idx not in status["vector_indexes"]:
                    missing.append(f"vector:{idx}")
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
    
    def _vector_search_chunks(self, query: str, k: int = None, 
                              score_threshold: float = None) -> List[Dict]:
        """Semantic search over Chunks with dynamic score-based filtering."""
        if k is None:
            k = config.VECTOR_SEARCH_K
        if score_threshold is None:
            score_threshold = config.VECTOR_SCORE_THRESHOLD
        
        query_embedding = self.embeddings.embed_query(query)
        
        results = []
        
        # Search chunks - fetch more results than k to allow filtering
        try:
            chunk_results = self.graph.query("""
                CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                YIELD node, score
                RETURN 'Chunk' AS type, node.chunk_id AS id, node.embedding_text AS text, 
                       node.title AS title, score
                ORDER BY score DESC
            """, {"index_name": f"{config.VECTOR_INDEX_NAME}", "k": k * 2, "embedding": query_embedding})
            
            # Apply score threshold - only return chunks above the bar
            filtered = [r for r in chunk_results if r["score"] >= score_threshold]
            results.extend(filtered)
        except Exception as e:
            logger.warning(f"Chunk vector search failed: {e}")
            # Fallback: try to get chunks without vector search
            try:
                fallback = self.graph.query("""
                    MATCH (c:Chunk) 
                    RETURN 'Chunk' AS type, c.chunk_id AS id, c.embedding_text AS text,
                           c.title AS title, 0.5 AS score
                    LIMIT $k
                """, {"k": k})
                results.extend(list(fallback))
                logger.info(f"Using fallback chunk retrieval: {len(fallback)} chunks")
            except:
                pass
        
        return results[:k]
    
    def _keyword_search_bm25(self, query: str, raw_entities: List[str] = None,
                            aligned_entities: List[Dict] = None, k: int = None) -> List[Dict]:
        """BM25-powered keyword search with entity-driven query expansion."""
        if k is None: 
            k = config.KEYWORD_SEARCH_K
        if raw_entities is None: 
            raw_entities = []
        if aligned_entities is None: 
            aligned_entities = []
        
        # Build search terms with entity expansion
        search_terms = set()
        for entity in raw_entities:
            for word in entity.split():
                if len(word) >= 3: 
                    search_terms.add(word)
        
        # ENTITY EXPANSION: Add aligned KG entity names
        for aligned in aligned_entities:
            kg_name = aligned.get("name") or aligned.get("matched")
            if kg_name:
                for word in kg_name.split():
                    if len(word) >= 3: 
                        search_terms.add(word)
        
        # Add important query words
        STOPWORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'in', 'for', 'on', 'with'}
        for word in query.lower().split():
            clean = ''.join(c for c in word if c.isalnum())
            if len(clean) >= 4 and clean not in STOPWORDS: 
                search_terms.add(clean)
        
        if not search_terms: 
            return []
        
        # Build Lucene query (removed fuzzy ~1 operator - causes parsing error in Neo4j)
        lucene_terms = list(search_terms)[:10]
        lucene_query = " OR ".join(lucene_terms)
        
        try:
            # Use Neo4j Full-Text Index (BM25)
            results = self.graph.query("""
                CALL db.index.fulltext.queryNodes("chunk_text_index", $lucene_query)
                YIELD node, score
                RETURN node.chunk_id AS id, node.embedding_text AS text, score
                ORDER BY score DESC LIMIT $top_k
            """, {"lucene_query": lucene_query, "top_k": k * 2})
            
            # GRAPH NEIGHBOR RETRIEVAL: Expand with NEXT_CHUNK context
            expanded, seen = [], set()
            for r in results[:k]:
                if r["id"] not in seen:
                    expanded.append(r)
                    seen.add(r["id"])
                    # Fetch neighboring chunks
                    neighbors = self.graph.query("""
                        MATCH (c:Chunk {chunk_id: $cid})-[:NEXT_CHUNK]->(next:Chunk)
                        RETURN next.chunk_id AS id, next.embedding_text AS text, $score * 0.5 AS score
                        UNION
                        MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c:Chunk {chunk_id: $cid})
                        RETURN prev.chunk_id AS id, prev.embedding_text AS text, $score * 0.5 AS score
                    """, {"cid": r["id"], "score": r["score"]})
                    for neighbor in neighbors:
                        if neighbor["id"] not in seen and len(expanded) < k * 1.5:
                            expanded.append(neighbor)
                            seen.add(neighbor["id"])
            return expanded[:k]
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []
    
    def reciprocal_rank_fusion(self, ranked_lists: List[List[Dict]], k: int = None) -> List[Dict]:
        """Merge multiple ranked result lists using Reciprocal Rank Fusion."""
        if k is None: 
            k = config.RRF_K
        rrf_scores, doc_data = {}, {}
        
        for result_list in ranked_lists:
            for rank, doc in enumerate(result_list):
                doc_id = doc.get('id')
                if not doc_id: 
                    continue
                # RRF formula: 1 / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                if doc_id not in doc_data: 
                    doc_data[doc_id] = doc
        
        # Build final ranked list
        merged = []
        for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_data[doc_id].copy()
            doc['rrf_score'] = rrf_score
            doc['score'] = rrf_score
            merged.append(doc)
        return merged
    
    def _build_farmer_answer(self, evidence: dict) -> str:
        """Generate farmer-friendly answer from evidence."""
        context_chunks = evidence.get("vector_context", [])[:6]
        context_text = "\n\n".join([f"[DOC-{i+1}] {c.get('text', '')[:800]}" for i, c in enumerate(context_chunks)])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly rice farming advisor. Answer using ONLY the context provided.
- Use simple language
- Cite sources as [DOC-N]
- If information mentions [VISUAL CONCEPT], include that knowledge in your answer
- If you don't know, say so honestly"""),
            ("user", """Question: {question}

Context:
{context}""")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({"question": evidence["question"], "context": context_text})
        return result.content
    
    # ============================================================
    # Main Pipeline Execution
    # ============================================================
    
    def run_query(self, question: str) -> PipelineResult:
        """
        Run the full rice farming advisor pipeline.
        Returns structured output with Langfuse tracing.
        """
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Root agent span
        with self.langfuse.start_as_current_observation(
            as_type="agent",
            name="Rice_Farming_Advisor_V7",
            input={"question": question}
        ) as agent:
            
            result = {"question": question}
            
            # Step 1: Extract Question Entities
            with self.langfuse.start_as_current_observation(
                as_type="generation",
                name="Extract_Question_Entities",
                input={"question": question}
            ) as gen:
                raw_entities = self._extract_raw_entities(question)
                gen.update(output={"entities": raw_entities, "count": len(raw_entities)})
            result["raw_entities"] = raw_entities
            
            # Step 2: Align Entities to KG
            with self.langfuse.start_as_current_observation(
                as_type="chain",
                name="Align_Entities_To_KG",
                input={"raw_entities": raw_entities}
            ) as chain:
                alignment = self._rank_and_align_entities(raw_entities)
                aligned = alignment["aligned_entities"]
                chain.update(output={"aligned": [e["name"] for e in aligned], "count": len(aligned)})
            result["aligned_entities"] = aligned
            
            # Step 3: Multi-Source Evidence Retrieval
            retrieval_span = self.langfuse.start_observation(
                name="Multi_Source_Evidence_Retrieval",
                as_type="chain",
                input={"aligned_entities": [e["name"] for e in aligned], "query": question}
            )
            
            # Start all tool observations upfront
            tool1 = retrieval_span.start_observation(
                name="KG_Graph_Traversal",
                as_type="tool",
                input={"entities": [e["name"] for e in aligned]}
            )
            tool2 = retrieval_span.start_observation(
                name="Vector_Semantic_Search",
                as_type="tool",
                input={"query": question}
            )
            tool3 = retrieval_span.start_observation(
                name="Keyword_BM25_Search",
                as_type="tool",
                input={"query": question, "raw_entities": raw_entities}
            )
            
            # Execute searches
            graph_facts = self._graph_traversal_search(aligned)
            vector_results = self._vector_search_chunks(question)
            keyword_results = self._keyword_search_bm25(question, raw_entities=raw_entities, aligned_entities=aligned)
            
            # Update and end tools
            tool1.update(output={"facts_count": len(graph_facts)})
            tool1.end()
            result["graph_facts"] = graph_facts
            
            tool2.update(output={"results_count": len(vector_results)})
            tool2.end()
            
            tool3.update(output={"results_count": len(keyword_results)})
            tool3.end()
            
            retrieval_span.update(output={
                "total_sources": len(graph_facts) + len(vector_results) + len(keyword_results)
            })
            retrieval_span.end()
            
            # Step 4: Aggregate Evidence using RRF
            with self.langfuse.start_as_current_observation(
                as_type="chain",
                name="Aggregate_Evidence_RRF",
                input={"vector_count": len(vector_results), "keyword_count": len(keyword_results)}
            ) as agg:
                # Use Reciprocal Rank Fusion
                merged = self.reciprocal_rank_fusion([vector_results, keyword_results])
                agg.update(output={"merged_count": len(merged)})
            
            result["vector_context"] = merged
            result["keyword_results"] = keyword_results

            
            # Step 5: Generate Farmer Answer
            with self.langfuse.start_as_current_observation(
                as_type="generation",
                name="Generate_Farmer_Answer",
                input={
                    "question": question,
                    "context_chunks": len(merged),
                    "graph_facts": len(graph_facts)
                }
            ) as gen:
                farmer_answer = self._build_farmer_answer(result)
                gen.update(output={"answer_length": len(farmer_answer)})
            
            result["farmer_answer"] = farmer_answer
            
            # Update root agent
            agent.update(output={
                "answer_preview": farmer_answer[:200],
                "entities_found": len(aligned),
                "evidence_sources": len(merged) + len(graph_facts)
            })
            
            trace_id = agent.trace_id
            result["trace_id"] = trace_id
        
        # Flush to Langfuse
        self.langfuse.flush()
        
        execution_time = time.time() - start_time
        
        # Build trace URL
        trace_url = f"{config.LANGFUSE_HOST}/traces/{trace_id}"
        
        return PipelineResult(
            question=question,
            farmer_answer=result.get("farmer_answer", ""),
            raw_entities=result.get("raw_entities", []),
            aligned_entities=result.get("aligned_entities", []),
            graph_facts=result.get("graph_facts", []),
            vector_context=result.get("vector_context", []),
            keyword_results=keyword_results,

            trace_id=trace_id,
            trace_url=trace_url,
            execution_time=execution_time
        )
