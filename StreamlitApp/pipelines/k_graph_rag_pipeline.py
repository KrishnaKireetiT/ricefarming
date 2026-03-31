"""
K-GraphRAG Pipeline — 4-channel Graph RAG with ColBERT + Qwen dense + BM25 + KG.

Converted from FinalizedPpipelines/phase5_rag_pipeline_colbert_qwen_server.ipynb.
Uses lightonai/Reason-ModernColBERT for late-interaction retrieval alongside
Qwen dense embeddings (text-embedding-3-small via OpenAI-compat API),
Lucene BM25 fulltext, and Neo4j knowledge-graph traversal.

All four channels are fused via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ColBERT — lightonai/Reason-ModernColBERT
from pipelines.colbert_retrieval import encode as colbert_encode, maxsim_score

# Langfuse tracing (optional)
_LANGFUSE_AVAILABLE = False
langfuse_context = None

def _noop_observe(**kwargs):
    return lambda f: f

observe = _noop_observe

def _get_langfuse_client():
    return None

try:
    # Langfuse v4+
    from langfuse import observe, get_client as _get_langfuse_client
    _LANGFUSE_AVAILABLE = True

    class _LangfuseContextShim:
        """Shim to provide langfuse_context-like API using v4 get_client()."""
        def get_current_trace_id(self):
            return _get_langfuse_client().get_current_trace_id()
        def get_current_trace_url(self):
            return _get_langfuse_client().get_trace_url()
        def update_current_trace(self, **kwargs):
            try:
                client = _get_langfuse_client()
                io_kwargs = {}
                if "input" in kwargs:
                    io_kwargs["input"] = kwargs["input"]
                if "output" in kwargs:
                    io_kwargs["output"] = kwargs["output"]
                if io_kwargs:
                    client.set_current_trace_io(**io_kwargs)
            except Exception:
                pass
        def update_current_observation(self, **kwargs):
            try:
                _get_langfuse_client().update_current_span(**kwargs)
            except Exception:
                pass

    langfuse_context = _LangfuseContextShim()
except ImportError:
    try:
        # Langfuse v2/v3 fallback
        from langfuse.decorators import observe, langfuse_context
        _LANGFUSE_AVAILABLE = True
    except ImportError:
        pass

# Suppress Pydantic serialization warnings from LangChain structured output
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

import config
from pipelines.base import BasePipeline, PipelineResult

logger = logging.getLogger("k_graphrag")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')

VIETNAMESE_PATTERN = re.compile(
    r"[\u00e0\u00e1\u1ea3\u00e3\u1ea1\u0103\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7"
    r"\u00e2\u1ea5\u1ea7\u1ea9\u1eab\u1ead\u00e8\u00e9\u1ebb\u1ebd\u1eb9"
    r"\u00ea\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7\u00ec\u00ed\u1ec9\u0129\u1ecb"
    r"\u00f2\u00f3\u1ecf\u00f5\u1ecd\u00f4\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9"
    r"\u01a1\u1edb\u1edd\u1edf\u1ee1\u1ee3\u00f9\u00fa\u1ee7\u0169\u1ee5"
    r"\u01b0\u1ee9\u1eeb\u1eed\u1eef\u1ef1\u1ef3\u00fd\u1ef7\u1ef9\u1ef5"
    r"\u0111\u0110]"
)

MAX_QUERY_LENGTH = 2000

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
FIGURE_REF_RE = re.compile(r"\[Figure\s+(C\d+-\d+)\]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pydantic model for structured entity extraction
# ---------------------------------------------------------------------------
class QueryEntities(BaseModel):
    """Entities extracted from a query for KG lookup."""
    entities: list[str] = Field(description="Entity names mentioned or implied in the query")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def detect_language(text: str) -> str:
    return "vi" if len(VIETNAMESE_PATTERN.findall(text)) >= 2 else "en"


def translate_to_english(text: str, llm: BaseChatModel) -> str:
    response = llm.invoke([
        SystemMessage(content="You are a translator. Translate the following Vietnamese text to English. Return only the translation."),
        HumanMessage(content=text),
    ])
    return response.content.strip()


def sanitize_lucene(query: str) -> str:
    return LUCENE_SPECIAL.sub(r"\\\1", query)


def _extract_json(text: str) -> str:
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    for sc, ec in [("{", "}"), ("[", "]")]:
        s, e = text.find(sc), text.rfind(ec)
        if s != -1 and e != -1 and e > s:
            return text[s: e + 1]
    return text.strip()


def _parse_json_entity_list(text: str) -> list[str]:
    try:
        data = json.loads(_extract_json(text))
        if isinstance(data, dict) and "entities" in data:
            return [str(e) for e in data["entities"]]
        if isinstance(data, list):
            return [str(e) for e in data]
    except Exception:
        pass
    return []


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[str, float]]],
    k: int = 60,
) -> tuple[list[tuple[str, float]], dict[str, set[str]]]:
    """Fuse N ranked lists via RRF. Returns (sorted_results, channel_map)."""
    scores: dict[str, float] = defaultdict(float)
    channels: dict[str, set[str]] = defaultdict(set)
    for channel, ranked_list in ranked_lists.items():
        for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
            channels[chunk_id].add(channel)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results, dict(channels)


# ---------------------------------------------------------------------------
# ColBERT Store — loads pre-computed per-chunk .npy embeddings
# ---------------------------------------------------------------------------
class ColBERTStore:
    """Loads per-chunk ColBERT .npy files into memory and runs MaxSim retrieval.

    Directory layout expected:
        <embed_base>/chunks_en/{chunk_id}.npy
        <embed_base>/chunks_vi/{chunk_id}.npy
        <embed_base>/chunks_en/ids.pkl
        <embed_base>/chunks_vi/ids.pkl
    Each .npy file has shape (num_tokens, 128), float32, L2-normalised.
    """

    def __init__(self, embed_base: str):
        logger.info("Loading ColBERT store from %s", embed_base)
        self._en = self._load_index(embed_base, "chunks_en")
        self._vi = self._load_index(embed_base, "chunks_vi")
        logger.info("ColBERT store ready — %d EN embeddings, %d VI embeddings",
                     len(self._en), len(self._vi))

    def _load_index(self, embed_base: str, index_name: str) -> dict[str, np.ndarray]:
        index_dir = os.path.join(embed_base, index_name)
        ids_file  = os.path.join(index_dir, "ids.pkl")
        if not os.path.exists(ids_file):
            logger.warning("ColBERT index not found: %s — channel will return empty", ids_file)
            return {}
        with open(ids_file, "rb") as f:
            ids = pickle.load(f)
        embeddings: dict[str, np.ndarray] = {}
        missing = 0
        for doc_id in ids:
            path = os.path.join(index_dir, f"{doc_id}.npy")
            if os.path.exists(path):
                embeddings[str(doc_id)] = np.load(path)
            else:
                missing += 1
        if missing:
            logger.warning("ColBERT index '%s': %d/%d .npy files missing",
                           index_name, missing, len(ids))
        return embeddings

    def search(self, query_emb: np.ndarray, lang: str, k: int) -> list[tuple[str, float]]:
        """Return top-k (chunk_id, score) pairs using MaxSim scoring."""
        store = self._vi if lang == "vi" else self._en
        if not store:
            return []
        scored = [
            (doc_id, maxsim_score(query_emb, doc_emb))
            for doc_id, doc_emb in store.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
class KGraphRAGPipeline(BasePipeline):
    """4-channel Graph RAG: ColBERT + Qwen dense + Lucene BM25 + KG.

    Retrieval channels:
      1. Qwen dense semantic (vector index in Neo4j)
      2. ColBERT late-interaction (pre-computed .npy MaxSim)
      3. Lucene BM25 fulltext
      4. Knowledge Graph traversal (entity extraction → relationship walk)

    Fusion: Reciprocal Rank Fusion across all channels.
    """

    def __init__(self):
        self._initialized = False
        self.driver = None
        self.embeddings: Optional[Embeddings] = None
        self.colbert_store: Optional[ColBERTStore] = None
        self.llm: Optional[BaseChatModel] = None
        self._neo4j_database = os.getenv("KGRAPHRAG_NEO4J_DATABASE", config.NEO4J_DATABASE)

        # Search parameters from config
        self._vector_search_k = config.VECTOR_SEARCH_K
        self._keyword_search_k = config.KEYWORD_SEARCH_K
        self._colbert_search_k = int(os.getenv("COLBERT_SEARCH_K", "10"))
        self._graph_traversal_top_k = config.GRAPH_TRAVERSAL_TOP_K
        self._rrf_k = config.RRF_K
        self._max_context_tokens = 16000
        self._include_full_text = True

    # ------------------------------------------------------------------ #
    # BasePipeline interface
    # ------------------------------------------------------------------ #
    def get_name(self) -> str:
        return "K-GraphRAG Pipeline"

    def get_version(self) -> str:
        return "2.0"

    def get_description(self) -> str:
        return (
            "4-channel Graph RAG with ColBERT + Qwen dense embeddings, "
            "Lucene BM25, KG traversal, and RRF fusion"
        )

    def initialize(self) -> None:
        if self._initialized:
            return

        # Neo4j — use K-GraphRAG-specific env vars, fall back to shared config
        _kg_uri = os.getenv("KGRAPHRAG_NEO4J_URI", config.NEO4J_URI)
        _kg_user = os.getenv("KGRAPHRAG_NEO4J_USER", config.NEO4J_USER)
        _kg_pass = os.getenv("KGRAPHRAG_NEO4J_PASSWORD", config.NEO4J_PASSWORD)
        self.driver = GraphDatabase.driver(
            _kg_uri,
            auth=(_kg_user, _kg_pass),
            max_connection_pool_size=10,
        )
        logger.info("Neo4j driver created: %s (db=%s)", _kg_uri, self._neo4j_database)

        # Dense embeddings — OpenAI-compatible (text-embedding-3-large with 1024 dims)
        embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        embed_dims = int(os.getenv("EMBED_DIMENSIONS", "1536"))
        # text-embedding-3-large supports custom dimensions via dimensions parameter
        if "text-embedding-3" in embed_model:
            self.embeddings = OpenAIEmbeddings(model=embed_model, dimensions=embed_dims)
        else:
            self.embeddings = OpenAIEmbeddings(model=embed_model)
        self._validate_embedding_dimension()

        # ColBERT store — pre-computed .npy embeddings
        colbert_base = os.getenv(
            "COLBERT_EMBED_BASE",
            str((Path(__file__).parent.parent.parent.parent / "FinalizedPpipelines" / "colbert-embeddings").resolve()),
        )
        self.colbert_store = ColBERTStore(colbert_base)

        # LLM — Qwen via OpenAI-compatible API
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            base_url=config.QWEN_BASE_URL,
            api_key=config.QWEN_API_KEY,
            temperature=0.0,
        )

        self._initialized = True
        logger.info("KGraphRAGPipeline initialized (4-channel, ColBERT EN=%d chunks)",
                     len(self.colbert_store._en))

    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------ #
    # Startup validation
    # ------------------------------------------------------------------ #
    def _validate_embedding_dimension(self):
        test_vec   = self.embeddings.embed_query("test")
        actual_dim = len(test_vec)
        expected_dim = 1536
        try:
            with self.driver.session(database=self._neo4j_database) as session:
                rec = session.run(
                    "SHOW INDEXES YIELD name, options "
                    "WHERE name = 'chunk_embedding_index' RETURN options"
                ).single()
                if rec:
                    expected_dim = rec["options"].get("indexConfig", {}).get(
                        "vector.dimensions", 1536
                    )
        except Exception as e:
            logger.warning("Could not query vector index dimension: %s", e)
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: model produces {actual_dim}d "
                f"but Neo4j index expects {expected_dim}d."
            )
        logger.info("Embedding dimension validated: %dd", actual_dim)

    # ------------------------------------------------------------------ #
    # Neo4j helpers
    # ------------------------------------------------------------------ #
    def _run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        with self.driver.session(database=self._neo4j_database) as session:
            return [r.data() for r in session.run(query, params or {})]

    def _fetch_chunks(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        records = self._run_cypher(
            "UNWIND $ids AS cid "
            "MATCH (c:Chunk {chunk_id: cid}) "
            "RETURN c.chunk_id AS chunk_id, "
            "  c.eng_summary AS eng_summary, c.vi_summary AS vi_summary, "
            "  c.eng_text AS eng_text, c.vi_text AS vi_text, "
            "  c.eng_title AS eng_title, c.vi_title AS vi_title, "
            "  c.eng_hierarchy AS eng_hierarchy, "
            "  c.start_page AS start_page, c.end_page AS end_page",
            {"ids": chunk_ids},
        )
        return {str(r["chunk_id"]): r for r in records}

    # ------------------------------------------------------------------ #
    # Channel 1: Qwen dense semantic (vector index)
    # ------------------------------------------------------------------ #
    @observe(name="semantic_search", capture_input=False, capture_output=False)
    def _semantic_search(
        self, query: str, lang: str, k: int | None = None
    ) -> list[tuple[str, float]]:
        k = k or self._vector_search_k
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "lang": lang, "k": k},
                metadata={"channel": "qwen_dense_semantic"},
            )
        embedding   = self.embeddings.embed_query(query)
        index_name  = "chunk_vi_embedding_index" if lang == "vi" else "chunk_embedding_index"
        try:
            records = self._run_cypher(
                f"CALL db.index.vector.queryNodes('{index_name}', $k, $embedding) "
                "YIELD node, score RETURN node.chunk_id AS chunk_id, score",
                {"k": k, "embedding": embedding},
            )
            results = [(str(r["chunk_id"]), float(r["score"])) for r in records]
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output={"count": len(results), "top_chunks": [r[0] for r in results[:3]]},
                    metadata={"channel": "qwen_dense_semantic", "index": index_name},
                )
            logger.debug("Dense semantic search (%s): %d results", index_name, len(results))
            return results
        except Exception as e:
            logger.warning("Dense semantic search failed (%s): %s", index_name, e)
            if langfuse_context:
                langfuse_context.update_current_observation(output={"count": 0, "error": str(e)})
            return []

    # ------------------------------------------------------------------ #
    # Channel 2: ColBERT semantic (numpy MaxSim)
    # ------------------------------------------------------------------ #
    @observe(name="colbert_search", capture_input=False, capture_output=False)
    def _colbert_search(
        self, query: str, lang: str, k: int | None = None
    ) -> list[tuple[str, float]]:
        k = k or self._colbert_search_k
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "lang": lang, "k": k},
                metadata={"channel": "colbert_semantic"},
            )
        try:
            query_emb = colbert_encode([query], is_query=True)[0]   # (Q, 128)
            results = self.colbert_store.search(query_emb, lang, k)
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output={"count": len(results), "top_chunks": [r[0] for r in results[:3]]},
                    metadata={"channel": "colbert_semantic"},
                )
            logger.debug("ColBERT search: %d results", len(results))
            return results
        except Exception as e:
            logger.warning("ColBERT search failed: %s", e)
            if langfuse_context:
                langfuse_context.update_current_observation(output={"count": 0, "error": str(e)})
            return []

    # ------------------------------------------------------------------ #
    # Channel 3: Lucene BM25 fulltext
    # ------------------------------------------------------------------ #
    @observe(name="fulltext_search", capture_input=False, capture_output=False)
    def _fulltext_search(
        self, query: str, k: int | None = None
    ) -> list[tuple[str, float]]:
        k = k or self._keyword_search_k
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "k": k},
                metadata={"channel": "lucene_bm25"},
            )
        sanitized = sanitize_lucene(query)
        if not sanitized.strip():
            return []
        try:
            records = self._run_cypher(
                "CALL db.index.fulltext.queryNodes('chunk_fulltext_index', $query) "
                "YIELD node, score RETURN node.chunk_id AS chunk_id, score LIMIT $k",
                {"query": sanitized, "k": k},
            )
            results = [(str(r["chunk_id"]), float(r["score"])) for r in records]
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output={"count": len(results), "top_chunks": [r[0] for r in results[:3]]},
                    metadata={"channel": "lucene_bm25"},
                )
            logger.debug("Fulltext search (Lucene): %d results", len(results))
            return results
        except Exception as e:
            logger.warning("Fulltext search failed: %s", e)
            if langfuse_context:
                langfuse_context.update_current_observation(output={"count": 0, "error": str(e)})
            return []

    # ------------------------------------------------------------------ #
    # Channel 4: Knowledge Graph retrieval
    # ------------------------------------------------------------------ #
    @observe(name="graph_retrieval", capture_input=False, capture_output=False)
    def _graph_retrieval(
        self, query_en: str, query_embedding: list[float],
        community_embedding: list[float] | None = None,
    ) -> tuple[list[tuple[str, float]], list[dict], list[dict], list[dict]]:
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query_en},
                metadata={"channel": "qwen_kg_graph"},
            )
        entities = self._extract_entities(query_en)
        if not entities:
            return [], [], [], []

        sanitized_names = [sanitize_lucene(e) for e in entities if sanitize_lucene(e).strip()]
        if not sanitized_names:
            return [], [], [], []

        try:
            matched = self._run_cypher(
                "UNWIND $names AS name "
                "CALL db.index.fulltext.queryNodes('entity_fulltext_index', name) "
                "YIELD node, score WHERE score > 0.5 "
                "RETURN name AS raw_query, node.name AS name, node.type AS type, "
                "node.description AS description, score "
                "ORDER BY score DESC LIMIT $limit",
                {"names": sanitized_names, "limit": len(sanitized_names) * 5},
            )
        except Exception as e:
            logger.warning("Entity fulltext search failed: %s", e)
            matched = []

        if not matched:
            return [], [], [], []

        seen: set[str] = set()
        unique_entities = [e for e in matched if not (e["name"] in seen or seen.add(e["name"]))]
        entity_names = [e["name"] for e in unique_entities]

        all_triples: list[dict] = []
        chunk_entity_counts: dict[str, int] = defaultdict(int)

        try:
            records = self._run_cypher(
                "UNWIND $names AS name "
                "MATCH (e:Entity {name: name})-[r:RELATES_TO]-(n:Entity) "
                "RETURN "
                "  CASE WHEN startNode(r) = e THEN e.name ELSE n.name END AS source, "
                "  r.relation AS relation, "
                "  CASE WHEN startNode(r) = e THEN n.name ELSE e.name END AS target, "
                "  r.detail AS detail, r.source_chunk AS source_chunk, r.pages AS pages",
                {"names": entity_names},
            )
            all_triples.extend(records)
        except Exception as e:
            logger.warning("Graph traversal failed: %s", e)

        try:
            records = self._run_cypher(
                "UNWIND $names AS name "
                "MATCH (e:Entity {name: name})-[:EXTRACTED_FROM]->(c:Chunk) "
                "RETURN c.chunk_id AS chunk_id",
                {"names": entity_names},
            )
            for r in records:
                chunk_entity_counts[str(r["chunk_id"])] += 1
        except Exception as e:
            logger.warning("Provenance query failed: %s", e)

        for t in all_triples:
            sc = t.get("source_chunk")
            if sc:
                chunk_entity_counts[str(sc)] += 1

        ranked = sorted(chunk_entity_counts.items(), key=lambda x: x[1], reverse=True)
        max_count = ranked[0][1] if ranked else 1
        chunk_results = [
            (cid, cnt / max_count) for cid, cnt in ranked
        ][: self._graph_traversal_top_k]

        communities: list[dict] = []
        try:
            comm_emb = community_embedding if community_embedding is not None else query_embedding
            records = self._run_cypher(
                "CALL db.index.vector.queryNodes('community_embedding_index', 3, $embedding) "
                "YIELD node, score WHERE score > 0.3 "
                "RETURN node.title AS title, node.summary AS summary, score",
                {"embedding": comm_emb},
            )
            communities = records
        except Exception as e:
            logger.warning("Community search failed: %s", e)

        seen_t: set[str] = set()
        unique_triples = []
        for t in all_triples:
            key = f"{t.get('source')}|{t.get('relation')}|{t.get('target')}"
            if key not in seen_t:
                seen_t.add(key)
                unique_triples.append(t)

        if langfuse_context:
            langfuse_context.update_current_observation(
                output={
                    "chunk_count": len(chunk_results),
                    "triple_count": len(unique_triples),
                    "community_count": len(communities),
                    "entity_count": len(entity_names),
                },
                metadata={"channel": "qwen_kg_graph"},
            )
        return chunk_results, unique_triples, communities, unique_entities

    @observe(name="entity_extraction", capture_input=False, capture_output=False)
    def _extract_entities(self, query_en: str) -> list[str]:
        if langfuse_context:
            langfuse_context.update_current_observation(input={"query": query_en})
        prompt = (
            "Extract the key entity names from this query that would be useful "
            "for searching a knowledge graph about rice production. "
            "Include equipment, processes, inputs, crop stages, pests, parameters, "
            "products, principles, and varieties.\n\n"
            f"Query: {query_en}"
        )
        try:
            result = self.llm.with_structured_output(QueryEntities).invoke(
                [HumanMessage(content=prompt)]
            )
            if result and result.entities:
                if langfuse_context:
                    langfuse_context.update_current_observation(output={"entities": result.entities})
                return result.entities
        except Exception as e:
            logger.warning("Structured entity extraction failed, trying JSON fallback: %s", e)

        # Fallback: prompt for JSON + manual parse
        try:
            raw = self.llm.invoke([
                HumanMessage(
                    content=prompt
                    + '\n\nReturn ONLY a JSON object like: {"entities": ["entity1", "entity2"]}'
                )
            ])
            entities = _parse_json_entity_list(raw.content)
            if langfuse_context:
                langfuse_context.update_current_observation(output={"entities": entities})
            return entities
        except Exception as e:
            logger.warning("Entity extraction failed completely: %s", e)

        if langfuse_context:
            langfuse_context.update_current_observation(output={"entities": []})
        return []

    # ------------------------------------------------------------------ #
    # Context Assembly (from Neo4j)
    # ------------------------------------------------------------------ #
    def _assemble_context(
        self,
        chunk_ids: list[str],
        lang: str,
        triples: list[dict],
        communities: list[dict],
    ) -> str:
        budget = self._max_context_tokens
        used   = 0
        context_parts: list[str] = []

        def tok(t): return len(t) // 3

        if triples:
            lines = []
            for t in triples[:30]:
                line = f"  {t.get('source','?')} --[{t.get('relation','?')}]--> {t.get('target','?')}"
                if t.get("detail"):
                    line += f": {t['detail']}"
                lines.append(line)
            g = "=== Knowledge Graph Triples ===\n" + "\n".join(lines)
            used += tok(g); context_parts.append(g)

        if communities:
            lines = [f"  [{c.get('title','')}]: {c.get('summary','')}" for c in communities[:5]]
            cm = "=== Community Summaries ===\n" + "\n".join(lines)
            used += tok(cm); context_parts.append(cm)

        # Fetch chunk data from Neo4j
        chunks_data = self._fetch_chunks(chunk_ids)

        title_key     = "eng_title" if lang == "en" else "vi_title"
        hierarchy_key = "eng_hierarchy"
        summary_key   = "eng_summary" if lang == "en" else "vi_summary"
        text_key      = "eng_text" if lang == "en" else "vi_text"

        for cid in chunk_ids:
            chunk = chunks_data.get(cid)
            if chunk is None:
                continue

            title    = chunk.get(title_key) or f"Chunk {cid}"
            hier     = chunk.get(hierarchy_key) or ""
            summary  = chunk.get(summary_key) or ""
            fulltext = chunk.get(text_key) or ""
            pages    = f"(pp. {chunk.get('start_page','?')}-{chunk.get('end_page','?')})"

            base = f"\n--- {title} {pages} ---\n[{hier}]\n{summary}"
            if used + tok(base) > budget:
                break
            used += tok(base)
            text = base

            ft = f"\n[Full text]\n{fulltext}" if self._include_full_text and fulltext else ""
            if ft and used + tok(ft) <= budget:
                text += ft; used += tok(ft)

            context_parts.append(text)

        return "\n".join(context_parts)

    # ------------------------------------------------------------------ #
    # Answer Generation
    # ------------------------------------------------------------------ #
    @observe(name="generate_answer", as_type="generation", capture_input=False, capture_output=False)
    def _generate_answer(self, query: str, context: str, lang: str) -> tuple[str, str]:
        if langfuse_context:
            langfuse_context.update_current_observation(
                model=config.LLM_MODEL_NAME,
                input={"query": query, "context_chars": len(context), "lang": lang},
                metadata={"llm_provider": "qwen", "lang": lang},
            )
        if lang == "vi":
            sys_prompt = (
                "Ban la mot tro ly nong nghiep thuc te. Tra loi CHI dua tren ngu canh duoc cung cap.\n\n"
                "CAU TRUC TRA LOI (bat buoc theo thu tu):\n"
                "**Tra loi nhanh:** 1-2 cau trai loi thang van de.\n"
                "**Huong dan tung buoc:** Danh so ro rang, su dung ✅ cho moi buoc kha thi.\n"
                "**Luong/Don vi:** Neu bai viet dung don vi khac voi cau hoi, doi ra don vi quen thuoc "
                "(vi du: kg/ha → kg/cong/sao, lit/ha → binh xit 16 lit). Hien thi ro phep tinh.\n"
                "**Trich dan nguon:** Ghi so trang sau moi thong tin quan trong, vi du (tr. 45-47).\n"
                "**Canh bao:** Neu co thuoc tru sau/hoa chat, ghi ro ⚠️ lieu luong an toan va thoi gian cach ly.\n"
                "**Giai thich tu ngu:** Neu dung thuat ngu ky thuat, giai thich trong ngoac don, vi du "
                "'NH4+ (dan dau amoni — chua nito cho lua)'.\n"
                "**Neu khong chac:** Neu ngu canh chua du thong tin, ghi ro 'Tai lieu khong neu ro diem nay — "
                "nen hoi can bo khuyen nong dia phuong.'\n"
                "**Bai hoc chinh:** Mot dong tom tat hanh dong quan trong nhat.\n\n"
                "TU KIEM TRA (lam tham truoc khi viet cau tra loi cuoi cung):\n"
                "- Doc lai moi con so, ngay thang, phep doi don vi va phep tinh ban sap viet.\n"
                "- Doi chieu tung con so voi ngu canh nguon — neu nguon ghi 70 kg/ha, khong duoc viet 700 kg/ha.\n"
                "- Kiem tra tung buoc chuyen doi don vi (vi du: kg/ha sang kg/cong).\n"
                "- Neu phat hien sai, sua ngay tai cho truoc khi xuat ket qua.\n"
                "- KHONG de cap buoc tu kiem tra nay trong cau tra loi.\n\n"
                "Tra loi bang tieng Viet, van phong gian di, thuc te."
            )
        else:
            sys_prompt = (
                "You are a practical agricultural advisor helping rice farmers. "
                "Answer ONLY based on the provided context. "
                "Write as if explaining to a farmer who is standing in their field — clear, direct, and actionable.\n\n"
                "REQUIRED ANSWER STRUCTURE (follow this order):\n"
                "**Quick Answer:** 1-2 sentences directly addressing the question.\n"
                "**Step-by-Step Guide:** Numbered steps, each starting with ✅. Use plain language.\n"
                "**Amounts & Units:** If the source uses different units than the question implies, "
                "convert them to practical field amounts (e.g., kg/ha → kg per standard plot, "
                "L/ha → litres per backpack sprayer tank). Show the conversion clearly.\n"
                "**Sources:** After each key fact, cite the page range: (pp. 45-47).\n"
                "**Warnings:** For any pesticide, fertiliser rate, or chemical — prefix with ⚠️ "
                "and include safe handling notes and pre-harvest interval if mentioned.\n"
                "**Plain language:** If you must use a technical term, explain it in parentheses: "
                "e.g., 'tillering (the stage when the rice plant produces side shoots)'.\n"
                "**If context is thin:** Say clearly: 'The manual does not cover this specifically — "
                "please consult your local extension officer.'\n"
                "**Key Takeaway:** One bold sentence summarising the single most important action.\n\n"
                "SELF-CHECK (do this silently before writing your final answer):\n"
                "- Re-read every number, date, unit conversion, and calculation you are about to write.\n"
                "- Cross-check each figure against the source context — if the context says 70 kg/ha, do not write 700 kg/ha.\n"
                "- Verify all unit conversions step-by-step (e.g. kg/ha to kg per plot).\n"
                "- If you spot an inconsistency, fix it in place before outputting.\n"
                "- Do NOT mention this self-check in your answer.\n\n"
                "Answer in English. Be concise but complete — a farmer should be able to act on this immediately."
            )

        full_prompt = f"{context}\n\nQuestion: {query}"
        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=full_prompt)]

        try:
            answer = self.llm.invoke(messages).content.strip()
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=answer[:500] if len(answer) > 500 else answer,
                    usage={"input": len(context) // 4, "output": len(answer) // 4},
                )
            return answer, sys_prompt
        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            if langfuse_context:
                langfuse_context.update_current_observation(output=f"[Generation failed: {e}]")
            return f"[Generation failed: {e}]", sys_prompt

    # ------------------------------------------------------------------ #
    # Main query pipeline → PipelineResult
    # ------------------------------------------------------------------ #
    @observe(name="rag_pipeline", capture_input=False, capture_output=False)
    def run_query(self, question: str, language: str = "vi") -> PipelineResult:
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        timings: dict[str, float] = {}

        # Langfuse root trace
        _lf = _get_langfuse_client() if _LANGFUSE_AVAILABLE else None
        if _lf and not hasattr(_lf, 'start_as_current_observation'):
            _lf = None
        if langfuse_context:
            langfuse_context.update_current_trace(
                input={"query": question},
                tags=["4-channel-rag"],
            )

        if not question or not question.strip():
            return PipelineResult(
                question=question or "",
                farmer_answer="Please provide a non-empty query.",
                raw_entities=[], aligned_entities=[],
                graph_facts=[], vector_context=[], keyword_results=[],
                trace_id="", trace_url="",
            )
        if len(question) > MAX_QUERY_LENGTH:
            question = question[:MAX_QUERY_LENGTH]

        # 1. Language detection
        t0 = time.time()
        lang = detect_language(question)
        timings["language_detection"] = time.time() - t0
        logger.info("Query language: %s", lang)

        # 2. Translation
        t0 = time.time()
        if lang == "vi":
            query_en = translate_to_english(question, self.llm)
        else:
            query_en = question
        timings["translation"] = time.time() - t0

        # 3. Dense embedding (shared by semantic + community search)
        t0 = time.time()
        query_embedding = self.embeddings.embed_query(
            question if lang == "en" else query_en
        )
        timings["embedding"] = time.time() - t0

        # 4. Four retrieval channels — run in parallel (all I/O-bound)
        t_retrieval = time.time()
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_semantic = ex.submit(copy_context().run, self._semantic_search, question, lang)
            f_colbert  = ex.submit(copy_context().run, self._colbert_search, query_en, "en")
            f_fulltext = ex.submit(copy_context().run, self._fulltext_search, question)
            f_graph    = ex.submit(
                copy_context().run, self._graph_retrieval,
                query_en, query_embedding, query_embedding,
            )
            semantic_ranked = f_semantic.result()
            colbert_ranked  = f_colbert.result()
            fulltext_ranked = f_fulltext.result()
            graph_chunks, triples, communities, aligned_ents = f_graph.result()

        timings["retrieval_parallel"] = time.time() - t_retrieval
        logger.info(
            "Retrieval counts — dense: %d, colbert: %d, bm25: %d, graph: %d  (parallel)",
            len(semantic_ranked), len(colbert_ranked),
            len(fulltext_ranked), len(graph_chunks),
        )

        # 5. RRF fusion across all 4 channels
        t0 = time.time()
        ranked_lists = {k: v for k, v in {
            "semantic_dense":   semantic_ranked,
            "semantic_colbert": colbert_ranked,
            "fulltext_bm25":    fulltext_ranked,
            "graph":            graph_chunks,
        }.items() if v}

        if not ranked_lists:
            return PipelineResult(
                question=question,
                farmer_answer="No relevant information found for your query.",
                raw_entities=[], aligned_entities=[],
                graph_facts=[{"source": t.get("source",""), "relation": t.get("relation",""),
                              "target": t.get("target",""), "detail": t.get("detail","")}
                             for t in triples] if triples else [],
                vector_context=[], keyword_results=[],
                trace_id="", trace_url="",
                execution_time=time.time() - start_time,
            )

        fused, channel_map = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k)
        timings["rrf_fusion"] = time.time() - t0

        # 6. Top-K selection by RRF score
        rrf_scores = dict(fused)
        top_ids = [cid for cid, _ in fused[: self._graph_traversal_top_k]]

        # 7. Assemble context
        t0 = time.time()
        context = self._assemble_context(top_ids, lang, triples, communities)
        timings["context_assembly"] = time.time() - t0

        # 8. Generate answer
        t0 = time.time()
        answer, sys_prompt = self._generate_answer(question, context, lang)
        timings["generation"] = time.time() - t0

        execution_time = time.time() - start_time
        timings["total"] = execution_time
        logger.info("Pipeline completed in %.2fs", execution_time)

        # Langfuse finalise
        if langfuse_context:
            langfuse_context.update_current_trace(
                output={"answer": answer[:500] if len(answer) > 500 else answer, "language": lang},
                metadata={
                    "timings": {k: round(v, 3) for k, v in timings.items()},
                    "chunks_retrieved": len(top_ids),
                    "channels_active": list(ranked_lists.keys()),
                },
                tags=["4-channel-rag", f"lang:{lang}"],
            )

        # Build PipelineResult fields — enrich with chunk content for display
        all_cids = set()
        for cid, _ in semantic_ranked:
            all_cids.add(cid)
        for cid, _ in colbert_ranked:
            all_cids.add(cid)
        for cid, _ in fulltext_ranked:
            all_cids.add(cid)
        for cid, _ in fused:
            all_cids.add(cid)
        all_chunks = self._fetch_chunks(list(all_cids))

        _text_key = "eng_text" if lang == "en" else "vi_text"
        _title_key = "eng_title" if lang == "en" else "vi_title"

        def _enrich(cid: str) -> dict:
            c = all_chunks.get(cid, {})
            return {
                "title": c.get(_title_key) or c.get("eng_title") or f"Chunk {cid}",
                "text": c.get(_text_key) or c.get("eng_text") or "",
                "eng_hierarchy": c.get("eng_hierarchy", ""),
                "start_page": c.get("start_page", ""),
                "end_page": c.get("end_page", ""),
            }

        vector_context_out = []
        for cid, score in semantic_ranked:
            d = {"chunk_id": cid, "score": score, "channel": "semantic_dense"}
            d.update(_enrich(cid))
            vector_context_out.append(d)
        for cid, score in colbert_ranked:
            d = {"chunk_id": cid, "score": score, "channel": "semantic_colbert"}
            d.update(_enrich(cid))
            vector_context_out.append(d)

        keyword_results_out = []
        for cid, score in fulltext_ranked:
            d = {"chunk_id": cid, "id": cid, "score": score,
                 "raw_bm25_score": score, "channel": "fulltext_bm25"}
            d.update(_enrich(cid))
            keyword_results_out.append(d)

        rrf_fused_out = []
        for cid, score in fused:
            channels = channel_map.get(cid, set())
            if len(channels) > 1:
                src = "both"
            elif "semantic_dense" in channels or "semantic_colbert" in channels:
                src = "semantic"
            elif "fulltext_bm25" in channels:
                src = "keyword"
            else:
                src = "both"
            d = {
                "chunk_id": cid, "id": cid, "score": score, "rrf_score": score,
                "channels": sorted(channels), "retrieval_source": src,
            }
            d.update(_enrich(cid))
            rrf_fused_out.append(d)

        # Build entity score lookup for graph facts
        _ent_scores: dict[str, float] = {}
        if aligned_ents:
            for e in aligned_ents:
                n = e.get("name", "")
                s = e.get("score", 0)
                if n and (n not in _ent_scores or s > _ent_scores[n]):
                    _ent_scores[n] = s

        graph_facts_out = []
        for t in triples:
            src = t.get("source", "")
            tgt = t.get("target", "")
            fact_score = max(_ent_scores.get(src, 0), _ent_scores.get(tgt, 0))
            graph_facts_out.append({
                "source": src,
                "relation": t.get("relation", ""),
                "target": tgt,
                "detail": t.get("detail", ""),
                "score": round(fact_score, 2),
            })

        raw_entities = list({
            e for t in triples
            for e in [t.get("source", ""), t.get("target", "")]
            if e
        }) if triples else []

        # Get Langfuse trace info
        trace_id = ""
        trace_url = ""
        if langfuse_context:
            try:
                trace_id = langfuse_context.get_current_trace_id() or ""
                trace_url = langfuse_context.get_current_trace_url() or ""
            except Exception:
                pass

        return PipelineResult(
            question=question,
            farmer_answer=answer,
            raw_entities=raw_entities,
            aligned_entities=[{"raw": e.get("raw_query", ""), "name": e.get("name", ""), "type": e.get("type", ""), "score": e.get("score", 0)} for e in aligned_ents] if aligned_ents else [],
            graph_facts=graph_facts_out,
            vector_context=vector_context_out,
            keyword_results=keyword_results_out,
            rrf_fused_results=rrf_fused_out,
            llm_context={
                "system_prompt": sys_prompt,
                "question": question,
                "full_context": context,
                "context": context,
            },
            trace_id=trace_id,
            trace_url=trace_url,
            execution_time=execution_time,
            query_topic="general",
        )

    # ------------------------------------------------------------------ #
    # Database verification
    # ------------------------------------------------------------------ #
    def verify_database(self) -> dict:
        status = {
            "connected": False,
            "chunk_count": 0,
            "entity_count": 0,
            "indexes": [],
            "missing_indexes": [],
            "ready": False,
        }

        required_indexes = [
            "chunk_embedding_index",
            "chunk_vi_embedding_index",
            "chunk_fulltext_index",
            "entity_fulltext_index",
            "community_embedding_index",
        ]

        try:
            with self.driver.session(database=self._neo4j_database) as session:
                session.run("RETURN 1")
                status["connected"] = True

                result = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt")
                status["chunk_count"] = result.single()["cnt"]

                result = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
                status["entity_count"] = result.single()["cnt"]

                result = session.run("SHOW INDEXES YIELD name RETURN name")
                existing = {r["name"] for r in result}
                status["indexes"] = sorted(existing)
                status["missing_indexes"] = [
                    idx for idx in required_indexes if idx not in existing
                ]

                status["ready"] = (
                    status["chunk_count"] > 0
                    and len(status["missing_indexes"]) == 0
                )
        except Exception as e:
            logger.error("Database verification failed: %s", e)
            status["error"] = str(e)

        return status

    # ------------------------------------------------------------------ #
    # Agent graph for visualization
    # ------------------------------------------------------------------ #
    def get_agent_graph(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": "start", "label": "User Query", "type": "start"},
                {"id": "detect_lang", "label": "Language Detection", "type": "process"},
                {"id": "translate", "label": "Translate (if VI)", "type": "process"},
                {"id": "embed", "label": "Embed Query (Qwen dense)", "type": "process"},
                {"id": "semantic", "label": "Qwen Dense Search", "type": "retrieval"},
                {"id": "colbert", "label": "ColBERT MaxSim", "type": "retrieval"},
                {"id": "fulltext", "label": "Lucene BM25", "type": "retrieval"},
                {"id": "graph", "label": "KG Retrieval", "type": "retrieval"},
                {"id": "rrf", "label": "RRF Fusion (4-way)", "type": "process"},
                {"id": "context", "label": "Context Assembly", "type": "process"},
                {"id": "generate", "label": "Generate Answer", "type": "process"},
                {"id": "end", "label": "Pipeline Result", "type": "end"},
            ],
            "edges": [
                {"from": "start", "to": "detect_lang"},
                {"from": "detect_lang", "to": "translate"},
                {"from": "translate", "to": "embed"},
                {"from": "embed", "to": "semantic"},
                {"from": "embed", "to": "colbert"},
                {"from": "embed", "to": "fulltext"},
                {"from": "embed", "to": "graph"},
                {"from": "semantic", "to": "rrf"},
                {"from": "colbert", "to": "rrf"},
                {"from": "fulltext", "to": "rrf"},
                {"from": "graph", "to": "rrf"},
                {"from": "rrf", "to": "context"},
                {"from": "context", "to": "generate"},
                {"from": "generate", "to": "end"},
            ],
        }

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("KGraphRAGPipeline closed")
