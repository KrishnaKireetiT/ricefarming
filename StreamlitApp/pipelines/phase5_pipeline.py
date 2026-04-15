"""
phase5_pipeline.py — 4-channel bilingual Graph RAG pipeline (research/4).

Differences from research/3:
  - ColBERT embeddings live in Neo4j (Chunk.eng_embedding + eng_emb_shape), not .npy.
    ColBERTStore pulls them via one Cypher MATCH at startup, reshapes to (T,128),
    runs brute-force MaxSim at query time.
  - Generation prompt has NO Quick Answer section (removes 3/150 arithmetic
    self-contradictions observed in research/3). Structure is Step-by-Step → Key Takeaway.
  - Graph channel has no timeout — wait indefinitely; _extract_entities retries with
    exponential backoff (plain-text + manual JSON parse, no structured output).
  - Config values pulled from config.py / env, not redefined.
  - LangChain abstractions preserved for component swapping (LLM, embeddings).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from neo4j import GraphDatabase
from pipelines.colbert_retrieval import encode as colbert_encode, maxsim_score
from config import (
    COLBERT_SEARCH_K,
    EMBED_MODEL,
    GENERATION_MODEL,
    GRAPH_TRAVERSAL_TOP_K,
    KEYWORD_SEARCH_K,
    MAX_GRAPH_HOPS,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OUTPUT_CHUNKS_VI,
    PHASE5_NEO4J_URI,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    RRF_K,
    RRF_MIN_SCORE,
    VECTOR_SEARCH_K,
)

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pipelines.base import BasePipeline, PipelineResult

warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

_HERE = Path(__file__).parent if "__file__" in dir() else Path(".").resolve()

# ── Langfuse tracing (optional) ──────────────────────────────────────────────
_LANGFUSE_AVAILABLE = False
langfuse_context = None
def observe(**kwargs):
    return lambda f: f
def _get_langfuse_client():
    return None
try:
    from langfuse.decorators import observe, langfuse_context  # noqa: F811
    _LANGFUSE_AVAILABLE = True
except ImportError:
    pass
if _LANGFUSE_AVAILABLE:
    try:
        from langfuse import get_client as _get_langfuse_client  # noqa: F811
    except ImportError:
        from langfuse import Langfuse as _LangfuseV2
        _langfuse_v2_singleton = None
        def _get_langfuse_client():  # noqa: F811
            global _langfuse_v2_singleton
            if _langfuse_v2_singleton is None:
                _langfuse_v2_singleton = _LangfuseV2()
            return _langfuse_v2_singleton

logger = logging.getLogger("graphrag4")

# ── Language / query hygiene ─────────────────────────────────────────────────
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

# ── Topic routing (keyword → topic filter for BM25 + dense channels) ─────────
TOPIC_KEYWORDS = {
    "ipm": ["pest", "insect", "disease", "threshold", "IPM", "biological control",
            "pesticide", "herbicide", "fungicide", "BPH", "leaf folder", "stem borer",
            "snail", "rat", "weed", "rodent", "fungal", "blast", "hopper",
            "infestation", "larva", "predator", "spray", "trap"],
    "water_management": ["AWD", "drainage", "irrigation", "water level", "flooding",
                         "alternate wetting", "water", "moist", "flooded",
                         "flood", "waterlog"],
    "fertilisation": ["fertiliser", "fertilizer", "NPK", "urea", "potassium", "nitrogen",
                      "phosphorus", "granule", "top-dress", "prilled",
                      "dap", "kcl", "lime", "acidic", "acid", "pH", "alluvial",
                      "leaf color chart", "nutrient", "manure"],
    "straw_management": ["straw", "stubble", "composting", "mushroom", "cattle feed",
                         "baling", "dry period", "burial", "decomposition", "baler",
                         "husk", "bedding", "ensiling", "ensiled", "circular economy",
                         "mulching", "compost pile", "compost", "cattle", "buffalo"],
    "harvest": ["harvest", "combine", "moisture content", "grain loss", "threshing",
                "reaping", "combine harvester"],
    "post_harvest": ["drying", "moisture", "storage", "grain quality", "husking",
                     "milling", "dryer", "sun-dry", "fumigation", "paddy",
                     "dried"],
    "sowing": ["seeder", "sowing", "seed rate", "row spacing", "cluster", "transplant",
               "sow", "direct-seeded", "direct seeded", "drone seeding",
               "row seeding", "cluster seeding", "pneumatic", "broadcasting"],
    "land_preparation": ["land preparation", "levelling", "ploughing", "tillage",
                         "puddling", "leveling", "level", "plow", "plowing",
                         "laser", "elevation", "soil preparation"],
    "seed_preparation": ["seed treatment", "germination", "seed selection", "salt solution",
                         "seed", "soaking", "soak", "sprout", "incubation",
                         "certified seed", "variety", "varieties", "hydrated"],
    "digital_tools": ["CS-MAP", "CF-Rice", "carbon", "digital", "LCA", "mrv",
                      "emission", "GHG", "carbon footprint"],
    "storage": ["storage", "warehouse", "silo", "bag", "stored", "store",
                "ventilation", "aeration", "fumigant"],
}


_FUZZY_THRESHOLD = 82


def _fuzzy_keyword_score(query_lower: str, keywords: list[str]) -> float:
    """Score how well *query_lower* matches a topic's keyword list.

    For single-word keywords: token-level fuzzy match against each query word.
    For multi-word keywords: partial_ratio against the full query string.
    Returns the sum of best match scores (0-1 each) across all keywords.
    """
    query_words = query_lower.split()
    score = 0.0
    for kw in keywords:
        kw_low = kw.lower()
        if " " in kw_low:
            # multi-word keyword → partial match against full query
            if fuzz.partial_ratio(kw_low, query_lower) >= _FUZZY_THRESHOLD:
                score += 1.0
        else:
            # single-word keyword → best token match
            best = max(
                (fuzz.ratio(kw_low, w) for w in query_words),
                default=0,
            )
            if best >= _FUZZY_THRESHOLD:
                score += 1.0
    return score


def detect_query_topic(query: str) -> str | None:
    q = query.lower()
    scores = {t: _fuzzy_keyword_score(q, kws) for t, kws in TOPIC_KEYWORDS.items()}
    scores = {t: s for t, s in scores.items() if s > 0}
    if not scores:
        return None
    return max(scores, key=scores.get)


def _is_valid_summary_for_context(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return False
    except (json.JSONDecodeError, ValueError):
        pass
    if re.search(r'"(eng_summary|eng_text|eng_chunk_title)"\s*:', text):
        return False
    if any(marker in text for marker in ["base64", "image_url", "data:image"]):
        return False
    return True


def detect_language(text: str) -> str:
    return "vi" if len(VIETNAMESE_PATTERN.findall(text)) >= 2 else "en"


def sanitize_lucene(q: str) -> str:
    return LUCENE_SPECIAL.sub(r"\\\1", q)


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class RAGConfig:
    neo4j_uri: str = NEO4J_URI
    neo4j_user: str = NEO4J_USER
    neo4j_password: str = NEO4J_PASSWORD
    neo4j_database: str = NEO4J_DATABASE

    vector_search_k: int = VECTOR_SEARCH_K
    keyword_search_k: int = KEYWORD_SEARCH_K
    colbert_search_k: int = COLBERT_SEARCH_K
    graph_traversal_top_k: int = GRAPH_TRAVERSAL_TOP_K
    max_graph_hops: int = MAX_GRAPH_HOPS
    rrf_k: int = RRF_K
    rrf_min_score: float = RRF_MIN_SCORE

    mode: str = "textonly"
    llm_provider: str = "qwen"
    llm_model: str = GENERATION_MODEL
    llm_temperature: float = 0.0
    llm_timeout: int = 300

    embed_provider: str = "qwen3_local"
    embed_model: str = EMBED_MODEL

    max_context_tokens: int = 16000
    include_full_text: bool = True
    enable_sibling_expansion: bool = False

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls()


# ── LLM + embeddings (LangChain) ─────────────────────────────────────────────
def create_llm(config: RAGConfig) -> BaseChatModel:
    if config.llm_provider == "qwen":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.llm_model,
            base_url=QWEN_BASE_URL,
            api_key=QWEN_API_KEY,
            temperature=config.llm_temperature,
            timeout=config.llm_timeout,
            max_retries=0,
        )
    raise ValueError(f"Unknown LLM provider: {config.llm_provider}")


class Qwen3LocalEmbeddings(Embeddings):
    """Qwen3-Embedding-0.6B via transformers (1024d, L2-normalised, last-token pool)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading %s ...", model_name)
        self._tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self._model.eval()
        self._torch = torch

    def _pool(self, hidden, mask):
        left_pad = (mask[:, -1].sum() == mask.shape[0])
        if left_pad:
            return hidden[:, -1]
        seq_len = mask.sum(dim=1) - 1
        return hidden[range(hidden.shape[0]), seq_len]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import torch.nn.functional as F
        texts = [t if t and t.strip() else " " for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            enc = self._tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with self._torch.no_grad():
                res = self._model(**enc, use_cache=False)
            embs = self._pool(res.last_hidden_state, enc["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
            out.extend(embs.cpu().numpy().tolist())
        return out

    def embed_documents(self, texts): return self._encode(texts)
    def embed_query(self, text):      return self._encode([text])[0]


def create_embeddings(config: RAGConfig) -> Embeddings:
    if config.embed_provider == "qwen3_local":
        return Qwen3LocalEmbeddings(config.embed_model)
    raise ValueError(f"Unknown embed provider: {config.embed_provider}")


# ── Chunk / ColBERT stores ───────────────────────────────────────────────────
class ChunkStore:
    def __init__(self, json_path: str):
        logger.info("Loading chunk store from %s", json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        self._by_id = {str(c["chunk_id"]): c for c in chunks}
        logger.info("Loaded %d chunks", len(self._by_id))

    def get(self, cid: str): return self._by_id.get(str(cid))
    def __len__(self): return len(self._by_id)


class ColBERTStore:
    """Loads ColBERT flat-list embeddings from Neo4j at startup and runs MaxSim in RAM.

    Reads Chunk.eng_embedding + eng_emb_shape / vi_embedding + vi_emb_shape written by phase 4.
    Corpus is small (41 chunks × ~T×128 × f32 ≈ few MB) — brute-force numpy MaxSim is sub-ms.
    """

    def __init__(self, driver, database: str):
        logger.info("Loading ColBERT store from Neo4j (%s)", database)
        self._en: dict[str, np.ndarray] = {}
        self._vi: dict[str, np.ndarray] = {}
        with driver.session(database=database) as session:
            recs = session.run(
                "MATCH (c:Chunk) "
                "WHERE c.eng_embedding IS NOT NULL OR c.vi_embedding IS NOT NULL "
                "RETURN c.chunk_id AS chunk_id, "
                "c.eng_embedding AS ee, c.eng_emb_shape AS es, "
                "c.vi_embedding  AS ve, c.vi_emb_shape  AS vs"
            )
            for r in recs:
                cid = str(r["chunk_id"])
                if r["ee"] and r["es"]:
                    self._en[cid] = np.asarray(r["ee"], dtype=np.float32).reshape(tuple(r["es"]))
                if r["ve"] and r["vs"]:
                    self._vi[cid] = np.asarray(r["ve"], dtype=np.float32).reshape(tuple(r["vs"]))
        logger.info("ColBERT store ready — %d EN, %d VI", len(self._en), len(self._vi))

    def search(self, query_emb: np.ndarray, lang: str, k: int) -> list[tuple[str, float]]:
        store = self._vi if lang == "vi" else self._en
        if not store:
            return []
        scored = [(cid, maxsim_score(query_emb, emb)) for cid, emb in store.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ── Entity extraction helpers ────────────────────────────────────────────────
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    for sc, ec in [("{", "}"), ("[", "]")]:
        s, e = text.find(sc), text.rfind(ec)
        if s != -1 and e != -1 and e > s:
            return text[s:e + 1]
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


# ── Trace dataclasses ────────────────────────────────────────────────────────
@dataclass
class ChunkTrace:
    chunk_id: str
    title: str
    hierarchy: str
    start_page: int
    end_page: int
    channels: list[str]
    rrf_score: float


@dataclass
class RAGResult:
    answer: str
    language: str
    traces: list[ChunkTrace]
    graph_triples: list[dict]
    community_summaries: list[str]
    timings: dict[str, float]
    config_snapshot: dict
    detected_topic: str | None = None


def reciprocal_rank_fusion(ranked_lists: dict[str, list[tuple[str, float]]], k: int = 60):
    scores: dict[str, float] = defaultdict(float)
    channels: dict[str, set[str]] = defaultdict(set)
    for ch, lst in ranked_lists.items():
        for rank, (cid, _) in enumerate(lst, start=1):
            scores[cid] += 1.0 / (k + rank)
            channels[cid].add(ch)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused, dict(channels)


def invoke_with_retry(llm: BaseChatModel, msgs: list, label: str = "llm") -> Any:
    """Invoke LLM, retrying forever with capped exponential backoff. Never gives up."""
    backoff_cap = 60
    attempt = 0
    while True:
        attempt += 1
        try:
            t0 = time.time()
            resp = llm.invoke(msgs)
            if attempt > 1:
                logger.info("%s succeeded on attempt %d (%.1fs)", label, attempt, time.time() - t0)
            return resp
        except Exception as e:
            delay = min(backoff_cap, 2 ** min(attempt - 1, 6))
            logger.warning("%s failed (attempt %d): %s — retrying in %ds",
                           label, attempt, str(e)[:200], delay)
            time.sleep(delay)


def embed_with_retry(embeddings: Embeddings, text: str, label: str = "embed",
                     max_attempts: int = 3) -> list[float]:
    """Embedding calls can fail transiently (OOM, CUDA hiccups). Short bounded retry."""
    for attempt in range(1, max_attempts + 1):
        try:
            return embeddings.embed_query(text)
        except Exception as e:
            if attempt == max_attempts:
                logger.error("%s failed after %d attempts: %s", label, attempt, str(e)[:200])
                raise
            delay = min(8, 2 ** (attempt - 1))
            logger.warning("%s failed (attempt %d): %s — retrying in %ds",
                           label, attempt, str(e)[:200], delay)
            time.sleep(delay)
    return []


def translate_to_english(text: str, llm: BaseChatModel) -> str:
    resp = invoke_with_retry(llm, [
        SystemMessage(content="You are a translator. Translate the Vietnamese text to English. Return only the translation."),
        HumanMessage(content=text),
    ], label="translate_to_english")
    return resp.content.strip()


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════════════
class GraphRAGPipeline:
    """4-channel Graph RAG: ColBERT MaxSim + Qwen dense ANN + Lucene BM25 + KG traversal."""

    def __init__(self, config: RAGConfig, chunks_json_path: str):
        self.config = config
        self.chunk_store = ChunkStore(chunks_json_path)
        self.driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )
        self.colbert_store = ColBERTStore(self.driver, config.neo4j_database)
        self.llm = create_llm(config)
        self.embeddings = create_embeddings(config)
        self._validate_embedding_dimension()
        logger.info(
            "GraphRAGPipeline ready (llm=%s, db=%s, colbert_en=%d)",
            config.llm_model, config.neo4j_database, len(self.colbert_store._en),
        )

    _embed_dim_validated: int | None = None

    def _validate_embedding_dimension(self):
        if GraphRAGPipeline._embed_dim_validated == id(type(self.embeddings)):
            return
        actual = len(self.embeddings.embed_query("test"))
        expected = 1024
        try:
            with self.driver.session(database=self.config.neo4j_database) as s:
                rec = s.run(
                    "SHOW INDEXES YIELD name, options "
                    "WHERE name = 'chunk_embedding_index' RETURN options"
                ).single()
                if rec:
                    expected = rec["options"].get("indexConfig", {}).get("vector.dimensions", 1024)
        except Exception as e:
            logger.warning("Could not query vector index dim: %s", e)
        if actual != expected:
            raise ValueError(f"Embed dim mismatch: model {actual}d vs index {expected}d")
        GraphRAGPipeline._embed_dim_validated = id(type(self.embeddings))
        logger.info("Embedding dimension OK: %dd", actual)

    def _run_cypher(self, query: str, params: dict | None = None,
                    max_attempts: int = 3) -> list[dict]:
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self.driver.session(database=self.config.neo4j_database) as s:
                    return [r.data() for r in s.run(query, params or {})]
            except Exception as e:
                last_err = e
                if attempt == max_attempts:
                    break
                delay = min(4, 2 ** (attempt - 1))
                logger.warning("Cypher failed (attempt %d): %s — retrying in %ds",
                               attempt, str(e)[:200], delay)
                time.sleep(delay)
        assert last_err is not None
        raise last_err

    # ── Channel 1: Qwen dense semantic (vector index) ─────────────────────────
    @observe(name="semantic_search", capture_input=False, capture_output=False)
    def _semantic_search(self, query: str, lang: str, k: int | None = None,
                         topic: str | None = None) -> list[tuple[str, float]]:
        k = k or self.config.vector_search_k
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "lang": lang, "k": k, "topic": topic},
                metadata={"channel": "qwen_dense_semantic"},
            )
        try:
            emb = embed_with_retry(self.embeddings, query, label="dense_query_embed")
            index = "chunk_vi_embedding_index" if lang == "vi" else "chunk_embedding_index"
            kb = k * 3 if topic else k
            recs = self._run_cypher(
                f"CALL db.index.vector.queryNodes('{index}', $kb, $emb) YIELD node, score "
                "RETURN node.chunk_id AS chunk_id, node.topic AS topic, score",
                {"kb": kb, "emb": emb},
            )
            scored = [(str(r["chunk_id"]), r.get("topic"), float(r["score"])) for r in recs]

            if topic:
                topic_boost = 0.15
                reranked = sorted(
                    scored,
                    key=lambda x: x[2] + (topic_boost if x[1] == topic else 0.0),
                    reverse=True,
                )
                results = [(cid, sc) for cid, _, sc in reranked[:k]]
            else:
                results = [(cid, sc) for cid, _, sc in scored[:k]]

            if langfuse_context:
                langfuse_context.update_current_observation(
                    output={"count": len(results), "top_chunks": [r[0] for r in results[:3]]},
                    metadata={"channel": "qwen_dense_semantic", "index": index,
                              "topic_boost_active": bool(topic)},
                )
            return results
        except Exception as e:
            logger.warning("Dense semantic failed: %s", e)
            if langfuse_context:
                langfuse_context.update_current_observation(output={"count": 0, "error": str(e)})
            return []

    # ── Channel 2: ColBERT semantic ──────────────────────────────────────────
    @observe(name="colbert_search", capture_input=False, capture_output=False)
    def _colbert_search(self, query: str, lang: str, k: int | None = None) -> list[tuple[str, float]]:
        k = k or self.config.colbert_search_k
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "lang": lang, "k": k},
                metadata={"channel": "colbert_semantic"},
            )
        try:
            q_emb = colbert_encode([query], is_query=True)[0]
            results = self.colbert_store.search(q_emb, lang, k)
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output={"count": len(results), "top_chunks": [r[0] for r in results[:3]]},
                    metadata={"channel": "colbert_semantic",
                              "query_token_count": int(q_emb.shape[0]) if hasattr(q_emb, "shape") else None},
                )
            return results
        except Exception as e:
            logger.warning("ColBERT failed: %s", e)
            if langfuse_context:
                langfuse_context.update_current_observation(output={"count": 0, "error": str(e)})
            return []

    # ── Channel 3: Lucene BM25 ───────────────────────────────────────────────
    @observe(name="fulltext_search", capture_input=False, capture_output=False)
    def _fulltext_search(self, query: str, k: int | None = None,
                         topic: str | None = None) -> list[tuple[str, float]]:
        k = k or self.config.keyword_search_k
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query, "k": k, "topic": topic},
                metadata={"channel": "lucene_bm25"},
            )
        q = sanitize_lucene(query)
        if not q.strip():
            return []
        try:
            kb = k * 3 if topic else k
            recs = self._run_cypher(
                "CALL db.index.fulltext.queryNodes('chunk_fulltext_index', $q) YIELD node, score "
                "RETURN node.chunk_id AS chunk_id, node.topic AS topic, score LIMIT $kb",
                {"q": q, "kb": kb},
            )
            scored = [(str(r["chunk_id"]), r.get("topic"), float(r["score"])) for r in recs]

            if topic and scored:
                max_score = max(s for _, _, s in scored) or 1.0
                topic_boost = 0.15 * max_score
                reranked = sorted(
                    scored,
                    key=lambda x: x[2] + (topic_boost if x[1] == topic else 0.0),
                    reverse=True,
                )
                results = [(cid, sc) for cid, _, sc in reranked[:k]]
            else:
                results = [(cid, sc) for cid, _, sc in scored[:k]]
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output={"count": len(results), "top_chunks": [r[0] for r in results[:3]]},
                    metadata={"channel": "lucene_bm25", "topic_filter_active": bool(topic)},
                )
            return results
        except Exception as e:
            logger.warning("Fulltext failed: %s", e)
            if langfuse_context:
                langfuse_context.update_current_observation(output={"count": 0, "error": str(e)})
            return []

    # ── Channel 4: KG graph retrieval ────────────────────────────────────────
    @observe(name="graph_retrieval", capture_input=False, capture_output=False)
    def _graph_retrieval(self, query_en: str, query_embedding: list[float]):
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"query": query_en},
                metadata={"channel": "qwen_kg_graph"},
            )
        entities = self._extract_entities(query_en)
        if not entities:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output={"chunk_count": 0, "triple_count": 0, "community_count": 0,
                            "entity_count": 0, "reason": "no_entities_extracted"},
                )
            return [], [], []
        sanitized = [sanitize_lucene(e) for e in entities if sanitize_lucene(e).strip()]
        if not sanitized:
            return [], [], []

        try:
            matched = self._run_cypher(
                "UNWIND $names AS name "
                "CALL db.index.fulltext.queryNodes('entity_fulltext_index', name) "
                "YIELD node, score WHERE score > 0.5 "
                "RETURN node.name AS name, node.type AS type, node.description AS description, score "
                "ORDER BY score DESC LIMIT $limit",
                {"names": sanitized, "limit": len(sanitized) * 5},
            )
        except Exception as e:
            logger.warning("Entity fulltext failed: %s", e)
            matched = []
        if not matched:
            return [], [], []

        seen: set[str] = set()
        entity_names = [e["name"] for e in matched if not (e["name"] in seen or seen.add(e["name"]))]

        all_triples: list[dict] = []
        chunk_counts: dict[str, int] = defaultdict(int)
        try:
            recs = self._run_cypher(
                "UNWIND $names AS name "
                "MATCH (e:Entity {name: name})-[r:RELATES_TO]-(n:Entity) "
                "RETURN "
                "  CASE WHEN startNode(r) = e THEN e.name ELSE n.name END AS source, "
                "  r.relation AS relation, "
                "  CASE WHEN startNode(r) = e THEN n.name ELSE e.name END AS target, "
                "  r.detail AS detail, r.source_chunk AS source_chunk, r.pages AS pages",
                {"names": entity_names},
            )
            all_triples.extend(recs)
        except Exception as e:
            logger.warning("Graph traversal failed: %s", e)

        try:
            recs = self._run_cypher(
                "UNWIND $names AS name "
                "MATCH (e:Entity {name: name})-[:EXTRACTED_FROM]->(c:Chunk) "
                "RETURN c.chunk_id AS chunk_id",
                {"names": entity_names},
            )
            for r in recs:
                chunk_counts[str(r["chunk_id"])] += 1
        except Exception as e:
            logger.warning("Provenance query failed: %s", e)

        for t in all_triples:
            sc = t.get("source_chunk")
            if sc:
                chunk_counts[str(sc)] += 1

        ranked = sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True)
        max_c = ranked[0][1] if ranked else 1
        chunk_results = [(cid, cnt / max_c) for cid, cnt in ranked][:self.config.graph_traversal_top_k]

        communities: list[dict] = []
        try:
            recs = self._run_cypher(
                "CALL db.index.vector.queryNodes('community_embedding_index', 3, $emb) "
                "YIELD node, score WHERE score > 0.3 "
                "RETURN node.title AS title, node.summary AS summary, score",
                {"emb": query_embedding},
            )
            communities = recs
        except Exception as e:
            logger.warning("Community search failed: %s", e)

        seen_t: set[str] = set()
        unique_triples: list[dict] = []
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
                    "top_entities": entity_names[:5],
                },
                metadata={"channel": "qwen_kg_graph"},
            )
        return chunk_results, unique_triples, communities

    @observe(name="entity_extraction", capture_input=False, capture_output=False)
    def _extract_entities(self, query_en: str) -> list[str]:
        if langfuse_context:
            langfuse_context.update_current_observation(input={"query": query_en})
        prompt = (
            "Extract up to 10 key entity names from this query for searching a "
            "knowledge graph about rice production (equipment, processes, inputs, "
            "crop stages, pests, parameters, products, principles, varieties).\n"
            "Rules: output ONLY the JSON object, no prose, no code fences, no repetition "
            "of the query. Normalize Unicode: write 'CO2' not 'CO\u2082', 'P2O5' not 'P\u2082O\u2085', "
            "'m2'/'m3' not 'm\u00b2'/'m\u00b3'. Strip Vietnamese unit words (công, sào, tấn) "
            "and keep only the concept.\n\n"
            f"Query: {query_en}"
            '\n\nJSON: {"entities": ["entity1", "entity2"]}'
        )
        bounded_llm = self.llm.bind(max_tokens=256)
        max_empty_attempts = 5
        for attempt in range(1, max_empty_attempts + 1):
            logger.info("Entity extraction LLM call (attempt %d/%d)...",
                        attempt, max_empty_attempts)
            t0 = time.time()
            raw = invoke_with_retry(bounded_llm, [HumanMessage(content=prompt)],
                                    label="entity_extraction")
            logger.info("Entity extraction returned in %.1fs", time.time() - t0)
            ents = _parse_json_entity_list(raw.content)
            if ents:
                if langfuse_context:
                    langfuse_context.update_current_observation(
                        output={"entities": ents, "attempt": attempt},
                    )
                return ents
            logger.warning("Entity extraction empty (attempt %d/%d)",
                           attempt, max_empty_attempts)
            if attempt < max_empty_attempts:
                time.sleep(min(8, 2 ** (attempt - 1)))

        logger.warning("Entity extraction gave up after %d empty results — "
                       "graph channel will contribute nothing for this query",
                       max_empty_attempts)
        if langfuse_context:
            langfuse_context.update_current_observation(
                output={"entities": [], "gave_up_after": max_empty_attempts,
                        "reason": "llm_returned_no_entities"},
            )
        return []

    # ── Sibling expansion ────────────────────────────────────────────────────
    def _expand_siblings(self, chunk_ids: list[str]) -> list[str]:
        if not self.config.enable_sibling_expansion:
            return chunk_ids
        expanded = list(chunk_ids)
        for cid in chunk_ids:
            try:
                recs = self._run_cypher(
                    "MATCH (parent)-[:CONTAINS]->(t:Chunk {chunk_id: $cid}) "
                    "MATCH (parent)-[:CONTAINS]->(s:Chunk) "
                    "WHERE s.chunk_id <> $cid "
                    "RETURN s.chunk_id AS chunk_id",
                    {"cid": cid},
                )
                for r in recs:
                    sib = str(r["chunk_id"])
                    if sib not in expanded:
                        expanded.append(sib)
            except Exception as e:
                logger.warning("Sibling expansion failed for %s: %s", cid, e)
        return expanded

    # ── Context assembly ─────────────────────────────────────────────────────
    def _assemble_context(self, chunk_ids: list[str], lang: str, mode: str,
                          triples: list[dict], communities: list[dict]) -> str:
        budget = self.config.max_context_tokens
        used = 0
        parts: list[str] = []

        def tok(t): return len(t) // 3

        if triples:
            lines = []
            for t in triples[:30]:
                line = f"  {t.get('source','?')} --[{t.get('relation','?')}]--> {t.get('target','?')}"
                if t.get("detail"):
                    line += f": {t['detail']}"
                lines.append(line)
            g = "=== Knowledge Graph Triples ===\n" + "\n".join(lines)
            used += tok(g); parts.append(g)

        if communities:
            lines = [f"  [{c.get('title','')}]: {c.get('summary','')}" for c in communities[:5]]
            cm = "=== Community Summaries ===\n" + "\n".join(lines)
            used += tok(cm); parts.append(cm)

        if lang == "en":
            tk, hk, sk, txk = "eng_chunk_title", "eng_chunk_hierarchy", "eng_summary", "eng_text"
        else:
            tk, hk, sk, txk = "vi_chunk_title", "hierarchy_id", "vi_summary", "vi_text"

        for cid in chunk_ids:
            chunk = self.chunk_store.get(cid)
            if chunk is None:
                continue
            title = chunk.get(tk, f"Chunk {cid}")
            hier = chunk.get(hk, "")
            summary = chunk.get(sk, "")
            if not _is_valid_summary_for_context(summary):
                summary = chunk.get(txk, "") or chunk.get("vi_summary", "")
                if not summary or not _is_valid_summary_for_context(summary):
                    continue
            fulltext = chunk.get(txk, "")
            tables = chunk.get("tables", [])
            pages = f"(pp. {chunk.get('start_page','?')}-{chunk.get('end_page','?')})"
            base = f"\n--- {title} {pages} ---\n[{hier}]\n{summary}"
            if used + tok(base) > budget:
                break
            used += tok(base)
            text = base
            ft = f"\n[Full text]\n{fulltext}" if self.config.include_full_text and fulltext else ""
            tb = "\n" + "\n".join(tables) if tables else ""
            if ft and used + tok(ft) <= budget:
                text += ft; used += tok(ft)
            if tb and used + tok(tb) <= budget:
                text += tb; used += tok(tb)
            parts.append(text)

        return "\n".join(parts)

    # ── Generation ───────────────────────────────────────────────────────────
    @observe(name="generate_answer", as_type="generation", capture_input=False, capture_output=False)
    def _generate_answer(self, query: str, context: str, lang: str, mode: str) -> str:
        if langfuse_context:
            langfuse_context.update_current_observation(
                model=self.config.llm_model,
                input={"query": query, "context_chars": len(context), "lang": lang, "mode": mode},
                metadata={"llm_provider": self.config.llm_provider, "lang": lang, "mode": mode},
            )
        if lang == "vi":
            sys_prompt = (
                "Ban la mot tro ly nong nghiep thuc te. Tra loi CHI dua tren ngu canh duoc cung cap.\n\n"
                "CAU TRUC TRA LOI (bat buoc theo thu tu):\n"
                "**Phan tich tung buoc:** Danh so ro rang, su dung ✅ cho moi buoc kha thi. "
                "Suy luan ro rang — neu co phep tinh, viet ra tung buoc.\n"
                "**Luong/Don vi:** Neu bai viet dung don vi khac voi cau hoi, doi ra don vi quen thuoc "
                "(vi du: kg/ha → kg/cong/sao). Hien thi ro phep tinh.\n"
                "**Trich dan nguon:** Ghi so trang sau moi thong tin quan trong, vi du (tr. 45-47).\n"
                "**Canh bao:** Neu co thuoc tru sau/hoa chat, ghi ro ⚠️ lieu luong an toan va thoi gian cach ly.\n"
                "**Giai thich tu ngu:** Neu dung thuat ngu ky thuat, giai thich trong ngoac don.\n"
                "**Neu khong chac:** Neu ngu canh chua du thong tin, ghi ro 'Tai lieu khong neu ro diem nay.'\n"
                "**Bai hoc chinh:** Mot cau tom tat DUY NHAT, rut ra TRUC TIEP tu phan phan tich tung buoc. "
                "Viet cau nay SAU CUNG — khong bao gio doan truoc khi hoan thanh cac buoc phan tich.\n\n"
                "TU KIEM TRA (lam tham truoc khi viet cau tra loi cuoi cung):\n"
                "- Doc lai moi con so, ngay thang, phep doi don vi va phep tinh.\n"
                "- Doi chieu tung con so voi ngu canh — neu nguon ghi 70 kg/ha, khong duoc viet 700 kg/ha.\n"
                "- Neu phat hien sai, sua ngay tai cho truoc khi xuat.\n"
                "- KHONG de cap buoc tu kiem tra nay trong cau tra loi.\n\n"
                "SUY LUAN SO LIEU: Khi ngu canh co nhieu gia tri khac nhau cho cung mot phep do, "
                "xac dinh gia tri nao ap dung cho cau hoi cu the — khong chon tuy y.\n\n"
                "PHAN BIET QUY TRINH: Khi ngu canh mo ta nhieu quy trinh, chi tra loi theo dung quy trinh "
                "cau hoi hoi. Khong tron lan cac buoc tu cac phuong phap khac nhau.\n\n"
                "KIEM TRA TRUOC KHI TU CHOI: Truoc khi noi tai lieu khong de cap, kiem tra ky moi chi tiet "
                "trong ngu canh (bao gom chu thich hinh, bang, gia tri so trong van ban). "
                "Neu thong tin that su khong co, noi ro. KHONG doan hoac suy dien.\n\n"
                "Tra loi bang tieng Viet, van phong gian di, thuc te."
            )
        else:
            sys_prompt = (
                "You are a practical agricultural advisor helping rice farmers. "
                "Answer ONLY based on the provided context. "
                "Write as if explaining to a farmer standing in their field — clear, direct, actionable.\n\n"
                "REQUIRED ANSWER STRUCTURE (follow this order):\n"
                "**Step-by-Step Analysis:** Numbered steps, each starting with ✅. Reason clearly — "
                "if there is a calculation, show every step. If the question involves a number, "
                "derive it here before stating it anywhere else.\n"
                "**Amounts & Units:** If the source uses different units than the question implies, "
                "convert to practical field amounts (e.g., kg/ha → kg per standard plot, "
                "L/ha → litres per backpack sprayer). Show the conversion.\n"
                "**Sources:** After each key fact, cite the page range: (pp. 45-47).\n"
                "**Warnings:** For any pesticide, fertiliser rate, or chemical — prefix with ⚠️ "
                "and include safe handling notes and pre-harvest interval if mentioned.\n"
                "**Plain language:** If you must use a technical term, explain it in parentheses.\n"
                "**If context is thin:** Say clearly: 'The manual does not cover this specifically — "
                "please consult your local extension officer.'\n"
                "**Key Takeaway:** ONE bold sentence summarising the single most important action or number. "
                "This must be derived DIRECTLY from your Step-by-Step Analysis above — never guess it "
                "before completing the steps, and never contradict a number you just derived.\n\n"
                "SELF-CHECK (silently, before writing the final answer):\n"
                "- Re-read every number, date, unit conversion, and calculation you are about to write.\n"
                "- Cross-check each figure against the source context — if context says 70 kg/ha, "
                "do not write 700 kg/ha.\n"
                "- Verify unit conversions step-by-step.\n"
                "- Confirm your Key Takeaway uses the SAME numbers as your Step-by-Step Analysis.\n"
                "- If you spot an inconsistency, fix it in place before outputting.\n"
                "- Do NOT mention this self-check in your answer.\n\n"
                "NUMERIC REASONING (critical): When the context contains multiple different numeric "
                "values for the same measurement (e.g., different durations, doses, thresholds), do not "
                "pick one arbitrarily. Reason about which value applies to the specific question by "
                "considering conditions, season, scenario, or source. Explain briefly before stating "
                "the final number.\n\n"
                "PROTOCOL DISTINCTION (critical): When the context describes multiple protocols for the "
                "same task (e.g., conventional broadcasting vs mechanised sowing), identify which "
                "protocol the question asks about and answer ONLY from that protocol. Do not mix "
                "stages, doses, or steps from different methods.\n\n"
                "ANTI-DENIAL CHECK (critical): Before stating that the manual 'does not specify' or "
                "'does not mention' something, cross-check every detail in the context thoroughly — "
                "equipment specifications, figure captions, table entries, parenthetical notes, and "
                "numeric values embedded in prose. Re-read the context at least twice for the specific "
                "piece of information before concluding absence. However, if after thorough checking "
                "the information is genuinely not present, state clearly that the manual does not "
                "contain it. Do NOT guess, infer, or fabricate values not explicitly stated. "
                "Accuracy is paramount — a correct 'not specified' is always better than a "
                "hallucinated answer.\n\n"
                "Answer in English. Be concise but complete — a farmer should be able to act on this immediately."
            )

        full = f"{context}\n\nQuestion: {query}"
        msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=full)]
        bounded_llm = self.llm.bind(max_tokens=1500)
        answer = invoke_with_retry(bounded_llm, msgs, label="answer_generation").content.strip()
        if langfuse_context:
            langfuse_context.update_current_observation(
                output=answer[:500] if len(answer) > 500 else answer,
                usage={"input": len(context) // 4, "output": len(answer) // 4},
            )
        return answer

    # ── Parallel retrieval (all 4 channels) ────────────────────────────────
    @observe(name="parallel_retrieval", capture_input=False, capture_output=False)
    def _parallel_retrieval(self, user_query: str, query_en: str, lang: str,
                            query_embedding: list[float], detected_topic: str | None):
        if langfuse_context:
            langfuse_context.update_current_observation(
                input={"channels": ["semantic_dense", "colbert_maxsim",
                                    "fulltext_bm25", "graph_kg"],
                       "topic": detected_topic or "standard"},
            )
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_sem  = ex.submit(copy_context().run, self._semantic_search,
                               user_query, lang, None, detected_topic)
            f_col  = ex.submit(copy_context().run, self._colbert_search, query_en, "en")
            f_bm25 = ex.submit(copy_context().run, self._fulltext_search,
                               query_en, None, detected_topic)
            f_gr   = ex.submit(copy_context().run, self._graph_retrieval,
                               query_en, query_embedding)
            semantic_ranked = f_sem.result()
            colbert_ranked  = f_col.result()
            fulltext_ranked = f_bm25.result()
            logger.info("Waiting for graph channel...")
            tw = time.time()
            graph_chunks, triples, communities = f_gr.result()
            logger.info("Graph done in %.1fs (triples: %d, chunks: %d)",
                        time.time() - tw, len(triples), len(graph_chunks))
        logger.info(
            "Retrieval — dense: %d, colbert: %d, bm25: %d, graph: %d",
            len(semantic_ranked), len(colbert_ranked),
            len(fulltext_ranked), len(graph_chunks),
        )
        if langfuse_context:
            langfuse_context.update_current_observation(
                output={
                    "dense_count": len(semantic_ranked),
                    "colbert_count": len(colbert_ranked),
                    "bm25_count": len(fulltext_ranked),
                    "graph_chunk_count": len(graph_chunks),
                    "triple_count": len(triples),
                    "community_count": len(communities),
                },
            )
        return semantic_ranked, colbert_ranked, fulltext_ranked, graph_chunks, triples, communities

    # ── Main query ───────────────────────────────────────────────────────────
    @observe(name="rag_pipeline", capture_input=False, capture_output=False)
    def query(self, user_query: str, mode: str | None = None) -> RAGResult:
        timings: dict[str, float] = {}
        mode = mode or self.config.mode

        _lf = _get_langfuse_client() if _LANGFUSE_AVAILABLE else None
        if _lf and not hasattr(_lf, "start_as_current_observation"):
            _lf = None  # v2 — inline spans unavailable, rely on @observe only

        if langfuse_context:
            langfuse_context.update_current_trace(
                input={"query": user_query, "mode": mode},
                metadata={"config": self._config_snapshot()},
                tags=["4-channel-rag", f"mode:{mode}"],
            )

        if not user_query or not user_query.strip():
            return RAGResult(answer="Please provide a non-empty query.", language="en",
                             traces=[], graph_triples=[], community_summaries=[],
                             timings={}, config_snapshot=self._config_snapshot())
        if len(user_query) > MAX_QUERY_LENGTH:
            user_query = user_query[:MAX_QUERY_LENGTH]

        # 1. Language detection (span)
        t0 = time.time()
        if _lf:
            with _lf.start_as_current_observation(
                as_type="span", name="language_detection",
                input={"query": user_query[:300]},
            ) as _obs:
                lang = detect_language(user_query)
                _obs.update(output={"lang": lang})
        else:
            lang = detect_language(user_query)
        timings["language_detection"] = time.time() - t0
        logger.info("Query language: %s", lang)

        # 2. Translation (span, only for VI queries)
        t0 = time.time()
        if lang == "vi":
            if _lf:
                with _lf.start_as_current_observation(
                    as_type="span", name="translation",
                    input={"query_vi": user_query[:300]},
                ) as _obs:
                    query_en = translate_to_english(user_query, self.llm)
                    _obs.update(output={"query_en": query_en[:300]})
            else:
                query_en = translate_to_english(user_query, self.llm)
        else:
            query_en = user_query
        timings["translation"] = time.time() - t0

        # 3. Topic detection (keyword match, fast)
        t0 = time.time()
        if _lf:
            with _lf.start_as_current_observation(
                as_type="span", name="topic_detection",
                input={"query_en": query_en[:300]},
            ) as _obs:
                detected_topic = detect_query_topic(query_en)
                _obs.update(output={"topic": detected_topic or "standard"})
        else:
            detected_topic = detect_query_topic(query_en)
        timings["topic_detection"] = time.time() - t0
        if detected_topic:
            logger.info("Detected topic: %s", detected_topic)
        if langfuse_context:
            langfuse_context.update_current_trace(
                tags=["4-channel-rag", f"mode:{mode}", f"topic:{detected_topic or 'standard'}",
                      f"lang:{lang}"],
            )

        # 4. Shared dense embedding (span — reused by dense semantic + community search)
        t0 = time.time()
        if _lf:
            with _lf.start_as_current_observation(
                as_type="span", name="query_embedding",
                input={"text": (user_query if lang == "en" else query_en)[:300]},
            ) as _obs:
                query_embedding = embed_with_retry(
                    self.embeddings, user_query if lang == "en" else query_en,
                    label="shared_query_embed",
                )
                _obs.update(
                    output={"embedding_dim": len(query_embedding)},
                    metadata={"model": self.config.embed_model},
                )
        else:
            query_embedding = embed_with_retry(
                self.embeddings, user_query if lang == "en" else query_en,
                label="shared_query_embed",
            )
        timings["embedding"] = time.time() - t0

        # 5. Four channels in parallel
        t_r = time.time()
        (semantic_ranked, colbert_ranked, fulltext_ranked,
         graph_chunks, triples, communities) = self._parallel_retrieval(
            user_query, query_en, lang, query_embedding, detected_topic,
        )
        timings["retrieval_parallel"] = time.time() - t_r

        # RRF fusion
        t0 = time.time()
        ranked = {k: v for k, v in {
            "semantic_dense":   semantic_ranked,
            "semantic_colbert": colbert_ranked,
            "fulltext_bm25":    fulltext_ranked,
            "graph":            graph_chunks,
        }.items() if v}
        if not ranked:
            return RAGResult(answer="No relevant information found.", language=lang,
                             traces=[], graph_triples=[], community_summaries=[],
                             timings=timings, config_snapshot=self._config_snapshot(),
                             detected_topic=detected_topic)
        if _lf:
            with _lf.start_as_current_observation(
                as_type="span", name="rrf_fusion",
                input={"active_channels": list(ranked.keys()),
                       "channel_counts": {k: len(v) for k, v in ranked.items()}},
            ) as _obs:
                fused, channel_map = reciprocal_rank_fusion(ranked, k=self.config.rrf_k)
                _obs.update(
                    output={"fused_count": len(fused),
                            "top_chunks": [c for c, _ in fused[:5]]},
                    metadata={"rrf_k": self.config.rrf_k,
                              "rrf_min_score": self.config.rrf_min_score},
                )
        else:
            fused, channel_map = reciprocal_rank_fusion(ranked, k=self.config.rrf_k)
        timings["rrf_fusion"] = time.time() - t0

        if self.config.rrf_min_score > 0:
            fused = [
                (cid, sc) for cid, sc in fused
                if sc >= self.config.rrf_min_score or len(channel_map.get(cid, set())) > 1
            ]
        rrf_scores = dict(fused)
        top_ids = [cid for cid, _ in fused[:self.config.graph_traversal_top_k]]

        t0 = time.time()
        final_ids = self._expand_siblings(top_ids)
        timings["sibling_expansion"] = time.time() - t0

        t0 = time.time()
        context = self._assemble_context(final_ids, lang, mode, triples, communities)
        timings["context_assembly"] = time.time() - t0

        t0 = time.time()
        answer = self._generate_answer(user_query, context, lang, mode)
        timings["generation"] = time.time() - t0

        traces = []
        for cid in final_ids:
            chunk = self.chunk_store.get(cid)
            if chunk is None:
                continue
            traces.append(ChunkTrace(
                chunk_id=cid,
                title=chunk.get("eng_chunk_title" if lang == "en" else "vi_chunk_title", f"Chunk {cid}"),
                hierarchy=chunk.get("eng_chunk_hierarchy" if lang == "en" else "hierarchy_id", ""),
                start_page=chunk.get("start_page", 0),
                end_page=chunk.get("end_page", 0),
                channels=sorted(channel_map.get(cid, set())),
                rrf_score=rrf_scores.get(cid, 0.0),
            ))

        timings["total"] = sum(timings.values())
        logger.info("Pipeline completed in %.2fs", timings["total"])

        if langfuse_context:
            langfuse_context.update_current_trace(
                output={
                    "answer": answer[:500] if len(answer) > 500 else answer,
                    "language": lang,
                },
                metadata={
                    "timings": {k: round(v, 3) for k, v in timings.items()},
                    "chunks_retrieved": len(final_ids),
                    "triples_found": len(triples),
                    "communities_found": len(communities),
                    "channels_active": list(ranked.keys()),
                    "detected_topic": detected_topic,
                },
                tags=["4-channel-rag", f"mode:{mode}", f"lang:{lang}",
                      f"topic:{detected_topic or 'standard'}"],
            )

        return RAGResult(
            answer=answer, language=lang, traces=traces,
            graph_triples=triples,
            community_summaries=[f"[{c.get('title','')}]: {c.get('summary','')}" for c in communities],
            timings=timings, config_snapshot=self._config_snapshot(),
            detected_topic=detected_topic,
        )

    def _config_snapshot(self) -> dict:
        c = self.config
        return {
            "llm_provider": c.llm_provider, "llm_model": c.llm_model,
            "embed_provider": c.embed_provider, "embed_model": c.embed_model,
            "neo4j_database": c.neo4j_database,
            "vector_search_k": c.vector_search_k,
            "colbert_search_k": c.colbert_search_k,
            "keyword_search_k": c.keyword_search_k,
            "graph_traversal_top_k": c.graph_traversal_top_k,
            "rrf_k": c.rrf_k, "rrf_min_score": c.rrf_min_score,
            "mode": c.mode, "max_context_tokens": c.max_context_tokens,
            "include_full_text": c.include_full_text,
            "enable_sibling_expansion": c.enable_sibling_expansion,
        }

    def close(self):
        self.driver.close()
        logger.info("Pipeline closed")


def init_pipeline(chunks_json: str | None = None) -> tuple[GraphRAGPipeline, RAGConfig]:
    """Convenience helper used by benchmark_runner.py and notebooks."""
    config = RAGConfig.from_env()
    chunks = chunks_json or str((_HERE / OUTPUT_CHUNKS_VI).resolve())
    return GraphRAGPipeline(config, chunks), config


# ═════════════════════════════════════════════════════════════════════════════
# Phase5Pipeline — BasePipeline wrapper for Streamlit app integration
# ═════════════════════════════════════════════════════════════════════════════
class Phase5Pipeline(BasePipeline):
    """BasePipeline wrapper around GraphRAGPipeline for Streamlit app integration."""

    def __init__(self):
        self._initialized = False
        self._inner: GraphRAGPipeline | None = None

    def get_name(self) -> str:
        return "Phase5 4-Channel Graph RAG"

    def get_version(self) -> str:
        return "5.0-v2"

    def get_description(self) -> str:
        return ("4-channel bilingual Graph RAG: ColBERT MaxSim + Qwen3 dense ANN "
                "+ Lucene BM25 + KG traversal with topic-aware retrieval and RRF fusion")

    def initialize(self) -> None:
        if self._initialized:
            return
        cfg = RAGConfig.from_env()
        cfg.neo4j_uri = PHASE5_NEO4J_URI
        chunks_path = str(Path(OUTPUT_CHUNKS_VI).resolve())
        self._inner = GraphRAGPipeline(cfg, chunks_path)
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def run_query(self, question: str, language: str = "vi") -> PipelineResult:
        if not self._initialized:
            self.initialize()

        t0 = time.time()
        result: RAGResult = self._inner.query(question)
        elapsed = time.time() - t0

        # --- Split traces by channel into vector_context vs keyword_results ---
        traces = result.traces or []
        vector_context = []
        keyword_results = []
        rrf_fused = []

        for tr in traces:
            entry = {
                "chunk_id": tr.chunk_id,
                "title": tr.title,
                "hierarchy": tr.hierarchy,
                "start_page": tr.start_page,
                "end_page": tr.end_page,
                "channels": tr.channels,
                "rrf_score": tr.rrf_score,
            }
            rrf_fused.append(entry)
            channels_set = set(tr.channels)
            if channels_set & {"semantic_dense", "semantic_colbert", "graph"}:
                vector_context.append(entry)
            if "fulltext_bm25" in channels_set:
                keyword_results.append(entry)

        # --- Map graph triples ---
        graph_facts = []
        for t in (result.graph_triples or []):
            graph_facts.append({
                "source": t.get("source", ""),
                "relation": t.get("relation", ""),
                "target": t.get("target", ""),
                "detail": t.get("detail", ""),
                "score": 1.0,
            })

        # --- Community summaries into llm_context ---
        community_text = "\n".join(result.community_summaries or [])
        llm_context = {"community_summaries": community_text} if community_text else {}

        # --- Langfuse trace info ---
        trace_id = ""
        trace_url = ""
        if _LANGFUSE_AVAILABLE and langfuse_context:
            try:
                trace_id = langfuse_context.get_current_trace_id() or ""
                trace_url = langfuse_context.get_current_trace_url() or ""
            except Exception:
                pass

        return PipelineResult(
            question=question,
            farmer_answer=result.answer or "",
            raw_entities=[],
            aligned_entities=[],
            graph_facts=graph_facts,
            vector_context=vector_context,
            keyword_results=keyword_results,
            trace_id=trace_id,
            trace_url=trace_url,
            rrf_fused_results=rrf_fused,
            llm_context=llm_context,
            trace_data=result.timings or {},
            execution_time=elapsed,
            query_topic=result.detected_topic or "general",
        )

    def verify_database(self) -> dict:
        status = {
            "connected": False,
            "chunk_count": 0,
            "entity_count": 0,
            "indexes": [],
            "missing_indexes": [],
            "ready": False,
            "colbert_en_count": 0,
            "colbert_vi_count": 0,
        }
        required_indexes = [
            "chunk_embedding_index",
            "chunk_vi_embedding_index",
            "chunk_fulltext_index",
            "entity_fulltext_index",
            "community_embedding_index",
        ]
        cfg = RAGConfig.from_env()
        cfg.neo4j_uri = PHASE5_NEO4J_URI
        try:
            driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
            with driver.session(database=cfg.neo4j_database) as session:
                session.run("RETURN 1")
                status["connected"] = True

                result = session.run("MATCH (c:Chunk) RETURN count(c) AS cnt")
                status["chunk_count"] = result.single()["cnt"]

                result = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
                status["entity_count"] = result.single()["cnt"]

                result = session.run(
                    "MATCH (c:Chunk) WHERE c.eng_embedding IS NOT NULL "
                    "RETURN count(c) AS cnt"
                )
                status["colbert_en_count"] = result.single()["cnt"]

                result = session.run(
                    "MATCH (c:Chunk) WHERE c.vi_embedding IS NOT NULL "
                    "RETURN count(c) AS cnt"
                )
                status["colbert_vi_count"] = result.single()["cnt"]

                result = session.run("SHOW INDEXES YIELD name RETURN name")
                existing = {r["name"] for r in result}
                status["indexes"] = sorted(existing)
                status["missing_indexes"] = [
                    idx for idx in required_indexes if idx not in existing
                ]

                status["ready"] = (
                    status["chunk_count"] > 0
                    and status["colbert_en_count"] > 0
                    and len(status["missing_indexes"]) == 0
                )
            driver.close()
        except Exception as e:
            logger.error("Database verification failed: %s", e)
            status["error"] = str(e)

        return status

    def get_agent_graph(self) -> dict:
        return {
            "nodes": [
                {"id": "start", "label": "User Query", "type": "start"},
                {"id": "detect_lang", "label": "Language Detection", "type": "process"},
                {"id": "translate", "label": "Translate (if VI)", "type": "process"},
                {"id": "topic", "label": "Topic Detection", "type": "process"},
                {"id": "embed", "label": "Embed Query (Qwen3 Local)", "type": "process"},
                {"id": "semantic", "label": "Qwen3 Dense ANN", "type": "retrieval"},
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
                {"from": "translate", "to": "topic"},
                {"from": "topic", "to": "embed"},
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
        if self._inner:
            self._inner.close()


# ── CLI smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipe, cfg = init_pipeline()
    try:
        q = " ".join(sys.argv[1:]) or "What is the recommended seed rate for mechanised row sowing?"
        print(f"\n=== Query: {q}\n")
        r = pipe.query(q)
        print(f"\n=== Answer ({r.language}):\n{r.answer}\n")
        print(f"=== Traces ({len(r.traces)}):")
        for t in r.traces[:10]:
            print(f"  [{','.join(t.channels)}] rrf={t.rrf_score:.4f}  {t.chunk_id}  {t.title}  (pp.{t.start_page}-{t.end_page})")
        print(f"\n=== Timings: {json.dumps({k: round(v,2) for k,v in r.timings.items()}, indent=2)}")
    finally:
        pipe.close()
