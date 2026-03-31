# Pipeline Testing App — Full Context Document

> **Purpose:** Give ChatGPT (or any LLM) enough context to design a rigorous, reproducible testing protocol for comparing the two RAG pipelines in this app.

---

## 1. What Is This App?

A **Streamlit web application** for testing and comparing two Knowledge-Graph-augmented RAG pipelines that answer rice-farming questions from the **1 Million Hectare Project Handbook** (Mekong Delta). The app lets agricultural extension officers:

- Run **single queries** against either pipeline and see the full retrieval + answer breakdown.
- Run **batch tests** (upload CSV/JSON/TXT of questions) and export results.
- **Switch pipelines** from the sidebar to compare V7 vs ADITI head-to-head.
- View **Langfuse traces** (embedded iframe) showing every LLM call, retrieval step, and timing.
- View the **agent graph** visualizing the pipeline's step-by-step flow.
- **Comment** on individual answer sections (answer, entities, context, etc.) for evaluation.
- View **query history** per user with full replay of all retrieved context.

**Tech stack:** Streamlit, Python 3.10, Neo4j (Knowledge Graph), Qwen LLM (remote, OpenAI-compatible API), Sentence Transformers (local embeddings), Langfuse (tracing/observability), SQLite (user auth & history).

---

## 2. The Two Pipelines Under Test

Both pipelines share the same Neo4j Knowledge Graph (121 chunks, ~1086 entities from the rice handbook) and the same Qwen LLM, but differ in their retrieval and fusion strategies.

### 2.1 KG Pipeline V7 (BM25 + RRF)

**File:** `pipelines/pipeline_v7.py` — class `PipelineV7`  
**Version string:** `7.0.0`

**Pipeline steps (in order):**

1. **Extract Question Entities** — LLM call to extract key entities as a JSON array.
2. **Align Entities to KG** — Embed extracted entities with `BAAI/bge-base-en-v1.5` (384-dim), compute cosine similarity against all KG entity embeddings (cached), threshold ≥ 0.6.
3. **Multi-Source Evidence Retrieval** (parallel):
   - **KG Graph Traversal** — For each aligned entity, MATCH `(n {name})-[r:REL|MENTIONS]-(m)` in Neo4j. Returns up to 20 scored facts. Weights: REL=1.0, MENTIONS=0.25.
   - **Vector Semantic Search** — Query Neo4j vector index (`multimodal_kg_index`) with question embedding. Returns top-K chunks (default K=5) filtered by score threshold ≥ 0.8.
   - **BM25 Keyword Search** — Build Lucene query from extracted entities + query words with fuzzy matching (`term~1`). Query Neo4j fulltext index (`chunk_text_index`). Expand results with NEXT_CHUNK neighbors for context continuity.
4. **Aggregate Evidence (RRF)** — Reciprocal Rank Fusion merges vector + keyword results (NOT graph facts). Formula: `score = Σ 1/(k + rank)` where k=60. Graph facts go directly to the answer generator.
5. **Generate Farmer Answer** — LLM call with system prompt ("friendly rice farming advisor"), top-10 graph facts + top-6 RRF-fused chunks as context. Cites sources as `[DOC-N]`.

**Key characteristics:**
- Embedding model: `BAAI/bge-base-en-v1.5` (384-dim, English-centric)
- Entity alignment: single-language (English embeddings only)
- Keyword search: Lucene BM25 with fuzzy matching + NEXT_CHUNK graph neighbor expansion
- Fusion: Reciprocal Rank Fusion (vector + keyword only; graph facts bypass fusion)
- No Vietnamese translation step
- No structured data nodes
- No HyDE
- No math verification
- No LLM-based relevance reranking of BM25 results
- 2 LLM calls total: 1 entity extraction + 1 answer generation

**Langfuse trace structure:**
```
Rice_Farming_Advisor_V7 (agent)
├── Extract_Question_Entities (generation)
├── Align_Entities_To_KG (chain)
├── Multi_Source_Evidence_Retrieval (chain)
│   ├── KG_Graph_Traversal (tool)
│   ├── Vector_Semantic_Search (tool)
│   └── Keyword_BM25_Search (tool)
├── Aggregate_Evidence_RRF (chain)
└── Generate_Farmer_Answer (generation)
```

---

### 2.2 ADITI Triple-Hybrid Pipeline

**File:** `pipelines/aditi_pipeline.py` — class `AditiPipeline`  
**Version string:** `1.0`

**Pipeline steps (in order):**

1. **Multi-Stage Entity Extraction** — LLM call supporting Vietnamese + English. Extracts entities AND classifies them by type (soil_types, nutrients, seasons, activities, measurements, other). Also does regex-based numeric extraction (công, hectare, kg/ha) with automatic unit conversions (1 công = 0.13 ha). Detects entity relationships (soil+nutrient, activity+season).
2. **Bilingual Entity Alignment** — Uses `intfloat/multilingual-e5-large` (1024-dim) for entity embeddings. Translates Vietnamese entities to English via LLM, tries BOTH Vietnamese and English embeddings against KG, picks the best match. Confidence tiers: high (≥0.92), medium (0.65–0.92), low (<0.65). Medium-confidence matches include alternatives.
3. **KG Graph Traversal** — Similar to V7 but with: acronym expansion (e.g., "AWD" from "Alternate Wetting and Drying (AWD)"), relevance filtering against aligned entities, and LLM-based reranking using English translation of query. Returns top-10 most relevant facts.
4. **Triple-Hybrid Retrieval** (combines three sources):
   - **Structured Data Search** — Queries dedicated `NutrientRequirement` and `LimeRequirement` nodes in Neo4j for factual lookups (nutrient rates, lime amounts). Only triggers when intent detection identifies a nutrient or lime query.
   - **BM25 Full-Text Search** — Translates Vietnamese query to English first (since index content is English). Gets raw BM25 results, then uses **LLM relevance reranking** (0-10 score per chunk, filters chunks scoring <6/10). This is a key differentiator — ADITI uses an extra LLM call to filter irrelevant BM25 hits.
   - **Vector Search with HyDE** — Generates a hypothetical answer first (HyDE technique), then searches using that as the query embedding. Also filters out generic overview/summary chunks.
   - **Hybrid Score Fusion** — Weighted combination: 40% structured + 15% BM25 (normalized) + 45% vector. Applies mild reranking multiplier based on title match (0.7–1.3x).
5. **Extract Structured Data from Text** — Regex extraction of nutrient/lime rates from retrieved chunk text, added as additional graph facts.
6. **Generate Farmer Answer** — LLM call with detailed system prompt including mandatory unit conversion rules (1 công = 0.13 ha), step-by-step math instructions, and pre-computed measurements injected into context. Sources cited by chapter/section.
7. **Math Verification** — Post-generation step that: regex-detects arithmetic expressions in the answer (×, ÷, +, −), evaluates them with Python, and if any are wrong (>1% tolerance), sends a correction prompt to the LLM.

**Key characteristics:**
- TWO embedding models: `intfloat/multilingual-e5-large` (1024-dim, for entity alignment) + `BAAI/bge-base-en-v1.5` (384-dim, for document retrieval)
- Bilingual entity alignment (Vietnamese + English, picks best)
- Vietnamese→English translation before BM25 and vector search
- HyDE (Hypothetical Document Embeddings) for vector search
- LLM-based relevance reranking of BM25 results (extra LLM call)
- Structured data nodes (NutrientRequirement, LimeRequirement) for factual queries
- Post-generation math verification with Python eval + LLM correction
- Pre-computed unit conversions injected into LLM context
- Query intent detection (nutrient_query, lime_query, general)
- 5-7 LLM calls total: 1 entity extraction + 1 entity translation + 1 Vietnamese→English translation + 1 LLM relevance reranking + 1 HyDE generation + 1 answer generation + (optional) 1 math correction

**Langfuse trace structure:**
```
ADITI_Triple_Hybrid_Pipeline (agent)
├── extract_entities_multistage (generation)
├── align_entities_to_kg (chain)
├── traverse_kg_graph (chain)
├── triple_hybrid_retrieval (chain)
└── generate_farmer_answer (generation)
```

---

## 3. Key Architectural Differences Summary

| Feature | V7 | ADITI |
|---|---|---|
| Entity extraction | Simple JSON array | Multi-stage with type classification + numeric extraction + relationship detection |
| Entity alignment embedding | bge-base-en-v1.5 (384-dim, English) | multilingual-e5-large (1024-dim, bilingual) |
| Bilingual entity alignment | No (English only) | Yes (tries both VI and EN, picks best) |
| Vietnamese query handling | None (assumes English) | Translates to English for retrieval |
| Graph traversal | Basic REL/MENTIONS | + acronym expansion + relevance filtering + reranking |
| Structured data retrieval | No | Yes (NutrientRequirement, LimeRequirement nodes) |
| BM25 post-processing | Raw BM25 scores | LLM relevance reranking (filters <6/10) |
| BM25 expansion | NEXT_CHUNK neighbor retrieval | None |
| Vector search | Direct query embedding | HyDE (hypothetical answer embedding) |
| Fusion method | RRF (vector + keyword) | Weighted hybrid (40% structured + 15% BM25 + 45% vector) |
| Unit conversion | None | Pre-computed công→hectare injected into prompt |
| Math verification | None | Post-generation Python eval + LLM correction |
| LLM calls per query | 2 | 5–7 |
| Expected latency | Faster (~5-10s) | Slower (~15-30s due to more LLM calls) |

---

## 4. Knowledge Graph Structure (Shared)

Both pipelines query the same Neo4j database:

- **Document structure:** Document → Chapter → Section → Chunk (PART_OF / IN_CHAPTER relationships)
- **Chunk connectivity:** Chunk → NEXT_CHUNK → Chunk (sequential page order)
- **Entity nodes:** Named entities (e.g., "Alluvial Soil", "Nitrogen", "AWD") with REL and MENTIONS relationships
- **Structured data nodes (ADITI only uses these):** NutrientRequirement, LimeRequirement with specific properties (soil_type, nutrient, min_kg_ha, max_kg_ha, season, ph_threshold)
- **Indexes:** 
  - Vector index: `multimodal_kg_index` on Chunk embeddings (384-dim, bge-base-en-v1.5)
  - Fulltext index: `chunk_text_index` on Chunk text (Lucene BM25)
- **Scale:** ~121 chunks, ~1086 entities
- **Content:** 1 Million Hectare Project Handbook — covers land preparation, seed selection, sowing, water management (AWD), fertilization, pest management (IPM), harvest, post-harvest, straw management, and technology

---

## 5. App Features Relevant to Testing

### 5.1 Single Query Tab
- Text area for question input
- Runs through selected pipeline
- Displays: execution time, graph facts count, semantic results count, keyword results count
- Shows full answer with comment capability
- Shows all retrieved context: raw entities, aligned entities (with confidence scores), graph facts, vector chunks, keyword chunks, RRF/hybrid fused results, and the exact LLM context (system prompt + full context sent to LLM)
- Embedded Langfuse trace viewer + agent graph visualization

### 5.2 Batch Testing Tab
- Upload CSV/JSON/TXT or paste questions
- Runs all questions sequentially through selected pipeline
- Progress bar
- Card view or detailed view of results
- Export as JSON/CSV

### 5.3 History Tab
- All previous queries saved per user
- Full replay with all context, entities, traces
- Delete individual queries

### 5.4 Sidebar
- Pipeline selector dropdown (V7, ADITI)
- DB status check (chunk count, entity count, index status)
- Recent query history shortcuts
- Theme toggle (light/dark)

### 5.5 Comment System
- Users can add comments on specific sections (answer, entities, context) per query
- Stored in SQLite, visible when replaying history
- Useful for recording evaluation notes

---

## 6. Infrastructure

| Component | Detail |
|---|---|
| **LLM** | Qwen3-30B-A3B (AWQ 4-bit) on TPU, served via OpenAI-compatible API at `http://hanoi2.ucd.ie/v1` |
| **Neo4j** | Docker container at `172.17.0.1:7687`, database: `neo4j` |
| **Langfuse** | Self-hosted at `http://35.186.40.29:8080` |
| **Embeddings** | Local SentenceTransformers: `BAAI/bge-base-en-v1.5` (384-dim) + `intfloat/multilingual-e5-large` (1024-dim, ADITI only) |
| **App DB** | SQLite at `data/app.db` (users, query history, comments, batch runs) |
| **Deployment** | Docker / docker-compose on TPU VM |

---

## 7. What the Testing Protocol Should Cover

The protocol should enable extension officers to rigorously compare V7 vs ADITI across:

1. **Answer quality** — correctness, completeness, specificity, citations, clarity
2. **Bilingual performance** — Vietnamese question understanding and entity extraction
3. **Retrieval quality** — are the right chunks/facts being retrieved?
4. **Calculation accuracy** — unit conversions (công→hectare), arithmetic in answers
5. **Structured data utilization** — does ADITI's structured data path improve factual queries?
6. **Latency** — execution time per pipeline
7. **Entity alignment quality** — confidence tiers, correct KG mapping
8. **Failure modes** — hallucination, irrelevant retrieval, wrong citations
9. **Edge cases** — mixed-language queries, ambiguous questions, out-of-scope questions

---

## 8. Existing Testing Protocol (Already Written)

There is already a testing protocol at `docs/pipeline_testing_protocol.md` that includes:
- 50 questions across 10 topics (3 EN + 2 VI per topic)
- 5 difficulty types: factual, multi-hop, numeric, procedural, bilingual
- 6 evaluation criteria with 1-5 scoring and weights
- Testing procedure (single query → batch → special attention tests)
- Deliverables specification

**Use this as a starting point** and expand/refine it based on the pipeline details above.
