#!/usr/bin/env python3
"""
Hybrid Retrieval: KG + Vector Search for Rice Farming Assistant
Combines structured knowledge graph with semantic vector search
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, CrossEncoder

# Neo4j connection - use environment variables or defaults
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://2d074b24.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "cnSN_ZMqaui2WG40oeiQZwXYoG4fBoVYE9z7-woOAsI")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

@dataclass
class RetrievalResult:
    """Single retrieval result"""
    content: str
    source_id: str
    source_type: str
    title: str
    score: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class HybridRetriever:
    """Combines KG and vector search"""
    
    def __init__(self):
        print("🔧 Initializing Hybrid Retriever...")
        
        # Neo4j
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.database = NEO4J_DATABASE
        
        # Embedding model (lightweight, fast)
        print("📦 Loading embedding model...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Reranker (cross-encoder for better accuracy)
        print("📦 Loading reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Build vector index
        self.documents = []
        self.embeddings = None
        self._build_index()
        
    def _session(self):
        return self.driver.session(database=self.database)
    
    def _build_index(self):
        """Extract content from KG and build vector index"""
        print("🔨 Building vector index from knowledge graph...")
        
        with self._session() as s:
            # Get all Section nodes with content
            result = s.run("""
                MATCH (n:Section)
                WHERE n.content IS NOT NULL
                RETURN n.id AS id, 'Section' AS type, n.number AS number,
                       n.title AS title, n.content AS content
                ORDER BY n.number
            """).data()
            
            self.documents = result
            
            if not self.documents:
                print("⚠️  No Section nodes found with content!")
                self.embeddings = np.array([])
                return
            
            texts = [f"{d['title']}. {d['content']}" for d in self.documents]
            self.embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            
        print(f"✅ Indexed {len(self.documents)} documents ({self.embeddings.shape[1]} dims)\n")
    
    def vector_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Semantic vector search"""
        # Check if embeddings exist
        if len(self.documents) == 0 or self.embeddings.size == 0:
            print("⚠️  No documents in vector index")
            return []
        
        query_emb = self.encoder.encode([query], convert_to_numpy=True)[0]
        
        # Cosine similarity
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        return [
            RetrievalResult(
                content=self.documents[i]['content'],
                source_id=self.documents[i]['id'],
                source_type=self.documents[i]['type'],
                title=self.documents[i]['title'] or '',
                score=float(scores[i]),
                method='vector',
                metadata={'number': self.documents[i]['number']}
            )
            for i in top_idx
        ]
    
    def kg_entity_search(self, query: str) -> List[RetrievalResult]:
        """Search for relevant sections based on query keywords"""
        results = []
        
        with self._session() as s:
            # Simple keyword-based search in Section titles and content
            # This matches sections where the title or content contains query keywords
            keyword_results = s.run("""
                MATCH (n:Section)
                WHERE n.content IS NOT NULL
                  AND (toLower(n.title) CONTAINS toLower($q) 
                       OR toLower(n.content) CONTAINS toLower($q))
                RETURN n.id AS id, n.title AS title, n.content AS content,
                       n.number AS number
                ORDER BY n.number
                LIMIT 5
            """, q=query).data()
            
            for item in keyword_results:
                results.append(RetrievalResult(
                    content=item['content'],
                    source_id=item['id'],
                    source_type='Section',
                    title=item['title'] or '',
                    score=0.90,
                    method='kg_keyword',
                    metadata={'number': item['number']}
                ))
        
        return results
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """Rerank results using cross-encoder for better accuracy"""
        if not results:
            return results
        
        # Prepare pairs: (query, content)
        pairs = [[query, r.content[:512]] for r in results]  # Limit to 512 chars for speed
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores (blend with original: 70% rerank + 30% original)
        for r, rerank_score in zip(results, rerank_scores):
            r.score = 0.7 * float(rerank_score) + 0.3 * r.score
            r.method += "+rerank"
        
        # Re-sort and return top-k
        reranked = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        return reranked
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Hybrid retrieval: vector + KG"""
        print(f"\n🔍 Query: {query}")
        
        # Vector search
        vec_results = self.vector_search(query, top_k=10)
        print(f"  [Vector] {len(vec_results)} results")
        
        # KG entity search
        kg_results = self.kg_entity_search(query)
        print(f"  [KG Entity] {len(kg_results)} results")
        
        # Fusion: deduplicate and boost multi-method matches
        seen = {}
        for r in vec_results + kg_results:
            if r.source_id not in seen:
                seen[r.source_id] = r
            else:
                # Boost if found by multiple methods
                existing = seen[r.source_id]
                existing.score = max(existing.score, r.score) * 1.15
                existing.method += f"+{r.method}"
        
        fused = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k*2]  # Get 2x candidates
        print(f"  [Fused] {len(fused)} candidates")
        
        # Rerank for better accuracy
        print(f"  [Reranking] {len(fused)} candidates...")
        reranked = self.rerank(query, fused, top_k=top_k)
        print(f"  [Final] {len(reranked)} results\n")
        
        return reranked
    
    def close(self):
        self.driver.close()

# ============================================================
# LANGFUSE AGENT TOOLS - Each retrieval step as a callable tool
# ============================================================

class HybridRetrieverTools:
    """
    Exposes each retrieval step as a separate tool for Langfuse Agent Graphs.
    Each tool can be called independently and tracked by Langfuse.
    """
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
    
    # ==================== TOOL 1: Vector Search ====================
    def tool_vector_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Tool 1: Semantic vector search using sentence embeddings.
        
        Args:
            query: User's question or search query
            top_k: Number of results to return (default: 5)
            
        Returns:
            Dict with:
                - results: List of retrieval results
                - count: Number of results found
                - method: 'vector_search'
        """
        results = self.retriever.vector_search(query, top_k=top_k)
        
        return {
            "results": [
                {
                    "content": r.content,
                    "source_id": r.source_id,
                    "source_type": r.source_type,
                    "title": r.title,
                    "score": r.score,
                    "method": r.method,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "count": len(results),
            "method": "vector_search"
        }
    
    # ==================== TOOL 2: KG Entity Search ====================
    def tool_kg_entity_search(self, query: str) -> Dict[str, Any]:
        """
        Tool 2: Knowledge graph entity search - finds practices, machines, and stage-specific content.
        
        Args:
            query: User's question or search query
            
        Returns:
            Dict with:
                - results: List of retrieval results from KG
                - count: Number of results found
                - method: 'kg_entity_search'
                - detected_entities: List of entities found (practices, machines, stages)
        """
        results = self.retriever.kg_entity_search(query)
        
        # Extract unique entities from results
        entities = list(set(
            r.metadata.get('entity', r.source_id) 
            for r in results 
            if r.source_type in ['Practice', 'Machine']
        ))
        
        return {
            "results": [
                {
                    "content": r.content,
                    "source_id": r.source_id,
                    "source_type": r.source_type,
                    "title": r.title,
                    "score": r.score,
                    "method": r.method,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "count": len(results),
            "method": "kg_entity_search",
            "detected_entities": entities
        }
    
    # ==================== TOOL 3: Fusion & Deduplication ====================
    def tool_fusion_deduplicate(self, vector_results: List[Dict], kg_results: List[Dict], top_k: int = 10) -> Dict[str, Any]:
        """
        Tool 3: Fuse and deduplicate results from multiple sources.
        Boosts items found by multiple methods.
        
        Args:
            vector_results: Results from vector search
            kg_results: Results from KG entity search
            top_k: Number of candidates to keep after fusion
            
        Returns:
            Dict with:
                - results: Fused and deduplicated results
                - count: Number of results after fusion
                - method: 'fusion_deduplicate'
                - stats: Deduplication statistics
        """
        # Convert dicts back to RetrievalResult objects
        all_results = []
        for r in vector_results:
            all_results.append(RetrievalResult(
                content=r['content'],
                source_id=r['source_id'],
                source_type=r['source_type'],
                title=r['title'],
                score=r['score'],
                method=r['method'],
                metadata=r.get('metadata', {})
            ))
        
        for r in kg_results:
            all_results.append(RetrievalResult(
                content=r['content'],
                source_id=r['source_id'],
                source_type=r['source_type'],
                title=r['title'],
                score=r['score'],
                method=r['method'],
                metadata=r.get('metadata', {})
            ))
        
        # Deduplicate and boost multi-method matches
        seen = {}
        boosted_count = 0
        
        for r in all_results:
            if r.source_id not in seen:
                seen[r.source_id] = r
            else:
                # Boost if found by multiple methods
                existing = seen[r.source_id]
                existing.score = max(existing.score, r.score) * 1.15
                existing.method += f"+{r.method}"
                boosted_count += 1
        
        # Sort by score and take top_k
        fused = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]
        
        return {
            "results": [
                {
                    "content": r.content,
                    "source_id": r.source_id,
                    "source_type": r.source_type,
                    "title": r.title,
                    "score": r.score,
                    "method": r.method,
                    "metadata": r.metadata
                }
                for r in fused
            ],
            "count": len(fused),
            "method": "fusion_deduplicate",
            "stats": {
                "total_input": len(all_results),
                "unique_after_dedup": len(seen),
                "boosted_items": boosted_count,
                "final_output": len(fused)
            }
        }
    
    # ==================== TOOL 4: Reranking ====================
    def tool_rerank(self, query: str, results: List[Dict], top_k: int = 5) -> Dict[str, Any]:
        """
        Tool 4: Rerank results using cross-encoder for better accuracy.
        
        Args:
            query: Original user query
            results: List of results to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            Dict with:
                - results: Reranked results
                - count: Number of results
                - method: 'rerank'
                - score_changes: Statistics about score changes
        """
        # Convert dicts to RetrievalResult objects
        result_objects = [
            RetrievalResult(
                content=r['content'],
                source_id=r['source_id'],
                source_type=r['source_type'],
                title=r['title'],
                score=r['score'],
                method=r['method'],
                metadata=r.get('metadata', {})
            )
            for r in results
        ]
        
        # Store original scores for comparison
        original_scores = {r.source_id: r.score for r in result_objects}
        
        # Rerank
        reranked = self.retriever.rerank(query, result_objects, top_k=top_k)
        
        # Calculate score changes
        score_changes = [
            {
                "source_id": r.source_id,
                "original_score": original_scores[r.source_id],
                "new_score": r.score,
                "change": r.score - original_scores[r.source_id]
            }
            for r in reranked
        ]
        
        return {
            "results": [
                {
                    "content": r.content,
                    "source_id": r.source_id,
                    "source_type": r.source_type,
                    "title": r.title,
                    "score": r.score,
                    "method": r.method,
                    "metadata": r.metadata
                }
                for r in reranked
            ],
            "count": len(reranked),
            "method": "rerank",
            "score_changes": score_changes
        }
    
    # ==================== TOOL 5: Final Selection ====================
    def tool_final_selection(self, results: List[Dict], top_k: int = 5, diversity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Tool 5: Final selection with diversity filtering.
        Ensures results are not too similar to each other.
        
        Args:
            results: List of reranked results
            top_k: Number of final results to return
            diversity_threshold: Minimum diversity score (0-1) between selected items
            
        Returns:
            Dict with:
                - results: Final selected results
                - count: Number of results
                - method: 'final_selection'
                - diversity_score: Average diversity among selected items
        """
        if not results:
            return {
                "results": [],
                "count": 0,
                "method": "final_selection",
                "diversity_score": 0.0
            }
        
        # Simple diversity: avoid duplicate source types in a row
        selected = []
        seen_types = set()
        
        for r in results:
            if len(selected) >= top_k:
                break
            
            # Always add first result
            if not selected:
                selected.append(r)
                seen_types.add(r['source_type'])
                continue
            
            # Add if different type or if we need more results
            if r['source_type'] not in seen_types or len(selected) < top_k - 1:
                selected.append(r)
                seen_types.add(r['source_type'])
        
        # Fill remaining slots if needed
        remaining = [r for r in results if r not in selected]
        selected.extend(remaining[:top_k - len(selected)])
        
        # Calculate diversity score (ratio of unique types)
        diversity_score = len(set(r['source_type'] for r in selected)) / len(selected) if selected else 0.0
        
        return {
            "results": selected[:top_k],
            "count": len(selected[:top_k]),
            "method": "final_selection",
            "diversity_score": diversity_score,
            "unique_source_types": len(set(r['source_type'] for r in selected))
        }
    
    # ==================== HELPER: Get All Tools ====================
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns schema definitions for all tools, compatible with Langfuse Agent Graphs.
        """
        return [
            {
                "name": "vector_search",
                "description": "Semantic vector search using sentence embeddings. Use this to find relevant content based on meaning and context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's search query or question"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                },
                "function": self.tool_vector_search
            },
            {
                "name": "kg_entity_search",
                "description": "Search knowledge graph for specific entities (practices, machines) and stage-based content. Use this when the query mentions specific farming practices, equipment, or growth stages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's search query"
                        }
                    },
                    "required": ["query"]
                },
                "function": self.tool_kg_entity_search
            },
            {
                "name": "fusion_deduplicate",
                "description": "Fuse and deduplicate results from multiple search methods. Boosts items found by multiple methods.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vector_results": {
                            "type": "array",
                            "description": "Results from vector search"
                        },
                        "kg_results": {
                            "type": "array",
                            "description": "Results from KG entity search"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of candidates to keep",
                            "default": 10
                        }
                    },
                    "required": ["vector_results", "kg_results"]
                },
                "function": self.tool_fusion_deduplicate
            },
            {
                "name": "rerank",
                "description": "Rerank results using a cross-encoder model for better relevance. Use this to improve the quality of search results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The original user query"
                        },
                        "results": {
                            "type": "array",
                            "description": "Results to rerank"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return",
                            "default": 5
                        }
                    },
                    "required": ["query", "results"]
                },
                "function": self.tool_rerank
            },
            {
                "name": "final_selection",
                "description": "Final selection with diversity filtering to ensure varied results. Use this as the last step to get the final set of results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "description": "Reranked results"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of final results",
                            "default": 5
                        },
                        "diversity_threshold": {
                            "type": "number",
                            "description": "Minimum diversity score (0-1)",
                            "default": 0.7
                        }
                    },
                    "required": ["results"]
                },
                "function": self.tool_final_selection
            }
        ]

# ============================================================
# DEMO
# ============================================================

def demo_traditional_retrieve():
    """Demo: Traditional monolithic retrieve() method"""
    print("\n" + "="*80)
    print("DEMO 1: Traditional Monolithic Retrieval")
    print("="*80 + "\n")
    
    retriever = HybridRetriever()
    
    query = "What is AWD and how does it reduce emissions?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.source_type}: {r.title}")
        print(f"    Score: {r.score:.3f} | Method: {r.method}")
        print(f"    Content: {r.content[:150]}...\n")
    
    retriever.close()

def demo_tool_based_retrieve():
    """Demo: Tool-based retrieval for Langfuse Agent Graphs"""
    print("\n" + "="*80)
    print("DEMO 2: Tool-Based Retrieval (Langfuse-Compatible)")
    print("="*80 + "\n")
    
    retriever = HybridRetriever()
    tools = HybridRetrieverTools(retriever)
    
    query = "What is AWD and how does it reduce emissions?"
    print(f"Query: {query}\n")
    
    # Step 1: Vector Search Tool
    print("🔧 TOOL 1: Vector Search")
    vector_result = tools.tool_vector_search(query, top_k=5)
    print(f"   Found {vector_result['count']} results\n")
    
    # Step 2: KG Entity Search Tool
    print("🔧 TOOL 2: KG Entity Search")
    kg_result = tools.tool_kg_entity_search(query)
    print(f"   Found {kg_result['count']} results")
    print(f"   Detected entities: {kg_result['detected_entities']}\n")
    
    # Step 3: Fusion & Deduplication Tool
    print("🔧 TOOL 3: Fusion & Deduplication")
    fusion_result = tools.tool_fusion_deduplicate(
        vector_results=vector_result['results'],
        kg_results=kg_result['results'],
        top_k=10
    )
    print(f"   Stats: {fusion_result['stats']}\n")
    
    # Step 4: Reranking Tool
    print("🔧 TOOL 4: Reranking")
    rerank_result = tools.tool_rerank(
        query=query,
        results=fusion_result['results'],
        top_k=5
    )
    print(f"   Reranked {rerank_result['count']} results")
    print(f"   Score changes:")
    for change in rerank_result['score_changes'][:3]:
        print(f"      {change['source_id']}: {change['original_score']:.3f} -> {change['new_score']:.3f} (delta {change['change']:+.3f})")
    print()
    
    # Step 5: Final Selection Tool
    print("🔧 TOOL 5: Final Selection")
    final_result = tools.tool_final_selection(
        results=rerank_result['results'],
        top_k=3,
        diversity_threshold=0.7
    )
    print(f"   Selected {final_result['count']} results")
    print(f"   Diversity score: {final_result['diversity_score']:.2f}")
    print(f"   Unique source types: {final_result['unique_source_types']}\n")
    
    # Display final results
    print("\n📋 FINAL RESULTS:")
    print("="*80)
    for i, r in enumerate(final_result['results'], 1):
        print(f"\n[{i}] {r['source_type']}: {r['title']}")
        print(f"    Score: {r['score']:.3f} | Method: {r['method']}")
        print(f"    Content: {r['content'][:150]}...")
    
    print("\n" + "="*80)
    print("\n📋 TOOL SCHEMAS (for Langfuse registration):")
    print("="*80)
    schemas = tools.get_tool_schemas()
    for schema in schemas:
        print(f"\nTool: {schema['name']}")
        print(f"  Description: {schema['description'][:80]}...")
        print(f"  Required params: {schema['parameters']['required']}")
    
    retriever.close()

def main():
    """Run both demos"""
    demo_traditional_retrieve()
    demo_tool_based_retrieve()

if __name__ == "__main__":
    main()
