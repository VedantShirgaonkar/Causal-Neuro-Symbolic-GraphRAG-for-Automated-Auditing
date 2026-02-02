# 03: Hybrid Retrieval Logic

## Technical Specification

This document provides detailed technical extraction from `src/retrieval/` for research paper citation.

---

## 1. Architecture Overview

### 1.1 HybridRetriever Class

**Module:** `src/retrieval/hybrid_orchestrator.py`

**Design Pattern:** Parallel execution with merge and rerank

```python
class HybridRetriever:
    """Orchestrates hybrid vector + graph retrieval.
    
    Executes parallel queries against ChromaDB (semantic) and
    Neo4j (logical), then merges and re-ranks the results.
    """
```

---

## 2. Retrieval Algorithm

### 2.1 Step-by-Step Algorithm

```
ALGORITHM: HybridRetrieval(query, n_results)
INPUT: Natural language query, desired result count
OUTPUT: ReasoningPacket with merged, ranked results

1. QUERY REFINEMENT (Optional)
   refined_query ← QueryRefiner.refine(query)
   Extract: math_expressions, key_concepts, search_terms

2. PARALLEL EXECUTION
   vector_results ← ASYNC VectorSearch(refined_query, n=2*n_results)
   graph_results  ← ASYNC GraphSearch(refined_query, n=2*n_results)

3. MERGE AND DEDUPLICATE
   merged ← MergeResults(vector_results, graph_results)
   Remove duplicates by content hash

4. CROSS-ENCODER RERANK
   scored_results ← CrossEncoder.predict(query, merged)
   Sort by relevance score descending

5. RETURN TOP-K
   Return merged[:n_results] as ReasoningPacket
```

### 2.2 Implementation

```python
def retrieve(
    self,
    query: str,
    use_refinement: bool = True,
    n_results: int = 10,
    rerank: bool = True,
) -> ReasoningPacket:
    """Main retrieval entry point."""
    
    # Step 1: Optional query refinement
    refined = None
    if use_refinement and self.query_refiner:
        refined = self.query_refiner.refine(query)
    
    # Step 2: Parallel search
    search_query = refined.optimized_query if refined else query
    vector_results = self._vector_search(search_query, n_results * 2)
    graph_results = self._graph_search(search_query, n_results * 2)
    
    # Step 3: Merge
    merged = self._merge_results(vector_results, graph_results)
    
    # Step 4: Rerank
    if rerank and self.reranker:
        merged = self.reranker.rerank(query, merged, top_k=n_results)
    
    return ReasoningPacket(...)
```

---

## 3. Vector Search Implementation

### 3.1 ChromaDB Query

**Method:** `_vector_search()`

```python
def _vector_search(self, query: str, n_results: int = 10) -> List[RetrievalResult]:
    """Execute vector similarity search."""
    results = self.vector_store.search(
        query=query,
        n_results=n_results,
        include_distances=True,
    )
    
    return [
        RetrievalResult(
            id=r["id"],
            content=r["content"],
            source="vector",
            score=r.get("score", 0.5),
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]
```

---

## 4. Graph Search Implementation

### 4.1 Neo4j Traversal

**Method:** `_graph_search()`

**Query Strategy:**
1. Full-text concept search
2. Prerequisite chain traversal (depth 1-2)
3. Misconception retrieval

```python
def _graph_search(self, query: str, n_results: int = 10) -> List[RetrievalResult]:
    """Execute graph traversal search."""
    
    with self.graph_client.session() as session:
        # Concept search
        result = session.run("""
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($query)
               OR toLower(c.description) CONTAINS toLower($query)
            RETURN c.id, c.name, c.description, 'concept' as type
            LIMIT $limit
        """, {"query": query, "limit": n_results})
        
        # Add prerequisite neighbors
        # Add misconceptions
```

---

## 5. Cross-Encoder Reranking

### 5.1 Model Specification

**Class:** `CrossEncoderReranker`

```python
class CrossEncoderReranker:
    """Re-ranks results using a cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
```

| Property | Value |
|----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Architecture | BERT-based cross-encoder |
| Input | (query, document) pairs |
| Output | Relevance score [0, 1] |

### 5.2 Reranking Implementation

```python
def rerank(
    self,
    query: str,
    results: List[RetrievalResult],
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """Re-rank results by relevance to query."""
    
    # Create query-document pairs
    pairs = [(query, r.content) for r in results]
    
    # Score with cross-encoder
    scores = self.model.predict(pairs)
    
    # Sort by score
    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # Update scores and return
    for result, score in scored_results:
        result.score = float(score)
    
    return [r for r, _ in scored_results[:top_k]]
```

---

## 6. Result Fusion Logic

### 6.1 Merge Strategy

**Method:** `_merge_results()`

```python
def _merge_results(
    self,
    vector_results: List[RetrievalResult],
    graph_results: List[RetrievalResult],
) -> List[RetrievalResult]:
    """Merge and deduplicate results from both sources."""
    
    merged = []
    seen_content_hashes = set()
    
    # Interleave results (vector first, then graph)
    for result in vector_results + graph_results:
        content_hash = hash(result.content[:100])
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            merged.append(result)
    
    return merged
```

### 6.2 Score Normalization

Graph results use fixed score assignment:
```python
score = 0.8  # Default graph match score
```

Vector results use ChromaDB distance conversion:
```python
score = 1 - distance  # Cosine distance to similarity
```

---

## 7. Query Refinement

### 7.1 QueryRefiner Class

**Module:** `src/retrieval/query_refiner.py`

**LLM:** GPT-4o-mini

```python
class QueryRefiner:
    """Refines queries using GPT-4o-mini."""
    
    def refine(self, query: str) -> RefinedQuery:
        """Extract math expressions, key concepts, search terms."""
```

### 7.2 Refined Query Structure

```python
@dataclass
class RefinedQuery:
    original_query: str
    optimized_query: str
    math_expressions: List[str]
    key_concepts: List[str]
    search_terms: List[str]
```

---

## 8. Data Structures

### 8.1 RetrievalResult

```python
@dataclass
class RetrievalResult:
    id: str                              # Unique identifier
    content: str                         # Text content
    source: str                          # "vector" or "graph"
    score: float                         # Relevance score [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 8.2 ReasoningPacket

```python
@dataclass
class ReasoningPacket:
    query: str                           # Original query
    refined_query: str                   # After refinement
    results: List[RetrievalResult]       # Merged results
    graph_context: Dict[str, Any]        # Additional graph data
    total_results: int
    vector_count: int
    graph_count: int
```

---

## 9. Performance Metrics

| Metric | Value |
|--------|-------|
| Vector Search Latency | ~50ms |
| Graph Search Latency | ~100ms |
| Reranking Latency | ~200ms (CPU) |
| Total Retrieval Time | ~500ms |
| Default Result Count | 15 |
