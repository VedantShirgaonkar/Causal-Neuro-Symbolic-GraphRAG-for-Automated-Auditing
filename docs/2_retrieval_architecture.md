# 2. Retrieval Architecture

## Primary Module: `src/retrieval/hybrid_orchestrator.py`

### Main Class: `HybridRetriever`

**Purpose:** Orchestrates hybrid vector + graph retrieval with cross-encoder re-ranking.

```python
class HybridRetriever:
    """Orchestrates hybrid vector + graph retrieval.
    
    Executes parallel queries against ChromaDB (semantic) and
    Neo4j (logical), then merges and re-ranks the results.
    """
```

---

## Retrieval Pipeline

```
Query → QueryRefiner → [Vector Search || Graph Search] → Merge → CrossEncoderReranker → Results
```

### 1. Vector Search (ChromaDB)
```python
def _vector_search(self, query: str, n_results: int = 10):
    # Standard semantic similarity search
    # NO chapter filtering - searches entire collection
```

### 2. Graph Search (Neo4j)

**Location:** `hybrid_orchestrator.py` lines 218-280

```python
def _graph_search(self, query: str, n_results: int = 10):
    # Search by LaTeX if query contains math
    if "\\" in query or "=" in query:
        latex_results = self.graph_client.search_by_latex(query)
    
    # Get prerequisites for concepts found
    for result in results[:3]:
        prereqs = self.graph_client.get_prerequisites(concept_id, depth=2)
        # Score decays with distance: 0.7 - (distance * 0.1)
    
    # Get related misconceptions
    misconceptions = self.graph_client.get_misconceptions(concept_id)
```

---

## Key Questions Answered

### Q: Do we have "Chapter Filtering" logic?

**Answer: NOT FOUND**

No chapter or section-based filtering exists. The vector search queries the **entire** ChromaDB collection. Graph search is concept-based, not chapter-based.

**Evidence:**
```bash
$ grep -ri "chapter" src/
# No results
```

### Q: Do we have "Prerequisite Checks"?

**Answer: YES - but only in graph search**

```python
# From _graph_search():
prereqs = self.graph_client.get_prerequisites(concept_id, depth=2)
for prereq in prereqs[:5]:
    results.append(RetrievalResult(
        content=f"[Prerequisite] {node.get('name')}: {node.get('description')}",
        score=0.7 - (distance * 0.1),  # Decrease score with distance
        metadata={"distance": distance, "type": "prerequisite"},
    ))
```

This fetches 1-2 hop prerequisites for concepts found in the initial search, but it does NOT enforce prerequisite checking before showing content.

---

## Re-Ranking

### Class: `CrossEncoderReranker`

```python
class CrossEncoderReranker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int):
        # Uses cross-encoder to score query-result pairs
        # Returns results sorted by relevance
```

---

## Summary

| Feature | Status |
|---------|--------|
| Semantic Vector Search | ✅ Implemented (ChromaDB) |
| Graph Traversal | ✅ Implemented (Neo4j) |
| Prerequisite Retrieval | ✅ Implemented (1-2 hops) |
| **Chapter Filtering** | ❌ **NOT FOUND** |
| **Prerequisite Enforcement** | ❌ **NOT FOUND** |
| Cross-Encoder Re-ranking | ✅ Implemented |
