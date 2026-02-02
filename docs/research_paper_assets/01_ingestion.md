# 01: Ingestion & Vectorization Pipeline

## Technical Specification

This document provides detailed technical extraction from `src/ingestion/` and `src/vector_store/` for research paper citation.

---

## 1. PDF Processing Strategy

### 1.1 PDF-to-Image Conversion

**Library:** `pdf2image` (poppler backend)

**Class:** `IngestionEngine` in `src/ingestion/ingestion_engine.py`

**Configuration Parameters:**
```python
class IngestionEngine:
    def __init__(
        self,
        dpi: int = 200,                    # Resolution for PDF rasterization
        confidence_threshold: float = 0.25, # Layout detection threshold
        use_gpu: bool = True,              # GPU acceleration for YOLO
    )
```

**Processing Pipeline:**
1. `_pdf_to_images()` → Converts PDF pages to PIL Images at 200 DPI
2. `_process_page()` → Runs layout analysis on each image
3. `process_pdf()` → Orchestrates full pipeline

---

## 2. Layout Analysis

### 2.1 DocLayout-YOLO

**Model:** Custom DocLayout-YOLO trained on scientific documents

**Detected Block Types:**
- Text paragraphs
- Mathematical equations (inline and display)
- Figures with captions
- Tables
- Section headers

**Implementation:** `src/ingestion/layout_parser.py`

---

## 3. Formula Extraction

### 3.1 LaTeX Extraction

**Module:** `src/ingestion/formula_extractor.py`

**OCR Integration:** Uses PyTesseract for text regions

**LaTeX Normalization:** `src/ingestion/latex_normalizer.py`
- Standardizes delimiters (`$...$`, `\(...\)`)
- Handles multi-line expressions
- Preserves equation numbering

---

## 4. Chunking Algorithm

### 4.1 Text Chunking Parameters

**Chunk Size:** Configurable, default handling in ChromaDB client

**Overlap Strategy:** Content blocks from layout analysis define natural chunk boundaries

**ID Generation:**
```python
def _generate_document_id(self, pdf_path: Path) -> str:
    """Generate a unique document ID based on file content."""
    # Uses SHA-256 hash of file content
```

---

## 5. Embedding Generation

### 5.1 Model Specification

**Class:** `EmbeddingService` in `src/vector_store/embeddings.py`

```python
class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,  # Auto-select: cuda/cpu
    )
```

**Model Details:**
| Property | Value |
|----------|-------|
| Model Name | `sentence-transformers/all-mpnet-base-v2` |
| Embedding Dimension | **768** |
| Context Window | 384 tokens |
| Normalization | L2 normalized by default |

### 5.2 LaTeX-Specific Embedding

```python
def embed_latex(self, latex_strings: List[str], include_description: bool = True):
    """Preprocesses LaTeX with natural language context."""
    for latex in latex_strings:
        text = f"Mathematical expression: {latex}"  # Context enhancement
```

---

## 6. Vector Storage

### 6.1 ChromaDB Configuration

**Class:** `ChromaVectorStore` in `src/vector_store/chroma_client.py`

```python
# Persistence Path (from settings.py)
chroma_persist_directory: str = "./data/chroma_db"
chroma_collection_name: str = "mathematest_chunks"
```

**API Version:** ChromaDB v0.4+ (PersistentClient API)

```python
self._client = chromadb.PersistentClient(path=str(persist_path))
```

### 6.2 Search Implementation

**Distance Metric:** Cosine distance (converted to similarity)

```python
item["score"] = 1 - item["distance"]  # Convert distance to similarity
```

---

## 7. Mathematical Formulas

### 7.1 Cosine Similarity

The embedding similarity between query vector **q** and document vector **d** is computed as:

$$\text{similarity}(q, d) = \frac{q \cdot d}{\|q\| \|d\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \cdot \sqrt{\sum_{i=1}^{n} d_i^2}}$$

Since embeddings are L2-normalized, this simplifies to:

$$\text{similarity}(q, d) = q \cdot d = \sum_{i=1}^{768} q_i d_i$$

### 7.2 ChromaDB Distance Conversion

ChromaDB returns L2 distance, converted to similarity score:

$$\text{score} = 1 - \text{distance}$$

Where distance is computed as:

$$\text{distance}(q, d) = \|q - d\|_2 = \sqrt{\sum_{i=1}^{768}(q_i - d_i)^2}$$

---

## 8. Statistics

| Metric | Value |
|--------|-------|
| Total Chunks Indexed | 1,014 |
| Embedding Dimension | 768 |
| Collection Name | `mathematest_chunks` |
| Storage Format | ChromaDB PersistentClient |
