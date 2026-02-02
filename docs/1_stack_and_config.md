# 1. Stack and Configuration

## Active Tech Stack

### Configuration File: `src/config/settings.py`

**LLM Configuration:**
```python
# STANDARDIZED TO GPT-4o-mini for cost efficiency
gpt4o_mini_model: str = "gpt-4o-mini"
gpt4o_model: str = "gpt-4o-mini"  # CHANGED: Now defaults to mini for budget
default_model: str = "gpt-4o-mini"

# Rate limiting
openai_max_retries: int = 3
openai_timeout: int = 60

# Budget tracking
budget_limit_usd: float = 5.0
```

**Databases:**
```python
# Neo4j (Knowledge Graph)
neo4j_uri: str = "bolt://localhost:7687"
neo4j_user: str = "neo4j"
neo4j_password: str = "password"
neo4j_database: str = "neo4j"

# ChromaDB (Vector Store)
chroma_persist_directory: str = "./data/chroma_db"
chroma_collection_name: str = "mathematest_chunks"
```

**Embedding Models (LOCAL - no API cost):**
```python
embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

---

## Dependencies: `requirements.txt`

| Category | Packages |
|----------|----------|
| **Layout Detection** | `doclayout-yolo>=0.0.2` |
| **Formula OCR** | `unimernet>=0.2.2`, `transformers>=4.35.0`, `sentencepiece>=0.1.99` |
| **Symbolic Math** | `sympy>=1.12`, `latex2sympy2>=1.9.1` |
| **PDF Processing** | `pdf2image>=1.16.0`, `opencv-python>=4.8.0`, `pillow>=9.5.0` |
| **Knowledge Graph** | `neo4j>=5.14.0` |
| **Vector Store** | `chromadb>=0.4.18`, `sentence-transformers>=2.2.2` |
| **LLM** | `openai>=1.6.0`, `langgraph>=0.0.60`, `langchain>=0.1.0` |
| **Core** | `pydantic>=2.5.0`, `torch>=2.0.0`, `numpy>=1.24.0` |

**Note:** Local models (doclayout-yolo, unimernet) are NOT actively used since pivot to GPT-4o-mini.

---

## Environment Variables Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key starting with `sk-` |
| `NEO4J_URI` | Optional override for Neo4j connection |
| `NEO4J_USER` | Optional override for Neo4j username |
| `NEO4J_PASSWORD` | Optional override for Neo4j password |

**Loaded from:** `.env` file via `pydantic_settings`
