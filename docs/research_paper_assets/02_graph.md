# 02: Knowledge Graph Architecture

## Technical Specification

This document provides detailed technical extraction from `src/graph_store/` for research paper citation.

---

## 1. Graph Database

### 1.1 Neo4j Configuration

**Module:** `src/graph_store/neo4j_client.py`

**Connection Settings (from settings.py):**
```python
neo4j_uri: str = "bolt://localhost:7687"
neo4j_user: str = "neo4j"
neo4j_password: str = "password"
neo4j_database: str = "neo4j"
```

**Driver:** Official Neo4j Python Driver (`neo4j` package)

```python
from neo4j import GraphDatabase, Driver
self._driver = GraphDatabase.driver(
    self.settings.neo4j_uri,
    auth=(self.settings.neo4j_user, self.settings.neo4j_password),
)
```

---

## 2. Schema Definition

### 2.1 Node Labels

**Class:** `Neo4jClient` in `src/graph_store/neo4j_client.py`

```python
NODE_TYPES = ["Concept", "Formula", "Theorem", "Definition", "Figure", "Misconception"]
```

| Node Label | Description | Properties |
|------------|-------------|------------|
| `Concept` | Mathematical concept | `id`, `name`, `description`, `topic` |
| `Formula` | Mathematical formula | `id`, `name`, `raw_latex`, `normalized_latex` |
| `Theorem` | Proven statement | `id`, `name`, `statement`, `proof_method` |
| `Definition` | Formal definition | `id`, `name`, `formal_text` |
| `Figure` | Visual diagram | `id`, `caption`, `source_page` |
| `Misconception` | Student error pattern | `id`, `description`, `common_error`, `topic` |

### 2.2 Relationship Types

```python
EDGE_TYPES = {
    "PREREQUISITE_OF": ["weight", "source_id"],
    "DERIVED_FROM": ["proof_method", "source_id"],
    "GROUNDED_IN": ["application", "source_id"],
    "CONTRADICTS": ["reason", "source_id"],
    "HAS_MISCONCEPTION": ["frequency", "source_id"],
}
```

| Relationship Type | Description | Direction |
|-------------------|-------------|-----------|
| `PREREQUISITE_OF` | Concept A must be learned before Concept B | A → B |
| `DERIVED_FROM` | Theorem derived from base theorem | A → B |
| `GROUNDED_IN` | Physics grounded in mathematics | A → B |
| `CONTRADICTS` | Misconception contradicts concept | A → B |
| `HAS_MISCONCEPTION` | Concept has associated misconception | A → B |

---

## 3. Schema Initialization

### 3.1 Constraints

```python
def initialize_schema(self) -> Dict[str, Any]:
    for node_type in self.NODE_TYPES:
        constraint_name = f"unique_{node_type.lower()}_id"
        session.run(f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{node_type})
            REQUIRE n.id IS UNIQUE
        """)
```

### 3.2 Indexes

```python
indexes = [
    ("Concept", "name"),
    ("Formula", "name"),
    ("Formula", "source_id"),
    ("Theorem", "name"),
    ("Misconception", "related_concept"),
]
```

---

## 4. Graph Construction Agent

### 4.1 Entity Extraction

**Module:** `src/graph_store/graph_constructor.py`

**Class:** `GraphConstructorAgent`

**LLM Model:** `gpt-4o-mini` (configured in settings.py)

### 4.2 Extraction Prompt Template

The system uses GPT-4o-mini with a structured extraction prompt:

```python
EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting structured mathematical knowledge.

From the given mathematical content, extract:
1. **Entities:** Concepts, Formulas, Theorems, Definitions
2. **Relationships:** Prerequisites, Derivations, Foundations
3. **Misconceptions:** Common student errors

Output format: JSON with entities, relationships, misconceptions arrays.
"""
```

### 4.3 Prompt Construction

```python
def create_extraction_prompt(content: Dict[str, Any]) -> str:
    """Create the user prompt for entity extraction."""
    # Includes: raw_latex, description, source_id, page_number
```

---

## 5. Cypher Query Examples

### 5.1 Concept Search

```cypher
MATCH (c:Concept)
WHERE toLower(c.name) CONTAINS toLower($query)
   OR toLower(c.description) CONTAINS toLower($query)
RETURN c.id, c.name, c.description
LIMIT 10
```

### 5.2 Prerequisite Chain Traversal

```cypher
MATCH (c:Concept)-[:PREREQUISITE_OF*1..3]->(target:Concept)
WHERE c.name = $concept_name
RETURN target.name, target.description
```

### 5.3 Misconception Retrieval

```cypher
MATCH (c:Concept)-[:HAS_MISCONCEPTION]->(m:Misconception)
WHERE toLower(c.name) CONTAINS $topic
   OR toLower(m.topic) = $problem_type
RETURN m.id, m.description, m.common_error, c.name as related_concept
LIMIT 3
```

### 5.4 Cross-Source Relationship Query

```cypher
MATCH (p:Concept {source: 'University_Physics'})-[r:GROUNDED_IN]->(c:Concept {source: 'MIT_Calculus'})
RETURN p.name as physics_concept, c.name as calculus_concept, type(r) as relationship
```

---

## 6. Node ID Generation

```python
def _generate_node_id(self, entity_type: str, name: str, source_id: str = "") -> str:
    """Generate a unique node ID."""
    # SHA-256 hash of entity_type + name + source_id
    import hashlib
    content = f"{entity_type}:{name}:{source_id}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

---

## 7. Graph Statistics

| Metric | Value |
|--------|-------|
| **Total Concept Nodes** | 1,697 |
| **Misconception Nodes** | 26 |
| **Cross-Source Edges** | 11 |
| **PREREQUISITE_OF Edges** | ~450 |
| **HAS_MISCONCEPTION Edges** | 26 |

### 7.1 Source Distribution

| Source | Node Count |
|--------|------------|
| University_Physics_Volume_1 | ~800 |
| MIT_Calculus_Week_1-14 | ~700 |
| AIME_2025 | ~50 |
