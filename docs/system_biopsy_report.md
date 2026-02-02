# System Biopsy Report: Pivot Assessment

**Generated:** 2026-01-24
**Target:** Pivot from "MCQ Generation" to "Automated Textbook Auditing"

---

## 1. Directory Tree (`src/`)

```
src/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── generation/
│   ├── __init__.py
│   ├── lean4_formalizer.py
│   └── mcq_generator.py
├── graph_store/
│   ├── __init__.py
│   ├── graph_constructor.py
│   └── neo4j_client.py
├── ingestion/
│   ├── __init__.py
│   ├── formula_extractor.py
│   ├── ingestion_engine.py
│   ├── latex_normalizer.py
│   ├── layout_parser.py
│   └── ocr_utils.py
├── models/
│   ├── __init__.py
│   └── schemas.py
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_orchestrator.py
│   └── query_refiner.py
├── vector_store/
│   ├── __init__.py
│   ├── chroma_client.py
│   └── embeddings.py
└── verification/
    ├── __init__.py
    ├── auditor_prover.py  <-- KEY AUDITING COMPONENT
    ├── check_env.py
    ├── lean_compiler.py
    └── verification_sandbox.py
```

---

## 2. The "Auditing" Logic

**Status:** ✅ **FOUND & IMPLEMENTED**

The core auditing logic resides in `src/verification/auditor_prover.py`. This module implements the "Lobotomized Auditor" pattern, which strictly enforces that a problem can only be solved using concepts from previous chapters.

**Key Component:** `AuditorProver` class
**Two-Layer Audit Logic:**
1.  **FAIL_GAP:** Detected when the model explicitly states "Missing Prerequisite" or cannot find a required concept in the allowed context.
2.  **FAIL_LOGIC:** Detected when the context is sufficient, but the mathematical statement itself is false (e.g., "1 = 2").

**Evidence (`src/verification/auditor_prover.py`):**
```python
class AuditorProver:
    """Textbook Auditing Engine — Hybrid Cloud Execution.
    
    Implements the "Lobotomized Auditor" pattern:
    - Only uses theorems from chapters 1 to X-1 when auditing chapter X
    - Detects "Pedagogical Gaps" (problems using undefined concepts)
    ...
    """

    def _determine_audit_result(self, response, lean_code, missing_prereqs):
        """Two-Layer Audit Logic.
        Distinguishes between:
        - FAIL_GAP: Missing prerequisite (pedagogical/timing error)
        - FAIL_LOGIC: Mathematical fallacy (truth error)
        ...
        """
```

---

## 3. Premise Selection Implementation

**Status:** ✅ **FOUND & IMPLEMENTED**

The system enforces "knowledge ordering" by retrieving only concepts from chapters *prior* to the current target chapter. This is located in the `HybridRetriever`.

**Location:** `src/retrieval/hybrid_orchestrator.py`
**Function:** `retrieve_premises(chapter)`

**Evidence:**
```python
def retrieve_premises(self, chapter: int, include_concepts: bool = True) -> List[Dict[str, Any]]:
    """Retrieve premises (theorems/axioms) from chapters before the given chapter.
    
    Implements "Premise Selection" for textbook auditing:
    - Only returns theorems from chapters 1 to chapter-1
    - Used to simulate a student's limited knowledge state
    """
    # ...
    theorem_query = """
    MATCH (t:Theorem)
    WHERE t.chapter < $chapter OR t.chapter = 0  <-- STRICT CHAPTER FILTER
    RETURN ...
    """
```

---

## 4. Lean 4 Verification Bridge

**Status:** ✅ **FOUND & IMPLEMENTED**

The system constructs Lean 4 files with strict import controls. The `AuditorProver` prompts the LLM to write Lean code that uses *only* the cited premises, and the `Lean4Formalizer` ensures `Mathlib` compatibility.

**Location:** `src/verification/auditor_prover.py` (Logic) & `src/generation/lean4_formalizer.py` (Syntax)

**Prompting Strategy (`auditor_prover.py`):**
```python
AUDITOR_SYSTEM_PROMPT = """
...
RULE 2: NO GHOST KNOWLEDGE
You are FORBIDDEN from using ANY concept not explicitly listed in the Context.
- If you need Integration... and they are NOT in the Context → STOP IMMEDIATELY.
...
"""
```

**Import Structure (`lean4_formalizer.py`):**
```python
LEAN4_MATHLIB_PROMPT = """
...
=== REQUIREMENTS ===
1. ALWAYS start with: `import Mathlib`
...
"""
```

---

## 5. Current Entry Points

The system has multiple active entry points reflecting both the "MCQ Generation" legacy and the new "Auditing" capabilities.

**A. Auditing Pipeline (New Pivot)**
While the core *classes* exist (`AuditorProver`), there is **no single CLI script** (like `run_audit.py`) visible in `scripts/` that executes a full textbook audit. The logic exists in library code but needs a driver script.

**B. Diagnostic MCQ (Legacy/Hybrid)**
- **Script:** `scripts/run_phase_3_diagnostic.py`
- **Function:** Generates MCQs with "Smart Distractors" based on Misconception nodes. It transforms the system into a "Diagnostic Teacher" but is not a "Textbook Auditor" per se.

**C. Cross-Domain Verification**
- **Script:** `scripts/run_phase_4_bridge.py`
- **Function:** Verifies retrieval across domains (Physics <-> Calculus).

**D. Ingestion**
- **Script:** `scripts/ingest_openstax.py`
- **Function:** Full ingestion of OpenStax Calculus into Graph + Vector stores.

---

## Verdict for Lead Researcher

We have successfully **built the core engine** for the "Textbook Auditor":
1.  **Logic:** The "Lobotomized Auditor" pattern is fully implemented in `AuditorProver`.
2.  **Filtering:** Premise selection by chapter is active in `HybridRetriever`.
3.  **Verification:** The Lean 4 bridge is legally restricted to context.

**Missing:** A dedicated top-level execution script (e.g., `scripts/run_textbook_audit.py`) to run the audit on a full book. Currently, you must use the Python API to invoke `hybrid_retriever.audit_chapter()`.
