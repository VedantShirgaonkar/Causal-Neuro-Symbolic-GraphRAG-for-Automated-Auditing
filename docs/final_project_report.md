# MathemaTest: A Neuro-Symbolic GraphRAG System for STEM Education

## Final Project Report — Semester VI

**Author:** MathemaTest Research Team  
**Date:** January 2026  
**Course:** Advanced AI Systems for Education

---

## Abstract

We present **MathemaTest**, a neuro-symbolic GraphRAG (Retrieval-Augmented Generation) system designed for STEM education and assessment. The system integrates:

1. **Multi-modal document ingestion** (PDFs, LaTeX, diagrams)
2. **Hybrid knowledge graph** (Neo4j concepts + ChromaDB vectors)
3. **Formal verification** (Lean 4 + Mathlib theorem proving)
4. **Diagnostic MCQ generation** (misconception-linked distractors)

Key results demonstrate **cross-domain retrieval** (Physics ↔ Calculus), **26.7% misconception coverage** in distractors, and **80% verification rate** on undergraduate mathematics—establishing the architecture's viability while highlighting current LLM formalization limitations.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MathemaTest Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │   Stage 1   │    │   Stage 2   │    │   Stage 3   │    │  Stage 4  │ │
│  │  INGESTION  │───▶│    GRAPH    │───▶│    LEAN     │───▶│  PEDAGOGY │ │
│  │             │    │             │    │             │    │           │ │
│  │ PDF/LaTeX   │    │ Neo4j Nodes │    │ Lean 4      │    │ MCQ Gen   │ │
│  │ OCR/Vision  │    │ ChromaDB    │    │ Mathlib     │    │ Distractors│
│  │ Chunking    │    │ Edges       │    │ Verification│    │ Diagnosis │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
│                                                                          │
│  Data Sources:          Knowledge Base:      Formal Proof:    Output:   │
│  - University Physics   - 1,697 Concepts     - lake build     - MCQs    │
│  - MIT Calculus (14)    - 1,014 Chunks       - ring/omega     - Reports │
│  - AIME 2025            - 11 Cross-Edges     - 80% pass rate  - JSON    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Innovation 1: Cross-Domain Retrieval

### The Challenge
STEM problems often require concepts from multiple disciplines. A work-energy problem needs both **Physics** (definition of work) and **Calculus** (line integrals).

### Our Solution
The hybrid retriever queries both vector similarity (ChromaDB) and graph traversal (Neo4j) simultaneously, then reranks with a cross-encoder.

### Evidence (Phase 4)

**Bridge Problem:** *"A force field F = (2xy, x²) acts on a particle moving along path C..."*

| Domain | Nodes Retrieved | Sources |
|--------|-----------------|---------|
| Physics | 13 | University_Physics_Volume_1 (Ch 7.1 Work) |
| Calculus | 2 | mit_calculus_lec_week8 (Line Integrals) |

**Verdict:** ✅ **CROSS-DOMAIN LINK VERIFIED**

The system successfully connected concepts across academic disciplines, enabling "inter-domain reasoning" for multi-disciplinary problems.

---

## Key Innovation 2: Diagnostic Distractors

### The Challenge
Traditional MCQ distractors are arbitrary wrong answers. Pedagogically effective distractors should reveal *why* a student erred.

### Our Solution
The **Smart Distractor Engine** queries Neo4j for `(Concept)-[:HAS_MISCONCEPTION]->(Misconception)` nodes and uses them to generate targeted wrong answers.

### Evidence (Phase 3)

| Metric | Value |
|--------|-------|
| Misconception Coverage | 26.7% |
| Graph-Backed Distractors | 4/15 |
| Unique Misconceptions Used | 4 |

**Qualitative Example (AIME 2025 Problem 1):**

> **Distractor B:** 3000  
> **Misconception ID:** `misc_combinatorics_057ffc57`  
> **Diagnosis:** *"Student likely counted all possible paths from (0,0) to (10,10) without restricting to paths that pass through prime coordinate points."*

This transforms the system from a *"Pure Solver"* into a *"Diagnostic Teacher"* capable of identifying conceptual gaps.

---

## Key Innovation 3: Formal Verification Infrastructure

### The Challenge
LLMs can produce mathematically incorrect solutions. Ground-truth verification requires formal proof systems.

### Our Solution
A Lean 4 + Mathlib integration using:
- `lake new mathematest math` project initialization
- `lake exe cache get` for pre-compiled binaries (7,727 oleans)
- `subprocess.run` bridge for theorem compilation
- Self-correction loop (3 attempts with error feedback)

### Evidence: The Complexity Ceiling

| Dataset | Difficulty | Pass Rate |
|---------|------------|-----------|
| **Easy Math** | Undergraduate | **80%** (4/5) |
| **AIME 2025** | Olympiad | 0% (0/5) |

**Easy Math Results:**

| Theorem | Topic | Status |
|---------|-------|--------|
| Distributivity: a*(b+c) = a*b + a*c | Algebra | ✅ |
| And Elimination: p ∧ q → p | Logic | ✅ |
| Power Derivative: d/dx(x²) = 2x | Calculus | ✅ |
| Set Intersection Commutativity | Set Theory | ✅ |
| Even Plus Two | Number Theory | ❌ |

**Interpretation:** The 80% rate on undergraduate math proves the *infrastructure* works. The 0% on AIME demonstrates a **complexity ceiling**—not a failure of verification, but of LLM formalization capabilities for competition-level problems.

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | GPT-4o-mini | Generation, correction |
| **Vector Store** | ChromaDB | Semantic similarity |
| **Graph Store** | Neo4j | Concept relationships |
| **Embeddings** | all-mpnet-base-v2 | Text embeddings |
| **Reranker** | cross-encoder/ms-marco-MiniLM | Result reranking |
| **Theorem Prover** | Lean 4 + Mathlib v4.26.0 | Formal verification |
| **Verification** | SymPy | Symbolic math checks |

---

## Quantitative Summary

| Metric | Value |
|--------|-------|
| **Concept Nodes** | 1,697 |
| **ChromaDB Chunks** | 1,014 |
| **Cross-Source Edges** | 11 |
| **Misconception Nodes** | 26 |
| **Cross-Domain Retrieval** | ✅ Verified |
| **MCQ Misconception Coverage** | 26.7% |
| **Easy Math Verification** | 80% |
| **AIME Verification** | 0% |

---

## Conclusion

MathemaTest demonstrates that a **neuro-symbolic GraphRAG architecture** can:

1. ✅ Enable **inter-disciplinary reasoning** by connecting concepts across Physics and Calculus
2. ✅ Generate **pedagogically-grounded distractors** linked to real misconception nodes
3. ✅ Provide **formal verification** for undergraduate-level theorems

The **complexity ceiling** on Olympiad problems (0% vs 80%) reflects current LLM formalization limitations, not architectural failure. As foundation models improve in mathematical reasoning, the verification infrastructure is ready to scale.

### Future Directions

1. **Fine-tune formalization LLM** on Lean 4 proof corpora
2. **Expand misconception ontology** for higher coverage
3. **Interactive student profiling** based on distractor patterns
4. **Multi-hop reasoning chains** with explicit prerequisite tracking

---

## Appendix: File Structure

```
MATHEMATEST/
├── src/
│   ├── ingestion/          # PDF parsing, OCR, chunking
│   ├── graph_store/        # Neo4j client, graph constructor
│   ├── retrieval/          # Hybrid orchestrator, reranker
│   ├── generation/         # MCQ generator, Lean formalizer
│   └── verification/       # Lean compiler, SymPy sandbox
├── scripts/
│   ├── run_phase_3_diagnostic.py
│   ├── run_phase_4_bridge.py
│   └── run_easy_verification.py
├── mathematest/            # Lean 4 Mathlib project
│   ├── lakefile.toml
│   └── Mathematest/Verification/
├── docs/
│   ├── phase_3_results.md
│   ├── phase_4_verification.md
│   └── easy_math_results.md
└── tests/
    ├── bridge_problem.json
    └── easy_math.json
```

---

*Report generated: January 2, 2026*
