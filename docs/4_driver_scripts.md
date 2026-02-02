# 4. Driver Scripts

## Scripts Directory: `scripts/`

| Script | Purpose | Lines |
|--------|---------|-------|
| `run_phase_3_diagnostic.py` | **MOST ACTIVE** - Diagnostic MCQ generation | 556 |
| `run_phase_c_live.py` | Live Lean 4 verification pipeline | 417 |
| `run_experiment.py` | General experiment runner | 280 |
| `seed_knowledge_base.py` | Initialize Neo4j/ChromaDB | 220 |
| `batch_ingestion.py` | PDF ingestion pipeline | 505 |
| `verify_aime_mathlib.py` | AIME Mathlib verification | 449 |

---

## Most Active Script: `run_phase_3_diagnostic.py`

### Purpose
Transforms MathemaTest from "Pure Solver" to "Diagnostic Teacher" using Neo4j misconception nodes.

### Main Execution Block

```python
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 3: Diagnostic MCQ Generation")
    parser.add_argument("-o", "--output", type=str, default="docs/phase_3_results.md")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    print("\n" + "=" * 60)
    print("PHASE 3: DIAGNOSTIC MCQ GENERATION")
    print("=" * 60 + "\n")
    
    pipeline = Phase3DiagnosticPipeline()
    
    try:
        # Generate diagnostic MCQs
        results = pipeline.run_all()
        
        # Generate report
        output_path = Path(args.output)
        pipeline.generate_report(output_path)
        
        # Summary
        total = sum(len(m.distractors) for m in results)
        graph = sum(m.misconceptions_from_graph for m in results)
        coverage = graph / total if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("PHASE 3 COMPLETE")
        print("=" * 60)
        print(f"MCQs Generated: {len(results)}")
        print(f"Total Distractors: {total}")
        print(f"Graph-Backed: {graph} ({coverage:.1%})")
        print(f"Report: {output_path}")
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
```

### Key Classes in Script

| Class | Purpose |
|-------|---------|
| `SmartDistractorEngine` | Queries Neo4j for misconceptions |
| `Phase3DiagnosticPipeline` | Orchestrates MCQ generation |
| `SmartDistractor` | Data class for misconception-backed distractors |
| `DiagnosticMCQ` | Data class for full MCQ with diagnostic metadata |

---

## Secondary Driver: `seed_knowledge_base.py`

**Purpose:** Initialize Neo4j and ChromaDB with textbook content.

```python
# Typical usage:
python scripts/seed_knowledge_base.py --pdf data/textbook.pdf
```

---

## Ingestion Driver: `batch_ingestion.py`

**Purpose:** Process PDFs, extract formulas, populate vector store.

```python
# Typical usage:
python scripts/batch_ingestion.py --input data/pdfs/ --output data/chroma_db/
```

---

## Summary

| Driver | Primary Use Case |
|--------|------------------|
| `run_phase_3_diagnostic.py` | MCQ generation with misconception distractors |
| `seed_knowledge_base.py` | Database initialization |
| `batch_ingestion.py` | PDF processing pipeline |
| `run_experiment.py` | End-to-end experiments |
| `verify_aime_mathlib.py` | Lean 4 theorem verification |
