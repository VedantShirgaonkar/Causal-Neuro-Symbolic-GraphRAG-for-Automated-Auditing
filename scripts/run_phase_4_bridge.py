#!/usr/bin/env python3
"""
Phase 4: Cross-Domain Retrieval Test.

Tests that the Knowledge Graph enables inter-domain reasoning by retrieving
context from BOTH Physics and Calculus sources for a bridge problem.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.retrieval.hybrid_orchestrator import HybridRetriever, ReasoningPacket
from src.graph_store.neo4j_client import Neo4jClient


# Set up logging to file
LOG_PATH = Path(__file__).parent.parent / "logs" / "bridge_trace.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalTrace:
    """Trace of retrieved nodes for verification."""
    physics_nodes: List[Dict[str, Any]]
    calculus_nodes: List[Dict[str, Any]]
    other_nodes: List[Dict[str, Any]]
    total_retrieved: int
    has_physics: bool
    has_calculus: bool
    cross_domain_verified: bool


def classify_source(source_id: str, content: str) -> str:
    """Classify a retrieved chunk by its source domain."""
    source_lower = (source_id or "").lower()
    content_lower = (content or "").lower()
    
    # Physics indicators
    physics_keywords = ["physics", "force", "work", "energy", "kinetic", "potential", "newton", "motion"]
    if "physics" in source_lower or any(kw in source_lower for kw in physics_keywords):
        return "physics"
    if any(kw in content_lower for kw in ["work done", "force field", "kinetic energy", "potential energy"]):
        return "physics"
    
    # Calculus indicators
    calculus_keywords = ["calculus", "integral", "derivative", "mit", "differentiation", "integration"]
    if "calculus" in source_lower or "mit" in source_lower or any(kw in source_lower for kw in calculus_keywords):
        return "calculus"
    if any(kw in content_lower for kw in ["line integral", "parameterization", "vector field", "∫", "integral"]):
        return "calculus"
    
    return "other"


def run_cross_domain_retrieval(problem: Dict[str, Any]) -> RetrievalTrace:
    """Run hybrid retrieval and trace the sources."""
    
    logger.info("=" * 60)
    logger.info("PHASE 4: CROSS-DOMAIN RETRIEVAL TEST")
    logger.info("=" * 60)
    logger.info(f"Problem: {problem['title']}")
    logger.info(f"Question: {problem['question']}")
    logger.info("")
    
    # Initialize retriever
    settings = get_settings()
    retriever = HybridRetriever(settings=settings)
    
    # Run retrieval
    query = problem["question"]
    logger.info(f"Running hybrid retrieval for query...")
    
    try:
        packet = retriever.retrieve(query, use_refinement=True, n_results=15)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        # Try without refinement
        packet = retriever.retrieve(query, use_refinement=False, n_results=15)
    
    logger.info(f"Retrieved {len(packet.results)} results")
    logger.info(f"Vector results: {packet.vector_count}")
    logger.info(f"Graph results: {packet.graph_count}")
    logger.info("")
    
    # Classify results
    physics_nodes = []
    calculus_nodes = []
    other_nodes = []
    
    logger.info("=" * 60)
    logger.info("RETRIEVAL TRACE - Node Classification")
    logger.info("=" * 60)
    
    for i, result in enumerate(packet.results):
        source_id = result.metadata.get("source_id", result.metadata.get("source", "unknown"))
        domain = classify_source(str(source_id), result.content)
        
        node_info = {
            "id": result.id,
            "source_id": source_id,
            "source_type": result.source,
            "score": result.score,
            "content_preview": result.content[:100] + "..." if len(result.content) > 100 else result.content,
            "domain": domain,
        }
        
        logger.info(f"Node {i+1}: [{domain.upper()}] source={source_id}")
        logger.info(f"  Content: {node_info['content_preview'][:80]}...")
        
        if domain == "physics":
            physics_nodes.append(node_info)
        elif domain == "calculus":
            calculus_nodes.append(node_info)
        else:
            other_nodes.append(node_info)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Physics nodes: {len(physics_nodes)}")
    logger.info(f"Calculus nodes: {len(calculus_nodes)}")
    logger.info(f"Other nodes: {len(other_nodes)}")
    
    # Determine if cross-domain verified
    has_physics = len(physics_nodes) > 0
    has_calculus = len(calculus_nodes) > 0
    cross_domain_verified = has_physics and has_calculus
    
    logger.info("")
    if cross_domain_verified:
        logger.info("✅ CROSS-DOMAIN LINK: VERIFIED")
        logger.info("   Retrieved nodes from BOTH Physics AND Calculus sources")
    else:
        logger.error("❌ CROSS-DOMAIN LINK: FAILED")
        if not has_physics:
            logger.error("   Missing: Physics nodes")
        if not has_calculus:
            logger.error("   Missing: Calculus nodes")
    
    retriever.close()
    
    return RetrievalTrace(
        physics_nodes=physics_nodes,
        calculus_nodes=calculus_nodes,
        other_nodes=other_nodes,
        total_retrieved=len(packet.results),
        has_physics=has_physics,
        has_calculus=has_calculus,
        cross_domain_verified=cross_domain_verified,
    )


def generate_report(problem: Dict, trace: RetrievalTrace, output_path: Path) -> str:
    """Generate Phase 4 verification report."""
    
    verdict = "✅ CROSS-DOMAIN LINK: VERIFIED" if trace.cross_domain_verified else "❌ CROSS-DOMAIN LINK: FAILED"
    
    report = f"""# Phase 4: Cross-Domain Retrieval Verification

## The Problem Statement

**Title:** {problem['title']}

**Question:** {problem['question']}

**Domain Requirements:**
- **Physics:** {problem['domain_requirements']['physics']}
- **Calculus:** {problem['domain_requirements']['calculus']}

---

## Retrieved Nodes

| # | Domain | Source ID | Content Preview |
|---|--------|-----------|-----------------|
"""
    
    all_nodes = [(n, "Physics") for n in trace.physics_nodes] + \
                [(n, "Calculus") for n in trace.calculus_nodes] + \
                [(n, "Other") for n in trace.other_nodes]
    
    for i, (node, domain) in enumerate(all_nodes[:10], 1):
        preview = node['content_preview'][:50].replace('|', '\\|').replace('\n', ' ')
        report += f"| {i} | {domain} | {node['source_id']} | {preview}... |\n"
    
    report += f"""
---

## Domain Coverage

| Domain | Nodes Retrieved | Status |
|--------|-----------------|--------|
| **Physics** | {len(trace.physics_nodes)} | {'✅ Found' if trace.has_physics else '❌ Missing'} |
| **Calculus** | {len(trace.calculus_nodes)} | {'✅ Found' if trace.has_calculus else '❌ Missing'} |
| **Other** | {len(trace.other_nodes)} | — |
| **Total** | {trace.total_retrieved} | — |

---

## The Solution

Given F = (2xy, x²) and path C: y = x² from (0,0) to (1,1):

1. **Parameterize:** x = t, y = t², t ∈ [0,1]
2. **dr = (dt, 2t dt)**
3. **F(t) = (2t³, t²)**
4. **F·dr = 2t³ dt + 2t³ dt = 4t³ dt**
5. **W = ∫₀¹ 4t³ dt = [t⁴]₀¹ = 1**

**Answer:** W = 1

---

## Verdict

# {verdict}

"""
    
    if trace.cross_domain_verified:
        report += """
The hybrid retrieval system successfully retrieved context from **both** Physics 
(Work-Energy concepts) and Calculus (Line Integral techniques) sources.

This demonstrates the Knowledge Graph's ability to perform **inter-domain reasoning**
by connecting concepts across different academic disciplines.
"""
    else:
        report += """
The retrieval system failed to connect concepts across domains.

**Action Required:** Ensure cross-source edges exist in Neo4j linking Physics 
concepts (Work, Energy) to Calculus concepts (Integration, Line Integrals).
"""
    
    # Save report
    with open(output_path, "w") as f:
        f.write(report)
    
    # Save trace JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "problem": problem,
            "trace": {
                "physics_nodes": trace.physics_nodes,
                "calculus_nodes": trace.calculus_nodes,
                "other_nodes": trace.other_nodes,
                "total": trace.total_retrieved,
                "has_physics": trace.has_physics,
                "has_calculus": trace.has_calculus,
                "verified": trace.cross_domain_verified,
            },
            "generated_at": datetime.now().isoformat(),
        }, f, indent=2)
    
    return report


def main():
    """Main entry point."""
    
    # Load bridge problem
    problem_path = Path(__file__).parent.parent / "tests" / "bridge_problem.json"
    with open(problem_path) as f:
        problem = json.load(f)
    
    # Run retrieval
    trace = run_cross_domain_retrieval(problem)
    
    # Generate report
    output_path = Path(__file__).parent.parent / "docs" / "phase_4_verification.md"
    generate_report(problem, trace, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print(f"Physics nodes: {len(trace.physics_nodes)}")
    print(f"Calculus nodes: {len(trace.calculus_nodes)}")
    print(f"Cross-Domain: {'VERIFIED ✅' if trace.cross_domain_verified else 'FAILED ❌'}")
    print(f"Report: {output_path}")
    print(f"Trace Log: {LOG_PATH}")
    
    # Return exit code
    return 0 if trace.cross_domain_verified else 1


if __name__ == "__main__":
    sys.exit(main())
