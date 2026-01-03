#!/usr/bin/env python3
"""
End-to-End Experiment Runner for MathemaTest.

Generates MCQs with full logging of:
- Neo4j prerequisite paths
- SymPy verification traces
- Misconception usage for distractors
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings, BudgetTracker
from src.graph_store.neo4j_client import Neo4jClient, MockNeo4jClient
from src.graph_store.graph_constructor import GraphConstructorAgent, MockGraphConstructorAgent
from src.vector_store.chroma_client import ChromaVectorStore, MockChromaVectorStore
from src.retrieval.hybrid_orchestrator import HybridRetriever, MockHybridRetriever
from src.generation.mcq_generator import MCQGenerator, GeneratedMCQ, MockMCQGenerator
from src.verification.verification_sandbox import SymbolicVerifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentLog:
    """Detailed log for a single MCQ generation."""
    topic: str
    mcq_id: str
    
    # Retrieval trace
    prerequisite_path: List[Dict[str, Any]] = field(default_factory=list)
    vector_results: int = 0
    graph_results: int = 0
    
    # Verification trace
    sympy_attempts: int = 0
    sympy_passed: bool = False
    sympy_errors: List[str] = field(default_factory=list)
    
    # Misconception usage
    misconceptions_used: List[str] = field(default_factory=list)
    distractor_sources: Dict[str, str] = field(default_factory=dict)
    
    # Final status
    is_verified: bool = False
    confidence_score: float = 0.0
    generation_time_ms: int = 0


@dataclass
class ExperimentResult:
    """Complete experiment results."""
    experiment_id: str
    timestamp: str
    topics: List[str]
    total_mcqs_requested: int
    total_mcqs_generated: int
    total_verified: int
    verification_rate: float
    avg_confidence: float
    total_sympy_attempts: int
    self_correction_count: int
    budget_used: float
    logs: List[ExperimentLog] = field(default_factory=list)
    mcqs: List[Dict] = field(default_factory=list)


EXPERIMENT_TOPICS = [
    {
        "name": "Recursive Matrix Determinants",
        "query": "matrix determinant recursion cofactor expansion",
        "keywords": ["det", "matrix", "recursion", "cofactor"],
    },
    {
        "name": "Variable Mass Systems",
        "query": "variable mass rocket snowplough momentum conservation",
        "keywords": ["dm/dt", "momentum", "variable mass", "thrust"],
    },
    {
        "name": "Non-Linear Path Work Integrals",
        "query": "work energy integral non-linear path line integral",
        "keywords": ["work", "integral", "path", "non-conservative"],
    },
]


def run_experiment(
    use_mock: bool = False,
    mcqs_per_topic: int = 5,
    output_dir: Optional[Path] = None,
) -> ExperimentResult:
    """Run the full MCQ generation experiment.
    
    Args:
        use_mock: Use mock components (no API calls).
        mcqs_per_topic: Number of MCQs per topic.
        output_dir: Output directory for results.
        
    Returns:
        ExperimentResult with all data.
    """
    settings = get_settings()
    budget = BudgetTracker(settings)
    output_dir = output_dir or Path("experiment_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    if use_mock:
        logger.info("Using MOCK components")
        retriever = MockHybridRetriever()
        generator = MockMCQGenerator()
        verifier = SymbolicVerifier()
    else:
        logger.info("Using REAL components")
        retriever = HybridRetriever(settings=settings)
        generator = MCQGenerator(
            settings=settings,
            retriever=retriever,
            budget_tracker=budget,
        )
        verifier = generator.verifier
    
    result = ExperimentResult(
        experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        topics=[t["name"] for t in EXPERIMENT_TOPICS],
        total_mcqs_requested=len(EXPERIMENT_TOPICS) * mcqs_per_topic,
        total_mcqs_generated=0,
        total_verified=0,
        verification_rate=0.0,
        avg_confidence=0.0,
        total_sympy_attempts=0,
        self_correction_count=0,
        budget_used=0.0,
    )
    
    all_mcqs = []
    all_logs = []
    confidences = []
    
    for topic_info in EXPERIMENT_TOPICS:
        topic_name = topic_info["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"TOPIC: {topic_name}")
        logger.info(f"{'='*60}")
        
        for i in range(mcqs_per_topic):
            logger.info(f"\n[{i+1}/{mcqs_per_topic}] Generating MCQ...")
            
            start_time = datetime.now()
            
            # Create log for this MCQ
            log = ExperimentLog(topic=topic_name, mcq_id="")
            
            # Step 1: Retrieve context
            if not use_mock:
                context = retriever.retrieve(topic_info["query"], n_results=5)
                log.vector_results = context.vector_count
                log.graph_results = context.graph_count
                
                # Extract prerequisite path
                for res in context.results:
                    if res.metadata.get("type") == "prerequisite":
                        log.prerequisite_path.append({
                            "node": res.content,
                            "distance": res.metadata.get("distance", 0),
                        })
                    elif res.metadata.get("type") == "misconception":
                        log.misconceptions_used.append(res.content)
            
            # Step 2: Generate MCQ
            difficulty = ["easy", "medium", "hard"][i % 3]
            mcq = generator.generate(topic_name, difficulty=difficulty)
            
            if mcq:
                log.mcq_id = mcq.id
                log.is_verified = mcq.is_verified
                log.confidence_score = mcq.confidence_score
                log.sympy_attempts = mcq.verification_attempts
                log.sympy_passed = mcq.is_verified
                log.sympy_errors = mcq.verification_errors
                
                # Track misconceptions used in distractors
                for misc in mcq.misconceptions_addressed:
                    log.distractor_sources[misc[:30]] = "graph_misconception"
                
                end_time = datetime.now()
                log.generation_time_ms = int((end_time - start_time).total_seconds() * 1000)
                
                result.total_mcqs_generated += 1
                if mcq.is_verified:
                    result.total_verified += 1
                result.total_sympy_attempts += mcq.verification_attempts
                if mcq.verification_attempts > 1:
                    result.self_correction_count += 1
                
                confidences.append(mcq.confidence_score)
                all_mcqs.append(mcq.to_dict())
                
                logger.info(f"  ✓ Generated: {mcq.id}")
                logger.info(f"    Verified: {mcq.is_verified} | Attempts: {mcq.verification_attempts}")
                logger.info(f"    Confidence: {mcq.confidence_score:.2f}")
            else:
                logger.warning(f"  ✗ Generation failed")
            
            all_logs.append(log)
    
    # Calculate final metrics
    result.logs = all_logs
    result.mcqs = all_mcqs
    result.verification_rate = result.total_verified / max(1, result.total_mcqs_generated)
    result.avg_confidence = sum(confidences) / max(1, len(confidences))
    result.budget_used = budget.total_spent if not use_mock else 0.0
    
    # Save results
    mcq_output_path = output_dir / "verified_mcq_bank.json"
    with open(mcq_output_path, "w") as f:
        json.dump({
            "metadata": {
                "experiment_id": result.experiment_id,
                "timestamp": result.timestamp,
                "total_mcqs": result.total_mcqs_generated,
                "verified_count": result.total_verified,
                "verification_rate": result.verification_rate,
            },
            "mcqs": [m for m in all_mcqs if m.get("is_verified", False)],
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Generated: {result.total_mcqs_generated}/{result.total_mcqs_requested}")
    logger.info(f"Verified: {result.total_verified} ({result.verification_rate:.1%})")
    logger.info(f"Avg Confidence: {result.avg_confidence:.2f}")
    logger.info(f"Self-Corrections: {result.self_correction_count}")
    logger.info(f"Saved to: {mcq_output_path}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MathemaTest Experiment")
    parser.add_argument("--mock", action="store_true", help="Use mock components")
    parser.add_argument("--mcqs-per-topic", type=int, default=5, help="MCQs per topic")
    parser.add_argument("--output-dir", type=str, default="experiment_output")
    
    args = parser.parse_args()
    
    result = run_experiment(
        use_mock=args.mock,
        mcqs_per_topic=args.mcqs_per_topic,
        output_dir=Path(args.output_dir),
    )
    
    # Print summary
    print(json.dumps({
        "experiment_id": result.experiment_id,
        "total_generated": result.total_mcqs_generated,
        "total_verified": result.total_verified,
        "verification_rate": f"{result.verification_rate:.1%}",
        "avg_confidence": f"{result.avg_confidence:.2f}",
    }, indent=2))


if __name__ == "__main__":
    main()
