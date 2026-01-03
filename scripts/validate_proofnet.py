#!/usr/bin/env python3
"""
ProofNet Validation with Lean 4 Formalization.

Selects theorems from GoldStandard ProofNet nodes and generates
Lean 4 formalizations with verification.

Phase C Task 4: Proof-Verification Success Rate tracking.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.graph_store.neo4j_client import Neo4jClient
from src.verification.lean_compiler import Lean4Compiler, LeanProofAttempt


logger = logging.getLogger(__name__)


@dataclass
class ProofNetValidationResult:
    """Result of validating a ProofNet theorem."""
    theorem_id: str
    natural_language: str
    original_lean: str
    generated_lean: str
    verification_success: bool
    attempts: int
    errors: List[str]
    time_seconds: float


class ProofNetValidator:
    """Validate ProofNet theorems with Lean 4 formalization."""
    
    def __init__(self):
        self.settings = get_settings()
        self.neo4j = Neo4jClient()
        self.lean_compiler = Lean4Compiler()
        self.results: List[ProofNetValidationResult] = []
    
    def get_proofnet_theorems(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get theorems from GoldStandard nodes.
        
        Args:
            n: Number of theorems to retrieve.
            
        Returns:
            List of theorem dictionaries.
        """
        query = """
        MATCH (g:GoldStandard)
        WHERE g.natural_language <> '' AND g.lean_statement <> ''
        RETURN g.theorem_id as id, g.natural_language as nl, 
               g.lean_statement as lean, g.topic as topic
        LIMIT $n
        """
        
        theorems = []
        with self.neo4j.session() as session:
            result = session.run(query, {"n": n})
            for r in result:
                theorems.append({
                    "id": r["id"],
                    "natural_language": r["nl"],
                    "original_lean": r["lean"],
                    "topic": r["topic"],
                })
        
        logger.info(f"Retrieved {len(theorems)} ProofNet theorems")
        return theorems
    
    def validate_theorem(
        self,
        theorem: Dict[str, Any],
        max_attempts: int = 3,
    ) -> ProofNetValidationResult:
        """Validate a single theorem with Lean 4.
        
        Args:
            theorem: Theorem dictionary.
            max_attempts: Maximum self-correction attempts.
            
        Returns:
            ProofNetValidationResult.
        """
        import time
        start_time = time.time()
        
        logger.info(f"Validating theorem: {theorem['id']}")
        
        # Try verification with self-correction
        success, attempts = self.lean_compiler.verify_with_self_correction(
            theorem_statement=theorem["natural_language"],
            max_attempts=max_attempts,
        )
        
        elapsed = time.time() - start_time
        
        # Get the final generated code
        final_code = attempts[-1].lean_code if attempts else ""
        final_errors = attempts[-1].result.errors if attempts else []
        
        return ProofNetValidationResult(
            theorem_id=theorem["id"],
            natural_language=theorem["natural_language"],
            original_lean=theorem.get("original_lean", ""),
            generated_lean=final_code,
            verification_success=success,
            attempts=len(attempts),
            errors=final_errors,
            time_seconds=elapsed,
        )
    
    def validate_batch(
        self,
        theorems: List[Dict[str, Any]],
        max_attempts: int = 2,
    ) -> List[ProofNetValidationResult]:
        """Validate a batch of theorems.
        
        Args:
            theorems: List of theorem dictionaries.
            max_attempts: Maximum self-correction attempts per theorem.
            
        Returns:
            List of validation results.
        """
        from tqdm import tqdm
        
        results = []
        for theorem in tqdm(theorems, desc="Validating theorems"):
            result = self.validate_theorem(theorem, max_attempts)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def compute_stats(self) -> Dict[str, Any]:
        """Compute validation statistics."""
        if not self.results:
            return {}
        
        total = len(self.results)
        success_count = sum(1 for r in self.results if r.verification_success)
        total_attempts = sum(r.attempts for r in self.results)
        total_time = sum(r.time_seconds for r in self.results)
        
        return {
            "total_theorems": total,
            "verified_count": success_count,
            "verification_rate": success_count / total if total > 0 else 0,
            "total_attempts": total_attempts,
            "avg_attempts": total_attempts / total if total > 0 else 0,
            "total_time_seconds": total_time,
            "avg_time_per_theorem": total_time / total if total > 0 else 0,
        }
    
    def generate_report(self, output_path: Path) -> str:
        """Generate validation report.
        
        Args:
            output_path: Path to save report.
            
        Returns:
            Report markdown content.
        """
        stats = self.compute_stats()
        
        report = f"""# ProofNet Lean 4 Validation Report

## Phase C: Formal Verification Integration

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Theorems Tested** | {stats.get('total_theorems', 0)} |
| **Verified Successfully** | {stats.get('verified_count', 0)} |
| **Verification Rate** | {stats.get('verification_rate', 0):.1%} |
| **Average Attempts** | {stats.get('avg_attempts', 0):.2f} |
| **Total Time** | {stats.get('total_time_seconds', 0):.1f}s |

---

## Detailed Results

"""
        
        for r in self.results:
            status = "✅" if r.verification_success else "❌"
            report += f"""### {status} {r.theorem_id}

**Natural Language:** {r.natural_language[:200]}...

**Verification:** {'Success' if r.verification_success else 'Failed'}
- Attempts: {r.attempts}
- Time: {r.time_seconds:.2f}s

"""
            if not r.verification_success and r.errors:
                report += f"**Errors:** {r.errors[0][:100]}...\n\n"
        
        report += """---

## Conclusion

"""
        if stats.get('verification_rate', 0) > 0.5:
            report += "The Lean 4 verification pipeline is working effectively.\n"
        else:
            report += "The Lean 4 verification pipeline requires further development.\n"
        
        # Save report
        with open(output_path, "w") as f:
            f.write(report)
        
        # Save results JSON
        results_json = output_path.with_suffix(".json")
        with open(results_json, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        return report
    
    def close(self):
        """Close connections."""
        self.neo4j.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ProofNet Lean 4 Validation")
    parser.add_argument("-n", "--num-theorems", type=int, default=5, help="Number of theorems to validate")
    parser.add_argument("--max-attempts", type=int, default=2, help="Max self-correction attempts")
    parser.add_argument("-o", "--output", type=str, default="docs/proofnet_validation_report.md", help="Output path")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    validator = ProofNetValidator()
    
    try:
        # Get theorems
        theorems = validator.get_proofnet_theorems(args.num_theorems)
        
        if not theorems:
            logger.error("No ProofNet theorems found!")
            return
        
        # Validate
        logger.info(f"Validating {len(theorems)} theorems...")
        results = validator.validate_batch(theorems, args.max_attempts)
        
        # Generate report
        output_path = Path(args.output)
        report = validator.generate_report(output_path)
        
        # Print summary
        stats = validator.compute_stats()
        
        print("\n" + "=" * 60)
        print("PROOFNET LEAN 4 VALIDATION COMPLETE")
        print("=" * 60)
        print(f"Theorems Tested: {stats['total_theorems']}")
        print(f"Verified: {stats['verified_count']}")
        print(f"Verification Rate: {stats['verification_rate']:.1%}")
        print(f"Report: {output_path}")
    
    finally:
        validator.close()


if __name__ == "__main__":
    main()
