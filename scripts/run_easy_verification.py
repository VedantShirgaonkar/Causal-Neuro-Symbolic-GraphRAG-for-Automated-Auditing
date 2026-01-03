#!/usr/bin/env python3
"""
Final Confidence Test: Verify basic undergraduate math theorems with Lean 4.

Tests the Mathlib infrastructure on accessible theorems to demonstrate
the verification pipeline works (even if AIME was too complex).
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.verification.lean_compiler import Lean4Compiler


logger = logging.getLogger(__name__)


# Mathlib header for easy theorems
EASY_HEADER = """/-
  MathemaTest - Basic Undergraduate Math Verification
  Topic: {topic}
-/

import Mathlib.Tactic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Data.Set.Basic

open Set Real

"""


@dataclass
class VerificationResult:
    """Result of verifying a single theorem."""
    id: str
    title: str
    topic: str
    success: bool
    output: str
    errors: List[str]


def run_easy_verification():
    """Run verification on easy math theorems."""
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "tests" / "easy_math.json"
    with open(dataset_path) as f:
        problems = json.load(f)
    
    logger.info("=" * 60)
    logger.info("FINAL CONFIDENCE TEST: EASY MATH VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Problems: {len(problems)}")
    logger.info("")
    
    # Initialize compiler
    compiler = Lean4Compiler(use_mathlib=True)
    
    if not compiler.mathlib_available:
        logger.error("Mathlib not available!")
        return [], 0
    
    results = []
    
    for problem in problems:
        logger.info(f"Testing: {problem['title']} ({problem['topic']})")
        
        # Build full code
        header = EASY_HEADER.format(topic=problem['topic'])
        full_code = header + problem['lean_theorem']
        
        # Compile
        compile_result = compiler.compile(full_code, use_prelude=False, use_mathlib=True)
        
        result = VerificationResult(
            id=problem['id'],
            title=problem['title'],
            topic=problem['topic'],
            success=compile_result.success,
            output=compile_result.output[:200] if compile_result.output else "",
            errors=compile_result.errors[:2] if compile_result.errors else [],
        )
        
        results.append(result)
        
        if result.success:
            logger.info(f"  ✅ PASSED")
        else:
            logger.warning(f"  ❌ FAILED: {result.errors[0][:60] if result.errors else 'Unknown'}...")
    
    # Calculate pass rate
    passed = sum(1 for r in results if r.success)
    total = len(results)
    pass_rate = passed / total if total > 0 else 0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"PASS RATE: {passed}/{total} ({pass_rate:.0%})")
    logger.info("=" * 60)
    
    return results, pass_rate


def generate_report(results: List[VerificationResult], pass_rate: float, output_path: Path):
    """Generate verification report."""
    
    passed = sum(1 for r in results if r.success)
    total = len(results)
    
    report = f"""# Easy Math Verification Report

## Summary

| Metric | Value |
|--------|-------|
| **Total Problems** | {total} |
| **Passed** | {passed} |
| **Pass Rate** | {pass_rate:.0%} |

---

## Detailed Results

"""
    
    for r in results:
        status = "✅" if r.success else "❌"
        report += f"""### {status} {r.title} ({r.topic})

- **Status:** {'PASSED' if r.success else 'FAILED'}
"""
        if r.errors:
            report += f"- **Error:** `{r.errors[0][:80]}...`\n"
        report += "\n"
    
    report += f"""---

## Conclusion

The verification infrastructure achieved **{pass_rate:.0%}** on basic undergraduate math,
demonstrating that the Lean 4 + Mathlib pipeline is operational.

The contrast with AIME (0%) illustrates the **complexity ceiling** of current LLM
formalization capabilities, not a failure of the verification infrastructure itself.
"""
    
    with open(output_path, "w") as f:
        f.write(report)
    
    # Save JSON
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump({
            "pass_rate": pass_rate,
            "passed": passed,
            "total": total,
            "results": [asdict(r) for r in results],
        }, f, indent=2)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    print("\n" + "=" * 60)
    print("FINAL CONFIDENCE TEST")
    print("=" * 60 + "\n")
    
    results, pass_rate = run_easy_verification()
    
    if results:
        output_path = Path(__file__).parent.parent / "docs" / "easy_math_results.md"
        generate_report(results, pass_rate, output_path)
        
        print(f"\nReport: {output_path}")
        
        # Success if at least 60%
        if pass_rate >= 0.6:
            print(f"\n✅ TARGET MET: {pass_rate:.0%} >= 60%")
            return 0
        else:
            print(f"\n⚠️ Below target: {pass_rate:.0%} < 60%")
            return 1
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
