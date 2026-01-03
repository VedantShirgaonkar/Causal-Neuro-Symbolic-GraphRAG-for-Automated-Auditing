#!/usr/bin/env python3
"""
Realistic Stress Test for MathemaTest Pipeline.

Tests the OCR validator and LaTeX normalizer against challenging
mathematical expressions matching the complexity of:
- STEP: Differential equations (Snowplough problem)
- Physics: Work-Energy integrals with dot products
- Putnam: Matrix determinants and complex expressions
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ocr_utils import OCRValidator
from src.ingestion.latex_normalizer import LaTeXNormalizer
from src.models.schemas import BlockType


@dataclass
class TestCase:
    """A test case for stress testing."""
    name: str
    source: str  # STEP, Physics, Putnam
    description: str
    raw_latex: str  # Simulated OCR output (may have errors)
    expected_valid: bool
    expected_fixes: list


# =============================================================================
# STRESS TEST CASES
# =============================================================================

STRESS_TEST_CASES = [
    # =========================================================================
    # STEP: DIFFERENTIAL EQUATIONS (Snowplough Problem Style)
    # =========================================================================
    TestCase(
        name="STEP_Snowplough_DE_1",
        source="STEP",
        description="Snowplough differential equation with derivative notation",
        raw_latex=r"\frac{dm}{dt} = \rho v A",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="STEP_Snowplough_DE_2",
        source="STEP",
        description="Momentum equation with product rule",
        raw_latex=r"\frac{d}{dt}(mv) = m\frac{dv}{dt} + v\frac{dm}{dt}",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="STEP_Snowplough_DE_3_OCR_Error",
        source="STEP",
        description="OCR error: missing closing brace in derivative",
        raw_latex=r"\frac{d}{dt}(mv = F",
        expected_valid=False,
        expected_fixes=["balanced_braces"],
    ),
    TestCase(
        name="STEP_Snowplough_Integral",
        source="STEP",
        description="Integration with limits",
        raw_latex=r"\int_{0}^{t} F \, dt = mv - m_0 v_0",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="STEP_Subscript_Error",
        source="STEP",
        description="OCR error: bare multi-digit subscript",
        raw_latex=r"v_12 + v_23 = v_13",
        expected_valid=False,  # Multi-char subscripts need braces
        expected_fixes=["braced_subscripts"],
    ),
    
    # =========================================================================
    # PHYSICS: WORK-ENERGY THEOREM (Dot-Product Integrals)
    # =========================================================================
    TestCase(
        name="Physics_WorkEnergy_Basic",
        source="Physics",
        description="Basic work-energy integral",
        raw_latex=r"W = \int_{x_1}^{x_2} \vec{F} \cdot d\vec{x}",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Physics_WorkEnergy_KE",
        source="Physics",
        description="Kinetic energy with fraction",
        raw_latex=r"\Delta KE = \frac{1}{2}mv_f^{2} - \frac{1}{2}mv_i^{2}",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Physics_WorkEnergy_OCR_Frac",
        source="Physics",
        description="OCR error: fraction shorthand",
        raw_latex=r"\Delta KE = \frac12 mv^2",
        expected_valid=True,  # Will be normalized
        expected_fixes=["fixed_frac_shorthand"],
    ),
    TestCase(
        name="Physics_Line_Integral",
        source="Physics",
        description="Line integral with limits",
        raw_latex=r"W = \int_{a}^{b} \vec{F} \cdot d\vec{r} = \int_{a}^{b} F\cos\theta \, ds",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Physics_Vector_Components",
        source="Physics",
        description="Vector components with subscripts",
        raw_latex=r"F_x = F\cos\theta, \quad F_y = F\sin\theta",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Physics_Power_Formula",
        source="Physics",
        description="Power as rate of work",
        raw_latex=r"P = \frac{dW}{dt} = \vec{F} \cdot \vec{v}",
        expected_valid=True,
        expected_fixes=[],
    ),
    
    # =========================================================================
    # PUTNAM: MATRIX DETERMINANTS AND COMPLEX EXPRESSIONS
    # =========================================================================
    TestCase(
        name="Putnam_Determinant_2x2",
        source="Putnam",
        description="2x2 matrix determinant",
        raw_latex=r"\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Putnam_Determinant_3x3",
        source="Putnam",
        description="3x3 determinant with indices",
        raw_latex=r"\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Putnam_Series_Sum",
        source="Putnam",
        description="Infinite series with factorial",
        raw_latex=r"\sum_{n=0}^{\infty} \frac{x^{n}}{n!} = e^{x}",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Putnam_Limit_Expression",
        source="Putnam",
        description="Limit with complex denominator",
        raw_latex=r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^{n} = e",
        expected_valid=True,
        expected_fixes=[],
    ),
    TestCase(
        name="Putnam_Index_OCR_Error",
        source="Putnam",
        description="OCR error: matrix index without braces",
        raw_latex=r"a_ij = b_ji^T",
        expected_valid=False,
        expected_fixes=["braced_subscripts"],
    ),
    TestCase(
        name="Putnam_Nested_Fractions",
        source="Putnam",
        description="Continued fraction",
        raw_latex=r"\frac{1}{1 + \frac{1}{1 + \frac{1}{x}}}",
        expected_valid=True,
        expected_fixes=[],
    ),
    
    # =========================================================================
    # EDGE CASES: COMMON OCR HALLUCINATIONS
    # =========================================================================
    TestCase(
        name="OCR_Empty_Braces",
        source="OCR_Error",
        description="Empty brace group (common OCR artifact)",
        raw_latex=r"x + {} + y",
        expected_valid=False,
        expected_fixes=[],
    ),
    TestCase(
        name="OCR_Triple_Operator",
        source="OCR_Error",
        description="Triple operator (OCR seeing lines as operators)",
        raw_latex=r"x +++ y",
        expected_valid=False,
        expected_fixes=[],
    ),
    TestCase(
        name="OCR_Backslash_Digit",
        source="OCR_Error",
        description="Backslash followed by digit (invalid command)",
        raw_latex=r"\1 + \2 = \3",
        expected_valid=False,
        expected_fixes=[],
    ),
    TestCase(
        name="OCR_Consecutive_Subscripts",
        source="OCR_Error",
        description="Double underscore (OCR artifact)",
        raw_latex=r"x__n",
        expected_valid=False,
        expected_fixes=[],
    ),
    TestCase(
        name="OCR_Long_Subscript",
        source="OCR_Error",
        description="Very long subscript without braces",
        raw_latex=r"x_initial + y_final",
        expected_valid=False,
        expected_fixes=["braced_subscripts"],
    ),
]


def run_stress_test():
    """Run all stress test cases and collect results."""
    validator = OCRValidator(strict_mode=True)
    normalizer = LaTeXNormalizer(validator=OCRValidator(strict_mode=False))
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(STRESS_TEST_CASES),
        "passed": 0,
        "failed": 0,
        "by_source": {},
        "cases": [],
    }
    
    print("=" * 70)
    print("MathemaTest Stress Test - Realistic STEM LaTeX")
    print("=" * 70)
    print()
    
    for tc in STRESS_TEST_CASES:
        # Initialize source bucket
        if tc.source not in results["by_source"]:
            results["by_source"][tc.source] = {"passed": 0, "failed": 0, "cases": []}
        
        # Run validation
        validation_result = validator.validate(tc.raw_latex)
        
        # Run normalization
        normalized_result = normalizer.normalize(tc.raw_latex)
        
        # Check if result matches expectation
        validation_matches = validation_result.is_valid == tc.expected_valid
        
        # Check if expected fixes were applied
        fixes_applied = set(normalized_result.normalization_applied)
        expected_fixes_set = set(tc.expected_fixes)
        fixes_match = expected_fixes_set.issubset(fixes_applied)
        
        passed = validation_matches and fixes_match
        
        if passed:
            results["passed"] += 1
            results["by_source"][tc.source]["passed"] += 1
            status = "✓ PASS"
        else:
            results["failed"] += 1
            results["by_source"][tc.source]["failed"] += 1
            status = "✗ FAIL"
        
        case_result = {
            "name": tc.name,
            "source": tc.source,
            "description": tc.description,
            "raw_latex": tc.raw_latex,
            "normalized_latex": normalized_result.normalized,
            "expected_valid": tc.expected_valid,
            "actual_valid": validation_result.is_valid,
            "validation_matches": validation_matches,
            "expected_fixes": tc.expected_fixes,
            "actual_fixes": list(fixes_applied),
            "sympy_compatible": normalized_result.sympy_compatible,
            "passed": passed,
            "errors": [e.model_dump() for e in validation_result.errors],
            "confidence": validation_result.confidence_score,
        }
        
        results["cases"].append(case_result)
        results["by_source"][tc.source]["cases"].append(case_result)
        
        # Print result
        print(f"{status} {tc.name}")
        print(f"     Source: {tc.source} | {tc.description}")
        print(f"     Raw:    {tc.raw_latex[:60]}{'...' if len(tc.raw_latex) > 60 else ''}")
        print(f"     Norm:   {normalized_result.normalized[:60]}{'...' if len(normalized_result.normalized) > 60 else ''}")
        print(f"     Valid:  {validation_result.is_valid} (expected {tc.expected_valid})")
        print(f"     Fixes:  {list(fixes_applied)}")
        print(f"     SymPy:  {'✓' if normalized_result.sympy_compatible else '✗'}")
        
        if validation_result.errors:
            for err in validation_result.errors[:2]:
                print(f"     Error:  [{err.error_type}] {err.message}")
        print()
    
    return results


def generate_markdown_report(results: dict) -> str:
    """Generate a markdown stress test report."""
    lines = [
        "# Stress Test Report: STEM LaTeX Validation",
        "",
        f"**Generated:** {results['timestamp']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Test Cases | {results['total_cases']} |",
        f"| Passed | {results['passed']} |",
        f"| Failed | {results['failed']} |",
        f"| Pass Rate | {100*results['passed']/results['total_cases']:.1f}% |",
        "",
        "## Results by Source",
        "",
    ]
    
    for source, data in results["by_source"].items():
        total = data["passed"] + data["failed"]
        rate = 100 * data["passed"] / max(1, total)
        lines.append(f"### {source}")
        lines.append(f"- Passed: {data['passed']}/{total} ({rate:.0f}%)")
        lines.append("")
    
    # Detailed results
    lines.extend([
        "## Detailed Results",
        "",
    ])
    
    for source in ["STEP", "Physics", "Putnam", "OCR_Error"]:
        if source not in results["by_source"]:
            continue
        
        lines.append(f"### {source} Cases")
        lines.append("")
        
        for case in results["by_source"][source]["cases"]:
            status = "✓" if case["passed"] else "✗"
            lines.append(f"#### {status} {case['name']}")
            lines.append(f"*{case['description']}*")
            lines.append("")
            lines.append("**Input:**")
            lines.append("```latex")
            lines.append(case["raw_latex"])
            lines.append("```")
            lines.append("")
            lines.append("**Normalized:**")
            lines.append("```latex")
            lines.append(case["normalized_latex"])
            lines.append("```")
            lines.append("")
            lines.append(f"- Valid: {case['actual_valid']} | SymPy: {'✓' if case['sympy_compatible'] else '✗'} | Confidence: {case['confidence']:.2f}")
            
            if case["actual_fixes"]:
                lines.append(f"- Fixes Applied: `{', '.join(case['actual_fixes'])}`")
            
            if case["errors"]:
                lines.append("- Errors Caught:")
                for err in case["errors"][:3]:
                    lines.append(f"  - `{err['error_type']}`: {err['message']}")
            
            lines.append("")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    results = run_stress_test()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {results['total_cases']} | Passed: {results['passed']} | Failed: {results['failed']}")
    print(f"Pass Rate: {100*results['passed']/results['total_cases']:.1f}%")
    print()
    
    for source, data in results["by_source"].items():
        total = data["passed"] + data["failed"]
        print(f"  {source}: {data['passed']}/{total} passed")
    
    # Save reports
    output_dir = Path(__file__).parent / "stress_test_output"
    output_dir.mkdir(exist_ok=True)
    
    # JSON
    json_path = output_dir / "stress_test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to: {json_path}")
    
    # Markdown
    md_report = generate_markdown_report(results)
    md_path = output_dir / "stress_test_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Markdown saved to: {md_path}")
    
    # Exit code
    if results["failed"] > 0:
        print(f"\n⚠️  {results['failed']} test cases failed!")
        return 1
    else:
        print("\n✓ All stress tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
