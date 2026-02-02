#!/usr/bin/env python3
"""
Test AuditorProver: Verify logical integrity auditing.

Tests:
1. Success: A simple theorem with sufficient prior context (should PASS).
2. Gap Detection: A theorem requiring concepts not in prior context (should FAIL_GAP).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verification.auditor_prover import AuditorProver


def test_auditor_prover():
    """Test the AuditorProver with success and gap scenarios."""
    
    print("=" * 60)
    print("AUDITORPROVER TEST")
    print("=" * 60)
    
    # Initialize the prover
    prover = AuditorProver()
    
    # =========================================================================
    # TEST 1: SUCCESS CASE
    # A simple theorem that should be provable with basic context
    # Using Chapter 2 context to verify a basic limit property
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("[TEST 1] BASIC THEOREM - Should PASS with prior context")
    print("-" * 60)
    
    simple_theorem = """
    If f(x) = c (a constant function), then the derivative f'(x) = 0.
    
    This follows from the definition of derivative as the limit of 
    the difference quotient: f'(x) = lim(h→0) [f(x+h) - f(x)] / h.
    For a constant function, f(x+h) = f(x) = c, so the numerator is 0.
    """
    
    # Using Chapter 5 context - derivatives are defined in Chapters 3-4
    result1 = prover.audit_theorem(
        theorem_text=simple_theorem,
        chapter=5,  # Testing with context from chapters 1-4 (includes derivatives)
    )
    
    print(f"Theorem: {simple_theorem[:80]}...")
    print(f"Chapter tested: {result1.chapter_tested}")
    print(f"Context items: {result1.context_count}")
    print(f"Context chapters: {result1.context_chapters}")
    print(f"\nSTATUS: {result1.status}")
    print(f"Reason: {result1.reason}")
    
    if result1.missing_prerequisites:
        print(f"Missing: {result1.missing_prerequisites}")
    
    # =========================================================================
    # TEST 2: GAP DETECTION
    # Fundamental Theorem of Calculus requires integration concepts
    # which wouldn't be available if we only use Chapter 1 context
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("[TEST 2] ADVANCED THEOREM - Should FAIL_GAP without context")
    print("-" * 60)
    
    advanced_theorem = """
    Fundamental Theorem of Calculus (Part 1):
    
    If f is continuous on [a, b] and F is defined by 
    F(x) = ∫[a to x] f(t) dt, then F is differentiable on (a, b) 
    and F'(x) = f(x).
    """
    
    result2 = prover.audit_theorem(
        theorem_text=advanced_theorem,
        chapter=2,  # Only Chapter 1 context - WAY before FTC is introduced
    )
    
    print(f"Theorem: {advanced_theorem[:80]}...")
    print(f"Chapter tested: {result2.chapter_tested}")
    print(f"Context items: {result2.context_count}")
    print(f"Context chapters: {result2.context_chapters}")
    print(f"\nSTATUS: {result2.status}")
    print(f"Reason: {result2.reason}")
    
    if result2.missing_prerequisites:
        print(f"Missing prerequisites: {result2.missing_prerequisites}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test1_passed = result1.status in ["VERIFIED", "VERIFIED_STRUCTURE", "PASS", "PASS_INFORMAL"]
    test2_passed = result2.status == "FAIL_GAP"
    
    print(f"Test 1 (Simple theorem should PASS): {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"    Status: {result1.status}")
    
    print(f"Test 2 (FTC should FAIL_GAP): {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"    Status: {result2.status}")
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("=" * 60)
    
    # Cleanup
    prover.close()
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    test_auditor_prover()
