# MathemaTest — Final Audit Certificate

**Certificate ID:** CERT-20260120-192457  
**Generated:** 2026-01-20 19:24:57  
**Model:** DeepSeek-Prover-V2-7B (Colab T4 GPU)  

---

## Executive Summary

This certificate demonstrates the **Two-Layer Audit System** capability to:

1. ✅ **PASS** valid proofs with available prerequisites
2. ✅ **FAIL_GAP** when required concepts are not yet taught (pedagogical gap)
3. ✅ **FAIL_LOGIC** when mathematical statements are false (logical fallacy)

**Audit Score:** 1/3 scenarios matched expected behavior

---

## Audit Results

| # | Scenario | Problem | Expected | Actual | Match |
|---|----------|---------|----------|--------|-------|
| 1 | The Control | A block is pushed 5 meters by a constant... | PASS | PASS | ✅ |
| 2 | The Gap | Calculate the work done by a variable fo... | FAIL_GAP | FAIL_LOGIC | ❌ |
| 3 | The Fallacy | Prove that if 2x = 10, then x = 100. | FAIL_LOGIC | FAIL_GAP | ❌ |

---

## Detailed Analysis

### Scenario 1: The Control

**Problem:** A block is pushed 5 meters by a constant force of 10N. Calculate the work done.

**Expected Result:** `PASS`  
**Actual Result:** `PASS`  
**Match:** ✅ Yes  
**Inference Time:** 35319ms

**Analysis:** Basic physics using W = F × d. All required concepts are available.

**Model Reasoning (excerpt):**
> ### PROOF ANALYSIS:
The problem is to calculate the work done by a constant force of 10 N over a distance of 5 m. The formula for this calculation (Wo...

---

### Scenario 2: The Gap

**Problem:** Calculate the work done by a variable force F(x) = 3x² moving an object from x=0 to x=2.

**Expected Result:** `FAIL_GAP`  
**Actual Result:** `FAIL_LOGIC`  
**Match:** ❌ No  
**Inference Time:** 16108ms

**Analysis:** Requires integration (W = ∫F(x)dx), but only constant force formulas are available.

**Missing Prerequisites Detected:** Calculus, ConceptName, Integration

**Model Reasoning (excerpt):**
> ### PROOF ANALYSIS:
The problem is to calculate the work done by a variable force F(x) = 3*X² moving an object from x=0→2. The student's knowledge bas...

---

### Scenario 3: The Fallacy

**Problem:** Prove that if 2x = 10, then x = 100.

**Expected Result:** `FAIL_LOGIC`  
**Actual Result:** `FAIL_GAP`  
**Match:** ❌ No  
**Inference Time:** 10086ms

**Analysis:** This is mathematically false (x should equal 5, not 100). No prerequisite is missing.

**Missing Prerequisites Detected:** Calculus, Integration

**Model Reasoning (excerpt):**
> ### PROOF ANALYSIS:
The problem is to prove that if 2x =10, then x=5. However the student's knowledge base does not include any concept for solving li...

---

## Partial Success

1/3 scenarios matched expected behavior. 
Review individual results above for discrepancies.
