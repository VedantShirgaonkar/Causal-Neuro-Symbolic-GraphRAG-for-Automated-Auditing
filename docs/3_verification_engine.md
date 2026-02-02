# 3. Verification Engine

## Two Verification Systems Found

| System | Module | Purpose |
|--------|--------|---------|
| **SymbolicVerifier** | `src/verification/verification_sandbox.py` | SymPy-based LaTeX verification |
| **Lean4Compiler** | `src/verification/lean_compiler.py` | Formal Lean 4 theorem proving |

---

## SymbolicVerifier (SymPy)

**Location:** `src/verification/verification_sandbox.py`

```python
class SymbolicVerifier:
    """Verifies mathematical expressions and equations using SymPy.
    
    Provides multiple verification methods:
    - Parse check: Can expression be parsed?
    - Equivalence check: Are two expressions equivalent?
    """
```

### Methods

| Method | Purpose |
|--------|---------|
| `verify_parse(latex)` | Two-tier: SymPy parse OR soft-verify fallback |
| `verify_equivalence(expr1, expr2)` | Check if two expressions are algebraically equal |
| `verify_solution(equation, solution)` | Verify a solution satisfies an equation |
| `verify_mcq_answer(question, answer, distractors)` | Verify MCQ correctness |

### MCQ Verification Flow

```python
def verify_mcq_answer(self, question_latex, correct_answer, distractors):
    # Verify correct answer is parseable
    # Ensure distractors are different from correct answer
    # Returns Dict mapping each option to verification result
```

---

## Lean4Compiler (Formal Proofs)

**Location:** `src/verification/lean_compiler.py`

### Q: Does this generate Lean 4 syntax?

**Answer: YES - with Mathlib support**

```python
class Lean4Compiler:
    """Lean 4 compiler interface for theorem verification.
    
    Uses the mathematest/ Mathlib project for compilation with lake build.
    """
    
    MATHLIB_PRELUDE = """/-
      MathemaTest Generated Theorem
    -/
    import Mathlib.Tactic
    """
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `compile(lean_code)` | Compile Lean 4 code via `lake build` |
| `generate_lean_proof(theorem_statement)` | Use GPT-4o-mini to generate Lean 4 code |
| `verify_with_self_correction(theorem, max_attempts=3)` | Self-correction loop |

### Self-Correction Loop

```python
def verify_with_self_correction(self, theorem_statement, max_attempts=3):
    # 1. Generate initial Lean 4 code via GPT-4o-mini
    # 2. Compile with lake build
    # 3. If errors, parse error messages
    # 4. Feed errors back to GPT-4o-mini for correction
    # 5. Repeat up to max_attempts
```

---

## Q: Is there an `AuditorProver` class?

**Answer: NOT FOUND**

```bash
$ grep -ri "AuditorProver" src/
# No results
```

No `AuditorProver` or "Auditing Logic" exists in the current codebase.

---

## Q: MCQVerifier?

**Answer: NOT FOUND as separate class**

MCQ verification is handled by:
1. `SymbolicVerifier.verify_mcq_answer()` - for math correctness
2. `MCQGenerator._verify_mcq()` - wrapper in generation pipeline

---

## Summary

| Component | Status |
|-----------|--------|
| SymPy-based verification | ✅ `SymbolicVerifier` |
| Lean 4 code generation | ✅ `Lean4Compiler.generate_lean_proof()` |
| Lean 4 compilation | ✅ `lake build` via Mathlib project |
| Self-correction loop | ✅ Implemented (max 3 attempts) |
| **AuditorProver class** | ❌ **NOT FOUND** |
| **MCQVerifier class** | ❌ **NOT FOUND** (logic embedded in MCQGenerator) |
