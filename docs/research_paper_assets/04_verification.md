# 04: Neuro-Symbolic Verification

## Technical Specification

This document provides detailed technical extraction from `src/verification/` and `src/generation/` for research paper citation.

---

## 1. Lean 4 Compiler Bridge

### 1.1 Architecture Overview

**Module:** `src/verification/lean_compiler.py`

**Class:** `Lean4Compiler`

**Bridge Pattern:** Python ↔ Lean 4 via subprocess and file I/O

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Python Code    │────▶│  Temporary .lean │────▶│   lake build    │
│  (lean_compiler)│     │  file in project │     │  (subprocess)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Parse Errors   │◀────│  stderr/stdout   │◀────│  Compilation    │
│  for Correction │     │  capture         │     │  Result         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### 1.2 Path Configuration

```python
class Lean4Compiler:
    # Lean binary path from elan installation
    LEAN_PATH = Path.home() / ".elan" / "bin" / "lean"
    LAKE_PATH = Path.home() / ".elan" / "bin" / "lake"
    
    # Path to Mathlib project (relative to MATHEMATEST root)
    MATHLIB_PROJECT_DIR = Path(__file__).parent.parent.parent / "mathematest"
    VERIFICATION_DIR = MATHLIB_PROJECT_DIR / "Mathematest" / "Verification"
```

---

## 2. Subprocess Communication

### 2.1 Environment Setup

```python
def __init__(self, ...):
    # Set up environment with elan PATH
    self.env = dict(os.environ)
    elan_bin = str(Path.home() / ".elan" / "bin")
    if elan_bin not in self.env.get("PATH", ""):
        self.env["PATH"] = f"{elan_bin}:{self.env.get('PATH', '')}"
```

### 2.2 Lake Build Execution

**Method:** `_run_lake_build()`

```python
def _run_lake_build(self, lean_code: str) -> LeanCompilationResult:
    """Run lake build on Lean code within Mathlib project."""
    
    # Generate unique filename
    import uuid
    temp_name = f"Temp_{uuid.uuid4().hex[:8]}"
    temp_file = self.VERIFICATION_DIR / f"{temp_name}.lean"
    
    # Write temporary file
    with open(temp_file, "w") as f:
        f.write(lean_code)
    
    try:
        # Run lake build
        result = subprocess.run(
            [str(self.LAKE_PATH), "build", f"Mathematest.Verification.{temp_name}"],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(self.lean_project_path),
            env=self.env,
        )
        
        return LeanCompilationResult(
            success=result.returncode == 0,
            code=lean_code,
            output=result.stdout + result.stderr,
            errors=self._parse_lean_errors(result.stderr),
            warnings=self._parse_lean_warnings(result.stderr),
            error_locations=self._extract_error_locations(result.stderr),
        )
        
    finally:
        # Cleanup temporary file
        temp_file.unlink(missing_ok=True)
```

### 2.3 Subprocess Parameters

| Parameter | Value |
|-----------|-------|
| `timeout` | 120 seconds |
| `capture_output` | True |
| `text` | True (string output) |
| `cwd` | `mathematest/` project root |

---

## 3. Mathlib Integration

### 3.1 Prelude Template

```python
MATHLIB_PRELUDE = """/-
  MathemaTest Generated Theorem
  Uses Mathlib for formal verification
-/

import Mathlib.Tactic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic

open Real Set Function

"""
```

### 3.2 Available Tactics

Key Mathlib tactics used in verification:
- `ring` - Polynomial ring solving
- `simp` - Simplification with simp lemmas
- `omega` - Linear arithmetic
- `nlinarith` - Nonlinear arithmetic
- `field_simp` - Field simplification

---

## 4. Error Parsing

### 4.1 Error Extraction

**Method:** `_parse_lean_errors()`

```python
def _parse_lean_errors(self, stderr: str) -> List[str]:
    """Extract error messages from Lean compiler output."""
    errors = []
    for line in stderr.split('\n'):
        if 'error:' in line.lower():
            errors.append(line.strip())
    return errors
```

### 4.2 Error Location Extraction

**Method:** `_extract_error_locations()`

```python
def _extract_error_locations(self, stderr: str) -> List[Dict[str, Any]]:
    """Extract file:line:col from error messages."""
    locations = []
    # Pattern: filename.lean:line:col: error: message
    pattern = r'(\w+\.lean):(\d+):(\d+):\s*error:\s*(.+)'
    for match in re.finditer(pattern, stderr):
        locations.append({
            "file": match.group(1),
            "line": int(match.group(2)),
            "column": int(match.group(3)),
            "message": match.group(4),
        })
    return locations
```

---

## 5. Prompt Engineering

### 5.1 Lean 4 Generation Prompt

**Module:** `src/generation/lean4_formalizer.py`

```python
LEAN4_SYSTEM_PROMPT = """You are an expert in Lean 4 and Mathlib theorem proving.

Generate VALID Lean 4 code that compiles with Mathlib v4.26.0.

RULES:
1. Use `import Mathlib.Tactic` for access to all tactics
2. Use `by` for tactic blocks
3. Prefer simple tactics: `ring`, `simp`, `omega`, `nlinarith`
4. Use `sorry` ONLY if proof is genuinely unknown
5. Do NOT use deprecated Lean 3 syntax

OUTPUT FORMAT:
Return ONLY valid Lean 4 code, no markdown, no explanations.
"""
```

### 5.2 Self-Correction Prompt

```python
CORRECTION_PROMPT = """The following Lean 4 code failed to compile:

```lean
{failed_code}
```

Compiler errors:
{errors}

Fix the code to address these specific errors. Return ONLY the corrected Lean 4 code.
"""
```

---

## 6. Self-Correction Loop

### 6.1 Algorithm

```
ALGORITHM: SelfCorrectionLoop(theorem_statement, max_attempts=3)
INPUT: Natural language theorem, maximum correction attempts
OUTPUT: LeanCompilationResult (success or final failure)

1. INITIALIZE attempt ← 1

2. GENERATE initial Lean code via GPT-4o-mini
   lean_code ← LLM.generate(theorem_statement, LEAN4_SYSTEM_PROMPT)

3. WHILE attempt ≤ max_attempts:
   a. COMPILE
      result ← Lean4Compiler.compile(lean_code)
   
   b. IF result.success:
      RETURN result  # Success!
   
   c. EXTRACT errors from result
   
   d. GENERATE correction prompt
      correction_prompt ← format(CORRECTION_PROMPT, 
                                 failed_code=lean_code,
                                 errors=result.errors)
   
   e. REGENERATE
      lean_code ← LLM.generate(correction_prompt)
   
   f. INCREMENT attempt ← attempt + 1

4. RETURN last failed result
```

### 6.2 Implementation

```python
def verify_with_correction(
    self,
    theorem: str,
    max_attempts: int = 3,
) -> List[LeanProofAttempt]:
    """Verify theorem with self-correction loop."""
    
    attempts = []
    
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            lean_code = self._generate_initial_proof(theorem)
        else:
            lean_code = self._generate_correction(
                previous_code=attempts[-1].lean_code,
                errors=attempts[-1].result.errors,
            )
        
        result = self.compile(lean_code, use_mathlib=True)
        
        attempts.append(LeanProofAttempt(
            attempt_number=attempt,
            lean_code=lean_code,
            result=result,
        ))
        
        if result.success:
            break
    
    return attempts
```

---

## 7. Data Structures

### 7.1 LeanCompilationResult

```python
@dataclass
class LeanCompilationResult:
    success: bool                      # True if compilation succeeded
    code: str                          # The Lean code that was compiled
    output: str                        # Full stdout + stderr
    errors: List[str]                  # Extracted error messages
    warnings: List[str]                # Extracted warnings
    error_locations: List[Dict]        # Line/column locations
```

### 7.2 LeanProofAttempt

```python
@dataclass
class LeanProofAttempt:
    attempt_number: int               # 1, 2, or 3
    lean_code: str                    # Code attempted
    result: LeanCompilationResult     # Compilation result
    correction_prompt: Optional[str]  # Prompt used for correction
```

---

## 8. SymPy Integration

### 8.1 Symbolic Verification

**Module:** `src/verification/verification_sandbox.py`

```python
class SymbolicVerifier:
    """Uses SymPy for algebraic verification."""
    
    def verify_equality(self, lhs: str, rhs: str) -> bool:
        """Check if two expressions are algebraically equal."""
        from sympy import simplify, sympify
        return simplify(sympify(lhs) - sympify(rhs)) == 0
```

---

## 9. Configuration

### 9.1 Timeout Settings

| Component | Timeout |
|-----------|---------|
| Lake Build | 120 seconds |
| Lean Version Check | 10 seconds |
| LLM Generation | 60 seconds |

### 9.2 Lean/Mathlib Versions

| Component | Version |
|-----------|---------|
| Lean 4 | v4.26.0 |
| Mathlib | v4.26.0 |
| Toolchain | `leanprover/lean4:v4.26.0` |
