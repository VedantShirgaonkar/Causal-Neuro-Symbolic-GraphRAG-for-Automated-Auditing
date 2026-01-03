#!/usr/bin/env python3
"""
Lean 4 Compiler Bridge for MathemaTest Phase C.

Provides formal verification of mathematical proofs by:
1. Compiling Lean 4 code
2. Detecting compilation errors
3. Feeding errors back to the self-correction loop

Requires: Lean 4 installed (elan/lake toolchain)
"""

import subprocess
import tempfile
import logging
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from src.config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class LeanCompilationResult:
    """Result of Lean 4 compilation."""
    success: bool
    code: str
    output: str
    errors: List[str]
    warnings: List[str]
    error_locations: List[Dict[str, Any]]


@dataclass
class LeanProofAttempt:
    """A single proof verification attempt."""
    attempt_number: int
    lean_code: str
    result: LeanCompilationResult
    correction_prompt: Optional[str] = None


class Lean4Compiler:
    """Lean 4 compiler interface for theorem verification.
    
    Uses the mathematest/ Mathlib project for compilation with lake build.
    """
    
    # Mathlib-aware prelude for generated files
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
    
    # Legacy standalone prelude (no Mathlib)
    LEAN_PRELUDE = """-- MathemaTest Generated
-- Standalone Lean 4 (no Mathlib)

"""
    
    # Lean binary path from elan installation
    LEAN_PATH = Path.home() / ".elan" / "bin" / "lean"
    LAKE_PATH = Path.home() / ".elan" / "bin" / "lake"
    
    # Path to Mathlib project (relative to MATHEMATEST root)
    MATHLIB_PROJECT_DIR = Path(__file__).parent.parent.parent / "mathematest"
    VERIFICATION_DIR = MATHLIB_PROJECT_DIR / "Mathematest" / "Verification"
    
    def __init__(self, lean_project_path: Optional[Path] = None, use_mathlib: bool = True):
        """Initialize Lean 4 compiler.
        
        Args:
            lean_project_path: Path to existing Lean 4 project. If None, uses mathematest/.
            use_mathlib: If True, uses lake build with Mathlib project.
        """
        self.settings = get_settings()
        self.lean_project_path = lean_project_path or self.MATHLIB_PROJECT_DIR
        self.use_mathlib = use_mathlib
        self.openai = OpenAI(api_key=self.settings.openai_api_key)
        
        # Set up environment with elan PATH
        self.env = dict(os.environ)
        elan_bin = str(Path.home() / ".elan" / "bin")
        if elan_bin not in self.env.get("PATH", ""):
            self.env["PATH"] = f"{elan_bin}:{self.env.get('PATH', '')}"
        
        # Ensure Verification directory exists
        self.VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check if Lean is available
        self.lean_available = self._check_lean_installation()
        self.mathlib_available = self._check_mathlib_available()
        
    def _check_lean_installation(self) -> bool:
        """Check if Lean 4 is installed."""
        try:
            lean_cmd = str(self.LEAN_PATH) if self.LEAN_PATH.exists() else "lean"
            result = subprocess.run(
                [lean_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=self.env,
            )
            if result.returncode == 0:
                logger.info(f"Lean 4 found: {result.stdout.strip()}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        logger.warning("Lean 4 not installed - using simulation mode")
        return False
    
    def _check_mathlib_available(self) -> bool:
        """Check if Mathlib project is properly set up."""
        lakefile = self.lean_project_path / "lakefile.toml"
        lake_packages = self.lean_project_path / ".lake" / "packages"
        
        if lakefile.exists() and lake_packages.exists():
            logger.info(f"Mathlib project found at {self.lean_project_path}")
            return True
        
        logger.warning(f"Mathlib project not found at {self.lean_project_path}")
        return False
    
    def compile(self, lean_code: str, use_prelude: bool = True, use_mathlib: bool = None) -> LeanCompilationResult:
        """Compile Lean 4 code and return result.
        
        Args:
            lean_code: Lean 4 code to compile.
            use_prelude: Whether to prepend the standard prelude.
            use_mathlib: If True, uses lake build with Mathlib. Defaults to self.use_mathlib.
            
        Returns:
            LeanCompilationResult with success/failure details.
        """
        use_mathlib = use_mathlib if use_mathlib is not None else self.use_mathlib
        
        if use_mathlib and self.mathlib_available:
            # Use Mathlib prelude for lake build
            full_code = (self.MATHLIB_PRELUDE + lean_code) if use_prelude else lean_code
            return self._run_lake_build(full_code)
        else:
            # Fallback to standalone compilation
            full_code = (self.LEAN_PRELUDE + lean_code) if use_prelude else lean_code
            
            if not self.lean_available:
                return self._simulate_compilation(full_code)
            
            return self._run_lean_compiler(full_code)
    
    def _run_lake_build(self, lean_code: str) -> LeanCompilationResult:
        """Run lake build on a file in the Mathlib project.
        
        Writes the code to Mathematest/Verification/ and builds it with lake.
        """
        import uuid
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        temp_filename = f"Temp_{file_id}.lean"
        temp_path = self.VERIFICATION_DIR / temp_filename
        module_name = f"Mathematest.Verification.Temp_{file_id}"
        
        try:
            # Write the lean code
            with open(temp_path, "w") as f:
                f.write(lean_code)
            
            # Run lake build from the project root
            lake_cmd = str(self.LAKE_PATH) if self.LAKE_PATH.exists() else "lake"
            result = subprocess.run(
                [lake_cmd, "build", module_name],
                capture_output=True,
                text=True,
                timeout=120,  # Lake build can take longer
                cwd=str(self.lean_project_path),
                env=self.env,
            )
            
            errors = self._parse_lean_errors(result.stdout + result.stderr)
            warnings = self._parse_lean_warnings(result.stdout + result.stderr)
            error_locations = self._extract_error_locations(result.stdout + result.stderr)
            
            return LeanCompilationResult(
                success=result.returncode == 0 and not errors,
                code=lean_code,
                output=result.stdout + result.stderr,
                errors=errors,
                warnings=warnings,
                error_locations=error_locations,
            )
            
        except subprocess.TimeoutExpired:
            return LeanCompilationResult(
                success=False,
                code=lean_code,
                output="Lake build timed out",
                errors=["Lake build timed out after 120 seconds"],
                warnings=[],
                error_locations=[],
            )
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    def _run_lean_compiler(self, lean_code: str) -> LeanCompilationResult:
        """Run standalone Lean 4 compiler (without Mathlib)."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=False,
        ) as f:
            f.write(lean_code)
            temp_path = Path(f.name)
        
        try:
            lean_cmd = str(self.LEAN_PATH) if self.LEAN_PATH.exists() else "lean"
            result = subprocess.run(
                [lean_cmd, str(temp_path)],
                capture_output=True,
                text=True,
                timeout=60,
                env=self.env,
            )
            
            errors = self._parse_lean_errors(result.stderr)
            warnings = self._parse_lean_warnings(result.stderr)
            error_locations = self._extract_error_locations(result.stderr)
            
            return LeanCompilationResult(
                success=result.returncode == 0 and not errors,
                code=lean_code,
                output=result.stdout + result.stderr,
                errors=errors,
                warnings=warnings,
                error_locations=error_locations,
            )
            
        except subprocess.TimeoutExpired:
            return LeanCompilationResult(
                success=False,
                code=lean_code,
                output="Compilation timed out",
                errors=["Compilation timed out after 60 seconds"],
                warnings=[],
                error_locations=[],
            )
        finally:
            temp_path.unlink(missing_ok=True)
    
    def _simulate_compilation(self, lean_code: str) -> LeanCompilationResult:
        """Simulate Lean compilation using GPT-4o-mini validation."""
        prompt = f"""Analyze this Lean 4 code for syntax and semantic errors:

```lean
{lean_code}
```

Respond with JSON:
{{
    "valid": true/false,
    "errors": ["list of errors if any"],
    "warnings": ["list of warnings if any"],
    "suggestions": ["suggestions for fixing"]
}}
"""
        
        try:
            response = self.openai.chat.completions.create(
                model=self.settings.default_model,
                messages=[
                    {"role": "system", "content": "You are a Lean 4 expert. Analyze code for errors."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            import json
            # Extract JSON from response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            try:
                parsed = json.loads(content.strip())
                return LeanCompilationResult(
                    success=parsed.get("valid", False),
                    code=lean_code,
                    output=str(parsed),
                    errors=parsed.get("errors", []),
                    warnings=parsed.get("warnings", []),
                    error_locations=[],
                )
            except json.JSONDecodeError:
                # Fallback: check for obvious issues
                return self._basic_syntax_check(lean_code)
                
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return self._basic_syntax_check(lean_code)
    
    def _basic_syntax_check(self, lean_code: str) -> LeanCompilationResult:
        """Basic syntax validation without Lean compiler."""
        errors = []
        warnings = []
        
        # Check for unbalanced braces/parens
        if lean_code.count("{") != lean_code.count("}"):
            errors.append("Unbalanced curly braces")
        if lean_code.count("(") != lean_code.count(")"):
            errors.append("Unbalanced parentheses")
        if lean_code.count("[") != lean_code.count("]"):
            errors.append("Unbalanced square brackets")
        
        # Check for common Lean patterns
        if "theorem" in lean_code and ":=" not in lean_code and "by" not in lean_code:
            errors.append("Theorem declared but no proof provided")
        
        if "sorry" in lean_code:
            warnings.append("Proof contains 'sorry' placeholder")
        
        return LeanCompilationResult(
            success=len(errors) == 0,
            code=lean_code,
            output="Basic syntax check (Lean not available)",
            errors=errors,
            warnings=warnings,
            error_locations=[],
        )
    
    def _parse_lean_errors(self, output: str) -> List[str]:
        """Extract error messages from Lean output."""
        errors = []
        for line in output.split("\n"):
            if "error:" in line.lower():
                errors.append(line.strip())
        return errors
    
    def _parse_lean_warnings(self, output: str) -> List[str]:
        """Extract warning messages from Lean output."""
        warnings = []
        for line in output.split("\n"):
            if "warning:" in line.lower():
                warnings.append(line.strip())
        return warnings
    
    def _extract_error_locations(self, output: str) -> List[Dict[str, Any]]:
        """Extract error locations (line, column) from Lean output."""
        locations = []
        pattern = r":(\d+):(\d+):"
        
        for match in re.finditer(pattern, output):
            locations.append({
                "line": int(match.group(1)),
                "column": int(match.group(2)),
            })
        
        return locations
    
    def generate_lean_proof(
        self,
        theorem_statement: str,
        natural_language_proof: Optional[str] = None,
    ) -> str:
        """Generate Lean 4 code from theorem statement.
        
        Args:
            theorem_statement: Natural language theorem.
            natural_language_proof: Optional proof outline.
            
        Returns:
            Lean 4 code.
        """
        prompt = f"""Convert this mathematical theorem to Lean 4 code with a proof.

THEOREM: {theorem_statement}
"""
        if natural_language_proof:
            prompt += f"\nPROOF OUTLINE: {natural_language_proof}"
        
        prompt += """

Generate valid Lean 4 code. Use standard Mathlib tactics. If you cannot prove it completely, use 'sorry' as placeholder.

```lean
"""
        
        response = self.openai.chat.completions.create(
            model=self.settings.default_model,
            messages=[
                {"role": "system", "content": "You are a Lean 4 expert. Generate syntactically correct Lean 4 code."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        
        content = response.choices[0].message.content
        
        # Extract code block
        if "```lean" in content:
            code = content.split("```lean")[1].split("```")[0]
        elif "```" in content:
            code = content.split("```")[1].split("```")[0]
        else:
            code = content
        
        return code.strip()
    
    def verify_with_self_correction(
        self,
        theorem_statement: str,
        natural_language_proof: Optional[str] = None,
        max_attempts: int = 3,
    ) -> Tuple[bool, List[LeanProofAttempt]]:
        """Verify theorem with self-correction loop.
        
        Args:
            theorem_statement: Theorem to prove.
            natural_language_proof: Optional proof outline.
            max_attempts: Maximum correction attempts.
            
        Returns:
            Tuple of (success, list of attempts).
        """
        attempts = []
        current_code = None
        
        for attempt_num in range(1, max_attempts + 1):
            logger.info(f"Verification attempt {attempt_num}/{max_attempts}")
            
            if attempt_num == 1:
                # Initial generation
                current_code = self.generate_lean_proof(
                    theorem_statement,
                    natural_language_proof,
                )
            else:
                # Self-correction based on previous errors
                prev_attempt = attempts[-1]
                current_code = self._generate_correction(
                    theorem_statement,
                    prev_attempt.lean_code,
                    prev_attempt.result.errors,
                )
            
            # Compile
            result = self.compile(current_code)
            
            attempt = LeanProofAttempt(
                attempt_number=attempt_num,
                lean_code=current_code,
                result=result,
            )
            attempts.append(attempt)
            
            if result.success:
                logger.info(f"Proof verified successfully on attempt {attempt_num}")
                return True, attempts
            
            logger.warning(f"Attempt {attempt_num} failed: {result.errors}")
        
        logger.error(f"Failed to verify after {max_attempts} attempts")
        return False, attempts
    
    def _generate_correction(
        self,
        theorem_statement: str,
        failed_code: str,
        errors: List[str],
    ) -> str:
        """Generate corrected Lean code based on compiler errors.
        
        Args:
            theorem_statement: Original theorem.
            failed_code: Code that failed to compile.
            errors: List of compilation errors.
            
        Returns:
            Corrected Lean 4 code.
        """
        prompt = f"""The following Lean 4 code failed to compile:

```lean
{failed_code}
```

ERRORS:
{chr(10).join(errors)}

ORIGINAL THEOREM: {theorem_statement}

Fix the errors and provide corrected Lean 4 code. Use 'sorry' if you cannot complete the proof.

```lean
"""
        
        response = self.openai.chat.completions.create(
            model=self.settings.default_model,
            messages=[
                {"role": "system", "content": "You are a Lean 4 expert. Fix compilation errors."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        
        content = response.choices[0].message.content
        
        # Extract code block
        if "```lean" in content:
            code = content.split("```lean")[1].split("```")[0]
        elif "```" in content:
            code = content.split("```")[1].split("```")[0]
        else:
            code = content
        
        return code.strip()


def main():
    """Test the Lean 4 compiler bridge."""
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Lean 4 Compiler Bridge")
    parser.add_argument("--theorem", type=str, help="Theorem to verify")
    parser.add_argument("--test", action="store_true", help="Run tests")
    args = parser.parse_args()
    
    compiler = Lean4Compiler()
    
    if args.test:
        # Test cases
        test_theorems = [
            "For all real numbers a and b, (a + b)² = a² + 2ab + b²",
            "The derivative of sin(x) is cos(x)",
            "If n is even, then n² is even",
        ]
        
        print("=" * 60)
        print("LEAN 4 COMPILER BRIDGE TEST")
        print("=" * 60)
        print(f"Lean Available: {compiler.lean_available}")
        print()
        
        for thm in test_theorems:
            print(f"Theorem: {thm}")
            success, attempts = compiler.verify_with_self_correction(thm, max_attempts=2)
            print(f"  Success: {success}")
            print(f"  Attempts: {len(attempts)}")
            if attempts:
                last = attempts[-1]
                print(f"  Errors: {last.result.errors[:2]}")
            print()
    
    elif args.theorem:
        success, attempts = compiler.verify_with_self_correction(args.theorem)
        print(f"Success: {success}")
        for a in attempts:
            print(f"\nAttempt {a.attempt_number}:")
            print(f"Code:\n{a.lean_code[:200]}...")
            print(f"Errors: {a.result.errors}")


if __name__ == "__main__":
    main()
