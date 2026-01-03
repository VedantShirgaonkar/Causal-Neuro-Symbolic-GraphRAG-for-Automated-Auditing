/-
  Mathlib "Hello World" Test
  This file verifies that Mathlib tactics are available.
-/

import Mathlib.Algebra.Group.Defs
import Mathlib.Tactic.Ring

-- Test 1: Basic natural number commutativity using ring tactic
example (a b : ℕ) : a + b = b + a := by ring

-- Test 2: Integer arithmetic
example (x y : ℤ) : (x + y)^2 = x^2 + 2*x*y + y^2 := by ring

-- Test 3: Polynomial identity
example (a b c : ℚ) : (a + b + c)^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*a*c := by ring

-- Success marker
#check (rfl : 1 + 1 = 2)
