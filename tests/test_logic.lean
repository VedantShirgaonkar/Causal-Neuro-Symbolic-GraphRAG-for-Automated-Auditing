-- Trivial Proof Test for MathemaTest Phase C
-- This file verifies Lean 4 compiler integration
-- Fixed for native Lean 4 (not Mathlib)

-- Simple commutativity theorem using Nat.add_comm from the standard library
theorem add_comm_test (a b : Nat) : a + b = b + a := by
  exact Nat.add_comm a b

-- Additional test: multiplication commutativity
theorem mul_comm_test (a b : Nat) : a * b = b * a := by
  exact Nat.mul_comm a b

-- Test with explicit proof terms
theorem add_assoc_test (a b c : Nat) : (a + b) + c = a + (b + c) := by
  exact Nat.add_assoc a b c

-- Simple logic test
theorem trivial_test : True := by
  trivial

-- Success marker
#check add_comm_test
#check mul_comm_test
#check add_assoc_test
#check trivial_test
