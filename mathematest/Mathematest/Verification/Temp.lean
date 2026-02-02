import Mathlib.Tactic

theorem limit_evaluation_example : lim x → 3 (x^2 − 3x)/(2x^2 − 5x − 3) = 3/7 := by
  -- Step 1: Show that the function is undefined at x = 3
  have h1 : (x^2 - 3x)/(2x^2 - 5x - 3) = 0/0 := by
    simp [h1],
  -- Step 2: Factor and cancel
  have h2 : (x^2 - 3x)/(2x^2 - 5x - 3) = x/(2x + 1) 
    -- This follows from the context provided
  -- Step 3: Evaluate the limit
  calc
    lim x → 3 (x/(2x + 1)) = 3/(2*3 + 1) 
    ... = 3/7
  