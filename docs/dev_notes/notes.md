Before you dive into the code, there are a few "ground-floor" technical realities and strategic pitfalls you should keep on your radar. If you're going for a journal-worthy implementation, these are the areas where most student projects fail.

### 1. The "LaTeX-to-SymPy" Bridge is Brittle

Writing the Python subprocess that executes LLM-generated math code is the hardest engineering task in this project.

* 
**The Pitfall:** LLMs often use different LaTeX packages or shorthand (e.g., `\frac{1}{2}` vs `1/2`) that `latex2sympy2` might not parse correctly.


* **The Note:** Build a **normalization layer** first. Before sending math to the symbolic solver, use a simple script to standardize the LaTeX strings. Don't let your "Logic Judge" fail just because of a missing curly brace.

### 2. Lean 4 is a "Semester VI" Boss

* **The Pitfall:** Integrating **Lean 4** for formal verification is a massive undertaking. It requires the LLM to understand a very strict functional programming syntax.
* **The Note:** Start with **SymPy** for your Semester V implementation. It handles 90% of undergraduate calculus and algebra perfectly. Keep Lean 4 as your "Advanced Feature" for the journal paper's conclusion or the final semester push to avoid getting stuck in "compiler hell" early on.



### 3. Graph "Hallucinations" During Ingestion

* 
**The Pitfall:** When your "Graph Constructor Agent" reads a PDF, it might create a relationship that doesn't exist (e.g., linking two unrelated formulas just because they are on the same page).


* **The Note:** Implement a **Schema Validator**. Ensure every triple (`Subject` -> `Predicate` -> `Object`) follows your Neo4j rules before it hits the database. If the LLM tries to create a relationship that doesn't make sense, flag it for a second pass.



### 4. The "Adversarial Loop" is Expensive

* 
**The Pitfall:** If your agent generates a question, fails verification, and retries 5 times, your API costs will skyrocket.


* **The Note:** Use **Small Language Models (SLMs)** like **Qwen-2.5-Math-7B** for the initial "Proposer" steps. Only call the "Frontier" models (GPT-5.2) for the final synthesis and complex re-ranking . This makes your system faster and more sustainable.



### 5. Benchmark Humility

* **The Pitfall:** You might run your system against **FrontierMath** and find it only solves 5% of the problems.
* **The Note:** Don't panic. Standard models solve 2%. In a research paper, a **150% improvement** (from 2% to 5%) is a massive victory. Focus on the *reliability*—show that when your system *does* give an answer, it is **guaranteed correct** by the Symbolic Solver.



### 6. Minimal Viable Innovation (MVI)

Before you build the whole UI, prove the **Neuro-Symbolic Loop** works:

1. Take one page of a math PDF.
2. Manually verify the Graph nodes.
3. Generate 1 MCQ.
4. Successfully run the **SymPy verification** in a Python subprocess.

**Once that loop closes, the rest is just scaling.** Good luck, Vedant. You have the blueprint; now it's time to build the engine.