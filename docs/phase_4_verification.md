# Phase 4: Cross-Domain Retrieval Verification

## The Problem Statement

**Title:** Cross-Domain: Work and Line Integrals

**Question:** A force field F = (2xy, x²) acts on a particle moving along the path C from (0,0) to (1,1), where C is the curve y = x². Calculate the work done.

**Domain Requirements:**
- **Physics:** Work done by variable force (W = ∫F·dr)
- **Calculus:** Line integrals and parameterization

---

## Retrieved Nodes

| # | Domain | Source ID | Content Preview |
|---|--------|-----------|-----------------|
| 1 | Physics | University_Physics_Volume_1_-_WEB | 8.4 Potential Energy Diagrams and Stability 45. A ... |
| 2 | Physics | University_Physics_Volume_1_-_WEB | 73. A mysterious force acts on all particles along... |
| 3 | Physics | University_Physics_Volume_1_-_WEB | is decreased. The same horizontal force is applied... |
| 4 | Physics | mit_calculus_lec_week8 | Z  R  R  R  R  R  4  rf d~r = f(P1) − f(P0) when C... |
| 5 | Physics | University_Physics_Volume_1_-_WEB | Let’s look at a specific example, choosing zero po... |
| 6 | Physics | University_Physics_Volume_1_-_WEB | 7.1 Work LEARNING OBJECTIVES By the end of this se... |
| 7 | Physics | University_Physics_Volume_1_-_WEB | Figure 7.3(b) shows a person holding a briefcase. ... |
| 8 | Physics | University_Physics_Volume_1_-_WEB | holds out a net to catch the pebble. (a) How much ... |
| 9 | Physics | University_Physics_Volume_1_-_WEB | How much instantaneous power does she exert at ? 8... |
| 10 | Physics | University_Physics_Volume_1_-_WEB | CHECK YOUR UNDERSTANDING 7.2 Can Earth’s gravity e... |

---

## Domain Coverage

| Domain | Nodes Retrieved | Status |
|--------|-----------------|--------|
| **Physics** | 13 | ✅ Found |
| **Calculus** | 2 | ✅ Found |
| **Other** | 0 | — |
| **Total** | 15 | — |

---

## The Solution

Given F = (2xy, x²) and path C: y = x² from (0,0) to (1,1):

1. **Parameterize:** x = t, y = t², t ∈ [0,1]
2. **dr = (dt, 2t dt)**
3. **F(t) = (2t³, t²)**
4. **F·dr = 2t³ dt + 2t³ dt = 4t³ dt**
5. **W = ∫₀¹ 4t³ dt = [t⁴]₀¹ = 1**

**Answer:** W = 1

---

## Verdict

# ✅ CROSS-DOMAIN LINK: VERIFIED


The hybrid retrieval system successfully retrieved context from **both** Physics 
(Work-Energy concepts) and Calculus (Line Integral techniques) sources.

This demonstrates the Knowledge Graph's ability to perform **inter-domain reasoning**
by connecting concepts across different academic disciplines.
