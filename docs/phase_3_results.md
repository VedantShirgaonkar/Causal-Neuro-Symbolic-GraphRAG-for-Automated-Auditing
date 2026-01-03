# Phase 3: Diagnostic MCQ Generation Results

## Executive Summary

**Generated:** 2026-01-02 19:18:37
**Problems:** 5 AIME 2025
**Total Distractors:** 15

---

## Misconception Coverage Metrics

| Metric | Value |
|--------|-------|
| **Misconception Coverage** | 26.7% |
| **Graph-Backed Distractors** | 4 |
| **Synthetic Distractors** | 11 |
| **Unique Graph Misconceptions Used** | 4 |

---

## Per-Problem Breakdown

### 🔸 AIME_2025_1 (combinatorics)

- **Topic:** Lattice Paths
- **Correct Answer:** 2100
- **Graph-Backed Distractors:** 1/3
- **Synthetic Distractors:** 2/3

### 🔸 AIME_2025_2 (geometry)

- **Topic:** Incircle Properties
- **Correct Answer:** 21
- **Graph-Backed Distractors:** 1/3
- **Synthetic Distractors:** 2/3

### 🔸 AIME_2025_3 (number_theory)

- **Topic:** GCD Properties
- **Correct Answer:** 1080
- **Graph-Backed Distractors:** 1/3
- **Synthetic Distractors:** 2/3

### 🔸 AIME_2025_4 (geometry)

- **Topic:** Inscribed Sphere
- **Correct Answer:** 8*pi
- **Graph-Backed Distractors:** 1/3
- **Synthetic Distractors:** 2/3

### 🔸 AIME_2025_5 (number_theory)

- **Topic:** Cube Differences
- **Correct Answer:** 2667
- **Graph-Backed Distractors:** 0/3
- **Synthetic Distractors:** 3/3

---

## Qualitative Example: AIME_2025_1

**Question:** A lattice path from (0,0) to (10,10) uses only steps Right (1,0) and Up (0,1). How many such paths p...

**Correct Answer (A):** 2100

### Distractors with Diagnostic Explanations

#### Option B: 3000

- **Misconception Source:** 📊 Graph
- **Misconception ID:** `misc_combinatorics_057ffc57`
- **Diagnosis:** *"Student likely counted all possible paths from (0,0) to (10,10) without restricting to paths that pass through prime coordinate points."*
- **Related Concept:** Lattice Paths

#### Option C: 1800

- **Misconception Source:** 🔧 Synthetic
- **Misconception ID:** `synthetic_combinatorics_0`
- **Diagnosis:** *"Student likely overcounted the arrangements of steps by not properly accounting for the constraints of the paths that pass through the required prime coordinates."*
- **Related Concept:** Lattice Paths

#### Option D: 1500

- **Misconception Source:** 🔧 Synthetic
- **Misconception ID:** `synthetic_combinatorics_1`
- **Diagnosis:** *"Student likely forgot to apply the multiplication principle when calculating the number of paths between the prime coordinates."*
- **Related Concept:** Lattice Paths

---

## Conclusion

Phase 3 achieved **26.7%** misconception coverage, with 4 out of 15
distractors backed by Neo4j misconception nodes.

The system now functions as a **Diagnostic Teacher**, able to:
1. Identify WHY a student got an answer wrong
2. Link errors to specific conceptual gaps
3. Provide targeted remediation based on prerequisite knowledge
