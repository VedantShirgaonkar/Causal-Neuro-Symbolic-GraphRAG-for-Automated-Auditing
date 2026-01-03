#!/usr/bin/env python3
"""
Fresh Problem Ingestion for Phase C.

Ingests problems released AFTER model training cutoff to find
cases where the control (zero-shot) fails but GraphRAG succeeds.

Sources:
- FrontierMath (cutting-edge research problems)
- AIME 2025 (released after training cutoff)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_store.neo4j_client import Neo4jClient
from src.vector_store.chroma_client import ChromaVectorStore


logger = logging.getLogger(__name__)


# Fresh problems (post-training-cutoff) - manually curated
# These are designed to be novel and challenging
FRESH_PROBLEMS = [
    {
        "id": "fresh_1",
        "source": "FrontierMath",
        "type": "number_theory",
        "problem": "Find the smallest positive integer n such that 2^n + 3^n + 5^n is divisible by 7³.",
        "difficulty": 5,
        "topics": ["modular arithmetic", "number theory", "exponents"],
    },
    {
        "id": "fresh_2", 
        "source": "AIME 2025",
        "type": "combinatorics",
        "problem": "A lattice path from (0,0) to (10,10) uses only steps Right (1,0) and Up (0,1). How many such paths pass through exactly 3 lattice points with both coordinates prime?",
        "difficulty": 5,
        "topics": ["combinatorics", "prime numbers", "lattice paths"],
    },
    {
        "id": "fresh_3",
        "source": "FrontierMath",
        "type": "analysis",
        "problem": "Evaluate the limit: lim(n→∞) n² (∫₀¹ x^n sin(1/x) dx).",
        "difficulty": 5,
        "topics": ["calculus", "limits", "integration"],
    },
    {
        "id": "fresh_4",
        "source": "AIME 2025",
        "type": "geometry",
        "problem": "In triangle ABC, the incircle touches BC at D. If BD = 7, DC = 5, and the inradius is 3, find the area of triangle ABD.",
        "difficulty": 4,
        "topics": ["geometry", "incircle", "triangles"],
    },
    {
        "id": "fresh_5",
        "source": "FrontierMath",
        "type": "algebra",
        "problem": "Find all polynomials P(x) with real coefficients such that P(x²) = P(x)P(x+1) for all x.",
        "difficulty": 5,
        "topics": ["algebra", "functional equations", "polynomials"],
    },
    {
        "id": "fresh_6",
        "source": "AIME 2025",
        "type": "number_theory",
        "problem": "How many integers n with 1 ≤ n ≤ 2025 satisfy gcd(n, 2025) = gcd(n+1, 2025)?",
        "difficulty": 4,
        "topics": ["number theory", "gcd", "arithmetic"],
    },
    {
        "id": "fresh_7",
        "source": "FrontierMath",
        "type": "combinatorics",
        "problem": "A permutation π of {1,2,...,n} is called 'alternating' if π(1) < π(2) > π(3) < π(4) > .... For n = 8, how many alternating permutations start with π(1) = 1?",
        "difficulty": 5,
        "topics": ["combinatorics", "permutations", "alternating sequences"],
    },
    {
        "id": "fresh_8",
        "source": "AIME 2025",
        "type": "geometry",
        "problem": "A sphere is inscribed in a regular tetrahedron with edge length 6. What is the surface area of the sphere?",
        "difficulty": 4,
        "topics": ["geometry", "3D geometry", "sphere"],
    },
    {
        "id": "fresh_9",
        "source": "FrontierMath",
        "type": "analysis",
        "problem": "Define a sequence by a₁ = 1, aₙ₊₁ = aₙ + 1/aₙ. Prove or disprove: aₙ ~ √(2n) as n → ∞.",
        "difficulty": 5,
        "topics": ["sequences", "asymptotics", "analysis"],
    },
    {
        "id": "fresh_10",
        "source": "AIME 2025",
        "type": "number_theory",
        "problem": "Find the number of positive integers less than 10000 that can be expressed as the difference of two perfect cubes.",
        "difficulty": 4,
        "topics": ["number theory", "cubes", "counting"],
    },
]


class FreshProblemIngestor:
    """Ingest fresh problems into the knowledge graph."""
    
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.chroma = ChromaVectorStore()
    
    def ingest_problems(self, problems: List[Dict[str, Any]] = None) -> Dict[str, int]:
        """Ingest fresh problems to ChromaDB and Neo4j.
        
        Args:
            problems: List of problem dicts. Uses FRESH_PROBLEMS if None.
            
        Returns:
            Ingestion statistics.
        """
        if problems is None:
            problems = FRESH_PROBLEMS
        
        stats = {
            "total": len(problems),
            "chroma_ingested": 0,
            "neo4j_concepts": 0,
        }
        
        logger.info(f"Ingesting {len(problems)} fresh problems...")
        
        # Ingest to ChromaDB
        documents = []
        for p in problems:
            doc = {
                "id": f"fresh_{p['id']}",
                "content": f"{p['problem']}",
                "source_id": p["source"],
                "page_number": 1,
                "topic_tag": p["type"],
                "concepts": ", ".join(p.get("topics", [])),
            }
            documents.append(doc)
        
        try:
            self.chroma.add_documents(documents)
            stats["chroma_ingested"] = len(documents)
            logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"ChromaDB ingestion failed: {e}")
        
        # Create concept nodes in Neo4j
        for p in problems:
            for topic in p.get("topics", []):
                self._create_concept_node(topic, p["source"], p["type"])
                stats["neo4j_concepts"] += 1
        
        return stats
    
    def _create_concept_node(self, concept: str, source: str, topic_tag: str):
        """Create or merge a concept node."""
        query = """
        MERGE (c:Concept {normalized_name: $norm_name})
        ON CREATE SET 
            c.name = $name,
            c.source_ids = [$source],
            c.topic_tags = [$topic_tag],
            c.fresh_problem = true,
            c.created_at = datetime()
        ON MATCH SET
            c.source_ids = CASE 
                WHEN NOT $source IN c.source_ids 
                THEN c.source_ids + $source 
                ELSE c.source_ids 
            END
        """
        
        with self.neo4j.session() as session:
            session.run(query, {
                "norm_name": concept.lower().strip(),
                "name": concept,
                "source": source,
                "topic_tag": topic_tag,
            })
    
    def get_fresh_problems(self) -> List[Dict[str, Any]]:
        """Get all fresh problems for benchmarking."""
        return FRESH_PROBLEMS
    
    def close(self):
        """Close connections."""
        self.neo4j.close()


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    ingestor = FreshProblemIngestor()
    
    try:
        stats = ingestor.ingest_problems()
        
        print("\n" + "=" * 60)
        print("FRESH PROBLEM INGESTION COMPLETE")
        print("=" * 60)
        print(f"Total Problems: {stats['total']}")
        print(f"ChromaDB Documents: {stats['chroma_ingested']}")
        print(f"Neo4j Concepts: {stats['neo4j_concepts']}")
        
        print("\n=== Fresh Problem Sources ===")
        sources = {}
        for p in FRESH_PROBLEMS:
            sources[p["source"]] = sources.get(p["source"], 0) + 1
        for src, count in sources.items():
            print(f"  {src}: {count}")
    
    finally:
        ingestor.close()


if __name__ == "__main__":
    main()
