#!/usr/bin/env python3
"""
ProofNet Import Script for MathemaTest - Real Data Version.

Downloads and imports the full ProofNet dataset from the official
GitHub repository (zhangir-azerbayev/ProofNet).

Target: 371 real theorems as :GoldStandard nodes in Neo4j.
"""

import argparse
import logging
import sys
import json
import requests
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.config.settings import get_settings
from src.graph_store.neo4j_client import Neo4jClient


logger = logging.getLogger(__name__)


# ProofNet GitHub URLs
PROOFNET_URLS = {
    "test": "https://raw.githubusercontent.com/zhangir-azerbayev/ProofNet/main/benchmark/test.jsonl",
    "valid": "https://raw.githubusercontent.com/zhangir-azerbayev/ProofNet/main/benchmark/valid.jsonl",
}


class ProofNetImporter:
    """Import real ProofNet theorems to Neo4j as GoldStandard nodes."""
    
    def __init__(self, neo4j_client: Neo4jClient = None):
        """Initialize importer."""
        self.settings = get_settings()
        self.neo4j = neo4j_client or Neo4jClient()
        self._ensure_constraints()
    
    def _ensure_constraints(self):
        """Ensure Neo4j constraints for GoldStandard nodes."""
        queries = [
            "CREATE CONSTRAINT gold_standard_id IF NOT EXISTS FOR (g:GoldStandard) REQUIRE g.theorem_id IS UNIQUE",
        ]
        
        for query in queries:
            try:
                with self.neo4j.session() as session:
                    session.run(query)
            except Exception as e:
                logger.debug(f"Constraint may already exist: {e}")
    
    def download_proofnet(self) -> List[Dict[str, Any]]:
        """Download ProofNet dataset from GitHub.
        
        Returns:
            List of theorem dictionaries.
        """
        theorems = []
        
        for split_name, url in PROOFNET_URLS.items():
            logger.info(f"Downloading ProofNet {split_name} from GitHub...")
            
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                # Parse JSONL (one JSON object per line)
                lines = response.text.strip().split('\n')
                
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    
                    try:
                        item = json.loads(line)
                        theorem = self._parse_theorem(item, f"proofnet_{split_name}_{i}", split_name)
                        if theorem:
                            theorems.append(theorem)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {i}: {e}")
                
                logger.info(f"Loaded {len(lines)} theorems from {split_name}")
                
            except requests.RequestException as e:
                logger.error(f"Failed to download {split_name}: {e}")
        
        logger.info(f"Total theorems loaded: {len(theorems)}")
        return theorems
    
    def _parse_theorem(self, item: Dict[str, Any], theorem_id: str, split: str) -> Dict[str, Any]:
        """Parse a theorem item from JSONL."""
        # ProofNet JSONL format has fields:
        # - nl_statement: Natural language statement
        # - formal_statement: Lean 3 formal statement
        # - src_header: Source header info
        # - (possibly others)
        
        nl_statement = item.get("nl_statement", "")
        formal_statement = item.get("formal_statement", "")
        
        # Try alternative field names
        if not nl_statement:
            nl_statement = item.get("informal_statement", item.get("statement", ""))
        if not formal_statement:
            formal_statement = item.get("lean_statement", item.get("formal", ""))
        
        # Skip if no content
        if not nl_statement and not formal_statement:
            return None
        
        # Extract topic from source header or content
        src_header = item.get("src_header", "")
        topic = self._extract_topic(nl_statement, src_header)
        
        return {
            "theorem_id": theorem_id,
            "natural_language": nl_statement,
            "lean_statement": formal_statement,
            "lean_proof": item.get("proof", ""),
            "topic": topic,
            "difficulty": self._estimate_difficulty(nl_statement),
            "source": "proofnet",
            "split": split,
            "src_header": src_header,
        }
    
    def _extract_topic(self, statement: str, header: str) -> str:
        """Extract topic from statement or header."""
        statement_lower = statement.lower()
        header_lower = header.lower()
        combined = statement_lower + " " + header_lower
        
        # Topic keywords
        topics = {
            "analysis": ["continuous", "limit", "derivative", "integral", "convergent", "series", "bounded"],
            "algebra": ["group", "ring", "field", "vector space", "linear", "polynomial", "matrix"],
            "topology": ["open", "closed", "compact", "connected", "metric", "topological"],
            "number_theory": ["prime", "divisible", "integer", "natural number", "modular"],
            "geometry": ["triangle", "circle", "angle", "parallel", "perpendicular", "distance"],
            "combinatorics": ["combinatorial", "counting", "permutation", "choose"],
        }
        
        for topic, keywords in topics.items():
            if any(kw in combined for kw in keywords):
                return topic
        
        return "mathematics"
    
    def _estimate_difficulty(self, statement: str) -> str:
        """Estimate difficulty based on statement complexity."""
        word_count = len(statement.split())
        
        if word_count < 15:
            return "easy"
        elif word_count < 40:
            return "medium"
        else:
            return "hard"
    
    def import_theorem(self, theorem: Dict[str, Any]) -> bool:
        """Import a single theorem to Neo4j."""
        query = """
        MERGE (g:GoldStandard {theorem_id: $theorem_id})
        SET g.natural_language = $natural_language,
            g.lean_statement = $lean_statement,
            g.lean_proof = $lean_proof,
            g.topic = $topic,
            g.difficulty = $difficulty,
            g.source = $source,
            g.split = $split,
            g.imported_at = datetime()
        """
        
        try:
            with self.neo4j.session() as session:
                session.run(query, theorem)
            return True
        except Exception as e:
            logger.error(f"Failed to import {theorem['theorem_id']}: {e}")
            return False
    
    def clear_existing(self) -> int:
        """Clear existing GoldStandard nodes (for re-import)."""
        query = "MATCH (g:GoldStandard) DETACH DELETE g RETURN count(g) as count"
        
        with self.neo4j.session() as session:
            result = session.run(query)
            count = result.single()["count"]
            logger.info(f"Cleared {count} existing GoldStandard nodes")
            return count
    
    def import_all(self, clear_existing: bool = True) -> Dict[str, Any]:
        """Import all ProofNet theorems.
        
        Args:
            clear_existing: Whether to clear existing GoldStandard nodes.
            
        Returns:
            Import statistics.
        """
        if clear_existing:
            self.clear_existing()
        
        theorems = self.download_proofnet()
        
        logger.info(f"Importing {len(theorems)} theorems to Neo4j...")
        
        stats = {
            "total": len(theorems),
            "imported": 0,
            "failed": 0,
            "topics": {},
        }
        
        for theorem in tqdm(theorems, desc="Importing theorems"):
            if self.import_theorem(theorem):
                stats["imported"] += 1
                topic = theorem.get("topic", "unknown")
                stats["topics"][topic] = stats["topics"].get(topic, 0) + 1
            else:
                stats["failed"] += 1
        
        logger.info(f"Import complete: {stats['imported']}/{stats['total']} theorems")
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about imported GoldStandard nodes."""
        queries = {
            "total_theorems": "MATCH (g:GoldStandard) RETURN count(g) as count",
            "with_lean": "MATCH (g:GoldStandard) WHERE g.lean_statement <> '' RETURN count(g) as count",
            "by_topic": """
                MATCH (g:GoldStandard) 
                RETURN g.topic as topic, count(g) as count 
                ORDER BY count DESC LIMIT 10
            """,
            "by_split": """
                MATCH (g:GoldStandard)
                RETURN g.split as split, count(g) as count
            """,
        }
        
        stats = {}
        with self.neo4j.session() as session:
            for name, query in queries.items():
                result = session.run(query)
                if name in ["by_topic", "by_split"]:
                    stats[name] = [dict(r) for r in result]
                else:
                    record = result.single()
                    stats[name] = record["count"] if record else 0
        
        return stats
    
    def close(self):
        """Close connections."""
        self.neo4j.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Import ProofNet (real data) to Neo4j")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear existing data")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    importer = ProofNetImporter()
    
    try:
        if args.stats_only:
            stats = importer.get_statistics()
            print("\n=== ProofNet Statistics ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            # Import theorems
            stats = importer.import_all(clear_existing=not args.no_clear)
            
            print("\n" + "=" * 60)
            print("PROOFNET IMPORT COMPLETE (REAL DATA)")
            print("=" * 60)
            print(f"Total Theorems: {stats['total']}")
            print(f"Imported: {stats['imported']}")
            print(f"Failed: {stats['failed']}")
            print(f"\nTopics Distribution:")
            for topic, count in sorted(stats['topics'].items(), key=lambda x: -x[1]):
                print(f"  {topic}: {count}")
    
    finally:
        importer.close()


if __name__ == "__main__":
    main()
