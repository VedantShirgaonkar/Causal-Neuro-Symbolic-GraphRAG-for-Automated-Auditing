#!/usr/bin/env python3
"""
Stress Test Script for MathemaTest Ingestion Pipeline.

Runs the pipeline on real STEM PDFs and generates a detailed audit report.
Tests:
- STEP: Differential equations, "Snowplough" problem
- Physics: Work-Energy dot-product integrals
- Putnam: Competition-level mathematical expressions
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingestion_engine import IngestionEngine, create_engine
from src.ingestion.ocr_utils import OCRValidator
from src.ingestion.latex_normalizer import LaTeXNormalizer
from src.models.schemas import BlockType, StructuredBlock


class StressTestReport:
    """Collects and formats stress test results."""
    
    def __init__(self):
        self.total_blocks = 0
        self.formula_blocks = 0
        self.valid_formulas = 0
        self.invalid_formulas = 0
        self.sympy_ready = 0
        self.hallucinations_caught = []
        self.normalization_fixes = []
        self.problematic_blocks = []
        self.clean_blocks = []
    
    def add_block_result(
        self,
        block: StructuredBlock,
        doc_name: str,
        page: int,
    ):
        """Record results for a processed block."""
        self.total_blocks += 1
        
        if block.block_type == BlockType.FORMULA:
            self.formula_blocks += 1
            
            latex = block.symbolic_metadata.latex_normalized
            if latex:
                if latex.validation.is_valid:
                    self.valid_formulas += 1
                    self.clean_blocks.append({
                        "document": doc_name,
                        "page": page,
                        "raw": latex.raw,
                        "normalized": latex.normalized,
                        "sympy_compatible": latex.sympy_compatible,
                    })
                else:
                    self.invalid_formulas += 1
                    self.problematic_blocks.append({
                        "document": doc_name,
                        "page": page,
                        "raw": latex.raw,
                        "normalized": latex.normalized,
                        "errors": [e.model_dump() for e in latex.validation.errors],
                        "warnings": [w.model_dump() for w in latex.validation.warnings],
                    })
                    for error in latex.validation.errors:
                        self.hallucinations_caught.append({
                            "document": doc_name,
                            "page": page,
                            "type": error.error_type,
                            "message": error.message,
                            "context": error.context,
                        })
                
                if latex.sympy_compatible:
                    self.sympy_ready += 1
                
                for fix in latex.normalization_applied:
                    self.normalization_fixes.append({
                        "document": doc_name,
                        "page": page,
                        "fix_type": fix,
                        "raw": latex.raw[:50] + "..." if len(latex.raw) > 50 else latex.raw,
                    })
    
    def generate_report(self) -> str:
        """Generate the stress test report as a markdown string."""
        lines = [
            "# Stress Test Report",
            f"\n**Generated:** {datetime.now().isoformat()}",
            "",
            "## Summary Statistics",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Total Blocks | {self.total_blocks} |",
            f"| Formula Blocks | {self.formula_blocks} |",
            f"| Valid Formulas | {self.valid_formulas} |",
            f"| Invalid Formulas | {self.invalid_formulas} |",
            f"| SymPy-Ready Formulas | {self.sympy_ready} |",
            f"| Hallucinations Caught | {len(self.hallucinations_caught)} |",
            f"| Normalizations Applied | {len(self.normalization_fixes)} |",
            "",
        ]
        
        if self.hallucinations_caught:
            lines.extend([
                "## Hallucinations Detected",
                "",
            ])
            for h in self.hallucinations_caught[:10]:  # Limit to first 10
                lines.append(f"### {h['document']} - Page {h['page']}")
                lines.append(f"- **Type:** `{h['type']}`")
                lines.append(f"- **Message:** {h['message']}")
                if h['context']:
                    lines.append(f"- **Context:** `{h['context']}`")
                lines.append("")
        
        if self.normalization_fixes:
            lines.extend([
                "## Normalizations Applied",
                "",
                "| Document | Page | Fix Type | Sample |",
                "|----------|------|----------|--------|",
            ])
            seen = set()
            for fix in self.normalization_fixes[:20]:
                key = (fix['document'], fix['fix_type'])
                if key not in seen:
                    seen.add(key)
                    lines.append(
                        f"| {fix['document'][:20]} | {fix['page']} | "
                        f"`{fix['fix_type']}` | `{fix['raw'][:30]}...` |"
                    )
            lines.append("")
        
        if self.clean_blocks:
            lines.extend([
                "## Sample Clean Formula Blocks",
                "",
            ])
            for block in self.clean_blocks[:5]:
                lines.append(f"### {block['document']} - Page {block['page']}")
                lines.append("```latex")
                lines.append(block['normalized'])
                lines.append("```")
                lines.append(f"- SymPy Compatible: {'✓' if block['sympy_compatible'] else '✗'}")
                lines.append("")
        
        if self.problematic_blocks:
            lines.extend([
                "## Problematic Blocks (Need Review)",
                "",
            ])
            for block in self.problematic_blocks[:5]:
                lines.append(f"### {block['document']} - Page {block['page']}")
                lines.append("**Raw:**")
                lines.append("```latex")
                lines.append(block['raw'][:200] + "..." if len(block['raw']) > 200 else block['raw'])
                lines.append("```")
                lines.append("**Errors:**")
                for error in block['errors'][:3]:
                    lines.append(f"- `{error['error_type']}`: {error['message']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export full report as JSON."""
        return json.dumps({
            "summary": {
                "total_blocks": self.total_blocks,
                "formula_blocks": self.formula_blocks,
                "valid_formulas": self.valid_formulas,
                "invalid_formulas": self.invalid_formulas,
                "sympy_ready": self.sympy_ready,
            },
            "hallucinations": self.hallucinations_caught,
            "normalizations": self.normalization_fixes[:50],
            "clean_blocks": self.clean_blocks[:20],
            "problematic_blocks": self.problematic_blocks[:20],
        }, indent=2)


def run_stress_test():
    """Run the stress test on all PDFs in the data directory."""
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return None
    
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        print("Error: No PDF files found in data directory")
        return None
    
    print(f"Found {len(pdfs)} PDF files to process:")
    for pdf in pdfs:
        print(f"  - {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
    print()
    
    # Create engine - use mock since we don't have GPU models installed
    # In production, use create_engine(use_mock=False)
    print("Initializing ingestion engine (using mock for stress test)...")
    engine = create_engine(use_mock=True)
    
    report = StressTestReport()
    
    for pdf_path in pdfs:
        print(f"\nProcessing: {pdf_path.name}")
        print("-" * 50)
        
        try:
            result = engine.process_pdf(pdf_path, page_range=(1, 3))  # First 3 pages
            
            print(f"  Pages processed: {result.total_pages}")
            print(f"  Blocks extracted: {len(result.blocks)}")
            print(f"  Formulas found: {result.formula_count}")
            print(f"  Processing time: {result.processing_time_seconds:.2f}s")
            
            for block in result.blocks:
                report.add_block_result(
                    block=block,
                    doc_name=pdf_path.name,
                    page=block.source_info.page_number,
                )
            
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for err in result.errors[:3]:
                    print(f"    - {err}")
                    
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return report


def main():
    """Main entry point for stress test."""
    print("=" * 60)
    print("MathemaTest Phase 1 - Stress Test")
    print("=" * 60)
    print()
    
    report = run_stress_test()
    
    if report is None:
        print("\nStress test failed!")
        sys.exit(1)
    
    # Generate reports
    output_dir = Path(__file__).parent / "stress_test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Markdown report
    md_report = report.generate_report()
    md_path = output_dir / "stress_test_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"\nMarkdown report saved to: {md_path}")
    
    # JSON report
    json_report = report.to_json()
    json_path = output_dir / "stress_test_report.json"
    with open(json_path, "w") as f:
        f.write(json_report)
    print(f"JSON report saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"Total blocks processed: {report.total_blocks}")
    print(f"Formula blocks: {report.formula_blocks}")
    print(f"Valid formulas: {report.valid_formulas} ({100*report.valid_formulas/max(1,report.formula_blocks):.1f}%)")
    print(f"SymPy-ready: {report.sympy_ready}")
    print(f"Hallucinations caught: {len(report.hallucinations_caught)}")
    print(f"Normalizations applied: {len(report.normalization_fixes)}")
    
    if report.invalid_formulas > 0:
        print(f"\n⚠️  {report.invalid_formulas} formulas have validation issues - review needed")
    else:
        print("\n✓ All formulas passed validation")
    
    return report


if __name__ == "__main__":
    main()
