#!/usr/bin/env python3
"""
Phase 4 Verification Gate.

Reads the bridge trace log and verifies that BOTH Physics and Calculus
sources were retrieved for the cross-domain problem.

Returns:
    PASS if both domains present
    FAIL if any domain missing
"""

import sys
import re
from pathlib import Path


LOG_PATH = Path(__file__).parent.parent / "logs" / "bridge_trace.log"


def verify_bridge() -> bool:
    """Verify cross-domain retrieval from log file.
    
    Returns:
        True if both Physics AND Calculus sources found, False otherwise.
    """
    if not LOG_PATH.exists():
        print(f"ERROR: Log file not found: {LOG_PATH}")
        return False
    
    with open(LOG_PATH) as f:
        log_content = f.read()
    
    # Check for Physics nodes
    physics_match = re.search(r'Physics nodes: (\d+)', log_content)
    physics_count = int(physics_match.group(1)) if physics_match else 0
    
    # Check for Calculus nodes
    calculus_match = re.search(r'Calculus nodes: (\d+)', log_content)
    calculus_count = int(calculus_match.group(1)) if calculus_match else 0
    
    # Check verdict
    verified = re.search(r'CROSS-DOMAIN LINK: VERIFIED', log_content)
    
    print("=" * 50)
    print("PHASE 4 VERIFICATION GATE")
    print("=" * 50)
    print(f"Physics nodes: {physics_count}")
    print(f"Calculus nodes: {calculus_count}")
    print("")
    
    has_physics = physics_count > 0
    has_calculus = calculus_count > 0
    
    if has_physics and has_calculus:
        print("✅ PASS: Cross-domain retrieval verified")
        print("   Both Physics AND Calculus sources retrieved")
        return True
    else:
        print("❌ FAIL: Cross-domain retrieval broken")
        if not has_physics:
            print("   ⚠️ Missing: Physics nodes (Work, Energy)")
        if not has_calculus:
            print("   ⚠️ Missing: Calculus nodes (Line Integrals)")
        return False


if __name__ == "__main__":
    result = verify_bridge()
    sys.exit(0 if result else 1)
