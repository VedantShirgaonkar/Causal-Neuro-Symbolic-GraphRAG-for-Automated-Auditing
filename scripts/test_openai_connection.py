#!/usr/bin/env python3
"""
OpenAI API Connection Smoke Test.

Verifies API connectivity before running live evaluation.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


def test_connection():
    """Test OpenAI API connection with a simple query."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Please add it to your .env file:")
        print("   OPENAI_API_KEY=sk-...")
        return False
    
    print(f"✓ API Key found: {api_key[:8]}...{api_key[-4:]}")
    print()
    
    try:
        client = OpenAI(api_key=api_key)
        
        print("Sending test query to GPT-4o-mini...")
        print("Query: 'What is the derivative of x^2?'")
        print("-" * 50)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "What is the derivative of x^2? Answer in one line."}
            ],
            temperature=0.3,
            max_tokens=100,
        )
        
        # Extract response
        content = response.choices[0].message.content
        usage = response.usage
        
        print()
        print("✓ API RESPONSE RECEIVED")
        print("=" * 50)
        print(f"Answer: {content}")
        print("=" * 50)
        print()
        print("USAGE METADATA:")
        print(f"  Model: {response.model}")
        print(f"  Prompt Tokens: {usage.prompt_tokens}")
        print(f"  Completion Tokens: {usage.completion_tokens}")
        print(f"  Total Tokens: {usage.total_tokens}")
        
        # Estimate cost (GPT-4o-mini pricing)
        input_cost = usage.prompt_tokens * 0.00015 / 1000  # $0.15 per 1M input
        output_cost = usage.completion_tokens * 0.0006 / 1000  # $0.60 per 1M output
        total_cost = input_cost + output_cost
        
        print(f"  Estimated Cost: ${total_cost:.6f}")
        print()
        print("✓ OpenAI API connection verified successfully!")
        print()
        
        # Log to file
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "api_usage.log"
        
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} | SMOKE_TEST | "
                   f"model={response.model} | "
                   f"tokens={usage.total_tokens} | "
                   f"cost=${total_cost:.6f}\n")
        
        print(f"✓ Usage logged to: {log_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
