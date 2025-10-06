#!/usr/bin/env python3
"""
Validate Wallet Report JSON
Week 0: JSON Schema validation

Usage:
    python report_json.py --wallet 0x... | python validate_report.py
"""
import sys
import json
from pathlib import Path

try:
    from jsonschema import validate, Draft202012Validator
except ImportError:
    print("❌ Missing jsonschema. Install:")
    print("   pip install jsonschema")
    sys.exit(1)

# Schema path (relative to project root)
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "report_v1.json"


def main():
    if not SCHEMA_PATH.exists():
        print(f"❌ Schema not found: {SCHEMA_PATH}")
        sys.exit(1)
    
    # Load schema
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    
    # Validate schema itself
    Draft202012Validator.check_schema(schema)
    
    # Read JSON from stdin
    try:
        data = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)
    
    # Validate data against schema
    try:
        validate(instance=data, schema=schema)
        print("✅ report_v1 schema valid")
        return True
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
