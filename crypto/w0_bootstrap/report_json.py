#!/usr/bin/env python3
"""
Wallet Report Generator (CLI)

Production-grade CLI for generating wallet activity reports.

Usage:
    python crypto/w0_bootstrap/report_json.py \
      --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 \
      --hours 24 \
      --db onchain.duckdb

Options:
    --wallet ADDR       Wallet address (required)
    --hours N           Time window in hours (default: 24, max: 720)
    --db PATH           DuckDB path (default: onchain.duckdb)
    --chain-id ID       Chain ID (default: 11155111 Sepolia)
    --validate          Validate against schema before output
    --output FILE       Write to file instead of stdout

Examples:
    # Generate 24h report
    python crypto/w0_bootstrap/report_json.py --wallet 0xABC...

    # Generate 7-day report with validation
    python crypto/w0_bootstrap/report_json.py \\
      --wallet 0xABC... --hours 168 --validate

    # Save to file
    python crypto/w0_bootstrap/report_json.py \\
      --wallet 0xABC... --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crypto.features.report_builder import ReportBuilder, ReportConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate wallet activity report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--wallet", 
        required=True, 
        help="Wallet address (0x...)"
    )
    
    parser.add_argument(
        "--hours", 
        type=int, 
        default=24, 
        help="Time window (1-720 hours, default: 24)"
    )
    
    parser.add_argument(
        "--db", 
        default="onchain.duckdb", 
        help="DuckDB path (default: onchain.duckdb)"
    )
    
    parser.add_argument(
        "--chain-id", 
        type=int, 
        default=11155111, 
        help="Chain ID (default: 11155111 Sepolia)"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true", 
        help="Validate against schema before output"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file path (default: stdout)"
    )
    
    args = parser.parse_args()
    
    try:
        # Build report
        config = ReportConfig(chain_id=args.chain_id)
        
        with ReportBuilder(args.db, config) as builder:
            report = builder.build(args.wallet, args.hours)
        
        # Optional validation
        if args.validate:
            from crypto.features.report_validator import ReportValidator
            validator = ReportValidator()
            validator.validate(report)
            print("✅ Report validated", file=sys.stderr)
        
        # Format JSON
        output_json = json.dumps(report, indent=2)
        
        # Output
        if args.output:
            Path(args.output).write_text(output_json, encoding='utf-8')
            print(f"✅ Report written to {args.output}", file=sys.stderr)
        else:
            print(output_json)
        
    except ValueError as e:
        print(f"❌ Input error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Database not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
