#!/usr/bin/env python3
"""
Wallet Report v0
Week 0: Basic wallet summary (skeleton)
"""
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_report(address: str, period_hours: int = 24):
    """
    Generate wallet report
    
    Week 0: Skeleton (mock data)
    Week 1: Real data from DuckDB
    """
    print(f"Generating report for: {address}")
    print(f"Period: last {period_hours}h\n")
    
    # Week 0: Mock data (placeholder)
    # Week 1: Replace with real DB queries
    
    report = {
        "wallet": address,
        "network": "Sepolia",
        "period": f"{period_hours}h",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        
        "summary": {
            "inbound": {
                "count": 0,  # TODO: Query transfers WHERE to_addr = address
                "total_eth": 0.0,
                "total_usd": 0.0
            },
            "outbound": {
                "count": 0,  # TODO: Query transfers WHERE from_addr = address
                "total_eth": 0.0,
                "total_usd": 0.0
            },
            "net_flow": {
                "eth": 0.0,
                "usd": 0.0
            }
        },
        
        "top_counterparties": [
            # TODO: Week 1
            # {
            #     "address": "0xabc...",
            #     "label": "Uniswap V3: Router",
            #     "interactions": 5
            # }
        ],
        
        "note": "Week 0: Skeleton report. Week 1: Real data."
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate wallet report")
    parser.add_argument(
        "--address",
        required=True,
        help="Wallet address (0x...)"
    )
    parser.add_argument(
        "--period",
        type=int,
        default=24,
        help="Period in hours (default: 24)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "pretty"],
        default="pretty",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Validate address
    address = args.address
    if not address.startswith("0x") or len(address) != 42:
        print("❌ Invalid address format")
        print("   Expected: 0x... (42 chars)")
        sys.exit(1)
    
    # Generate report
    report = generate_report(address, period_hours=args.period)
    
    # Output
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("=" * 50)
        print(f"Wallet Report: {report['wallet']}")
        print("=" * 50)
        print(f"Network: {report['network']}")
        print(f"Period: {report['period']}")
        print(f"Generated: {report['generated_at']}")
        print()
        
        print("Summary:")
        summary = report["summary"]
        print(f"  Inbound:  {summary['inbound']['count']} tx, "
              f"${summary['inbound']['total_usd']:.2f}")
        print(f"  Outbound: {summary['outbound']['count']} tx, "
              f"${summary['outbound']['total_usd']:.2f}")
        print(f"  Net flow: ${summary['net_flow']['usd']:.2f}")
        print()
        
        if report["top_counterparties"]:
            print("Top Counterparties:")
            for cp in report["top_counterparties"]:
                print(f"  • {cp['address']} ({cp['label']}): {cp['interactions']} interactions")
        else:
            print("Top Counterparties: (none - Week 0 skeleton)")
        
        print("\n" + "=" * 50)
        print(f"Note: {report['note']}")
        print("=" * 50)


if __name__ == "__main__":
    main()
