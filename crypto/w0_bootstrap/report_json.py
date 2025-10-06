#!/usr/bin/env python3
"""
Wallet Report JSON
Week 0 → Week 1: JSON output for API consumption
"""
import os
import sys
import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    import duckdb
except ImportError:
    print("❌ Missing dependencies. Install:")
    print("   pip install -e '.[crypto]'")
    sys.exit(1)

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")


def generate_report(wallet: str, hours: int = 24):
    """Generate wallet report as JSON"""
    db_path = HERE / "onchain.duckdb"
    if not db_path.exists():
        return {
            "error": "Database not found. Run capture first.",
            "wallet": wallet,
            "window_hours": hours
        }
    
    con = duckdb.connect(str(db_path))
    w = wallet.lower()
    
    # Aggregate query
    q_agg = """
    WITH recent AS (
      SELECT *
      FROM transfers
      WHERE block_time >= now() - INTERVAL ? HOUR
        AND (lower(from_addr) = ? OR lower(to_addr) = ?)
    )
    SELECT 
      COALESCE(SUM(CASE WHEN lower(to_addr) = ? THEN value_unit ELSE 0 END), 0) AS inbound,
      COALESCE(SUM(CASE WHEN lower(from_addr) = ? THEN value_unit ELSE 0 END), 0) AS outbound,
      COUNT(*) AS tx_count
    FROM recent
    """
    
    try:
        result = con.execute(q_agg, [hours, w, w, w, w]).fetchone()
        inbound, outbound, tx_count = result if result else (0.0, 0.0, 0)
    except Exception as e:
        return {
            "error": str(e),
            "wallet": wallet,
            "window_hours": hours
        }
    
    # Top counterparties
    q_top = """
    WITH recent AS (
      SELECT *
      FROM transfers
      WHERE block_time >= now() - INTERVAL ? HOUR
        AND (lower(from_addr) = ? OR lower(to_addr) = ?)
    )
    SELECT 
      CASE WHEN lower(to_addr) = ? THEN from_addr ELSE to_addr END AS counterparty,
      SUM(value_unit) AS amount
    FROM recent
    GROUP BY 1
    ORDER BY amount DESC
    LIMIT 3
    """
    
    try:
        tops = con.execute(q_top, [hours, w, w, w]).fetchall()
    except Exception as e:
        tops = []
    
    return {
        "wallet": w,
        "window_hours": hours,
        "inbound": float(inbound or 0.0),
        "outbound": float(outbound or 0.0),
        "net_flow": float((inbound or 0.0) - (outbound or 0.0)),
        "tx_count": int(tx_count or 0),
        "top_counterparties": [
            {
                "address": addr,
                "amount": float(amt)
            }
            for addr, amt in tops
        ]
    }


def main():
    ap = argparse.ArgumentParser(description="Generate wallet report (JSON)")
    ap.add_argument("--wallet", required=True, help="Wallet address (0x...)")
    ap.add_argument("--hours", type=int, default=24, help="Time window (hours)")
    ap.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    args = ap.parse_args()
    
    # Validate address
    wallet = args.wallet
    if not wallet.startswith("0x") or len(wallet) != 42:
        print(json.dumps({
            "error": "Invalid address format (expected: 0x... 42 chars)"
        }))
        return False
    
    # Generate report
    report = generate_report(wallet, hours=args.hours)
    
    # Output JSON
    if args.pretty:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False))
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
