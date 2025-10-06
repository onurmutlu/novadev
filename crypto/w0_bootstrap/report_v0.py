#!/usr/bin/env python3
"""
Wallet Report v0
Week 0: Basic wallet summary from DuckDB
"""
import os
import sys
import argparse
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
DECIMALS = int(os.getenv("TOKEN_DECIMALS", "18"))


def generate_report(wallet: str, hours: int = 24):
    """
    Generate wallet report from DuckDB
    
    Returns:
        dict with inbound, outbound, tx_count, top_counterparties
    """
    db_path = HERE / "onchain.duckdb"
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run capture_transfers.py first:")
        print("   python crypto/w0_bootstrap/capture_transfers.py --blocks 5000")
        return None
    
    con = duckdb.connect(str(db_path))
    wallet = wallet.lower()
    
    # Aggregate query
    q_agg = """
    WITH recent AS (
      SELECT *
      FROM transfers
      WHERE block_time >= now() - INTERVAL ? HOUR
        AND (lower(from_addr) = ? OR lower(to_addr) = ?)
    )
    SELECT
        SUM(CASE WHEN lower(to_addr) = ? THEN value_unit ELSE 0 END) AS inbound,
        SUM(CASE WHEN lower(from_addr) = ? THEN value_unit ELSE 0 END) AS outbound,
        COUNT(*) AS n_tx
    FROM recent
    """
    
    try:
        agg = con.execute(q_agg, [hours, wallet, wallet, wallet, wallet]).fetchone()
    except Exception as e:
        print(f"❌ Query error: {e}")
        return None
    
    if not agg:
        inbound, outbound, n_tx = 0.0, 0.0, 0
    else:
        inbound, outbound, n_tx = agg
        inbound = inbound or 0.0
        outbound = outbound or 0.0
        n_tx = n_tx or 0
    
    # Top counterparties
    q_tops = """
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
        tops = con.execute(q_tops, [hours, wallet, wallet, wallet]).fetchall()
    except Exception as e:
        print(f"⚠️  Top counterparties error: {e}")
        tops = []
    
    return {
        "wallet": wallet,
        "hours": hours,
        "inbound": inbound,
        "outbound": outbound,
        "net_flow": inbound - outbound,
        "tx_count": n_tx,
        "top_counterparties": [
            {"address": cp, "amount": amt}
            for cp, amt in tops
        ]
    }


def print_report(report):
    """Pretty print report"""
    if not report:
        return
    
    print("=" * 60)
    print(f"✅ Wallet Report: {report['wallet']}")
    print("=" * 60)
    print(f"Window:     last {report['hours']}h")
    print(f"Inbound:    {report['inbound']:.6f}")
    print(f"Outbound:   {report['outbound']:.6f}")
    print(f"Net flow:   {report['net_flow']:.6f}")
    print(f"Tx count:   {report['tx_count']}")
    
    if report['top_counterparties']:
        print("\nTop Counterparties:")
        for i, cp in enumerate(report['top_counterparties'], 1):
            print(f"  {i}. {cp['address']}: {cp['amount']:.6f}")
    else:
        print("\nTop Counterparties: (none)")
    
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Generate wallet report")
    ap.add_argument("--wallet", required=True, help="Wallet address (0x...)")
    ap.add_argument("--hours", type=int, default=24, help="Time window (hours)")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()
    
    # Validate address
    wallet = args.wallet
    if not wallet.startswith("0x") or len(wallet) != 42:
        print("❌ Invalid address format")
        print("   Expected: 0x... (42 characters)")
        return False
    
    # Generate report
    report = generate_report(wallet, hours=args.hours)
    if not report:
        return False
    
    # Output
    if args.json:
        import json
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
