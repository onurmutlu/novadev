#!/usr/bin/env python3
"""
Capture Transfer Events
Week 0: Fetch ERC-20 Transfer events and store in DuckDB
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

try:
    import requests
    from dotenv import load_dotenv
    import duckdb
except ImportError:
    print("❌ Missing dependencies. Install:")
    print("   pip install -e '.[crypto]'")
    sys.exit(1)

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

RPC_URL = os.getenv("RPC_URL")
TOKEN = os.getenv("TOKEN_ADDRESS", "").strip()
DECIMALS = int(os.getenv("TOKEN_DECIMALS", "18"))
START_BLOCK = int(os.getenv("START_BLOCK", "0"))

# ERC-20 Transfer(address indexed from, address indexed to, uint256 value)
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def rpc(method, params=None):
    """Call JSON-RPC"""
    r = requests.post(
        RPC_URL,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []},
        timeout=20
    )
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j["result"]


def hex_to_int(x):
    """Convert hex string to int"""
    if isinstance(x, str):
        return int(x, 16)
    return int(x)


def ensure_db():
    """Create DuckDB table if not exists"""
    db_path = HERE / "onchain.duckdb"
    con = duckdb.connect(str(db_path))
    con.sql("""
        CREATE TABLE IF NOT EXISTS transfers (
            block_number BIGINT,
            block_time   TIMESTAMP,
            tx_hash      TEXT,
            log_index    INTEGER,
            token        TEXT,
            from_addr    TEXT,
            to_addr      TEXT,
            raw_value    DECIMAL(38,0),
            value_unit   DOUBLE
        )
    """)
    con.sql("CREATE INDEX IF NOT EXISTS idx_tx ON transfers(tx_hash)")
    con.sql("CREATE INDEX IF NOT EXISTS idx_block ON transfers(block_number)")
    con.sql("CREATE INDEX IF NOT EXISTS idx_from ON transfers(from_addr)")
    con.sql("CREATE INDEX IF NOT EXISTS idx_to ON transfers(to_addr)")
    return con


def get_latest_block():
    """Get latest block number"""
    return hex_to_int(rpc("eth_blockNumber"))


def get_block_time(block_numbers):
    """Get block timestamps"""
    out = {}
    for b in block_numbers:
        try:
            blk = rpc("eth_getBlockByNumber", [hex(b), False])
            ts = hex_to_int(blk["timestamp"])
            out[b] = ts
            time.sleep(0.03)  # Rate limit
        except Exception as e:
            print(f"⚠️  Block {b} error: {e}")
            out[b] = 0
    return out


def fetch_logs(start, end):
    """Fetch logs in batches"""
    step = 1_500  # Avoid "query returned more than 10000 results"
    
    for a in range(start, end + 1, step):
        b = min(a + step - 1, end)
        
        filter_params = {
            "fromBlock": hex(a),
            "toBlock": hex(b),
            "topics": [TRANSFER_TOPIC0]
        }
        
        # Filter by token if specified
        if TOKEN and TOKEN != "0x0000000000000000000000000000000000000000":
            filter_params["address"] = TOKEN
        
        try:
            res = rpc("eth_getLogs", [filter_params])
            yield res
        except Exception as e:
            print(f"⚠️  Log fetch error [{a}..{b}]: {e}")
            yield []


def main():
    ap = argparse.ArgumentParser(description="Capture Transfer events")
    ap.add_argument("--blocks", type=int, default=5000, help="Scan last N blocks")
    args = ap.parse_args()
    
    if not RPC_URL:
        print("❌ RPC_URL yok. .env doldur.")
        return False
    
    print("=== Transfer Event Capture ===\n")
    
    # Get block range
    latest = get_latest_block()
    start = max(START_BLOCK, latest - args.blocks)
    end = latest
    
    print(f"Scanning logs {start}..{end} (latest={latest})")
    if TOKEN and TOKEN != "0x0000000000000000000000000000000000000000":
        print(f"Token filter: {TOKEN}")
    else:
        print("Token filter: All (native + ERC-20)")
    print()
    
    # Setup DB
    con = ensure_db()
    
    seen = 0
    for batch in fetch_logs(start, end):
        if not batch:
            continue
        
        # Get block timestamps
        blocks = sorted(set(hex_to_int(l["blockNumber"]) for l in batch))
        ts_map = get_block_time(blocks) if blocks else {}
        
        rows = []
        for l in batch:
            topics = l.get("topics", [])
            if not topics or topics[0].lower() != TRANSFER_TOPIC0:
                continue
            
            # Parse Transfer(from, to, value)
            if len(topics) < 3:
                continue  # Invalid Transfer event
            
            from_addr = "0x" + topics[1][-40:]
            to_addr = "0x" + topics[2][-40:]
            
            # Value from data field
            data = l.get("data", "0x0")
            try:
                val = hex_to_int(data)
            except:
                val = 0
            
            val_unit = val / (10 ** DECIMALS)
            
            bn = hex_to_int(l["blockNumber"])
            ts = ts_map.get(bn, 0)
            
            rows.append((
                bn,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)),
                l["transactionHash"],
                hex_to_int(l["logIndex"]),
                (TOKEN or l.get("address", "")).lower(),
                from_addr.lower(),
                to_addr.lower(),
                val,
                val_unit
            ))
        
        if rows:
            con.executemany(
                "INSERT INTO transfers VALUES (?,?,?,?,?,?,?,?,?)",
                rows
            )
            seen += len(rows)
            print(f"+{len(rows)} logs (total {seen})")
    
    print(f"\n✅ Done. Inserted {seen} transfer logs into {HERE/'onchain.duckdb'}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
