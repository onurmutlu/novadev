#!/usr/bin/env python3
"""
Idempotent Transfer Event Capture
Week 0 → Week 1: Production-ready ingest

Features:
- Idempotent (no duplicate inserts)
- State tracking (resume from last block)
- Reorg buffer (CONFIRMATIONS)
- Anti-join pattern (DuckDB friendly)
"""
import os
import sys
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
TOKEN = os.getenv("TOKEN_ADDRESS", "")  # Optional filter
DECIMALS = int(os.getenv("TOKEN_DECIMALS", "18"))
STATE_KEY = "transfers_v1"
CONFIRMATIONS = int(os.getenv("CONFIRMATIONS", "5"))  # Reorg buffer
STEP = 1500  # getLogs batch size

TOPIC0_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def rpc(method, params=None, timeout=25):
    """Call JSON-RPC"""
    r = requests.post(
        RPC_URL,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []},
        timeout=timeout
    )
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j["result"]


def h2i(x):
    """Hex to int"""
    return int(x, 16)


def ensure_db():
    """Create DB tables if not exist"""
    db_path = HERE / "onchain.duckdb"
    con = duckdb.connect(str(db_path))
    
    # Main table
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
    
    # Unique constraint (idempotent inserts)
    con.sql("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_txlog 
        ON transfers(tx_hash, log_index)
    """)
    
    # State table
    con.sql("""
        CREATE TABLE IF NOT EXISTS scan_state (
            skey TEXT PRIMARY KEY,
            last_scanned_block BIGINT,
            updated_at TIMESTAMP
        )
    """)
    
    return con


def get_latest_block():
    """Get latest block number"""
    return h2i(rpc("eth_blockNumber"))


def get_block_ts(b):
    """Get block timestamp"""
    blk = rpc("eth_getBlockByNumber", [hex(b), False])
    return h2i(blk["timestamp"])


def get_start_block(con, fallback_start):
    """Get last scanned block from state"""
    row = con.execute(
        "SELECT last_scanned_block FROM scan_state WHERE skey=?",
        [STATE_KEY]
    ).fetchone()
    
    if row and row[0] is not None:
        return int(row[0])
    return fallback_start


def set_state(con, bn):
    """Update scan state"""
    con.execute(
        "INSERT OR REPLACE INTO scan_state VALUES (?, ?, CURRENT_TIMESTAMP)",
        [STATE_KEY, int(bn)]
    )


def fetch_logs_range(a, b):
    """Fetch logs for block range"""
    flt = {
        "fromBlock": hex(a),
        "toBlock": hex(b),
        "topics": [TOPIC0_TRANSFER]
    }
    
    # Optional token filter
    if TOKEN and TOKEN != "0x0000000000000000000000000000000000000000":
        flt["address"] = TOKEN
    
    return rpc("eth_getLogs", [flt])


def main():
    ap = argparse.ArgumentParser(description="Idempotent transfer capture")
    ap.add_argument(
        "--backfill",
        type=int,
        default=5_000,
        help="Backfill last N blocks (first run)"
    )
    ap.add_argument(
        "--max_batches",
        type=int,
        default=10,
        help="Max batches per run"
    )
    args = ap.parse_args()
    
    if not RPC_URL:
        print("❌ RPC_URL yok. .env doldur.")
        return False
    
    print("=== Idempotent Transfer Capture ===\n")
    
    con = ensure_db()
    
    # Get block range
    latest = get_latest_block()
    safe_latest = latest - CONFIRMATIONS
    start_fallback = max(0, safe_latest - args.backfill)
    start = get_start_block(con, start_fallback)
    
    if start >= safe_latest:
        print(f"👍 Up to date. start={start}, safe_latest={safe_latest}")
        return True
    
    print(f"Scanning {start+1}..{safe_latest}")
    print(f"Latest: {latest}, Buffer: {CONFIRMATIONS} blocks\n")
    
    processed = 0
    a = start + 1
    batches = 0
    
    while a <= safe_latest and batches < args.max_batches:
        b = min(a + STEP - 1, safe_latest)
        
        try:
            logs = fetch_logs_range(a, b)
        except Exception as e:
            print(f"⚠️  Error [{a}-{b}]: {e}")
            time.sleep(1)
            continue
        
        # Prepare staging data
        rows = []
        ts_cache = {}
        
        for l in logs:
            topics = l.get("topics", [])
            if not topics or topics[0].lower() != TOPIC0_TRANSFER:
                continue
            
            bn = h2i(l["blockNumber"])
            
            # Cache block timestamp
            if bn not in ts_cache:
                try:
                    ts_cache[bn] = get_block_ts(bn)
                    time.sleep(0.02)  # Rate limit
                except Exception as e:
                    print(f"⚠️  Block {bn} timestamp error: {e}")
                    ts_cache[bn] = 0
            
            # Parse Transfer event
            if len(topics) < 3:
                continue
            
            from_addr = "0x" + topics[1][-40:]
            to_addr = "0x" + topics[2][-40:]
            
            # Value from data field
            data = l.get("data", "0x0")
            try:
                val = h2i(data)
            except:
                val = 0
            
            rows.append((
                bn,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts_cache[bn])),
                l["transactionHash"].lower(),
                h2i(l["logIndex"]),
                (TOKEN or l.get("address", "")).lower(),
                from_addr.lower(),
                to_addr.lower(),
                val,
                val / (10 ** DECIMALS)
            ))
        
        # Idempotent insert via anti-join
        if rows:
            # Staging table
            con.sql("""
                CREATE TEMP TABLE IF NOT EXISTS _staging 
                AS SELECT * FROM transfers WHERE 1=0
            """)
            
            con.executemany(
                "INSERT INTO _staging VALUES (?,?,?,?,?,?,?,?,?)",
                rows
            )
            
            # Anti-join insert (only new rows)
            con.sql("""
                INSERT INTO transfers
                SELECT s.*
                FROM _staging s
                LEFT JOIN transfers t
                ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
                WHERE t.tx_hash IS NULL
            """)
            
            inserted = con.sql("SELECT changes()").fetchone()[0]
            
            con.sql("DROP TABLE _staging")
            
            processed += inserted
            print(f"[{a:>8}-{b:>8}] {len(rows):>4} logs → {inserted:>4} inserted (total: {processed})")
        else:
            print(f"[{a:>8}-{b:>8}] 0 logs")
        
        # Update state
        set_state(con, b)
        
        a = b + 1
        batches += 1
        time.sleep(0.05)  # Rate limit between batches
    
    print(f"\n✅ Done. State: last_scanned_block={a-1}")
    print(f"Total inserted: {processed} transfers (idempotent)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
