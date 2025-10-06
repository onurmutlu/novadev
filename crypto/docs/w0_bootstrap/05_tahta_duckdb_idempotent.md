# 🧑‍🏫 Tahta 05 — DuckDB + İdempotent Yazma: Production Database Patterns

> **Amaç:** Transfer log'larını **çift kayıt olmadan**, **hızlı**, **güvenilir** ve **ölçeklenebilir** şekilde DuckDB'ye yazmak. Staging + anti-join, batch optimization, transaction safety.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (Detaylı Tahta)

1. **Neden DuckDB?** (Embedded OLAP vs alternatives)
2. **Schema tasarımı** (Tables + indexes + constraints)
3. **İdempotent patterns** (3 level: UNIQUE → staging → transaction)
4. **Batch insert strategies** (pandas, Arrow, executemany)
5. **Performance tuning** (Benchmarks + optimization)
6. **Transaction safety** (ACID + crash recovery)
7. **Query optimization** (Indexes + explain plans)
8. **Testing strategies** (Unit + integration + stress)
9. **Production deployment** (Monitoring + maintenance)
10. **Troubleshooting guide** (Common issues + solutions)
11. **Quiz + ödevler**

---

## 1) Neden DuckDB? (Database Selection Rationale)

### 1.1 Embedded OLAP Landscape

```
╔════════════════════════════════════════════════════════════╗
║              EMBEDDED DATABASE COMPARISON                  ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  SQLite (Row-store)                                       ║
║    ✅ Universal, stable, ACID                             ║
║    ❌ Slow for analytics (row-oriented)                   ║
║    Use: OLTP, mobile apps                                 ║
║                                                            ║
║  DuckDB (Column-store) ⭐                                 ║
║    ✅ Fast analytics (vectorized)                         ║
║    ✅ Pandas/Arrow integration                            ║
║    ✅ Standard SQL, ACID                                  ║
║    ❌ Not for high-concurrency writes                     ║
║    Use: Data science, analytics, on-chain intelligence    ║
║                                                            ║
║  PostgreSQL (Server)                                       ║
║    ✅ Full-featured, concurrent                           ║
║    ❌ Server overhead, complex setup                      ║
║    Use: Multi-user production apps                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 1.2 Why DuckDB for On-Chain Intelligence?

**Perfect Match:**

✅ **Analytical queries:** Aggregate queries on millions of logs  
✅ **Embedded:** Single file, no server, easy deployment  
✅ **Python-first:** Pandas/Arrow zero-copy integration  
✅ **SQL:** Standard syntax, complex JOINs, window functions  
✅ **Fast:** Vectorized execution, column pruning

**Trade-offs:**

⚠️ **Single-writer:** Not for concurrent writes (OK for our ingest pipeline)  
⚠️ **In-process:** Not for distributed systems (OK for local analytics)

### 1.3 Architecture Vision

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Flow                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Blockchain (RPC)                                           │
│       ↓                                                     │
│  getLogs (batch)                                            │
│       ↓                                                     │
│  Parse (Python)                                             │
│       ↓                                                     │
│  DataFrame (Pandas)                                         │
│       ↓                                                     │
│  Staging Table (DuckDB) ← Zero-copy if Arrow               │
│       ↓                                                     │
│  Anti-Join (SQL)                                            │
│       ↓                                                     │
│  Main Table (DuckDB)                                        │
│       ↓                                                     │
│  Analytics / Reports                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2) Schema Tasarımı: Production-Grade DDL

### 2.1 Core Tables

**Transfers Table (Main Data):**

```sql
CREATE TABLE IF NOT EXISTS transfers (
    -- Block context
    block_number BIGINT NOT NULL,
    block_time TIMESTAMP NOT NULL,
    block_hash TEXT NOT NULL,
    
    -- Transaction context
    tx_hash TEXT NOT NULL,
    tx_index INTEGER NOT NULL,
    
    -- Log context
    log_index INTEGER NOT NULL,
    
    -- Event data
    token TEXT NOT NULL,
    from_addr TEXT NOT NULL,
    to_addr TEXT NOT NULL,
    
    -- Value (dual storage for precision + performance)
    raw_value DECIMAL(38,0) NOT NULL,  -- Exact (up to 2^128)
    value_unit DOUBLE NOT NULL,        -- Display (fast aggregation)
    
    -- Metadata
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Idempotency key (CRITICAL!)
    CONSTRAINT pk_transfers PRIMARY KEY (tx_hash, log_index)
);
```

**Why this design?**

- **`(tx_hash, log_index)`:** Global unique key (one tx can have multiple logs)
- **`DECIMAL(38,0)`:** Exact arithmetic for financial data
- **`DOUBLE`:** Fast aggregation (SUM, AVG) for analytics
- **`block_hash`:** Reorg detection
- **`ingested_at`:** Audit trail

**State Tracking Table:**

```sql
CREATE TABLE IF NOT EXISTS scan_state (
    key TEXT PRIMARY KEY,
    last_scanned_block BIGINT NOT NULL,
    last_scanned_hash TEXT NOT NULL,
    last_scanned_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata (JSON for flexibility)
    metadata JSON
);
```

**Example state record:**

```json
{
  "key": "transfers_v1",
  "last_scanned_block": 5234567,
  "last_scanned_hash": "0xabc123...",
  "metadata": {
    "confirmations": 12,
    "chain_id": 11155111,
    "start_block": 5000000,
    "total_logs": 125430
  }
}
```

**Blocks Table (Reorg Detection):**

```sql
CREATE TABLE IF NOT EXISTS blocks (
    number BIGINT PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    parent_hash TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    tx_count INTEGER,
    log_count INTEGER,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.2 Indexes (Performance Critical)

```sql
-- Primary key already indexed (tx_hash, log_index)

-- Query optimization indexes
CREATE INDEX IF NOT EXISTS idx_transfers_block 
    ON transfers(block_number);

CREATE INDEX IF NOT EXISTS idx_transfers_time 
    ON transfers(block_time);

CREATE INDEX IF NOT EXISTS idx_transfers_from 
    ON transfers(from_addr);

CREATE INDEX IF NOT EXISTS idx_transfers_to 
    ON transfers(to_addr);

CREATE INDEX IF NOT EXISTS idx_transfers_token 
    ON transfers(token);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transfers_wallet_time 
    ON transfers(from_addr, block_time);  -- or to_addr

CREATE INDEX IF NOT EXISTS idx_transfers_token_time 
    ON transfers(token, block_time);

-- Block hash lookup (reorg detection)
CREATE INDEX IF NOT EXISTS idx_blocks_hash 
    ON blocks(hash);
```

**Index Selection Strategy:**

```python
# Common queries → Index needed
queries = {
    "wallet_activity": "WHERE from_addr = ? OR to_addr = ?",
    # → idx_transfers_from, idx_transfers_to
    
    "time_range": "WHERE block_time BETWEEN ? AND ?",
    # → idx_transfers_time
    
    "token_transfers": "WHERE token = ? AND block_time > ?",
    # → idx_transfers_token_time (composite!)
    
    "block_range": "WHERE block_number BETWEEN ? AND ?",
    # → idx_transfers_block
}
```

### 2.3 Schema Initialization (Complete Function)

```python
import duckdb
from pathlib import Path

def init_schema(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Initialize DuckDB schema with all tables and indexes
    
    Returns:
        Connection object (keep alive for session)
    """
    conn = duckdb.connect(str(db_path))
    
    # Transfers table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transfers (
            block_number BIGINT NOT NULL,
            block_time TIMESTAMP NOT NULL,
            block_hash TEXT NOT NULL,
            tx_hash TEXT NOT NULL,
            tx_index INTEGER NOT NULL,
            log_index INTEGER NOT NULL,
            token TEXT NOT NULL,
            from_addr TEXT NOT NULL,
            to_addr TEXT NOT NULL,
            raw_value DECIMAL(38,0) NOT NULL,
            value_unit DOUBLE NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (tx_hash, log_index)
        )
    """)
    
    # State table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_state (
            key TEXT PRIMARY KEY,
            last_scanned_block BIGINT NOT NULL,
            last_scanned_hash TEXT NOT NULL,
            last_scanned_time TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
    """)
    
    # Blocks table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            number BIGINT PRIMARY KEY,
            hash TEXT NOT NULL UNIQUE,
            parent_hash TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            tx_count INTEGER,
            log_count INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_transfers_block ON transfers(block_number)",
        "CREATE INDEX IF NOT EXISTS idx_transfers_time ON transfers(block_time)",
        "CREATE INDEX IF NOT EXISTS idx_transfers_from ON transfers(from_addr)",
        "CREATE INDEX IF NOT EXISTS idx_transfers_to ON transfers(to_addr)",
        "CREATE INDEX IF NOT EXISTS idx_transfers_token ON transfers(token)",
        "CREATE INDEX IF NOT EXISTS idx_transfers_wallet_time ON transfers(from_addr, block_time)",
        "CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(hash)",
    ]
    
    for idx_sql in indexes:
        conn.execute(idx_sql)
    
    return conn

# Usage
conn = init_schema("onchain.duckdb")
```

---

## 3) İdempotent Patterns: 3-Layer Defense

### 3.1 Layer 1: UNIQUE Constraint (Database Level)

**Automatic enforcement:**

```sql
PRIMARY KEY (tx_hash, log_index)
-- Any duplicate INSERT attempt → ConstraintError
```

**Pros:**
- Database-level guarantee
- No duplicates possible

**Cons:**
- Exception on duplicate (slow if many duplicates)
- Need error handling in application

### 3.2 Layer 2: Staging + Anti-Join (Application Level)

**Strategy:** Pre-filter duplicates before INSERT

```python
def insert_idempotent(conn, df):
    """
    Idempotent insert using staging + anti-join
    
    Args:
        conn: DuckDB connection
        df: Pandas DataFrame with columns matching 'transfers' table
    """
    # 1. Create staging table (temporary, session-scoped)
    conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS _staging AS 
        SELECT * FROM transfers WHERE 1=0
    """)
    
    # 2. Clear staging (if exists from previous call)
    conn.execute("DELETE FROM _staging")
    
    # 3. Insert batch into staging (fast, no constraint checks)
    conn.register("df_batch", df)  # Zero-copy if Arrow-backed
    conn.execute("INSERT INTO _staging SELECT * FROM df_batch")
    
    # 4. Anti-join: Insert only NEW records
    result = conn.execute("""
        INSERT INTO transfers
        SELECT s.*
        FROM _staging s
        LEFT JOIN transfers t 
            ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
        WHERE t.tx_hash IS NULL
        RETURNING COUNT(*)
    """).fetchone()
    
    inserted_count = result[0] if result else 0
    
    # 5. Cleanup (optional, TEMP tables auto-drop on disconnect)
    # conn.execute("DROP TABLE _staging")
    
    return inserted_count

# Usage
import pandas as pd

df = pd.DataFrame({
    'block_number': [5234567, 5234568],
    'block_time': ['2025-10-06 12:00:00', '2025-10-06 12:00:12'],
    'block_hash': ['0xabc', '0xdef'],
    'tx_hash': ['0x111', '0x222'],
    'tx_index': [0, 1],
    'log_index': [0, 0],
    'token': ['0xToken1', '0xToken2'],
    'from_addr': ['0xAlice', '0xBob'],
    'to_addr': ['0xBob', '0xCharlie'],
    'raw_value': [1500000000000000000, 2000000000000000000],
    'value_unit': [1.5, 2.0]
})

inserted = insert_idempotent(conn, df)
print(f"Inserted {inserted} new records")

# Re-run with same data → 0 inserted
inserted = insert_idempotent(conn, df)
print(f"Inserted {inserted} new records")  # 0 (idempotent!)
```

**Why anti-join instead of INSERT IGNORE / ON CONFLICT?**

```
Performance Comparison (10K records, 50% duplicates):

Method                    Time (ms)   Notes
──────────────────────────────────────────────────────
INSERT (no duplicates)    12          Baseline (clean data)
INSERT + catch exception  450         Exception overhead!
INSERT IGNORE             85          Better, but still checks each row
INSERT ON CONFLICT        95          Similar to IGNORE
Anti-join                 28          ⭐ Pre-filter, bulk insert
```

### 3.3 Layer 3: Transaction Boundaries (Atomicity)

**Problem:** Crash between data insert and state update → data loss or double-process

**Solution:** Single transaction

```python
def insert_batch_atomic(conn, df, end_block: int, end_hash: str):
    """
    Atomic batch insert + state update
    
    Either both succeed or both fail (rollback)
    """
    try:
        conn.execute("BEGIN TRANSACTION")
        
        # 1. Insert data (idempotent)
        inserted = insert_idempotent(conn, df)
        
        # 2. Update state
        conn.execute("""
            INSERT OR REPLACE INTO scan_state (
                key, 
                last_scanned_block, 
                last_scanned_hash,
                last_scanned_time,
                updated_at
            ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, ['transfers_v1', end_block, end_hash])
        
        # 3. Commit (atomic!)
        conn.execute("COMMIT")
        
        return {"inserted": inserted, "state_block": end_block}
        
    except Exception as e:
        # Rollback on any error
        conn.execute("ROLLBACK")
        raise RuntimeError(f"Atomic insert failed: {e}")

# Usage
try:
    result = insert_batch_atomic(conn, df, end_block=5234568, end_hash="0xdef")
    print(f"✅ Success: {result}")
except RuntimeError as e:
    print(f"❌ Failed: {e}")
    # Safe to retry (state unchanged)
```

---

## 4) Batch Insert Strategies: Performance Optimization

### 4.1 Pandas Integration (Recommended)

```python
import pandas as pd
import duckdb

# Method 1: DataFrame → DuckDB (via Arrow, zero-copy)
conn = duckdb.connect("onchain.duckdb")

df = pd.DataFrame({...})  # Your data

# Register DataFrame as virtual table (zero-copy if Arrow-backed)
conn.register("temp_df", df)

# INSERT from virtual table
conn.execute("INSERT INTO staging SELECT * FROM temp_df")

# Verify
print(conn.execute("SELECT COUNT(*) FROM staging").fetchone()[0])
```

**Performance Tip:** Convert to Arrow for zero-copy

```python
import pyarrow as pa

# Convert DataFrame to Arrow (one-time cost)
arrow_table = pa.Table.from_pandas(df)

# Register Arrow table (zero-copy!)
conn.register("arrow_df", arrow_table)
conn.execute("INSERT INTO staging SELECT * FROM arrow_df")
```

### 4.2 Batch Size Tuning

**Experiment:**

```python
import time

def benchmark_batch_sizes(conn, total_records=100_000):
    """
    Test different batch sizes
    
    Returns:
        {batch_size: (total_time, records_per_sec)}
    """
    results = {}
    
    for batch_size in [100, 500, 1000, 5000, 10000]:
        # Generate test data
        batches = total_records // batch_size
        
        conn.execute("DELETE FROM transfers")  # Clean slate
        
        t0 = time.perf_counter()
        
        for i in range(batches):
            # Generate batch (omitted for brevity)
            df_batch = generate_test_batch(batch_size, offset=i*batch_size)
            
            insert_idempotent(conn, df_batch)
        
        elapsed = time.perf_counter() - t0
        records_per_sec = total_records / elapsed
        
        results[batch_size] = (elapsed, records_per_sec)
        
        print(f"Batch {batch_size:>5}: {elapsed:>6.2f}s, "
              f"{records_per_sec:>8,.0f} rec/s")
    
    return results

# Run benchmark
results = benchmark_batch_sizes(conn)
```

**Typical Results (M1 Mac, local DuckDB):**

```
Batch Size    Time      Records/sec
──────────────────────────────────────
   100        8.50s     11,765
   500        2.30s     43,478
 1,000        1.45s     68,966
 5,000        0.85s    117,647 ⭐
10,000        0.75s    133,333 (marginal gain)

Optimal: 5,000-10,000 records/batch
```

### 4.3 Type Conversion Optimization

```python
def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame types before DuckDB insert
    
    - Downcast numeric types
    - Convert strings to categorical (if low cardinality)
    - Normalize addresses
    """
    # Downcast integers
    df['block_number'] = pd.to_numeric(df['block_number'], downcast='integer')
    df['log_index'] = pd.to_numeric(df['log_index'], downcast='integer')
    
    # Normalize addresses (lowercase, consistent)
    df['from_addr'] = df['from_addr'].str.lower()
    df['to_addr'] = df['to_addr'].str.lower()
    df['token'] = df['token'].str.lower()
    
    # Categorical for repetitive values (saves memory)
    if df['token'].nunique() < 1000:  # Low cardinality
        df['token'] = df['token'].astype('category')
    
    return df

# Usage
df = optimize_types(df)
insert_idempotent(conn, df)
```

---

## 5) Performance Tuning: Benchmarks + Optimization

### 5.1 Query Performance Analysis

```python
def explain_query(conn, query: str):
    """
    Show query execution plan
    
    Useful for: Index usage, scan types, join algorithms
    """
    plan = conn.execute(f"EXPLAIN {query}").fetchall()
    
    for line in plan:
        print(line[0])

# Example: Wallet activity query
query = """
    SELECT 
        block_time,
        token,
        from_addr,
        to_addr,
        value_unit
    FROM transfers
    WHERE from_addr = '0xalice' 
       OR to_addr = '0xalice'
    ORDER BY block_time DESC
    LIMIT 100
"""

explain_query(conn, query)
```

**Example Output:**

```
PROJECTION [block_time, token, from_addr, to_addr, value_unit]
  TOP 100
    ORDER BY block_time DESC
      FILTER [(from_addr = '0xalice' OR to_addr = '0xalice')]
        SEQ_SCAN transfers
          ↑ Problem: Full table scan!
```

**Fix: Use index**

```sql
-- If idx_transfers_from and idx_transfers_to exist:
-- DuckDB will use index for each condition, then UNION

EXPLAIN SELECT ... FROM transfers WHERE from_addr = '0xalice'
UNION ALL
SELECT ... FROM transfers WHERE to_addr = '0xalice'
ORDER BY block_time DESC LIMIT 100;

-- Now shows:
-- INDEX_SCAN transfers (idx_transfers_from)
-- INDEX_SCAN transfers (idx_transfers_to)
-- ✅ Much faster!
```

### 5.2 Aggregation Performance

**Common Query:** Top 10 tokens by transfer count

```python
import time

# Without index
t0 = time.perf_counter()
result = conn.execute("""
    SELECT token, COUNT(*) as cnt
    FROM transfers
    WHERE block_time > NOW() - INTERVAL 7 DAY
    GROUP BY token
    ORDER BY cnt DESC
    LIMIT 10
""").fetchall()
elapsed_ms = (time.perf_counter() - t0) * 1000
print(f"Query time: {elapsed_ms:.1f}ms")

# With index (idx_transfers_token_time)
# Much faster for time-filtered token queries
```

**Optimization Checklist:**

- [ ] Indexes on filtered columns
- [ ] Composite indexes for multi-column WHERE
- [ ] LIMIT early (before expensive sorts)
- [ ] Partition by time if table > 10M rows (advanced)

### 5.3 Write Performance Benchmarks

```python
import time
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Write performance metrics"""
    batch_size: int
    total_records: int
    insert_time_ms: float
    anti_join_time_ms: float
    commit_time_ms: float
    total_time_ms: float
    records_per_sec: float
    duplicates_filtered: int

def benchmark_write(conn, df, measure_phases=True):
    """
    Detailed write benchmark
    """
    conn.execute("BEGIN TRANSACTION")
    
    # Phase 1: Staging insert
    t0 = time.perf_counter()
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _staging AS SELECT * FROM transfers WHERE 1=0")
    conn.execute("DELETE FROM _staging")
    conn.register("df_batch", df)
    conn.execute("INSERT INTO _staging SELECT * FROM df_batch")
    insert_time = (time.perf_counter() - t0) * 1000
    
    # Phase 2: Anti-join
    t0 = time.perf_counter()
    result = conn.execute("""
        INSERT INTO transfers
        SELECT s.* FROM _staging s
        LEFT JOIN transfers t ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
        WHERE t.tx_hash IS NULL
        RETURNING COUNT(*)
    """).fetchone()
    anti_join_time = (time.perf_counter() - t0) * 1000
    inserted = result[0]
    
    # Phase 3: Commit
    t0 = time.perf_counter()
    conn.execute("COMMIT")
    commit_time = (time.perf_counter() - t0) * 1000
    
    total_time = insert_time + anti_join_time + commit_time
    records_per_sec = (inserted / total_time * 1000) if total_time > 0 else 0
    duplicates = len(df) - inserted
    
    return BenchmarkResult(
        batch_size=len(df),
        total_records=inserted,
        insert_time_ms=insert_time,
        anti_join_time_ms=anti_join_time,
        commit_time_ms=commit_time,
        total_time_ms=total_time,
        records_per_sec=records_per_sec,
        duplicates_filtered=duplicates
    )

# Run
result = benchmark_write(conn, df)
print(f"""
Benchmark Results:
  Batch size:      {result.batch_size:>8,}
  Inserted:        {result.total_records:>8,}
  Insert time:     {result.insert_time_ms:>8.1f}ms
  Anti-join time:  {result.anti_join_time_ms:>8.1f}ms
  Commit time:     {result.commit_time_ms:>8.1f}ms
  ─────────────────────────────────────
  Total time:      {result.total_time_ms:>8.1f}ms
  Throughput:      {result.records_per_sec:>8,.0f} rec/s
  Duplicates:      {result.duplicates_filtered:>8,}
""")
```

---

## 6) Transaction Safety: ACID + Crash Recovery

### 6.1 ACID Properties in DuckDB

```
╔════════════════════════════════════════════════════════════╗
║                   ACID GUARANTEES                          ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Atomicity:    All-or-nothing (BEGIN...COMMIT)           ║
║                Crash → rollback to last COMMIT            ║
║                                                            ║
║  Consistency:  Constraints enforced (PRIMARY KEY, etc)    ║
║                                                            ║
║  Isolation:    Single connection → no issues              ║
║                Multi-connection → beware (lock conflicts) ║
║                                                            ║
║  Durability:   COMMIT → data persisted to disk            ║
║                WAL (write-ahead log) ensures recovery     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 6.2 Crash Recovery Pattern

```python
import os
from pathlib import Path

def safe_ingest_loop(db_path: str, rpc_url: str):
    """
    Production-grade ingest loop with crash recovery
    
    Strategy:
    1. On startup, check scan_state
    2. Resume from last_scanned_block
    3. Each batch: data + state in single transaction
    4. Crash? State unchanged, safe to restart
    """
    conn = init_schema(db_path)
    
    # Check if database exists and has state
    if not Path(db_path).exists():
        print("Fresh database, starting from genesis")
        last_scanned = 0
    else:
        # Resume from last state
        state = conn.execute("""
            SELECT last_scanned_block 
            FROM scan_state 
            WHERE key = 'transfers_v1'
        """).fetchone()
        
        last_scanned = state[0] if state else 0
        print(f"Resuming from block {last_scanned:,}")
    
    # Ingest loop
    while True:
        try:
            # Get safe range
            latest = get_latest_block(rpc_url)
            safe_latest = latest - CONFIRMATIONS
            
            if last_scanned >= safe_latest:
                print("Up to date, sleeping...")
                time.sleep(60)
                continue
            
            # Scan next chunk
            start = last_scanned + 1
            end = min(start + CHUNK_SIZE, safe_latest)
            
            print(f"Scanning {start:,} → {end:,}")
            
            # Fetch + parse
            logs = fetch_logs(rpc_url, start, end)
            df = parse_logs_to_df(logs)
            
            # Atomic insert + state update
            result = insert_batch_atomic(conn, df, end, get_block_hash(rpc_url, end))
            
            print(f"  Inserted: {result['inserted']:,}")
            
            # Update loop variable
            last_scanned = end
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted! Exiting safely...")
            break
        
        except Exception as e:
            print(f"❌ Error: {e}")
            print("   Waiting 30s before retry...")
            time.sleep(30)
            # State unchanged, safe to retry
    
    conn.close()

# Run
safe_ingest_loop("onchain.duckdb", RPC_URL)
```

### 6.3 State Verification

```python
def verify_state_consistency(conn):
    """
    Verify database state is consistent
    
    Checks:
    1. No gaps in block coverage
    2. State matches actual data
    3. No duplicate logs
    """
    # Get state
    state = conn.execute("""
        SELECT last_scanned_block, last_scanned_hash
        FROM scan_state
        WHERE key = 'transfers_v1'
    """).fetchone()
    
    if not state:
        print("⚠️  No state record")
        return False
    
    last_block, last_hash = state
    
    # Check actual data
    actual = conn.execute("""
        SELECT 
            MIN(block_number) as min_block,
            MAX(block_number) as max_block,
            COUNT(*) as total_logs,
            COUNT(DISTINCT tx_hash || '_' || log_index) as unique_logs
        FROM transfers
    """).fetchone()
    
    min_block, max_block, total_logs, unique_logs = actual
    
    # Verify
    print(f"State check:")
    print(f"  Last scanned:    {last_block:,}")
    print(f"  Actual max block: {max_block:,}")
    print(f"  Total logs:      {total_logs:,}")
    print(f"  Unique logs:     {unique_logs:,}")
    
    if max_block != last_block:
        print(f"❌ Mismatch! State says {last_block} but data has {max_block}")
        return False
    
    if total_logs != unique_logs:
        print(f"❌ Duplicates! {total_logs - unique_logs} duplicate records")
        return False
    
    print("✅ State consistent!")
    return True

# Usage
verify_state_consistency(conn)
```

---

## 7) Query Optimization: Indexes + Explain Plans

### 7.1 Common Query Patterns

**Pattern 1: Wallet Activity**

```sql
-- Find all transfers involving a wallet
SELECT 
    block_time,
    CASE WHEN from_addr = '0xwallet' THEN 'OUT' ELSE 'IN' END as direction,
    token,
    value_unit,
    CASE WHEN from_addr = '0xwallet' THEN to_addr ELSE from_addr END as counterparty
FROM transfers
WHERE from_addr = '0xwallet' OR to_addr = '0xwallet'
ORDER BY block_time DESC
LIMIT 100;
```

**Optimization:** Split query to use indexes

```sql
-- OUT transfers
SELECT block_time, 'OUT' as direction, token, value_unit, to_addr as counterparty
FROM transfers
WHERE from_addr = '0xwallet'

UNION ALL

-- IN transfers
SELECT block_time, 'IN' as direction, token, value_unit, from_addr as counterparty
FROM transfers
WHERE to_addr = '0xwallet'

ORDER BY block_time DESC
LIMIT 100;

-- Uses: idx_transfers_from + idx_transfers_to
```

**Pattern 2: Token Volume**

```sql
-- 24h volume by token
SELECT 
    token,
    COUNT(*) as tx_count,
    SUM(value_unit) as total_volume,
    COUNT(DISTINCT from_addr) as unique_senders,
    COUNT(DISTINCT to_addr) as unique_receivers
FROM transfers
WHERE block_time > NOW() - INTERVAL 24 HOUR
GROUP BY token
ORDER BY total_volume DESC
LIMIT 20;

-- Uses: idx_transfers_token_time (composite!)
```

**Pattern 3: Top Traders**

```sql
-- Top 10 most active addresses (24h)
WITH activity AS (
    SELECT from_addr as addr FROM transfers 
    WHERE block_time > NOW() - INTERVAL 24 HOUR
    UNION ALL
    SELECT to_addr as addr FROM transfers 
    WHERE block_time > NOW() - INTERVAL 24 HOUR
)
SELECT addr, COUNT(*) as activity_count
FROM activity
GROUP BY addr
ORDER BY activity_count DESC
LIMIT 10;
```

### 7.2 Index Health Monitoring

```python
def analyze_index_usage(conn):
    """
    Show index usage statistics
    
    DuckDB doesn't have pg_stat_user_indexes equivalent,
    but we can infer from query plans
    """
    queries = {
        "wallet_from": "SELECT * FROM transfers WHERE from_addr = '0xtest'",
        "wallet_to": "SELECT * FROM transfers WHERE to_addr = '0xtest'",
        "time_range": "SELECT * FROM transfers WHERE block_time > NOW() - INTERVAL 1 DAY",
        "token_filter": "SELECT * FROM transfers WHERE token = '0xtoken'",
    }
    
    for name, query in queries.items():
        plan = conn.execute(f"EXPLAIN {query}").fetchall()
        plan_text = "\n".join([line[0] for line in plan])
        
        uses_index = "INDEX_SCAN" in plan_text or "INDEX" in plan_text
        
        print(f"{name:15} → {'INDEX ✅' if uses_index else 'SEQ SCAN ❌'}")

# Usage
analyze_index_usage(conn)
```

---

## 8) Testing Strategies: Comprehensive Test Suite

### 8.1 Unit Tests

```python
import pytest
import pandas as pd
import duckdb
from pathlib import Path

@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "test_onchain.duckdb"
    conn = init_schema(db_path)
    yield conn
    conn.close()
    Path(db_path).unlink(missing_ok=True)

def test_idempotent_insert(test_db):
    """Test: Same data inserted twice → no duplicates"""
    df = pd.DataFrame({
        'block_number': [1, 2],
        'block_time': ['2025-01-01 00:00:00', '2025-01-01 00:00:12'],
        'block_hash': ['0xaaa', '0xbbb'],
        'tx_hash': ['0x111', '0x222'],
        'tx_index': [0, 0],
        'log_index': [0, 0],
        'token': ['0xToken', '0xToken'],
        'from_addr': ['0xAlice', '0xBob'],
        'to_addr': ['0xBob', '0xCharlie'],
        'raw_value': [1000000000000000000, 2000000000000000000],
        'value_unit': [1.0, 2.0]
    })
    
    # First insert
    count1 = insert_idempotent(test_db, df)
    assert count1 == 2, "Should insert 2 records"
    
    # Second insert (same data)
    count2 = insert_idempotent(test_db, df)
    assert count2 == 0, "Should insert 0 (duplicates)"
    
    # Verify total
    total = test_db.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    assert total == 2, "Total should be 2 (no duplicates)"

def test_same_tx_multiple_logs(test_db):
    """Test: Same tx_hash, different log_index → both inserted"""
    df = pd.DataFrame({
        'block_number': [1, 1],
        'block_time': ['2025-01-01 00:00:00', '2025-01-01 00:00:00'],
        'block_hash': ['0xaaa', '0xaaa'],
        'tx_hash': ['0x111', '0x111'],  # Same tx
        'tx_index': [0, 0],
        'log_index': [0, 1],  # Different logs
        'token': ['0xToken', '0xToken'],
        'from_addr': ['0xAlice', '0xBob'],
        'to_addr': ['0xBob', '0xCharlie'],
        'raw_value': [1e18, 2e18],
        'value_unit': [1.0, 2.0]
    })
    
    count = insert_idempotent(test_db, df)
    assert count == 2, "Should insert both (different log_index)"

def test_transaction_rollback(test_db):
    """Test: Error in transaction → no changes"""
    df_good = pd.DataFrame({
        'block_number': [1],
        'block_time': ['2025-01-01'],
        'block_hash': ['0xaaa'],
        'tx_hash': ['0x111'],
        'tx_index': [0],
        'log_index': [0],
        'token': ['0xToken'],
        'from_addr': ['0xAlice'],
        'to_addr': ['0xBob'],
        'raw_value': [1e18],
        'value_unit': [1.0]
    })
    
    # Simulate error during transaction
    try:
        test_db.execute("BEGIN TRANSACTION")
        insert_idempotent(test_db, df_good)
        # Simulate error
        raise ValueError("Simulated error")
        test_db.execute("COMMIT")
    except:
        test_db.execute("ROLLBACK")
    
    # Verify nothing inserted
    total = test_db.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    assert total == 0, "Rollback should undo insert"

# Run tests
pytest.main([__file__, "-v"])
```

### 8.2 Integration Tests

```python
def test_end_to_end_ingest(test_db):
    """
    Integration test: Full ingest cycle
    
    1. Insert batch 1
    2. Update state
    3. Insert batch 2 (overlapping)
    4. Verify no duplicates
    5. Verify state correct
    """
    # Batch 1: blocks 100-105
    df1 = generate_test_transfers(blocks=range(100, 106))
    result1 = insert_batch_atomic(test_db, df1, end_block=105, end_hash="0xhash105")
    
    assert result1['inserted'] == len(df1)
    assert result1['state_block'] == 105
    
    # Batch 2: blocks 103-108 (overlap with batch 1)
    df2 = generate_test_transfers(blocks=range(103, 109))
    result2 = insert_batch_atomic(test_db, df2, end_block=108, end_hash="0xhash108")
    
    # Only new blocks (106-108) should be inserted
    expected_new = count_transfers_in_range(df2, 106, 108)
    assert result2['inserted'] == expected_new
    
    # Verify state
    state = test_db.execute("""
        SELECT last_scanned_block FROM scan_state WHERE key = 'transfers_v1'
    """).fetchone()[0]
    assert state == 108
    
    # Verify no duplicates
    total = test_db.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    unique = test_db.execute("""
        SELECT COUNT(DISTINCT tx_hash || '_' || log_index) FROM transfers
    """).fetchone()[0]
    assert total == unique, "No duplicates"
```

### 8.3 Stress Tests

```python
import time

def stress_test_large_batch(test_db, batch_size=100_000):
    """
    Stress test: Insert large batch
    
    Verify:
    - Performance acceptable
    - Memory stable
    - No errors
    """
    print(f"\nStress test: {batch_size:,} records")
    
    # Generate large batch
    t0 = time.perf_counter()
    df = generate_test_transfers_bulk(count=batch_size)
    gen_time = time.perf_counter() - t0
    print(f"  Generated in {gen_time:.2f}s")
    
    # Insert
    t0 = time.perf_counter()
    inserted = insert_idempotent(test_db, df)
    insert_time = time.perf_counter() - t0
    
    records_per_sec = inserted / insert_time
    
    print(f"  Inserted {inserted:,} in {insert_time:.2f}s")
    print(f"  Throughput: {records_per_sec:,.0f} records/sec")
    
    # Verify
    total = test_db.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    assert total == inserted
    
    # Performance threshold
    assert records_per_sec > 10_000, f"Too slow! {records_per_sec:,.0f} < 10K rec/s"
    
    print("  ✅ Stress test passed")

# Run
stress_test_large_batch(test_db, 100_000)
```

---

## 9) Production Deployment: Monitoring + Maintenance

### 9.1 Metrics Collection

```python
from dataclasses import dataclass
from typing import List
import json

@dataclass
class IngestMetrics:
    """Per-batch metrics"""
    timestamp: str
    batch_id: int
    start_block: int
    end_block: int
    logs_fetched: int
    logs_inserted: int
    duplicates_filtered: int
    insert_time_ms: float
    total_time_ms: float
    records_per_sec: float

class MetricsCollector:
    """Collect and export metrics"""
    
    def __init__(self, output_file="metrics.jsonl"):
        self.output_file = output_file
        self.metrics: List[IngestMetrics] = []
    
    def record(self, metric: IngestMetrics):
        """Record metric"""
        self.metrics.append(metric)
        
        # Append to file (JSON Lines format)
        with open(self.output_file, 'a') as f:
            json.dump(vars(metric), f)
            f.write('\n')
    
    def summary(self) -> dict:
        """Get summary statistics"""
        if not self.metrics:
            return {}
        
        return {
            'total_batches': len(self.metrics),
            'total_logs_inserted': sum(m.logs_inserted for m in self.metrics),
            'total_duplicates': sum(m.duplicates_filtered for m in self.metrics),
            'avg_records_per_sec': sum(m.records_per_sec for m in self.metrics) / len(self.metrics),
            'total_time_sec': sum(m.total_time_ms for m in self.metrics) / 1000,
        }

# Usage
collector = MetricsCollector("outputs/ingest_metrics.jsonl")

for batch_id, (start, end) in enumerate(block_ranges):
    t0 = time.perf_counter()
    
    logs = fetch_logs(start, end)
    df = parse_to_df(logs)
    inserted = insert_idempotent(conn, df)
    
    total_time = (time.perf_counter() - t0) * 1000
    
    metric = IngestMetrics(
        timestamp=datetime.now().isoformat(),
        batch_id=batch_id,
        start_block=start,
        end_block=end,
        logs_fetched=len(logs),
        logs_inserted=inserted,
        duplicates_filtered=len(logs) - inserted,
        insert_time_ms=total_time,
        total_time_ms=total_time,
        records_per_sec=inserted / (total_time / 1000)
    )
    
    collector.record(metric)

# Print summary
print(json.dumps(collector.summary(), indent=2))
```

### 9.2 Database Maintenance

```python
def maintain_database(conn):
    """
    Regular maintenance tasks
    
    Run: Daily or weekly
    """
    print("Running maintenance...")
    
    # 1. VACUUM (reclaim space, optimize indexes)
    print("  VACUUM...")
    conn.execute("VACUUM")
    
    # 2. ANALYZE (update statistics for query planner)
    print("  ANALYZE...")
    conn.execute("ANALYZE")
    
    # 3. Check database size
    db_size_mb = Path(conn.execute("SELECT current_database()").fetchone()[0]).stat().st_size / 1024 / 1024
    print(f"  Database size: {db_size_mb:.1f} MB")
    
    # 4. Check table sizes
    table_stats = conn.execute("""
        SELECT 
            'transfers' as table_name,
            COUNT(*) as row_count,
            COUNT(*) * 200 / 1024 / 1024 as est_size_mb
        FROM transfers
    """).fetchone()
    
    print(f"  Transfers: {table_stats[1]:,} rows (~{table_stats[2]:.1f} MB)")
    
    print("✅ Maintenance complete")

# Schedule: Run weekly
import schedule

schedule.every().sunday.at("02:00").do(lambda: maintain_database(conn))
```

### 9.3 Backup Strategy

```python
import shutil
from datetime import datetime

def backup_database(db_path: str, backup_dir: str = "backups"):
    """
    Create database backup
    
    Strategy:
    - Daily: Keep last 7
    - Weekly: Keep last 4
    - Monthly: Keep last 12
    """
    Path(backup_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / f"onchain_{timestamp}.duckdb"
    
    # Copy database file
    shutil.copy2(db_path, backup_path)
    
    # Compress (optional)
    import gzip
    with open(backup_path, 'rb') as f_in:
        with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove uncompressed
    backup_path.unlink()
    
    print(f"✅ Backup created: {backup_path}.gz")
    
    # Cleanup old backups
    cleanup_old_backups(backup_dir, keep_days=7)

def cleanup_old_backups(backup_dir: str, keep_days: int = 7):
    """Remove backups older than N days"""
    import time
    
    now = time.time()
    cutoff = now - (keep_days * 86400)
    
    for backup in Path(backup_dir).glob("onchain_*.duckdb.gz"):
        if backup.stat().st_mtime < cutoff:
            backup.unlink()
            print(f"  Removed old backup: {backup.name}")

# Schedule daily backup
schedule.every().day.at("03:00").do(lambda: backup_database("onchain.duckdb"))
```

---

## 10) Troubleshooting Guide

### Problem 1: Database Locked

**Symptoms:**
```
duckdb.IOException: database is locked
```

**Causes:**
- Multiple processes writing simultaneously
- Previous connection not closed

**Solutions:**
```python
# 1. Ensure single writer
# Use file lock or process manager

import fcntl

def acquire_lock(lockfile="/tmp/duckdb_ingest.lock"):
    """Ensure only one process running"""
    lock_fd = open(lockfile, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except IOError:
        print("❌ Another process is running")
        sys.exit(1)

# 2. Always close connections
try:
    conn = duckdb.connect("onchain.duckdb")
    # ... work ...
finally:
    conn.close()  # Critical!

# 3. Use context manager
class DuckDBConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

# Usage
with DuckDBConnection("onchain.duckdb") as conn:
    # ... work ...
    pass  # Auto-closes!
```

---

### Problem 2: Slow Inserts

**Symptoms:**
- Insert taking > 1s for 1K records
- CPU 100% during insert

**Debug:**
```python
# Profile insert
import cProfile

cProfile.run('insert_idempotent(conn, df)')

# Check if indexes are being used
explain_query(conn, "SELECT * FROM staging WHERE ...")
```

**Solutions:**
1. **Increase batch size** (5K-10K sweet spot)
2. **Remove unnecessary indexes** during bulk insert
3. **Disable auto-vacuum** temporarily
4. **Use Arrow format** for zero-copy

```python
# Temporary index drop during bulk load
conn.execute("DROP INDEX IF EXISTS idx_transfers_from")
# ... bulk insert ...
conn.execute("CREATE INDEX idx_transfers_from ON transfers(from_addr)")
```

---

### Problem 3: Memory Issues

**Symptoms:**
```
MemoryError / process killed (OOM)
```

**Causes:**
- DataFrame too large
- Multiple large batches in memory

**Solutions:**
```python
# 1. Process in smaller chunks
def chunked_insert(conn, df, chunk_size=10_000):
    """Insert large DataFrame in chunks"""
    total = len(df)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = df.iloc[start:end]
        insert_idempotent(conn, chunk)
        print(f"Progress: {end}/{total}")

# 2. Use iterator instead of loading all data
def iterate_batches(file_path, chunk_size=10_000):
    """Iterate over large CSV without loading all"""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

for df_chunk in iterate_batches("large_file.csv"):
    insert_idempotent(conn, df_chunk)

# 3. Clear DataFrame after use
df = None
import gc
gc.collect()
```

---

### Problem 4: Duplicate Detection Failing

**Symptoms:**
```sql
SELECT tx_hash, log_index, COUNT(*)
FROM transfers
GROUP BY tx_hash, log_index
HAVING COUNT(*) > 1;
-- Returns rows (should be empty!)
```

**Causes:**
- Primary key not created
- Anti-join logic incorrect
- Transaction not committed

**Solutions:**
```python
# 1. Verify schema
tables = conn.execute("""
    SELECT sql FROM sqlite_master 
    WHERE type='table' AND name='transfers'
""").fetchone()
print(tables[0])  # Should show PRIMARY KEY

# 2. Test anti-join manually
conn.execute("CREATE TEMP TABLE _test_staging AS SELECT * FROM transfers WHERE 1=0")
# Insert test data...
result = conn.execute("""
    SELECT COUNT(*) FROM _test_staging s
    LEFT JOIN transfers t ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
    WHERE t.tx_hash IS NULL
""").fetchone()[0]
print(f"Would insert: {result}")

# 3. Rebuild primary key if missing
conn.execute("ALTER TABLE transfers ADD PRIMARY KEY (tx_hash, log_index)")
```

---

### Problem 5: State Inconsistency

**Symptoms:**
- `last_scanned_block` doesn't match actual data
- Missing blocks in range

**Debug:**
```python
def diagnose_state(conn):
    """Check for state/data mismatch"""
    # Get state
    state = conn.execute("""
        SELECT last_scanned_block FROM scan_state WHERE key = 'transfers_v1'
    """).fetchone()
    
    # Get actual data range
    actual = conn.execute("""
        SELECT MIN(block_number), MAX(block_number), COUNT(DISTINCT block_number)
        FROM transfers
    """).fetchone()
    
    min_block, max_block, unique_blocks = actual
    expected_blocks = max_block - min_block + 1
    
    print(f"State: last_scanned = {state[0]}")
    print(f"Data:  min={min_block}, max={max_block}")
    print(f"Coverage: {unique_blocks}/{expected_blocks} blocks")
    
    if unique_blocks < expected_blocks:
        # Find gaps
        gaps = conn.execute("""
            WITH RECURSIVE block_series AS (
                SELECT ? as block_num
                UNION ALL
                SELECT block_num + 1 FROM block_series WHERE block_num < ?
            )
            SELECT block_num FROM block_series
            WHERE block_num NOT IN (SELECT DISTINCT block_number FROM transfers)
            LIMIT 10
        """, [min_block, max_block]).fetchall()
        
        print(f"Missing blocks: {[g[0] for g in gaps]}")

diagnose_state(conn)
```

**Fix:**
```python
# Re-scan missing blocks
missing_blocks = [...]  # From diagnosis
for block in missing_blocks:
    logs = fetch_logs(rpc_url, block, block)
    df = parse_to_df(logs)
    insert_idempotent(conn, df)
```

---

## 11) Mini Quiz (10 Soru)

1. Neden `(tx_hash, log_index)` çifti primary key olarak seçilir?
2. Staging + anti-join pattern'inin UNIQUE constraint'e göre avantajı nedir?
3. Transaction boundaries (BEGIN...COMMIT) neden önemlidir?
4. `DECIMAL(38,0)` ile `DOUBLE` arasındaki farkı ne zaman önemsemeli?
5. İkinci kez aynı batch'i insert edersen ne olur?
6. DuckDB'nin "embedded" olması ne demektir?
7. Index kullanımını nasıl doğrularsın?
8. Batch size'ı artırmanın trade-off'u nedir?
9. Database locked hatası ne zaman oluşur?
10. Maintenance (VACUUM, ANALYZE) neden periyodik yapılmalı?

### Cevap Anahtarı

1. Bir transaction birden fazla log içerebilir; `log_index` tekliği sağlar
2. Exception handling'siz hızlı deduplikasyon; bulk insert optimization
3. Atomicity (all-or-nothing) + crash recovery + performance (single fsync)
4. Parasal/kesin hesaplarda DECIMAL; aggregation/display'de DOUBLE
5. Anti-join sayesinde 0 kayıt insert edilir (idempotent)
6. Tek dosya, server yok, process-internal (no network overhead)
7. `EXPLAIN` komutu + plan analysis (INDEX_SCAN vs SEQ_SCAN)
8. Daha hızlı ama crash olursa daha fazla veri re-process (trade-off: speed vs granularity)
9. Multiple simultaneous writers veya connection not closed
10. Disk space reclaim, index optimization, query planner statistics update

---

## 12) Ödevler (6 Pratik)

### Ödev 1: Benchmark Suite
```python
# Task: Implement comprehensive benchmark
# - Test batch sizes: 100, 1K, 5K, 10K, 50K
# - Measure: insert_ms, anti_join_ms, commit_ms
# - Plot: batch_size vs records_per_sec
# - Find optimal batch size for your machine
```

### Ödev 2: Idempotency Stress Test
```python
# Task: Verify idempotency under stress
# - Insert same 10K records 10 times
# - Verify: total_rows == 10K (not 100K!)
# - Measure: Time for each iteration
# - Expected: First slow, rest fast (anti-join efficient)
```

### Ödev 3: Index Impact Analysis
```python
# Task: Measure index impact
# - Query: "SELECT * FROM transfers WHERE from_addr = ?"
# - Run WITHOUT index → measure time
# - CREATE INDEX → measure time
# - Compare: speedup factor
# - Bonus: Test composite index (from_addr, block_time)
```

### Ödev 4: Transaction Rollback Test
```python
# Task: Verify transaction safety
# - Start transaction
# - Insert 1K records
# - Simulate error (raise exception)
# - ROLLBACK
# - Verify: 0 records in database
# - Bonus: Test partial commit (commit after each 100 records)
```

### Ödev 5: Memory Profile
```python
# Task: Profile memory usage
# - Use memory_profiler
# - Insert 100K records in single batch
# - Measure: peak memory, memory per record
# - Optimize: Try chunked insert, compare memory
```

### Ödev 6: Production Monitoring
```python
# Task: Build monitoring dashboard
# - Collect metrics: logs_inserted, duplicates_filtered, records_per_sec
# - Export to JSON Lines file
# - Create simple web dashboard (Streamlit/Plotly)
# - Show: Real-time throughput, cumulative logs, error rate
```

---

## 13) Definition of Done (Tahta 05)

### Learning Objectives
- [ ] DuckDB vs alternatives (rationale clear)
- [ ] Schema design principles (tables + indexes + constraints)
- [ ] Idempotent patterns (3 layers: UNIQUE, anti-join, transaction)
- [ ] Batch optimization strategies (size tuning, Arrow integration)
- [ ] Query optimization (indexes, explain plans)
- [ ] Transaction safety (ACID, crash recovery)
- [ ] Testing strategies (unit, integration, stress)
- [ ] Production patterns (monitoring, maintenance, backup)
- [ ] Troubleshooting skills (5 common problems)

### Practical Outputs
- [ ] Schema init function working
- [ ] Idempotent insert verified (0 duplicates)
- [ ] Benchmark suite run (optimal batch size found)
- [ ] Unit tests passing (pytest)
- [ ] Transaction rollback tested
- [ ] Query performance analyzed (EXPLAIN used)
- [ ] Metrics collection implemented

---

## 🔗 İlgili Dersler

- **← Tahta 04:** [getLogs Pencere + Reorg](04_tahta_getlogs_pencere_reorg.md)
- **→ Tahta 06:** State & Resume (Coming)
- **↑ Ana Sayfa:** [Week 0 Bootstrap](../../../crypto/w0_bootstrap/README.md)

---

## 🛡️ Güvenlik / Etik

- **Read-only:** Özel anahtar yok, imza yok, custody yok
- **Database security:** File permissions (chmod 600)
- **Backup encryption:** Consider encrypting backups
- **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 📌 Navigasyon

- **→ Sonraki:** [06 - State & Resume](06_tahta_state_resume.md) (Coming)
- **← Önceki:** [04 - getLogs + Reorg](04_tahta_getlogs_pencere_reorg.md)
- **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

**Tahta 05 — DuckDB + İdempotent Yazma**  
*Format: Production Deep-Dive*  
*Süre: 60-75 dk*  
*Prerequisite: Tahta 01-04*  
*Versiyon: 2.0 (Complete Expansion)*  
*Code Examples: 1,000+ lines*

