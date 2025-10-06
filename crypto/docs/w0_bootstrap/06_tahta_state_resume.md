# 🧑‍🏫 Tahta 06 — State & Resume: Production Checkpoint Patterns

> **Amaç:** Log tarama pipeline'ını **kaldığı yerden güvenle devam ettirmek**: Atomic checkpoint, tail re-scan, crash recovery, reorg detection. Production-grade state management.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (Genişletilmiş Tahta)

1. **State management rationale** (Why checkpoint?)
2. **Checkpoint strategies** (Atomic, durable, observable)
3. **Tail re-scan patterns** (Reorg protection + overlap)
4. **Crash recovery scenarios** (Before/after commit)
5. **Schema design** (scan_state + blocks + metadata)
6. **Production implementation** (3 core classes + integration)
7. **Transaction safety** (ACID + isolation)
8. **Resume algorithms** (Cold start, warm restart, gap detection)
9. **Performance optimization** (Checkpoint frequency, state queries)
10. **Monitoring & alerting** (Lag tracking, health checks)
11. **Testing strategies** (Unit + integration + chaos)
12. **Troubleshooting guide** (10 common scenarios)
13. **Quiz + ödevler**

---

## 1) State Management Rationale: Why Checkpoint?

### 1.1 The Exactly-Once Illusion

```
╔════════════════════════════════════════════════════════════╗
║          EXACTLY-ONCE IN DISTRIBUTED SYSTEMS               ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Theoretical Guarantee:                                   ║
║    Each event processed exactly once                      ║
║                                                            ║
║  Reality:                                                 ║
║    Impossible to guarantee at system boundaries           ║
║    (network failures, crashes, Byzantine faults)          ║
║                                                            ║
║  Practical Solution:                                      ║
║    At-least-once delivery                                 ║
║      + Idempotent processing                              ║
║      + Durable checkpoints                                ║
║    ────────────────────────────                           ║
║    = Exactly-once semantics (end-to-end effect)           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**Our Strategy:**

```
RPC (at-least-once)
  → Parse (stateless)
  → DB Write (idempotent via PRIMARY KEY)
  → Checkpoint (atomic with write)
  ────────────────────────────────────
  = Exactly-once effect
```

### 1.2 Why Not Restart from Genesis?

**Problem: Full Scan Cost**

```python
# Cost analysis
BLOCKS_PER_YEAR = 365 * 24 * 3600 // 12  # ~2.6M blocks
LOGS_PER_BLOCK_AVG = 50  # Conservative estimate
RPC_COST_PER_REQUEST = 0.001  # $0.001 per request (Alchemy CU)

# Full rescan
total_requests = BLOCKS_PER_YEAR // 1500  # 1500 blocks per request
total_cost = total_requests * RPC_COST_PER_REQUEST
total_time_hours = total_requests * 0.5 / 3600  # 0.5s per request

print(f"Full rescan cost: ${total_cost:.2f}")
print(f"Full rescan time: {total_time_hours:.1f} hours")

# Output:
# Full rescan cost: $1.73
# Full rescan time: 0.24 hours (14 minutes)
```

**With checkpointing:**
- Daily incremental: ~7,200 blocks (12s × 86,400s)
- Cost: ~$0.005/day
- Time: ~2.4 minutes

**Savings:** 99.7% cost reduction, 84% time reduction

### 1.3 State Management Goals

**ACID Properties for Checkpoint:**

1. **Atomicity:** Data + state update in single transaction
2. **Consistency:** State always reflects successfully processed data
3. **Isolation:** Concurrent readers see consistent state
4. **Durability:** Checkpoint survives crashes

---

## 2) Checkpoint Strategies: Atomic, Durable, Observable

### 2.1 Checkpoint Location Options

```
╔════════════════════════════════════════════════════════════╗
║            CHECKPOINT STORAGE STRATEGIES                   ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Option 1: In-Database (DuckDB table) ⭐ RECOMMENDED      ║
║    ✅ Atomic with data writes (single TX)                 ║
║    ✅ Durable (persisted to disk)                         ║
║    ✅ Queryable (SQL)                                     ║
║    ❌ Coupled to database                                 ║
║                                                            ║
║  Option 2: Separate File (.state.json)                   ║
║    ✅ Simple                                              ║
║    ✅ Human-readable                                      ║
║    ❌ NOT atomic with data                                ║
║    ❌ Race conditions                                     ║
║    Use case: Prototyping only                            ║
║                                                            ║
║  Option 3: Redis/External Store                          ║
║    ✅ Distributed coordination                            ║
║    ✅ Fast reads                                          ║
║    ❌ Network dependency                                  ║
║    ❌ NOT atomic with data                                ║
║    Use case: Distributed pipelines                       ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**Our Choice:** Option 1 (In-Database)

### 2.2 scan_state Table Design

```sql
CREATE TABLE IF NOT EXISTS scan_state (
    -- Primary key
    key TEXT PRIMARY KEY,              -- e.g., 'transfers_v1'
    
    -- Checkpoint data
    last_scanned_block BIGINT NOT NULL,
    last_scanned_hash TEXT NOT NULL,   -- For reorg detection
    last_scanned_time TIMESTAMP NOT NULL,  -- Block timestamp (not system time!)
    
    -- Audit trail
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata (JSON for flexibility)
    metadata JSON,
    
    -- Constraints
    CHECK (last_scanned_block >= 0),
    CHECK (last_scanned_hash ~ '^0x[a-fA-F0-9]{64}$')  -- Valid hash format
);
```

**Metadata Schema (JSON):**

```json
{
  "confirmations": 12,
  "tail_resync": 12,
  "chain_id": 11155111,
  "start_block": 5000000,
  "total_logs_ingested": 125430,
  "last_error": null,
  "last_error_time": null,
  "version": "1.0.0",
  "host": "ingest-worker-01"
}
```

### 2.3 Checkpoint Update Pattern

**❌ Wrong: State first, data second**

```python
# DANGEROUS!
conn.execute("UPDATE scan_state SET last_scanned_block = ?", [end_block])
conn.commit()

# Then insert data (separate transaction)
conn.execute("INSERT INTO transfers VALUES (...)")
conn.commit()

# Problem: Crash between commits → state advanced but data missing!
```

**✅ Right: Atomic update**

```python
# SAFE: Single transaction
conn.execute("BEGIN TRANSACTION")

try:
    # 1. Insert data (idempotent)
    insert_transfers(conn, df)
    
    # 2. Update state (same TX!)
    conn.execute("""
        INSERT OR REPLACE INTO scan_state 
        (key, last_scanned_block, last_scanned_hash, last_scanned_time, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, ['transfers_v1', end_block, end_hash, end_time])
    
    # 3. Commit (atomic!)
    conn.execute("COMMIT")
    
except Exception as e:
    # Rollback on any error
    conn.execute("ROLLBACK")
    raise
```

### 2.4 Observability: State Health Metrics

```python
def get_checkpoint_health(conn) -> dict:
    """
    Get checkpoint health metrics
    
    Returns:
        {
            'exists': bool,
            'last_block': int,
            'last_hash': str,
            'age_seconds': float,  # Time since last update
            'lag_blocks': int,     # Blocks behind chain tip
            'is_stale': bool       # age > threshold
        }
    """
    # Get state
    state = conn.execute("""
        SELECT 
            last_scanned_block,
            last_scanned_hash,
            updated_at
        FROM scan_state
        WHERE key = 'transfers_v1'
    """).fetchone()
    
    if not state:
        return {'exists': False}
    
    last_block, last_hash, updated_at = state
    
    # Get chain tip
    latest_block = get_latest_block(rpc_url)
    
    # Calculate metrics
    import time
    age_seconds = (time.time() - updated_at.timestamp())
    lag_blocks = latest_block - last_block
    is_stale = age_seconds > 600  # 10 minutes threshold
    
    return {
        'exists': True,
        'last_block': last_block,
        'last_hash': last_hash,
        'age_seconds': age_seconds,
        'lag_blocks': lag_blocks,
        'is_stale': is_stale,
        'health': 'STALE' if is_stale else 'HEALTHY'
    }

# Usage
health = get_checkpoint_health(conn)
if health['is_stale']:
    print(f"⚠️  Pipeline stale! Last update {health['age_seconds']:.0f}s ago")
else:
    print(f"✅ Pipeline healthy, lag: {health['lag_blocks']} blocks")
```

---

## 3) Tail Re-scan Patterns: Reorg Protection

### 3.1 Why Tail Re-scan?

**Blockchain Finality Reality:**

```
Block Timeline:
──────────────────────────────────────────────────────────
                          ↓ Reorg boundary
[Old confirmed] ──────── [Recent unconfirmed] ── [Latest]
                          ← CONFIRMATIONS →

Safe to process:  [Old confirmed]
Buffer zone:      [Recent unconfirmed]  ← Re-scan each run
Danger zone:      [Latest]              ← Don't process yet
```

**Reorg Scenario:**

```
Run 1 (block 1000):
  Process: 988-1000 (latest=1012, confirmations=12, safe=1000)
  State: last_scanned=1000

--- REORG HAPPENS (blocks 995-1000 change) ---

Run 2 (block 1015):
  Without tail re-scan:
    Start: 1001
    Miss: Changed logs in 995-1000 ❌
  
  With tail re-scan (tail=12):
    Start: 1000 - 12 = 988
    Re-process: 988-1000 (catches reorg) ✅
    Continue: 1001-1003 (safe=1015-12=1003)
```

### 3.2 Tail Re-scan Algorithm

```python
def calculate_scan_range(
    last_scanned: Optional[int],
    latest_block: int,
    confirmations: int = 12,
    tail_resync: Optional[int] = None
) -> tuple[int, int]:
    """
    Calculate safe scan range with tail re-scan
    
    Args:
        last_scanned: Last successfully scanned block (from state)
        latest_block: Current chain tip
        confirmations: Confirmation buffer
        tail_resync: Number of blocks to re-scan (default: confirmations)
    
    Returns:
        (start_block, end_block) for next scan
    """
    if tail_resync is None:
        tail_resync = confirmations
    
    # Safe latest (reorg protection)
    safe_latest = latest_block - confirmations
    
    # Calculate start with tail re-scan
    if last_scanned is None:
        # Cold start
        start_block = 0
    else:
        # Warm restart: Go back tail_resync blocks
        start_block = max(0, last_scanned - tail_resync + 1)
    
    # End at safe latest
    end_block = safe_latest
    
    return (start_block, end_block)

# Examples
print(calculate_scan_range(None, 1000, 12))
# (0, 988)  # Cold start

print(calculate_scan_range(1000, 1020, 12))
# (989, 1008)  # Warm restart: re-scan 989-1000, continue 1001-1008
```

### 3.3 Overlap Visualization

```
╔════════════════════════════════════════════════════════════╗
║              TAIL RE-SCAN OVERLAP PATTERN                  ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Run 1:                                                   ║
║    ├─────────────────────┤                                ║
║    0                     1000 (last_scanned)              ║
║                                                            ║
║  Run 2: (tail_resync=12)                                  ║
║                    ├─────────────────────┤                ║
║                   989                   1020              ║
║                    ↑─ overlap ─↑                          ║
║                   (989-1000)                              ║
║                                                            ║
║  Idempotent Insert:                                       ║
║    989-1000: Already in DB → Skip (anti-join)            ║
║    1001-1020: New data → Insert                           ║
║                                                            ║
║  Result:                                                  ║
║    ├─────────────────────────────────────┤               ║
║    0                                     1020             ║
║    (No duplicates, reorg-safe)                           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 3.4 Dynamic Tail Sizing

```python
class AdaptiveTailResync:
    """
    Dynamically adjust tail_resync based on reorg history
    
    Strategy:
    - Increase if reorgs detected
    - Decrease if stable (to reduce redundant work)
    """
    
    def __init__(self, base_confirmations: int = 12):
        self.base = base_confirmations
        self.current = base_confirmations
        self.min_tail = max(5, base_confirmations // 2)
        self.max_tail = base_confirmations * 2
        self.reorg_count = 0
        self.stable_runs = 0
    
    def on_reorg_detected(self):
        """Increase tail when reorg detected"""
        self.reorg_count += 1
        self.stable_runs = 0
        
        # Increase tail
        self.current = min(self.current + 5, self.max_tail)
        
        print(f"⚠️  Reorg detected! Increasing tail to {self.current}")
    
    def on_stable_run(self):
        """Decrease tail when stable"""
        self.stable_runs += 1
        
        # After 10 stable runs, decrease tail
        if self.stable_runs >= 10:
            self.current = max(self.current - 1, self.min_tail)
            self.stable_runs = 0
            
            print(f"✅ Stable, decreasing tail to {self.current}")
    
    def get_tail(self) -> int:
        return self.current

# Usage
tail_manager = AdaptiveTailResync(base_confirmations=12)

for run in range(100):
    tail = tail_manager.get_tail()
    
    # Scan with current tail
    start, end = calculate_scan_range(last_scanned, latest, CONFIRMATIONS, tail)
    
    # Check for reorg
    if reorg_detected(conn, start, end):
        tail_manager.on_reorg_detected()
    else:
        tail_manager.on_stable_run()
```

---

## 4) Crash Recovery Scenarios: Before/After Commit

### 4.1 Transaction Timeline

```
╔════════════════════════════════════════════════════════════╗
║            TRANSACTION LIFECYCLE & CRASH POINTS            ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  T0: BEGIN TRANSACTION                                    ║
║    ↓                                                       ║
║  T1: INSERT INTO staging (batch data)                     ║
║    ↓                                                       ║
║  T2: INSERT INTO transfers (anti-join)                    ║
║    ↓                                                       ║
║  T3: UPDATE scan_state (checkpoint)                       ║
║    ↓                                                       ║
║  T4: COMMIT  ← Atomic point!                              ║
║    ↓                                                       ║
║  T5: Transaction complete                                 ║
║                                                            ║
║  Crash Scenarios:                                         ║
║  ─────────────────────────────────────────                ║
║                                                            ║
║  Crash at T1-T3 (before COMMIT):                          ║
║    → ROLLBACK automatic                                   ║
║    → No data written                                      ║
║    → No state update                                      ║
║    → Safe to retry from last checkpoint                   ║
║                                                            ║
║  Crash at T4 (during COMMIT):                             ║
║    → Database handles atomically                          ║
║    → Either fully committed OR fully rolled back          ║
║    → On restart: check if committed                       ║
║                                                            ║
║  Crash after T4 (after COMMIT):                           ║
║    → Data + state both written                            ║
║    → Checkpoint advanced                                  ║
║    → On restart: resume from new checkpoint               ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 4.2 Crash Recovery Implementation

```python
import signal
import sys
from contextlib import contextmanager

class CrashSafeIngester:
    """
    Crash-safe ingest pipeline with signal handling
    
    Handles:
    - Graceful shutdown (SIGTERM, SIGINT)
    - Crash recovery on restart
    - Transaction rollback on error
    """
    
    def __init__(self, conn, state_store, skey='transfers_v1'):
        self.conn = conn
        self.state = state_store
        self.skey = skey
        self.shutdown_requested = False
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print(f"\n⚠️  Shutdown signal received ({signum})")
        print("   Waiting for current batch to complete...")
        self.shutdown_requested = True
    
    @contextmanager
    def atomic_batch(self, end_block: int, end_hash: str):
        """
        Context manager for atomic batch processing
        
        Usage:
            with ingester.atomic_batch(end_block, end_hash) as batch:
                batch.insert_data(df)
                # ... other operations
                # Commit happens automatically on context exit
        """
        self.conn.execute("BEGIN TRANSACTION")
        
        class BatchContext:
            def __init__(self, conn, state, skey, end_block, end_hash):
                self.conn = conn
                self.state = state
                self.skey = skey
                self.end_block = end_block
                self.end_hash = end_hash
            
            def insert_data(self, df):
                """Insert data (idempotent)"""
                insert_idempotent(self.conn, df)
            
            def update_metadata(self, meta: dict):
                """Update state metadata"""
                self.metadata = meta
        
        batch_ctx = BatchContext(self.conn, self.state, self.skey, 
                                 end_block, end_hash)
        
        try:
            yield batch_ctx
            
            # Update state (same transaction!)
            self.state.set(self.skey, end_block, 
                          getattr(batch_ctx, 'metadata', None))
            
            # Commit
            self.conn.execute("COMMIT")
            
        except Exception as e:
            # Rollback on error
            self.conn.execute("ROLLBACK")
            print(f"❌ Batch failed, rolled back: {e}")
            raise
    
    def run_ingest_loop(self, rpc_url: str):
        """
        Main ingest loop with crash recovery
        
        On startup:
        1. Check last checkpoint
        2. Calculate resume range (with tail re-scan)
        3. Process batches atomically
        4. Handle shutdown gracefully
        """
        # Recover from last checkpoint
        state = self.state.get(self.skey)
        last_scanned = state.last_scanned_block if state else None
        
        if last_scanned is None:
            print("🆕 Cold start (no previous state)")
        else:
            print(f"🔄 Resuming from block {last_scanned:,}")
        
        # Ingest loop
        while not self.shutdown_requested:
            try:
                # Calculate range
                latest = get_latest_block(rpc_url)
                start, end = calculate_scan_range(
                    last_scanned, latest, 
                    confirmations=12, tail_resync=12
                )
                
                if start > end:
                    print("✅ Up to date, sleeping...")
                    time.sleep(60)
                    continue
                
                # Process batch
                print(f"📥 Processing {start:,} → {end:,}")
                
                logs = fetch_logs(rpc_url, start, end)
                df = parse_logs_to_df(logs)
                end_hash = get_block_hash(rpc_url, end)
                
                # Atomic batch
                with self.atomic_batch(end, end_hash) as batch:
                    batch.insert_data(df)
                    batch.update_metadata({
                        'logs_processed': len(logs),
                        'batch_start': start,
                        'batch_end': end
                    })
                
                print(f"   ✅ {len(logs)} logs processed")
                
                # Update loop variable
                last_scanned = end
                
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            
            except Exception as e:
                print(f"❌ Error: {e}")
                print("   Waiting 30s before retry...")
                time.sleep(30)
                # State unchanged, safe to retry
        
        print("👋 Shutdown complete")

# Usage
ingester = CrashSafeIngester(conn, state_store)
ingester.run_ingest_loop(RPC_URL)
```

### 4.3 Crash Recovery Testing

```python
import pytest
import os
import signal

def test_crash_before_commit(tmp_db):
    """Test: Crash before COMMIT → no data/state change"""
    conn, state = init_test_db(tmp_db)
    
    df = generate_test_transfers(count=100)
    
    # Simulate crash before commit
    try:
        conn.execute("BEGIN TRANSACTION")
        insert_idempotent(conn, df)
        state.set('transfers_v1', 1000)
        
        # Simulate crash (raise exception before COMMIT)
        raise RuntimeError("Simulated crash")
        
        conn.execute("COMMIT")  # Never reached
        
    except RuntimeError:
        conn.execute("ROLLBACK")
    
    # Verify: No data written
    count = conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    assert count == 0, "No data should be written"
    
    # Verify: No state update
    checkpoint = state.get('transfers_v1')
    assert checkpoint is None, "State should be unchanged"

def test_crash_after_commit(tmp_db):
    """Test: Crash after COMMIT → data/state persisted"""
    conn, state = init_test_db(tmp_db)
    
    df = generate_test_transfers(count=100)
    
    # Successful commit
    conn.execute("BEGIN TRANSACTION")
    insert_idempotent(conn, df)
    state.set('transfers_v1', 1000)
    conn.execute("COMMIT")
    
    # Simulate crash after commit
    # (In real scenario, process would terminate here)
    
    # Verify: Data written
    count = conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    assert count == 100, "Data should be persisted"
    
    # Verify: State updated
    checkpoint = state.get('transfers_v1')
    assert checkpoint.last_scanned_block == 1000

def test_graceful_shutdown(tmp_db):
    """Test: SIGTERM → graceful shutdown"""
    conn, state = init_test_db(tmp_db)
    ingester = CrashSafeIngester(conn, state)
    
    # Start ingest in background thread
    import threading
    thread = threading.Thread(target=ingester.run_ingest_loop, args=(RPC_URL,))
    thread.start()
    
    # Wait for first batch
    time.sleep(2)
    
    # Send SIGTERM
    os.kill(os.getpid(), signal.SIGTERM)
    
    # Wait for shutdown
    thread.join(timeout=10)
    
    # Verify: State is consistent
    checkpoint = state.get('transfers_v1')
    assert checkpoint is not None, "Checkpoint should exist"
    
    # Verify: No partial data
    verify_state_consistency(conn)
```

---

## 5) Schema Design: Complete Production Schema

### 5.1 scan_state Table (Enhanced)

```sql
CREATE TABLE IF NOT EXISTS scan_state (
    -- Primary key
    key TEXT PRIMARY KEY,
    
    -- Checkpoint data
    last_scanned_block BIGINT NOT NULL,
    last_scanned_hash TEXT NOT NULL,
    last_scanned_time TIMESTAMP NOT NULL,
    
    -- Progress tracking
    start_block BIGINT NOT NULL DEFAULT 0,
    total_blocks_processed BIGINT NOT NULL DEFAULT 0,
    total_logs_ingested BIGINT NOT NULL DEFAULT 0,
    
    -- Audit trail
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Error tracking
    last_error TEXT,
    last_error_time TIMESTAMP,
    error_count INTEGER DEFAULT 0,
    
    -- Metadata (JSON)
    metadata JSON,
    
    -- Constraints
    CHECK (last_scanned_block >= start_block),
    CHECK (total_blocks_processed >= 0),
    CHECK (total_logs_ingested >= 0),
    CHECK (last_scanned_hash ~ '^0x[a-fA-F0-9]{64}$')
);
```

### 5.2 blocks Table (Reorg Detection)

```sql
CREATE TABLE IF NOT EXISTS blocks (
    -- Block identity
    number BIGINT PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    parent_hash TEXT NOT NULL,
    
    -- Block metadata
    timestamp BIGINT NOT NULL,
    tx_count INTEGER,
    log_count INTEGER,
    
    -- Processing metadata
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reorg_detected BOOLEAN DEFAULT FALSE,
    reorg_detected_at TIMESTAMP,
    
    -- Constraints
    CHECK (number >= 0),
    CHECK (hash ~ '^0x[a-fA-F0-9]{64}$'),
    CHECK (parent_hash ~ '^0x[a-fA-F0-9]{64}$')
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(hash);
CREATE INDEX IF NOT EXISTS idx_blocks_parent ON blocks(parent_hash);
CREATE INDEX IF NOT EXISTS idx_blocks_processed ON blocks(processed_at);
```

### 5.3 ingest_metrics Table (Monitoring)

```sql
CREATE TABLE IF NOT EXISTS ingest_metrics (
    id BIGSERIAL PRIMARY KEY,
    
    -- Batch info
    batch_start_block BIGINT NOT NULL,
    batch_end_block BIGINT NOT NULL,
    
    -- Timing
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP NOT NULL,
    duration_ms BIGINT NOT NULL,
    
    -- Volume
    logs_fetched INTEGER NOT NULL,
    logs_inserted INTEGER NOT NULL,
    duplicates_filtered INTEGER NOT NULL,
    
    -- Performance
    rpc_latency_ms BIGINT,
    insert_latency_ms BIGINT,
    throughput_logs_per_sec DOUBLE,
    
    -- Status
    status TEXT NOT NULL,  -- 'success', 'error', 'timeout'
    error_message TEXT,
    
    -- Metadata
    metadata JSON
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_metrics_time ON ingest_metrics(completed_at);
CREATE INDEX IF NOT EXISTS idx_metrics_status ON ingest_metrics(status);
```

---

## 6) Production Implementation: Complete Classes

### 6.1 StateStore Class

```python
import duckdb
import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class ScanState:
    """State checkpoint"""
    key: str
    last_scanned_block: int
    last_scanned_hash: str
    last_scanned_time: datetime
    start_block: int
    total_blocks_processed: int
    total_logs_ingested: int
    created_at: datetime
    updated_at: datetime
    last_error: Optional[str]
    last_error_time: Optional[datetime]
    error_count: int
    metadata: dict

class StateStore:
    """
    Durable state management with DuckDB
    
    Features:
    - Atomic updates
    - Metadata tracking
    - Error logging
    - Progress metrics
    """
    
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Initialize schema if not exists"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_state (
                key TEXT PRIMARY KEY,
                last_scanned_block BIGINT NOT NULL,
                last_scanned_hash TEXT NOT NULL,
                last_scanned_time TIMESTAMP NOT NULL,
                start_block BIGINT NOT NULL DEFAULT 0,
                total_blocks_processed BIGINT NOT NULL DEFAULT 0,
                total_logs_ingested BIGINT NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_error TEXT,
                last_error_time TIMESTAMP,
                error_count INTEGER DEFAULT 0,
                metadata JSON
            )
        """)
    
    def get(self, key: str) -> Optional[ScanState]:
        """Get state by key"""
        row = self.conn.execute("""
            SELECT 
                key, last_scanned_block, last_scanned_hash, last_scanned_time,
                start_block, total_blocks_processed, total_logs_ingested,
                created_at, updated_at,
                last_error, last_error_time, error_count,
                metadata
            FROM scan_state
            WHERE key = ?
        """, [key]).fetchone()
        
        if not row:
            return None
        
        return ScanState(
            key=row[0],
            last_scanned_block=row[1],
            last_scanned_hash=row[2],
            last_scanned_time=row[3],
            start_block=row[4],
            total_blocks_processed=row[5],
            total_logs_ingested=row[6],
            created_at=row[7],
            updated_at=row[8],
            last_error=row[9],
            last_error_time=row[10],
            error_count=row[11],
            metadata=json.loads(row[12]) if row[12] else {}
        )
    
    def set(self, 
            key: str, 
            last_scanned_block: int,
            last_scanned_hash: str,
            last_scanned_time: datetime,
            logs_ingested: int = 0,
            metadata: Optional[dict] = None):
        """
        Update state (upsert)
        
        NOTE: Should be called within same transaction as data insert!
        """
        self.conn.execute("""
            INSERT INTO scan_state AS s (
                key, 
                last_scanned_block, 
                last_scanned_hash,
                last_scanned_time,
                total_logs_ingested,
                updated_at,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            ON CONFLICT (key) DO UPDATE SET
                last_scanned_block = excluded.last_scanned_block,
                last_scanned_hash = excluded.last_scanned_hash,
                last_scanned_time = excluded.last_scanned_time,
                total_blocks_processed = s.total_blocks_processed + 
                    (excluded.last_scanned_block - s.last_scanned_block),
                total_logs_ingested = s.total_logs_ingested + excluded.total_logs_ingested,
                updated_at = CURRENT_TIMESTAMP,
                metadata = excluded.metadata
        """, [
            key,
            last_scanned_block,
            last_scanned_hash,
            last_scanned_time,
            logs_ingested,
            json.dumps(metadata or {})
        ])
    
    def log_error(self, key: str, error: str):
        """Log error (separate transaction OK)"""
        self.conn.execute("""
            UPDATE scan_state
            SET 
                last_error = ?,
                last_error_time = CURRENT_TIMESTAMP,
                error_count = error_count + 1
            WHERE key = ?
        """, [error, key])
        self.conn.commit()
    
    def get_progress(self, key: str) -> dict:
        """Get progress statistics"""
        state = self.get(key)
        if not state:
            return {'exists': False}
        
        blocks_processed = state.total_blocks_processed
        logs_per_block = (state.total_logs_ingested / blocks_processed 
                         if blocks_processed > 0 else 0)
        
        return {
            'exists': True,
            'last_block': state.last_scanned_block,
            'start_block': state.start_block,
            'blocks_processed': blocks_processed,
            'logs_ingested': state.total_logs_ingested,
            'logs_per_block': logs_per_block,
            'error_count': state.error_count,
            'last_update': state.updated_at
        }
```

### 6.2 Checkpointer Class

```python
class Checkpointer:
    """
    Atomic checkpoint manager
    
    Usage:
        with Checkpointer(conn, state, 'transfers_v1') as cp:
            cp.insert_data(df)
            cp.commit(end_block, end_hash, end_time)
    """
    
    def __init__(self, 
                 conn: duckdb.DuckDBPyConnection,
                 state: StateStore,
                 key: str):
        self.conn = conn
        self.state = state
        self.key = key
        self._in_transaction = False
    
    def __enter__(self):
        """Start transaction"""
        self.conn.execute("BEGIN TRANSACTION")
        self._in_transaction = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Rollback on error, commit on success"""
        if exc_type is not None:
            # Error occurred
            if self._in_transaction:
                self.conn.execute("ROLLBACK")
            return False  # Propagate exception
        
        # Success handled by explicit commit() call
        if self._in_transaction:
            # If commit() not called, rollback
            self.conn.execute("ROLLBACK")
    
    def insert_data(self, df):
        """Insert data (idempotent)"""
        insert_idempotent(self.conn, df)
    
    def commit(self, 
               end_block: int,
               end_hash: str,
               end_time: datetime,
               logs_count: int,
               metadata: Optional[dict] = None):
        """
        Commit transaction with state update
        
        NOTE: Must be called before exiting context manager
        """
        # Update state (same transaction!)
        self.state.set(
            self.key,
            end_block,
            end_hash,
            end_time,
            logs_count,
            metadata
        )
        
        # Commit
        self.conn.execute("COMMIT")
        self._in_transaction = False

# Usage example
with Checkpointer(conn, state, 'transfers_v1') as cp:
    cp.insert_data(df)
    cp.commit(
        end_block=1000,
        end_hash="0xabc...",
        end_time=datetime.now(),
        logs_count=len(df),
        metadata={'batch_duration_ms': 1250}
    )
```

### 6.3 ResumePlanner Class

```python
class ResumePlanner:
    """
    Calculate safe resume ranges with tail re-scan
    
    Features:
    - Cold start detection
    - Tail re-scan calculation
    - Adaptive step sizing
    - Reorg protection
    """
    
    def __init__(self,
                 confirmations: int = 12,
                 tail_resync: Optional[int] = None,
                 initial_step: int = 1500,
                 min_step: int = 256,
                 max_step: int = 5000):
        self.confirmations = confirmations
        self.tail_resync = tail_resync or confirmations
        self.step = initial_step
        self.min_step = min_step
        self.max_step = max_step
    
    def get_next_range(self,
                       last_scanned: Optional[int],
                       latest_block: int,
                       start_block: int = 0) -> Optional[tuple[int, int]]:
        """
        Calculate next scan range
        
        Returns:
            (start, end) or None if up to date
        """
        # Calculate safe latest
        safe_latest = latest_block - self.confirmations
        
        # Calculate start with tail re-scan
        if last_scanned is None:
            # Cold start
            start = start_block
        else:
            # Warm restart
            start = max(start_block, last_scanned - self.tail_resync + 1)
        
        # Check if up to date
        if start > safe_latest:
            return None
        
        # Calculate end
        end = min(start + self.step - 1, safe_latest)
        
        return (start, end)
    
    def feedback(self, success: bool, latency_ms: float):
        """
        Adapt step size based on feedback
        
        Strategy: AIMD (Additive Increase, Multiplicative Decrease)
        """
        if success and latency_ms < 1000:
            # Increase step (additive)
            self.step = min(int(self.step * 1.2), self.max_step)
        elif not success:
            # Decrease step (multiplicative)
            self.step = max(int(self.step * 0.5), self.min_step)
    
    def get_overlap_size(self, last_scanned: Optional[int]) -> int:
        """Get overlap size for current scan"""
        if last_scanned is None:
            return 0
        return min(self.tail_resync, last_scanned)

# Usage
planner = ResumePlanner(confirmations=12, tail_resync=12, initial_step=1500)

state = state_store.get('transfers_v1')
last_scanned = state.last_scanned_block if state else None

scan_range = planner.get_next_range(last_scanned, latest_block)

if scan_range:
    start, end = scan_range
    overlap = planner.get_overlap_size(last_scanned)
    print(f"Scan: {start:,} → {end:,} (overlap: {overlap})")
else:
    print("Up to date")
```

---

## 7) Transaction Safety: ACID Guarantees

### 7.1 Isolation Levels

```python
def set_isolation_level(conn, level='SERIALIZABLE'):
    """
    Set transaction isolation level
    
    Levels:
    - READ UNCOMMITTED: Lowest isolation (dirty reads possible)
    - READ COMMITTED: Default in most databases
    - REPEATABLE READ: Prevents non-repeatable reads
    - SERIALIZABLE: Highest isolation (prevents phantoms)
    
    For our use case:
    - Single writer → SERIALIZABLE not needed
    - Use READ COMMITTED (DuckDB default)
    """
    # DuckDB uses SERIALIZABLE by default for single connection
    # No need to change for our use case
    pass

# Example: Verify isolation
def test_isolation(conn1, conn2):
    """Test that uncommitted changes are not visible"""
    # Connection 1: Start transaction
    conn1.execute("BEGIN TRANSACTION")
    conn1.execute("INSERT INTO transfers VALUES (...)")
    
    # Connection 2: Query (should not see uncommitted data)
    count = conn2.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    assert count == 0, "Uncommitted data should not be visible"
    
    # Connection 1: Commit
    conn1.execute("COMMIT")
    
    # Connection 2: Query again (now should see data)
    count = conn2.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    assert count > 0, "Committed data should be visible"
```

### 7.2 Deadlock Prevention

```python
class DeadlockSafeCheckpointer:
    """
    Checkpointer with deadlock prevention
    
    Strategy:
    - Single writer pattern (no concurrent writes to same table)
    - Timeout on lock acquisition
    - Retry with exponential backoff
    """
    
    def __init__(self, conn, state, key, lock_timeout_ms=5000):
        self.conn = conn
        self.state = state
        self.key = key
        self.lock_timeout = lock_timeout_ms
    
    def commit_with_retry(self, df, end_block, end_hash, end_time, max_retries=3):
        """
        Commit with retry on deadlock
        """
        for attempt in range(max_retries):
            try:
                # Set lock timeout
                self.conn.execute(f"SET lock_timeout = {self.lock_timeout}")
                
                # Attempt commit
                with Checkpointer(self.conn, self.state, self.key) as cp:
                    cp.insert_data(df)
                    cp.commit(end_block, end_hash, end_time, len(df))
                
                return True  # Success
                
            except Exception as e:
                if 'lock' in str(e).lower() or 'timeout' in str(e).lower():
                    # Deadlock or timeout
                    wait_time = (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  Lock conflict (attempt {attempt+1}/{max_retries}), "
                          f"waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Other error, don't retry
                    raise
        
        raise RuntimeError(f"Failed after {max_retries} retries")
```

---

## 8) Resume Algorithms: Cold Start, Warm Restart, Gap Detection

### 8.1 Cold Start Algorithm

```python
def cold_start_scan(conn, state, rpc_url, start_block=0):
    """
    Cold start: No previous state
    
    Strategy:
    1. Start from genesis (or specified start_block)
    2. Scan incrementally with adaptive step
    3. Build state from scratch
    """
    print(f"🆕 Cold start from block {start_block:,}")
    
    planner = ResumePlanner(initial_step=1500)
    current = start_block
    
    while True:
        latest = get_latest_block(rpc_url)
        scan_range = planner.get_next_range(current, latest, start_block)
        
        if not scan_range:
            print("✅ Caught up to chain tip")
            break
        
        start, end = scan_range
        
        print(f"📥 Scanning {start:,} → {end:,}")
        
        t0 = time.perf_counter()
        logs = fetch_logs(rpc_url, start, end)
        df = parse_logs_to_df(logs)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        # Atomic commit
        end_hash = get_block_hash(rpc_url, end)
        end_time = get_block_time(rpc_url, end)
        
        with Checkpointer(conn, state, 'transfers_v1') as cp:
            cp.insert_data(df)
            cp.commit(end, end_hash, end_time, len(logs))
        
        # Feedback
        planner.feedback(True, latency_ms)
        
        # Progress
        progress = (end - start_block) / (latest - start_block) * 100
        print(f"   ✅ {len(logs)} logs, {latency_ms:.0f}ms, {progress:.1f}% complete")
        
        current = end
```

### 8.2 Warm Restart Algorithm

```python
def warm_restart_scan(conn, state, rpc_url):
    """
    Warm restart: Resume from checkpoint
    
    Strategy:
    1. Load last checkpoint
    2. Apply tail re-scan (reorg protection)
    3. Continue incrementally
    """
    checkpoint = state.get('transfers_v1')
    
    if not checkpoint:
        print("⚠️  No checkpoint found, falling back to cold start")
        return cold_start_scan(conn, state, rpc_url)
    
    last_scanned = checkpoint.last_scanned_block
    print(f"🔄 Resuming from block {last_scanned:,}")
    
    planner = ResumePlanner(tail_resync=12)
    overlap = planner.get_overlap_size(last_scanned)
    
    print(f"   Tail re-scan: {overlap} blocks ({last_scanned - overlap + 1:,} → {last_scanned:,})")
    
    current = last_scanned
    
    while True:
        latest = get_latest_block(rpc_url)
        scan_range = planner.get_next_range(current, latest)
        
        if not scan_range:
            print("✅ Up to date")
            break
        
        start, end = scan_range
        
        print(f"📥 Scanning {start:,} → {end:,}")
        
        # Process batch...
        # (Same as cold start)
        
        current = end
```

### 8.3 Gap Detection Algorithm

```python
def detect_gaps(conn) -> list[tuple[int, int]]:
    """
    Detect missing block ranges
    
    Returns:
        List of (start, end) gaps
    """
    # Get min/max blocks
    bounds = conn.execute("""
        SELECT MIN(block_number), MAX(block_number)
        FROM transfers
    """).fetchone()
    
    if not bounds or bounds[0] is None:
        return []
    
    min_block, max_block = bounds
    
    # Find gaps using recursive CTE
    gaps = conn.execute("""
        WITH RECURSIVE block_series AS (
            SELECT ? as block_num
            UNION ALL
            SELECT block_num + 1 
            FROM block_series 
            WHERE block_num < ?
        ),
        existing_blocks AS (
            SELECT DISTINCT block_number as block_num
            FROM transfers
        ),
        missing_blocks AS (
            SELECT s.block_num
            FROM block_series s
            LEFT JOIN existing_blocks e ON s.block_num = e.block_num
            WHERE e.block_num IS NULL
        ),
        gap_groups AS (
            SELECT 
                block_num,
                block_num - ROW_NUMBER() OVER (ORDER BY block_num) as grp
            FROM missing_blocks
        )
        SELECT 
            MIN(block_num) as gap_start,
            MAX(block_num) as gap_end
        FROM gap_groups
        GROUP BY grp
        ORDER BY gap_start
    """, [min_block, max_block]).fetchall()
    
    return [(start, end) for start, end in gaps]

# Usage
gaps = detect_gaps(conn)
if gaps:
    print(f"⚠️  Found {len(gaps)} gap(s):")
    for start, end in gaps:
        size = end - start + 1
        print(f"   {start:,} → {end:,} ({size} blocks)")
    
    # Fill gaps
    for start, end in gaps:
        print(f"📥 Filling gap {start:,} → {end:,}")
        logs = fetch_logs(rpc_url, start, end)
        df = parse_logs_to_df(logs)
        insert_idempotent(conn, df)
else:
    print("✅ No gaps detected")
```

---

## 9) Performance Optimization: Checkpoint Frequency

### 9.1 Checkpoint Frequency Trade-offs

```
╔════════════════════════════════════════════════════════════╗
║         CHECKPOINT FREQUENCY ANALYSIS                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Strategy 1: Per-Block Checkpoint                         ║
║    Frequency: Every block                                 ║
║    Pros: Maximum granularity, minimal re-work on crash    ║
║    Cons: Slow (commit overhead), high I/O                 ║
║    Use case: Critical data, low volume                    ║
║                                                            ║
║  Strategy 2: Per-Batch Checkpoint ⭐ RECOMMENDED          ║
║    Frequency: Every N blocks (1000-2000)                  ║
║    Pros: Good balance, fast, reasonable re-work           ║
║    Cons: Some re-work on crash (up to N blocks)           ║
║    Use case: High volume, acceptable re-work              ║
║                                                            ║
║  Strategy 3: Time-Based Checkpoint                        ║
║    Frequency: Every T seconds (60s)                       ║
║    Pros: Predictable, good for monitoring                 ║
║    Cons: Variable batch size                              ║
║    Use case: Real-time dashboards                         ║
║                                                            ║
║  Strategy 4: Hybrid                                       ║
║    Frequency: Every N blocks OR T seconds (whichever first) ║
║    Pros: Best of both worlds                              ║
║    Cons: More complex logic                               ║
║    Use case: Production systems                           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 9.2 Hybrid Checkpoint Strategy

```python
class HybridCheckpointStrategy:
    """
    Checkpoint on block count OR time threshold
    
    Example: Checkpoint every 1500 blocks OR 60 seconds
    """
    
    def __init__(self, 
                 block_threshold: int = 1500,
                 time_threshold_seconds: float = 60.0):
        self.block_threshold = block_threshold
        self.time_threshold = time_threshold_seconds
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_block = 0
        self.blocks_since_checkpoint = 0
    
    def should_checkpoint(self, current_block: int) -> tuple[bool, str]:
        """
        Check if checkpoint needed
        
        Returns:
            (should_checkpoint, reason)
        """
        now = time.time()
        elapsed = now - self.last_checkpoint_time
        self.blocks_since_checkpoint = current_block - self.last_checkpoint_block
        
        # Check block threshold
        if self.blocks_since_checkpoint >= self.block_threshold:
            return True, f"block_threshold ({self.blocks_since_checkpoint} blocks)"
        
        # Check time threshold
        if elapsed >= self.time_threshold:
            return True, f"time_threshold ({elapsed:.1f}s)"
        
        return False, "no threshold met"
    
    def on_checkpoint(self, block: int):
        """Record checkpoint"""
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_block = block
        self.blocks_since_checkpoint = 0

# Usage
strategy = HybridCheckpointStrategy(block_threshold=1500, time_threshold_seconds=60)

current_block = 0
for batch in batches:
    # Process batch
    current_block += batch.size
    
    # Check checkpoint
    should_cp, reason = strategy.should_checkpoint(current_block)
    
    if should_cp:
        print(f"📌 Checkpointing (reason: {reason})")
        # Commit with state update
        with Checkpointer(conn, state, 'transfers_v1') as cp:
            cp.insert_data(batch.df)
            cp.commit(current_block, batch.hash, batch.time, len(batch.df))
        
        strategy.on_checkpoint(current_block)
    else:
        # Accumulate (no checkpoint yet)
        accumulate_batch(batch)
```

---

## 10) Monitoring & Alerting: Lag Tracking

### 10.1 Lag Metrics

```python
class LagMonitor:
    """
    Monitor pipeline lag and health
    
    Metrics:
    - Block lag: blocks behind chain tip
    - Time lag: time since last update
    - Throughput: blocks/second
    """
    
    def __init__(self, conn, state, rpc_url, alert_threshold_blocks=1000):
        self.conn = conn
        self.state = state
        self.rpc_url = rpc_url
        self.alert_threshold = alert_threshold_blocks
    
    def get_metrics(self) -> dict:
        """Get current lag metrics"""
        # Get checkpoint
        checkpoint = self.state.get('transfers_v1')
        if not checkpoint:
            return {'status': 'NO_CHECKPOINT'}
        
        # Get chain tip
        latest_block = get_latest_block(self.rpc_url)
        
        # Calculate lag
        block_lag = latest_block - checkpoint.last_scanned_block
        time_lag_seconds = (datetime.now() - checkpoint.updated_at).total_seconds()
        
        # Calculate throughput
        if checkpoint.total_blocks_processed > 0:
            runtime_seconds = (checkpoint.updated_at - checkpoint.created_at).total_seconds()
            throughput = checkpoint.total_blocks_processed / runtime_seconds if runtime_seconds > 0 else 0
        else:
            throughput = 0
        
        # Health status
        if block_lag > self.alert_threshold:
            status = 'CRITICAL'
        elif block_lag > self.alert_threshold // 2:
            status = 'WARNING'
        elif time_lag_seconds > 600:  # 10 minutes
            status = 'STALE'
        else:
            status = 'HEALTHY'
        
        return {
            'status': status,
            'block_lag': block_lag,
            'time_lag_seconds': time_lag_seconds,
            'latest_block': latest_block,
            'checkpoint_block': checkpoint.last_scanned_block,
            'throughput_blocks_per_sec': throughput,
            'total_blocks_processed': checkpoint.total_blocks_processed,
            'total_logs_ingested': checkpoint.total_logs_ingested,
            'error_count': checkpoint.error_count
        }
    
    def check_health(self) -> bool:
        """Check if pipeline is healthy (for alerts)"""
        metrics = self.get_metrics()
        
        if metrics['status'] in ['CRITICAL', 'STALE']:
            self.alert(metrics)
            return False
        
        return True
    
    def alert(self, metrics: dict):
        """Send alert (implement based on your alerting system)"""
        message = f"""
        ⚠️  Pipeline Health Alert
        
        Status: {metrics['status']}
        Block lag: {metrics['block_lag']:,} blocks
        Time lag: {metrics['time_lag_seconds']:.0f} seconds
        Latest block: {metrics['latest_block']:,}
        Checkpoint: {metrics['checkpoint_block']:,}
        
        Action required: Check ingest pipeline
        """
        
        print(message)
        
        # TODO: Send to alerting system (email, Slack, PagerDuty, etc.)

# Usage
monitor = LagMonitor(conn, state, RPC_URL, alert_threshold_blocks=1000)

# Periodic health check
import schedule

schedule.every(5).minutes.do(lambda: monitor.check_health())

# Run metrics dashboard
metrics = monitor.get_metrics()
print(f"Status: {metrics['status']}")
print(f"Block lag: {metrics['block_lag']:,}")
print(f"Throughput: {metrics['throughput_blocks_per_sec']:.2f} blocks/s")
```

---

## 11) Testing Strategies: Comprehensive Test Suite

### 11.1 Unit Tests

```python
import pytest
from datetime import datetime

@pytest.fixture
def test_state_store(tmp_path):
    """Create temporary state store"""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    state = StateStore(conn)
    return conn, state

def test_state_create_and_retrieve(test_state_store):
    """Test: Create state → retrieve → verify"""
    conn, state = test_state_store
    
    # Create state
    state.set(
        'test_key',
        last_scanned_block=1000,
        last_scanned_hash="0xabc",
        last_scanned_time=datetime.now(),
        logs_ingested=100
    )
    
    # Retrieve
    retrieved = state.get('test_key')
    
    # Verify
    assert retrieved is not None
    assert retrieved.last_scanned_block == 1000
    assert retrieved.last_scanned_hash == "0xabc"
    assert retrieved.total_logs_ingested == 100

def test_state_update(test_state_store):
    """Test: Update state → verify incremental counters"""
    conn, state = test_state_store
    
    # Initial
    state.set('test_key', 1000, "0xabc", datetime.now(), 100)
    
    # Update
    state.set('test_key', 2000, "0xdef", datetime.now(), 150)
    
    # Verify
    retrieved = state.get('test_key')
    assert retrieved.last_scanned_block == 2000
    assert retrieved.total_blocks_processed == 1000  # 2000 - 1000
    assert retrieved.total_logs_ingested == 250  # 100 + 150

def test_tail_rescan_calculation():
    """Test: Tail re-scan overlap calculation"""
    # No previous state (cold start)
    start, end = calculate_scan_range(None, 1000, 12, 12)
    assert start == 0
    assert end == 988  # 1000 - 12
    
    # With previous state (warm restart)
    start, end = calculate_scan_range(1000, 1020, 12, 12)
    assert start == 989  # 1000 - 12 + 1
    assert end == 1008  # 1020 - 12
    
    # Already up to date
    start, end = calculate_scan_range(1000, 1005, 12, 12)
    # start (989) > end (993) → no scan needed
```

### 11.2 Integration Tests

```python
def test_end_to_end_checkpoint_resume(tmp_path):
    """
    Integration test: Full checkpoint → crash → resume cycle
    
    Steps:
    1. Ingest batch 1, checkpoint
    2. Simulate crash
    3. Restart, verify resume from checkpoint
    4. Ingest batch 2, verify no duplicates
    """
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    state = StateStore(conn)
    
    # Initialize schema
    init_schema(conn)
    
    # Batch 1: blocks 100-200
    df1 = generate_test_transfers(blocks=range(100, 201))
    
    with Checkpointer(conn, state, 'transfers_v1') as cp:
        cp.insert_data(df1)
        cp.commit(200, "0xhash200", datetime.now(), len(df1))
    
    # Verify checkpoint
    checkpoint1 = state.get('transfers_v1')
    assert checkpoint1.last_scanned_block == 200
    
    # Simulate crash: Close connection
    conn.close()
    
    # Restart: New connection
    conn = duckdb.connect(str(db_path))
    state = StateStore(conn)
    
    # Resume: Calculate range
    checkpoint_after_restart = state.get('transfers_v1')
    assert checkpoint_after_restart.last_scanned_block == 200  # State persisted!
    
    # Batch 2: blocks 189-250 (overlap with batch 1)
    df2 = generate_test_transfers(blocks=range(189, 251))
    
    with Checkpointer(conn, state, 'transfers_v1') as cp:
        cp.insert_data(df2)
        cp.commit(250, "0xhash250", datetime.now(), len(df2))
    
    # Verify no duplicates
    total = conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
    unique = conn.execute("""
        SELECT COUNT(DISTINCT tx_hash || '_' || log_index) FROM transfers
    """).fetchone()[0]
    
    assert total == unique, "No duplicates (idempotency worked)"
    
    # Verify state advanced
    checkpoint2 = state.get('transfers_v1')
    assert checkpoint2.last_scanned_block == 250
```

### 11.3 Chaos Engineering Tests

```python
import random

def test_random_failures(tmp_path):
    """
    Chaos test: Random failures during ingest
    
    Verify: State remains consistent despite failures
    """
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    state = StateStore(conn)
    init_schema(conn)
    
    successful_batches = 0
    failed_batches = 0
    
    for batch_num in range(50):
        df = generate_test_transfers(count=100)
        
        # Random failure (30% chance)
        if random.random() < 0.3:
            try:
                with Checkpointer(conn, state, 'transfers_v1') as cp:
                    cp.insert_data(df)
                    
                    # Inject failure before commit
                    raise RuntimeError("Simulated random failure")
                    
                    cp.commit(batch_num * 100, f"0xhash{batch_num}", datetime.now(), len(df))
            except RuntimeError:
                failed_batches += 1
                continue  # State unchanged, safe to continue
        
        # Success
        try:
            with Checkpointer(conn, state, 'transfers_v1') as cp:
                cp.insert_data(df)
                cp.commit(batch_num * 100, f"0xhash{batch_num}", datetime.now(), len(df))
            successful_batches += 1
        except:
            failed_batches += 1
    
    print(f"Successful: {successful_batches}, Failed: {failed_batches}")
    
    # Verify state consistency
    checkpoint = state.get('transfers_v1')
    assert checkpoint is not None
    
    # Verify no duplicates
    verify_no_duplicates(conn)
    
    # Verify data matches state
    max_block_in_data = conn.execute("""
        SELECT MAX(block_number) FROM transfers
    """).fetchone()[0]
    
    assert max_block_in_data <= checkpoint.last_scanned_block

def verify_no_duplicates(conn):
    """Verify no duplicate (tx_hash, log_index) pairs"""
    duplicates = conn.execute("""
        SELECT tx_hash, log_index, COUNT(*) as cnt
        FROM transfers
        GROUP BY tx_hash, log_index
        HAVING cnt > 1
    """).fetchall()
    
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicates"
```

---

## 12) Troubleshooting Guide: 10 Common Scenarios

### Problem 1: State Not Advancing

**Symptoms:**
```python
checkpoint = state.get('transfers_v1')
print(checkpoint.last_scanned_block)  # Always same value
```

**Causes:**
1. Transaction not committed
2. State update outside transaction
3. Database locked

**Solutions:**
```python
# 1. Verify commit
conn.execute("BEGIN TRANSACTION")
# ... insert data ...
state.set(...)
conn.execute("COMMIT")  # ← Must have this!

# 2. Check transaction scope
# State.set() must be INSIDE same transaction as data insert

# 3. Check for locks
# Use single writer pattern
```

---

### Problem 2: Duplicate Logs After Restart

**Symptoms:**
```sql
SELECT tx_hash, log_index, COUNT(*)
FROM transfers
GROUP BY tx_hash, log_index
HAVING COUNT(*) > 1;
-- Returns rows
```

**Causes:**
1. PRIMARY KEY not enforced
2. Anti-join logic incorrect
3. Tail re-scan too large without idempotency

**Solutions:**
```sql
-- 1. Verify PRIMARY KEY exists
SELECT sql FROM sqlite_master WHERE name = 'transfers';
-- Should show PRIMARY KEY (tx_hash, log_index)

-- 2. Test anti-join manually
CREATE TEMP TABLE _test AS SELECT * FROM transfers LIMIT 10;
-- Run anti-join, verify count
```

---

### Problem 3: Missing Blocks (Gaps)

**Symptoms:**
```python
gaps = detect_gaps(conn)
print(gaps)  # [(1005, 1010), (1200, 1205)]
```

**Causes:**
1. Tail re-scan too small (missed reorg)
2. Crash during batch (partial commit)
3. Network error (logs fetched but not all processed)

**Solutions:**
```python
# 1. Increase tail_resync
tail_resync = 20  # Was 12

# 2. Fill gaps manually
for start, end in gaps:
    logs = fetch_logs(rpc_url, start, end)
    df = parse_logs_to_df(logs)
    insert_idempotent(conn, df)

# 3. Add gap detection to monitoring
schedule.every(1).hour.do(lambda: detect_and_fill_gaps(conn))
```

---

### Problem 4: High Lag (Blocks Behind)

**Symptoms:**
```python
metrics = monitor.get_metrics()
print(metrics['block_lag'])  # 5000+ blocks
```

**Causes:**
1. Slow RPC provider
2. Small batch size
3. Too frequent checkpoints

**Solutions:**
```python
# 1. Increase batch size
planner = ResumePlanner(initial_step=3000)  # Was 1500

# 2. Optimize checkpoint frequency
strategy = HybridCheckpointStrategy(
    block_threshold=3000,  # Was 1500
    time_threshold_seconds=120  # Was 60
)

# 3. Parallel fetching (advanced)
# Fetch logs for multiple ranges concurrently
```

---

### Problem 5: Reorg Not Detected

**Symptoms:**
- Logs in DB don't match chain
- Block hash mismatch

**Causes:**
1. No `blocks` table
2. Tail re-scan too small
3. Block hash not verified

**Solutions:**
```python
# 1. Implement block hash tracking
def store_block_hash(conn, block_num, block_hash, parent_hash):
    conn.execute("""
        INSERT OR REPLACE INTO blocks (number, hash, parent_hash, timestamp)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, [block_num, block_hash, parent_hash])

# 2. Verify hash on resume
def verify_block_hash(conn, block_num, expected_hash):
    stored = conn.execute("""
        SELECT hash FROM blocks WHERE number = ?
    """, [block_num]).fetchone()
    
    if stored and stored[0] != expected_hash:
        print(f"⚠️  Reorg detected at {block_num}")
        return False
    return True

# 3. Increase tail_resync if reorgs frequent
```

---

### Problem 6: Slow Checkpoint Queries

**Symptoms:**
```python
# Takes > 1s
state.get('transfers_v1')
```

**Causes:**
1. No index on `key` column
2. Large `metadata` JSON
3. Too many rows in `scan_state`

**Solutions:**
```sql
-- 1. Verify PRIMARY KEY (auto-indexed)
CREATE TABLE scan_state (
    key TEXT PRIMARY KEY,  -- ← Indexed!
    ...
);

-- 2. Minimize metadata size
# Only store essential info in metadata

-- 3. Clean old records
DELETE FROM scan_state WHERE updated_at < NOW() - INTERVAL 30 DAY;
```

---

### Problem 7: Memory Leak During Ingest

**Symptoms:**
- Memory usage grows unbounded
- Process killed (OOM)

**Causes:**
1. DataFrames not released
2. Connection leaks
3. Accumulating metrics

**Solutions:**
```python
# 1. Explicit cleanup
df = None
gc.collect()

# 2. Use context managers
with DuckDBConnection(db_path) as conn:
    # ... work ...
    pass  # Auto-closes

# 3. Limit metrics history
if len(metrics_list) > 10000:
    metrics_list = metrics_list[-10000:]  # Keep last 10K
```

---

### Problem 8: Transaction Deadlock

**Symptoms:**
```
duckdb.IOException: database is locked
```

**Causes:**
1. Multiple concurrent writers
2. Long-running transaction
3. Unfinished transaction (no COMMIT/ROLLBACK)

**Solutions:**
```python
# 1. Single writer pattern
import fcntl
lock_file = open('/tmp/ingest.lock', 'w')
fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

# 2. Timeout long transactions
conn.execute("SET lock_timeout = 5000")  # 5s

# 3. Always cleanup
try:
    conn.execute("BEGIN TRANSACTION")
    # ... work ...
    conn.execute("COMMIT")
except:
    conn.execute("ROLLBACK")
    raise
```

---

### Problem 9: Incorrect Tail Overlap

**Symptoms:**
- Re-scanning too many blocks (slow)
- Or not enough (missing reorgs)

**Causes:**
1. Tail_resync misconfigured
2. Adaptive sizing bug

**Solutions:**
```python
# 1. Verify calculation
last_scanned = 1000
tail_resync = 12
start = max(0, last_scanned - tail_resync + 1)
assert start == 989  # 1000 - 12 + 1

# 2. Log overlap size
overlap = min(tail_resync, last_scanned)
print(f"Overlap: {overlap} blocks")

# 3. Monitor overlap vs reorg frequency
# If reorgs > 0 but overlap < reorg depth → increase tail_resync
```

---

### Problem 10: State Corruption After Crash

**Symptoms:**
- State exists but data missing
- Inconsistent counters

**Causes:**
1. State updated BEFORE data (wrong order!)
2. Partial commit
3. File system corruption

**Solutions:**
```python
# 1. ALWAYS update state INSIDE same TX as data
# ❌ Wrong order:
state.set(...)
conn.commit()
insert_data(...)
conn.commit()

# ✅ Correct order:
conn.execute("BEGIN TRANSACTION")
insert_data(...)
state.set(...)  # Same TX!
conn.execute("COMMIT")

# 2. Verify state consistency on startup
verify_state_consistency(conn)

# 3. Backup state periodically
backup_database(db_path)
```

---

## 13) Mini Quiz (10 Soru)

1. Checkpoint'i neden data insert ile **aynı transaksiyonda** güncelliyoruz?
2. `TAIL_RESYNC` parametresi tipik olarak neden `CONFIRMATIONS` ile eşit alınır?
3. At-least-once delivery + idempotent processing birleşince ne elde ederiz?
4. Crash COMMIT'ten **önce** olursa beklenen durum nedir?
5. `blocks` tablosunu reorg tespiti için nasıl kullanırız?
6. Checkpoint frequency trade-off nedir? (sık vs seyrek)
7. Warm restart ile cold start arasındaki fark nedir?
8. Gap detection algoritması nasıl çalışır?
9. Lag monitoring için hangi metrikleri izleriz?
10. State corruption'dan nasıl korunuruz?

### Cevap Anahtarı

1. **Atomicity** → Data ve state ayrışmasın, crash-safe olsun
2. Son N blok en oynak bölge; güvenli tampon için overlap gerekir
3. **Exactly-once semantics** (end-to-end effect)
4. Veri **ve** state yazılmaz (ROLLBACK) → yeniden dene, tutarlı
5. `parent_hash` ≠ önceki `block_hash` → reorg detected → geri sar
6. Sık: Az re-work ama yavaş (commit overhead); Seyrek: Hızlı ama crash'te fazla re-work
7. Cold: Baştan başla; Warm: Checkpoint'ten devam (tail re-scan ile)
8. Recursive CTE ile missing blocks bul → GROUP BY gap'ler
9. block_lag, time_lag, throughput, error_count
10. State'i **data ile aynı TX**'te güncelle + backup + verify

---

## 14) Ödevler (6 Pratik)

### Ödev 1: State Persistence Test
```python
# Task: Verify state persists across restarts
# 1. Insert batch, checkpoint
# 2. Close connection
# 3. Reopen connection
# 4. Verify state unchanged
# 5. Resume and continue
```

### Ödev 2: Tail Re-scan Verification
```python
# Task: Verify tail re-scan catches reorg
# 1. Insert batch 1 (blocks 100-200)
# 2. Manually modify DB (simulate reorg: change some logs)
# 3. Run batch 2 with tail_resync=20
# 4. Verify modified logs detected and re-processed
```

### Ödev 3: Crash Recovery Simulation
```python
# Task: Simulate crash at different points
# 1. Inject failure BEFORE commit → verify no data/state
# 2. Inject failure AFTER commit → verify data/state persisted
# 3. Inject failure DURING commit (hard!) → verify atomic
```

### Ödev 4: Gap Detection & Fill
```python
# Task: Implement gap detection and filling
# 1. Ingest blocks 100-200, 250-300 (intentional gap)
# 2. Run detect_gaps() → should find [201, 249]
# 3. Implement fill_gaps() to fetch and insert missing
# 4. Verify no gaps after fill
```

### Ödev 5: Lag Monitoring Dashboard
```python
# Task: Build monitoring dashboard
# 1. Collect metrics: block_lag, time_lag, throughput
# 2. Plot over time (matplotlib or Streamlit)
# 3. Add alerting: if lag > threshold, print warning
# 4. Bonus: Send alert to Telegram/Slack
```

### Ödev 6: Checkpoint Frequency Benchmark
```python
# Task: Compare checkpoint strategies
# 1. Strategy A: Checkpoint every 500 blocks
# 2. Strategy B: Checkpoint every 2000 blocks
# 3. Strategy C: Checkpoint every 60 seconds
# Measure: Total time, commit overhead, re-work on simulated crash
# Find optimal strategy
```

---

## 15) Definition of Done (Tahta 06)

### Learning Objectives
- [ ] Exactly-once semantics understanding (at-least-once + idempotent)
- [ ] Checkpoint strategies (atomic, durable, observable)
- [ ] Tail re-scan rationale and calculation
- [ ] Crash recovery scenarios (before/after commit)
- [ ] Schema design (scan_state + blocks + metrics)
- [ ] Production patterns (StateStore, Checkpointer, ResumePlanner)
- [ ] Transaction safety (ACID, isolation, deadlock prevention)
- [ ] Resume algorithms (cold start, warm restart, gap detection)
- [ ] Performance optimization (checkpoint frequency)
- [ ] Monitoring & alerting (lag tracking, health checks)

### Practical Outputs
- [ ] StateStore class implemented and tested
- [ ] Checkpointer context manager working
- [ ] Atomic checkpoint verified (crash test passing)
- [ ] Tail re-scan calculation correct
- [ ] Gap detection function working
- [ ] Lag monitoring implemented
- [ ] Unit tests passing (5+ tests)
- [ ] Integration test passing (end-to-end)
- [ ] Chaos test passing (random failures)

---

## 🔗 İlgili Dersler

- **← Tahta 05:** [DuckDB + İdempotent](05_tahta_duckdb_idempotent.md)
- **→ Tahta 07:** JSON Rapor + Schema (Coming)
- **↑ Ana Sayfa:** [Week 0 Bootstrap](../../../crypto/w0_bootstrap/README.md)

---

## 🛡️ Güvenlik / Etik

- **Read-only:** Özel anahtar yok, imza yok, custody yok
- **Backup:** State corruption'a karşı günlük backup
- **Monitoring:** Production'da lag alerting şart
- **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 📌 Navigasyon

- **→ Sonraki:** [07 - JSON Rapor + Schema](07_tahta_rapor_json_schema.md) (Coming)
- **← Önceki:** [05 - DuckDB + İdempotent](05_tahta_duckdb_idempotent.md)
- **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

**Tahta 06 — State & Resume: Production Checkpoint Patterns**  
*Format: Production Deep-Dive*  
*Süre: 60-75 dk*  
*Prerequisite: Tahta 01-05*  
*Versiyon: 2.0 (Complete Expansion)*  
*Code Examples: 1,500+ lines*

