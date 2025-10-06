# 🧑‍🏫 Tahta 10 — Troubleshooting & Runbooks: Production Ops Guide

> **Amaç:** NovaDev Crypto hattında sahada karşılaşacağın problemleri **hızla teşhis edip gidermek**. Bu ders; **triage matrisi**, **golden signal metrikleri**, **standart tanılama kiti**, **12+ olay için runbook**, **doğruluk kontrolleri**, **performans tüningi**, **disaster recovery**, **postmortem şablonu** ve **release preflight** kontrol listeleri içerir.
> 
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (Genişletilmiş Tahta)

1. **Triage Matrix** (İlk 5 dakika response)
2. **Golden Signals & SLOs** (Monitoring targets)
3. **Diagnostic Toolkit** (Essential commands)
4. **Incident Runbooks** (12+ scenarios, step-by-step)
5. **Data Correctness** (Invariants & SQL checks)
6. **Performance Tuning** (Optimization checklist)
7. **Disaster Recovery** (Backup & restore)
8. **Postmortem Template** (Blameless culture)
9. **Release Preflight** (Go-live checklist)
10. **Operations Cheatsheet** (Quick commands)
11. **Monitoring & Alerting** (Setup guide)
12. **Quiz + Ödevler**

---

## 0) İlk 5 Dakika: Triage Matrix

### 0.1 SEV (Severity) Sınıflandırması

```
╔════════════════════════════════════════════════════════════╗
║            SEVERITY CLASSIFICATION                         ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  SEV-1 (CRITICAL)                                          ║
║    • API completely down (all endpoints 5xx)               ║
║    • Data corruption (wrong values in reports)             ║
║    • Security breach                                       ║
║    Response time: Immediate (< 15 min)                     ║
║    Escalation: Team lead + on-call                         ║
║                                                            ║
║  SEV-2 (HIGH)                                              ║
║    • p95 latency > SLO (> 1s)                              ║
║    • Partial data delay (> 1 hour lag)                     ║
║    • 429 rate > 5% sustained                               ║
║    Response time: < 1 hour                                 ║
║    Escalation: On-call engineer                            ║
║                                                            ║
║  SEV-3 (MEDIUM)                                            ║
║    • Single wallet errors                                  ║
║    • Edge case failures                                    ║
║    • Dev/staging issues                                    ║
║    Response time: Next business day                        ║
║    Escalation: Ticket queue                                ║
║                                                            ║
║  SEV-4 (LOW)                                               ║
║    • Documentation issues                                  ║
║    • Nice-to-have features                                 ║
║    Response time: Backlog                                  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 0.2 5-Minute Decision Tree

```
┌─────────────────────────────────────────────────────────┐
│              INCIDENT RESPONSE FLOWCHART                │
└─────────────────────────────────────────────────────────┘

START
  │
  ├─► 1. Check /healthz endpoint
  │   ├─ 200 OK? → Go to step 2
  │   └─ 5xx/timeout? → SEV-1: API Down (Runbook R13)
  │
  ├─► 2. Check error logs (last 1 hour)
  │   ├─ Error rate > 1%? → SEV-2: High error rate
  │   └─ Normal? → Go to step 3
  │
  ├─► 3. Check metrics dashboard
  │   ├─ p95 > 1s? → SEV-2: Performance degradation (R10)
  │   ├─ cache_hit < 50%? → SEV-2: Cache issue (R8)
  │   ├─ 429_rate > 5%? → SEV-2: RPC throttling (R1)
  │   └─ All normal? → Go to step 4
  │
  ├─► 4. Reproduce issue
  │   ├─ Try different wallet
  │   ├─ Try different time window
  │   └─ Compare with expected output
  │
  └─► 5. Determine scope
      ├─ All wallets affected? → SEV-1/SEV-2
      ├─ Specific wallet? → SEV-3
      └─ Cannot reproduce? → Request more info
```

### 0.3 Initial Response Template

**When incident is detected, immediately post:**

```
🚨 INCIDENT DETECTED

Severity: SEV-X
Component: [API / Ingest / Database / RPC]
Impact: [All users / Specific wallets / Internal only]

Initial Assessment:
• /healthz status: [OK / DEGRADED / DOWN]
• p95 latency: XXX ms
• Error rate: X.X%
• 429 rate: X.X%
• Affected wallets: [All / Specific / Unknown]

Action Plan:
1. [First action]
2. [Second action]
3. [ETA for next update]

ETA: [15 minutes for next update]
Incident Commander: @username
```

---

## 1) Golden Signals & SLOs

### 1.1 Key Metrics

| Metric | Description | Target (SLO) | Alert Threshold |
|--------|-------------|--------------|-----------------|
| `api_p50_ms` | Median API latency | < 200 ms | > 500 ms (5 min) |
| `api_p95_ms` | 95th percentile API latency | **< 1000 ms** | **> 1500 ms (5 min)** |
| `api_p99_ms` | 99th percentile API latency | < 2000 ms | > 3000 ms (5 min) |
| `api_error_rate` | 5xx errors / total requests | **< 0.1%** | **> 1%** (5 min) |
| `cache_hit_ratio` | Cache hits / total requests | **> 70%** | **< 50%** (10 min) |
| `cache_size` | Current entries in cache | < capacity | > 95% capacity |
| `builder_ms_p95` | ReportBuilder duration | **< 250 ms** | > 500 ms (5 min) |
| `duckdb_query_ms_p95` | Slowest query p95 | **< 150 ms** | > 300 ms (5 min) |
| `rpc_429_rate` | RPC 429 errors / total RPC calls | **≈ 0%** | **> 1%** (5 min) |
| `rpc_timeout_rate` | RPC timeouts / total | < 0.5% | > 2% (5 min) |
| `ingest_lag_blocks` | Latest chain block - last scanned | < 100 blocks | > 1000 blocks |
| `ingest_rate_blocks_s` | Blocks ingested per second | > 10 | < 5 (5 min) |
| `reorg_detected_count` | Reorg events detected | N/A (info) | N/A |
| `duplicate_dropped_count` | Idempotent drops | Low (< 1%) | > 10% (alert config issue) |

### 1.2 SLO Targets Summary

```
╔════════════════════════════════════════════════════════════╗
║                  SLO DASHBOARD                             ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  🎯 Availability:     99.0% (monthly)                     ║
║  ⚡ Latency (p95):    < 1000 ms                           ║
║  📊 Error Rate:       < 0.1%                              ║
║  💾 Cache Hit Ratio:  > 70%                               ║
║  🔄 Ingest Lag:       < 100 blocks (~20 minutes)          ║
║                                                            ║
║  Weekly SLO Review: Every Monday 10:00 AM                 ║
║  Monthly Postmortem: First Friday of month                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 2) Diagnostic Toolkit: Essential Commands

### 2.1 Health Check

```bash
# Basic health check
curl -sS http://localhost:8000/healthz | jq

# Expected output:
{
  "status": "ok",
  "uptime_s": 12345.67,
  "db_status": "ok",
  "cache_status": "active",
  "cache_size": 42,
  "cache_hits": 1234,
  "cache_misses": 567
}

# Check with timeout
timeout 5s curl -sS http://localhost:8000/healthz || echo "TIMEOUT"
```

### 2.2 API Testing

```bash
# Test wallet report (force cache miss with varying hours)
WALLET="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

# Cold cache test
curl -sS "http://localhost:8000/wallet/$WALLET/report?hours=24" | jq

# Warm cache test (repeat immediately)
time curl -sS "http://localhost:8000/wallet/$WALLET/report?hours=24" | jq '.meta'

# Test different wallets
for W in 0x1111... 0x2222... 0x3333...; do
  echo "Testing $W..."
  curl -sS "http://localhost:8000/wallet/$W/report?hours=24" | jq '.tx_count'
done

# Test error handling (invalid address)
curl -v "http://localhost:8000/wallet/invalid/report?hours=24" 2>&1 | grep "< HTTP"
# Should return 422
```

### 2.3 DuckDB Diagnostics

```bash
# Connect to database
duckdb /path/to/onchain.duckdb

-- Check table stats
.tables
PRAGMA table_info('transfers');

-- Row counts
SELECT COUNT(*) AS total_transfers FROM transfers;
SELECT COUNT(DISTINCT tx_hash) AS unique_txs FROM transfers;
SELECT COUNT(DISTINCT tx_hash || '#' || log_index) AS unique_logs FROM transfers;

-- Should match: total_transfers == unique_logs (PK uniqueness)

-- Data quality checks
SELECT 
  COUNT(*) AS total,
  COUNT(CASE WHEN value_unit < 0 THEN 1 END) AS negative_values,
  COUNT(CASE WHEN from_addr = to_addr THEN 1 END) AS self_transfers,
  COUNT(CASE WHEN symbol IS NULL THEN 1 END) AS null_symbols,
  COUNT(CASE WHEN decimals IS NULL THEN 1 END) AS null_decimals
FROM transfers;

-- Recent activity
SELECT 
  block_number,
  block_time,
  tx_hash,
  from_addr,
  to_addr,
  value_unit,
  symbol
FROM transfers
ORDER BY block_number DESC, log_index DESC
LIMIT 20;

-- Top active wallets
SELECT 
  addr,
  COUNT(*) AS tx_count
FROM (
  SELECT from_addr AS addr FROM transfers
  UNION ALL
  SELECT to_addr AS addr FROM transfers
) 
GROUP BY addr
ORDER BY tx_count DESC
LIMIT 10;

-- Block coverage (detect gaps)
SELECT 
  MIN(block_number) AS min_block,
  MAX(block_number) AS max_block,
  COUNT(DISTINCT block_number) AS unique_blocks,
  (MAX(block_number) - MIN(block_number) + 1) AS expected_blocks
FROM transfers;
-- If unique_blocks < expected_blocks, there are gaps

-- Performance: Query plans
EXPLAIN ANALYZE
SELECT COUNT(*) FROM transfers 
WHERE block_time >= '2025-10-06' AND block_time < '2025-10-07'
  AND to_addr = '0x...';
```

### 2.4 Schema Validation

```bash
# Validate report JSON against schema
WALLET="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

python crypto/w0_bootstrap/report_json.py \
  --wallet $WALLET \
  --hours 24 \
  --validate

# Expected output:
# Report successfully validated against schema.
# Report saved to stdout

# Or pipe to validator directly
python crypto/w0_bootstrap/report_json.py --wallet $WALLET --hours 24 | \
  python crypto/features/report_validator.py

# Batch validation (multiple wallets)
for W in 0x1111... 0x2222... 0x3333...; do
  echo "Validating $W..."
  python crypto/w0_bootstrap/report_json.py --wallet $W --hours 24 --validate \
    || echo "FAILED: $W"
done
```

### 2.5 Performance Testing

```bash
# Install hey (HTTP load testing tool)
# brew install hey (macOS)
# or download from: https://github.com/rakyll/hey

WALLET="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
URL="http://localhost:8000/wallet/$WALLET/report?hours=24"

# Warm up cache
curl -sS "$URL" > /dev/null

# Load test: 100 requests, 10 concurrent
hey -n 100 -c 10 "$URL"

# Expected output:
# Summary:
#   Total:    2.1234 secs
#   Slowest:  0.1234 secs
#   Fastest:  0.0123 secs
#   Average:  0.0212 secs
#   p50:      0.0200 secs
#   p95:      0.0450 secs  (should be < 0.100 for warm cache)
#   p99:      0.0800 secs

# Cold cache test (different hours each time to bypass cache)
for i in {1..20}; do
  HOURS=$((24 + i))
  hey -n 1 "http://localhost:8000/wallet/$WALLET/report?hours=$HOURS"
done

# Concurrent users simulation
hey -n 1000 -c 50 -q 10 "$URL"  # 50 concurrent users, 10 req/s each
```

### 2.6 Log Analysis

```bash
# Assuming JSONL logs in /var/log/novadev/api.log

# Error rate (last hour)
tail -n 10000 /var/log/novadev/api.log | \
  grep -c '"level":"ERROR"'

# Slow requests (> 1s)
tail -n 10000 /var/log/novadev/api.log | \
  jq 'select(.latency_ms > 1000) | {ts, route, latency_ms, address}'

# Cache miss rate
tail -n 10000 /var/log/novadev/api.log | \
  jq -s '
    group_by(.cache_hit) | 
    map({cache_hit: .[0].cache_hit, count: length}) | 
    .[]
  '

# Top 10 slowest requests
tail -n 10000 /var/log/novadev/api.log | \
  jq -s 'sort_by(.latency_ms) | reverse | .[0:10] | .[] | {ts, latency_ms, address}'

# 429 errors
tail -n 10000 /var/log/novadev/api.log | \
  grep -c '"status":429'
```

---

## 3) Incident Runbooks: 12+ Step-by-Step Scenarios

### R1 — RPC 429 Rate Limiting / Timeout

**Severity:** SEV-2 (High impact on ingest and API)

**Symptoms:**
- Error logs show `429 Too Many Requests` or `timeout` from RPC calls
- `rpc_429_rate` metric > 1%
- Reports fail to generate or are slow
- Ingest lag increasing

**Root Causes:**
- RPC provider rate limit exceeded
- Too aggressive polling (small delay, large window)
- Burst traffic (many concurrent requests)
- No exponential backoff on retries

**Diagnosis:**

```bash
# 1. Check 429 rate in logs
grep "429" /var/log/novadev/rpc.log | wc -l

# 2. Check RPC provider dashboard
# (Alchemy/Infura: check usage vs. limit)

# 3. Check current window size and request rate
# (If using AdaptiveWindowManager from Tahta 04)
grep "window_size" /var/log/novadev/ingest.log | tail -20

# 4. Test RPC directly
curl -X POST https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# If 429: rate limit confirmed
```

**Resolution:**

1. **Immediate (< 5 min):**
   ```bash
   # Reduce ingest rate
   # Edit config or restart with lower rate
   export INGEST_DELAY_MS=2000  # Increase delay
   export WINDOW_SIZE=100       # Decrease window
   
   # Restart ingest process
   pkill -f capture_transfers_idempotent.py
   python crypto/w0_bootstrap/capture_transfers_idempotent.py &
   ```

2. **Short-term (< 1 hour):**
   - Enable exponential backoff in RPC client (Tahta 02: `ExponentialBackoff`)
   - Implement token bucket rate limiting
   - Switch to backup RPC provider (if configured)

3. **Long-term (< 1 day):**
   - Upgrade RPC provider plan (paid tier with higher limits)
   - Implement adaptive window sizing (AIMD algorithm, Tahta 04)
   - Add circuit breaker pattern
   - Set up monitoring alerts for 429 rate

**Prevention:**
- Always use exponential backoff with jitter
- Monitor RPC usage proactively
- Have backup RPC providers configured
- Implement graceful degradation

**Verification:**
```bash
# Check 429 rate dropped
grep "429" /var/log/novadev/rpc.log | wc -l

# Check ingest lag decreasing
duckdb onchain.duckdb -c "SELECT MAX(block_number) FROM transfers;"
# Compare with chain head: curl https://sepolia.etherscan.io/
```

---

### R2 — getLogs "Too Many Results" (10k Limit)

**Severity:** SEV-2

**Symptoms:**
- Error: `query returned more than 10000 results`
- Ingest stuck at specific block range
- Logs show repeated failures on same range

**Root Causes:**
- Window size too large for high-activity blocks
- Popular token with many transfers
- No adaptive windowing

**Diagnosis:**

```bash
# 1. Identify failing block range
grep "Too Many Results" /var/log/novadev/ingest.log | tail -5

# 2. Check transfer density in that range
duckdb onchain.duckdb -c "
  SELECT block_number, COUNT(*) AS transfer_count
  FROM transfers
  WHERE block_number BETWEEN 12345000 AND 12346000
  GROUP BY block_number
  ORDER BY transfer_count DESC
  LIMIT 10;
"

# 3. Test problematic range directly
python -c "
from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://...'))
logs = w3.eth.get_logs({
    'fromBlock': 12345000,
    'toBlock': 12345999,  # 1000 block range
    'topics': ['0xddf252...']  # Transfer event
})
print(f'Found {len(logs)} logs')
"
```

**Resolution:**

1. **Immediate:**
   ```python
   # Implement binary search window reduction
   # crypto/collector/adaptive_window.py (from Tahta 04)
   
   def find_safe_window(from_block, to_block, max_results=10000):
       """Binary search to find largest safe window"""
       while from_block < to_block:
           mid = (from_block + to_block) // 2
           try:
               logs = get_logs(from_block, mid)
               if len(logs) < max_results:
                   from_block = mid + 1
               else:
                   to_block = mid - 1
           except Exception as e:
               if "too many results" in str(e):
                   to_block = mid - 1
               else:
                   raise
       return from_block
   ```

2. **Short-term:**
   - Reduce default window size to 500 blocks
   - Add token address filter to topics (if scanning specific token)
   - Implement progressive window reduction on failure

3. **Long-term:**
   - Use `AdaptiveWindowManager` from Tahta 04
   - Monitor transfer density and adjust proactively
   - Consider using archive node with pagination

**Prevention:**
- Start with conservative window size (500-1000 blocks)
- Use adaptive windowing from day 1
- Add metrics for logs-per-block

**Verification:**
```bash
# Check ingest resumed
tail -f /var/log/novadev/ingest.log | grep "Scanned blocks"

# Verify no gaps
duckdb onchain.duckdb -c "
  WITH blocks_range AS (
    SELECT generate_series(12345000, 12346000) AS block_num
  )
  SELECT br.block_num
  FROM blocks_range br
  LEFT JOIN transfers t ON br.block_num = t.block_number
  WHERE t.block_number IS NULL;
"
```

---

### R3 — Reorg Detected: Duplicate or Missing Transfers

**Severity:** SEV-2 (Data correctness issue)

**Symptoms:**
- Duplicate `tx_hash#log_index` detected
- Gaps in block sequence
- Reports show inconsistent totals after reorg
- Alerts: `reorg_detected_count` increased

**Root Causes:**
- Chain reorganization (uncle block replaced)
- Insufficient confirmation depth (`CONFIRMATIONS` too low)
- No tail re-scan after reorg

**Diagnosis:**

```bash
# 1. Check for duplicates
duckdb onchain.duckdb -c "
  SELECT tx_hash, log_index, COUNT(*) AS dup_count
  FROM transfers
  GROUP BY tx_hash, log_index
  HAVING COUNT(*) > 1;
"

# 2. Check reorg events in logs
grep "reorg" /var/log/novadev/ingest.log | tail -20

# 3. Check block hash consistency
duckdb onchain.duckdb -c "
  SELECT block_number, block_hash, COUNT(*) AS versions
  FROM blocks
  GROUP BY block_number
  HAVING COUNT(*) > 1;
"
# Multiple versions of same block = reorg happened

# 4. Verify against Etherscan
# Get block hash from DB
BLOCK_NUM=12345678
DB_HASH=$(duckdb onchain.duckdb -c "SELECT block_hash FROM blocks WHERE block_number=$BLOCK_NUM;")
# Get canonical hash from Etherscan
CANONICAL_HASH=$(curl -s "https://sepolia.etherscan.io/block/$BLOCK_NUM" | grep -oP 'hash-tag">\K[^<]+')
# Compare
if [ "$DB_HASH" != "$CANONICAL_HASH" ]; then
  echo "REORG DETECTED: Block $BLOCK_NUM hash mismatch"
fi
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Stop ingest
   pkill -f capture_transfers_idempotent.py
   
   # Identify reorg range
   # (Usually last 12-100 blocks depending on depth)
   
   # Delete affected blocks from DB
   duckdb onchain.duckdb -c "
     DELETE FROM transfers WHERE block_number >= 12345678;
     DELETE FROM blocks WHERE block_number >= 12345678;
     UPDATE scan_state SET last_scanned_block = 12345677;
   "
   
   # Restart ingest (will re-scan from checkpoint)
   python crypto/w0_bootstrap/capture_transfers_idempotent.py &
   ```

2. **Short-term:**
   - Increase `CONFIRMATIONS` parameter (e.g., from 5 to 12)
   - Enable tail re-scan (Tahta 06: `ResumePlanner` with `tail=12`)
   - Verify idempotent insert is working (PRIMARY KEY constraint)

3. **Long-term:**
   - Implement automatic reorg detection and recovery
   - Store block hashes and validate chain continuity
   - Add metrics: `reorg_depth_blocks`, `reorg_recovery_time_s`
   - Set up alerts for deep reorgs (> 12 blocks)

**Prevention:**
```python
# crypto/collector/reorg_detector.py (from Tahta 04)

class ReorgDetector:
    def detect_reorg(self, current_block_num, current_block_hash):
        """Check if current block matches DB"""
        db_hash = self.get_db_block_hash(current_block_num)
        if db_hash and db_hash != current_block_hash:
            # Reorg detected!
            logger.warning(f"Reorg at block {current_block_num}")
            self.trigger_recovery(current_block_num)
            return True
        return False
    
    def trigger_recovery(self, reorg_block):
        """Roll back to safe depth and re-scan"""
        safe_block = reorg_block - CONFIRMATIONS - 10
        self.rollback_to_block(safe_block)
        self.resume_from_block(safe_block)
```

**Verification:**
```bash
# 1. Check no duplicates
duckdb onchain.duckdb -c "
  SELECT COUNT(*) AS total, COUNT(DISTINCT tx_hash||'#'||log_index) AS unique
  FROM transfers;
"
# total should equal unique

# 2. Check no gaps
duckdb onchain.duckdb -c "
  WITH expected AS (
    SELECT generate_series(
      (SELECT MIN(block_number) FROM transfers),
      (SELECT MAX(block_number) FROM transfers)
    ) AS block_num
  )
  SELECT COUNT(*) AS gap_blocks
  FROM expected e
  LEFT JOIN transfers t ON e.block_num = t.block_number
  WHERE t.block_number IS NULL;
"
# Should be 0 (or low if some blocks have no transfers)

# 3. Verify reports consistent
python crypto/w0_bootstrap/report_json.py --wallet $WALLET --hours 24 --validate
```

---

### R4 — Ingest State Stuck (Not Progressing)

**Severity:** SEV-2

**Symptoms:**
- `scan_state.last_scanned_block` not increasing
- `ingest_lag_blocks` growing over time
- No new transfers in database
- Ingest process appears running but idle

**Root Causes:**
- Exception during ingest causing rollback loop
- Transaction not committing (missing `COMMIT`)
- Database lock preventing writes
- Checkpoint file corrupted

**Diagnosis:**

```bash
# 1. Check current state
duckdb onchain.duckdb -c "SELECT * FROM scan_state;"

# 2. Check ingest process logs
tail -100 /var/log/novadev/ingest.log | grep -E "(ERROR|Exception|Rollback)"

# 3. Check if process is actually running
ps aux | grep capture_transfers

# 4. Check DB lock status
lsof | grep onchain.duckdb

# 5. Test write manually
duckdb onchain.duckdb -c "
  BEGIN TRANSACTION;
  UPDATE scan_state SET last_scanned_block = last_scanned_block + 1;
  SELECT * FROM scan_state;
  ROLLBACK;
"
# If this fails, DB has a lock issue
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Stop ingest process
   pkill -f capture_transfers_idempotent.py
   
   # Check for abandoned connections
   lsof | grep onchain.duckdb
   # Kill any orphaned processes
   
   # Test manual write
   duckdb onchain.duckdb -c "
     BEGIN;
     UPDATE scan_state SET last_scanned_block = last_scanned_block + 1;
     COMMIT;
   "
   
   # Restart ingest
   python crypto/w0_bootstrap/capture_transfers_idempotent.py &
   ```

2. **Short-term:**
   - Review ingest code for exception handling (Tahta 06: `Checkpointer`)
   - Ensure atomic transactions (`BEGIN` → `INSERT/UPDATE` → `COMMIT`)
   - Add connection pool with auto-close on error
   - Implement gap detection and backfill

3. **Long-term:**
   - Add state progression metrics: `blocks_per_minute`
   - Implement dead-man switch (alert if no progress in 10 min)
   - Use separate process for read (API) vs. write (ingest)
   - Store detailed checkpoint metadata (timestamp, hash, etc.)

**Prevention:**

```python
# crypto/collector/checkpointer.py (from Tahta 06)

class AtomicCheckpointer:
    def checkpoint(self, transfers, new_block_num):
        """Atomic write of data + state"""
        conn = self.get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            
            # Insert transfers (idempotent)
            conn.executemany(
                "INSERT OR IGNORE INTO transfers VALUES (...)",
                transfers
            )
            
            # Update state
            conn.execute(
                "UPDATE scan_state SET last_scanned_block = ?, updated_at = NOW()",
                [new_block_num]
            )
            
            conn.execute("COMMIT")
            logger.info(f"Checkpoint saved: block {new_block_num}")
            
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Checkpoint failed: {e}")
            raise
```

**Verification:**
```bash
# Watch state progress in real-time
watch -n 5 'duckdb onchain.duckdb -c "SELECT last_scanned_block FROM scan_state;"'

# Should increment every 5-10 seconds

# Check lag vs chain head
CHAIN_HEAD=$(curl -s https://sepolia.etherscan.io/ | grep -oP 'Block #\K[0-9]+')
DB_HEAD=$(duckdb onchain.duckdb -c "SELECT last_scanned_block FROM scan_state;")
LAG=$((CHAIN_HEAD - DB_HEAD))
echo "Ingest lag: $LAG blocks"
# Should be < 100
```

---

---

### R5 — DuckDB "Database is Locked"

**Severity:** SEV-1 (API down)

**Symptoms:**
- API returns 500 errors
- Error: `database is locked` in logs
- Cannot write to database

**Root Causes:**
- Multiple processes trying to write simultaneously
- API opened connection in write mode
- Abandoned lock from crashed process

**Diagnosis:**

```bash
# 1. Check processes accessing DB
lsof | grep onchain.duckdb

# Output example:
# python  12345  user  3r  REG  /path/to/onchain.duckdb
# python  12346  user  4w  REG  /path/to/onchain.duckdb

# 2. Check DB file permissions
ls -la onchain.duckdb*

# 3. Test read-only access
duckdb onchain.duckdb -readonly -c "SELECT COUNT(*) FROM transfers;"

# 4. Test write access
duckdb onchain.duckdb -c "SELECT 1;" || echo "LOCKED"
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Kill all processes accessing DB
   pkill -f onchain.duckdb
   
   # Remove lock file if exists
   rm -f onchain.duckdb.wal onchain.duckdb.lock
   
   # Restart services in correct order:
   # 1. Ingest (write mode)
   python crypto/w0_bootstrap/capture_transfers_idempotent.py &
   
   # 2. API (read-only mode)
   export NOVA_DB_PATH=/path/to/onchain.duckdb
   uvicorn crypto.service.app:app --reload &
   ```

2. **Short-term:**
   - Ensure API uses read-only connections
   - Separate ingest process (single writer)
   - Close connections properly (use context managers)

3. **Long-term:**
   - Implement connection pool with read-only flag
   - Use separate databases for read/write (replication)
   - Monitor open file handles

**Prevention:**

```python
# crypto/service/deps.py (ensure read-only)

def get_conn() -> duckdb.DuckDBPyConnection:
    """Get thread-local read-only connection"""
    if _pool is None:
        raise RuntimeError("Pool not initialized")
    
    # CRITICAL: Always use read_only=True for API
    conn = duckdb.connect(
        database=settings.db_path,
        read_only=True  # ← CRITICAL
    )
    return conn
```

**Verification:**
```bash
# API should work
curl http://localhost:8000/healthz

# Ingest should work
tail -f /var/log/novadev/ingest.log | grep "Scanned blocks"
```

---

### R6 — Slow Queries: p95 > SLO

**Severity:** SEV-2

**Symptoms:**
- `duckdb_query_ms_p95 > 300 ms`
- API p95 latency increasing
- Slow reports for specific wallets

**Root Causes:**
- Missing indexes
- Inefficient SQL queries
- Large table scans
- No predicate pushdown

**Diagnosis:**

```bash
# 1. Enable query profiling
duckdb onchain.duckdb -c "
  PRAGMA enable_profiling = 'query_tree';
  PRAGMA profiling_output = 'query_plan.txt';
"

# 2. Run suspect query with EXPLAIN ANALYZE
duckdb onchain.duckdb -c "
  EXPLAIN ANALYZE
  SELECT 
    SUM(CASE WHEN to_addr = '0x...' THEN value_unit ELSE 0 END) AS inbound
  FROM transfers
  WHERE block_time >= '2025-10-06' AND block_time < '2025-10-07';
"

# Look for:
# - SEQ_SCAN (bad - means full table scan)
# - Missing filters
# - Large row counts

# 3. Check table stats
duckdb onchain.duckdb -c "
  SELECT 
    COUNT(*) AS total_rows,
    COUNT(DISTINCT block_number) AS unique_blocks,
    MIN(block_time) AS min_time,
    MAX(block_time) AS max_time,
    pg_size_pretty(pg_total_relation_size('transfers')) AS table_size
  FROM transfers;
"

# 4. Identify slow queries from logs
grep '"db_ms"' /var/log/novadev/api.log | \
  jq 'select(.db_ms > 300) | {wallet: .address, db_ms, query_type}' | \
  head -20
```

**Resolution:**

1. **Immediate:**
   - Add missing indexes
   ```sql
   -- If not already exists (from Tahta 05)
   CREATE INDEX IF NOT EXISTS idx_transfers_block_time 
     ON transfers(block_time);
   
   CREATE INDEX IF NOT EXISTS idx_transfers_to_addr 
     ON transfers(to_addr);
   
   CREATE INDEX IF NOT EXISTS idx_transfers_from_addr 
     ON transfers(from_addr);
   
   CREATE INDEX IF NOT EXISTS idx_transfers_composite 
     ON transfers(block_time, to_addr, from_addr);
   ```

2. **Short-term:**
   - Optimize queries in ReportBuilder (Tahta 07)
   - Use `LIMIT` where appropriate
   - Pre-aggregate common queries
   - Add query timeout

3. **Long-term:**
   - Implement materialized views (summary tables)
   - Partition large tables by time
   - Use columnar storage optimization
   - Cache pre-computed summaries

**Query Optimization Example:**

```python
# BAD: Full table scan
query = """
SELECT * FROM transfers 
WHERE to_addr = ? OR from_addr = ?
"""

# GOOD: Indexed filter + projection
query = """
SELECT 
  block_time, 
  token, 
  from_addr, 
  to_addr, 
  value_unit
FROM transfers
WHERE 
  block_time >= ? AND block_time < ?
  AND (to_addr = ? OR from_addr = ?)
"""
```

**Verification:**
```bash
# Re-run query with EXPLAIN
duckdb onchain.duckdb -c "
  EXPLAIN 
  SELECT ... FROM transfers WHERE ...;
"

# Should show INDEX_SCAN instead of SEQ_SCAN

# Check p95 improved
hey -n 100 -c 10 "http://localhost:8000/wallet/$WALLET/report?hours=24"
# p95 should be < 500ms (cold cache)
```

---

### R7 — Memory Bloat (OOM Risk)

**Severity:** SEV-2

**Symptoms:**
- RSS (Resident Set Size) growing linearly
- Process killed by OOM killer
- Swap usage increasing
- API becomes unresponsive

**Root Causes:**
- Loading entire result set into memory
- Memory leaks in Python objects
- Large batch sizes
- Unclosed connections

**Diagnosis:**

```bash
# 1. Check memory usage
ps aux | grep -E "(python|uvicorn)" | awk '{print $2, $4, $6, $11}'

# 2. Monitor over time
watch -n 5 'ps aux | grep python | awk "{print \$6}" | awk "{sum+=\$1} END {print sum/1024 \" MB\"}"'

# 3. Python memory profiler
pip install memory-profiler

python -m memory_profiler crypto/features/report_builder.py

# 4. Check for leaks (tracemalloc)
python -c "
import tracemalloc
tracemalloc.start()

# Your code here
from crypto.features.report_builder import ReportBuilder
builder = ReportBuilder('onchain.duckdb')
for i in range(100):
    report = builder.build(wallet='0x...', window_hours=24)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
"
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Restart service to reclaim memory
   pkill -f uvicorn
   uvicorn crypto.service.app:app --reload &
   
   # Reduce cache capacity temporarily
   export NOVA_CACHE_CAPACITY=512  # Down from 2048
   ```

2. **Short-term:**
   - Use iterators instead of loading full result sets
   - Implement streaming for large queries
   - Close connections explicitly
   - Reduce batch sizes

3. **Long-term:**
   - Add memory limits to Docker containers
   - Implement graceful degradation on memory pressure
   - Use generators for large datasets
   - Profile and optimize hot paths

**Code Optimization:**

```python
# BAD: Load all results into memory
def get_transfers(wallet):
    result = conn.execute("SELECT * FROM transfers WHERE to_addr = ?", [wallet])
    return result.fetchall()  # ← Loads everything

# GOOD: Use iterator
def get_transfers(wallet):
    result = conn.execute("SELECT * FROM transfers WHERE to_addr = ?", [wallet])
    for row in result.fetchmany(1000):  # ← Batch of 1000
        yield row

# BETTER: Aggregate in SQL
def get_transfer_summary(wallet):
    result = conn.execute("""
        SELECT 
            COUNT(*) AS tx_count,
            SUM(value_unit) AS total_value
        FROM transfers 
        WHERE to_addr = ?
    """, [wallet])
    return result.fetchone()  # ← Only 1 row
```

**Verification:**
```bash
# Check memory stable
watch -n 10 'ps aux | grep python'

# Run load test
hey -n 1000 -c 50 "http://localhost:8000/wallet/$WALLET/report?hours=24"

# Memory should stay flat (not grow linearly)
```

---

### R8 — Cache Stampede (Cold Start Spike)

**Severity:** SEV-2

**Symptoms:**
- p95 latency spike after deployment
- Cache hit rate drops to 0%
- Multiple concurrent requests for same key
- Database overload

**Root Causes:**
- Cache cleared on restart
- No warmup strategy
- Thundering herd problem
- Popular wallets requested simultaneously

**Diagnosis:**

```bash
# 1. Check cache hit rate
curl -s http://localhost:8000/healthz | jq '.cache_hit_rate'

# Should be > 0.7, if 0.0 = cold cache

# 2. Check concurrent requests for same key
tail -100 /var/log/novadev/api.log | \
  jq -r '.address' | \
  sort | uniq -c | sort -rn | head -10

# High counts = thundering herd

# 3. Monitor p95 during cold start
while true; do
  P95=$(hey -n 20 -c 5 "http://localhost:8000/wallet/$WALLET/report?hours=24" 2>&1 | grep "95%" | awk '{print $2}')
  echo "$(date) - p95: $P95"
  sleep 10
done
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Warmup cache with popular wallets
   for wallet in 0xAAA... 0xBBB... 0xCCC...; do
     echo "Warming up $wallet..."
     curl -s "http://localhost:8000/wallet/$wallet/report?hours=24" > /dev/null
     sleep 0.5
   done
   ```

2. **Short-term:**
   - Implement single-flight pattern (deduplicate concurrent requests)
   - Increase cache capacity
   - Extend TTL temporarily
   - Add request coalescing

3. **Long-term:**
   - Automatic cache warmup on startup
   - Persistent cache (Redis/file-based)
   - Staggered deployments (canary)
   - Pre-compute popular reports

**Single-Flight Implementation:**

```python
# crypto/service/cache.py

import asyncio
from typing import Dict, Any
from collections import defaultdict

class SingleFlightCache:
    """Prevents cache stampede by deduplicating concurrent requests"""
    
    def __init__(self, cache: TTLLRUCache):
        self.cache = cache
        self._inflight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_compute(self, key: str, compute_fn):
        """Get from cache or compute once for concurrent requests"""
        
        # Try cache first
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        # Check if computation already in flight
        async with self._lock:
            if key in self._inflight:
                # Wait for in-flight computation
                return await self._inflight[key]
            
            # Start new computation
            future = asyncio.create_task(compute_fn())
            self._inflight[key] = future
        
        try:
            result = await future
            self.cache.set(key, result)
            return result
        finally:
            async with self._lock:
                del self._inflight[key]
```

**Verification:**
```bash
# Test concurrent requests
seq 1 50 | xargs -P 50 -I {} \
  curl -s "http://localhost:8000/wallet/$WALLET/report?hours=24" > /dev/null

# Check logs: should see single computation, not 50
grep "Building report" /var/log/novadev/api.log | wc -l
# Should be 1, not 50

# Check cache hit rate recovered
curl -s http://localhost:8000/healthz | jq '.cache_hit_rate'
# Should be > 0.7
```

---

### R9 — Schema Validation Fail

**Severity:** SEV-2

**Symptoms:**
- 422 errors from `/wallet/{addr}/report`
- `jsonschema.ValidationError` in logs
- Reports fail validation after code change

**Root Causes:**
- `report_v1.json` schema changed
- ReportBuilder generates non-compliant JSON
- Extra fields added (violates `additionalProperties: false`)
- Field type mismatch

**Diagnosis:**

```bash
# 1. Generate report and validate
WALLET="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

python crypto/w0_bootstrap/report_json.py \
  --wallet $WALLET \
  --hours 24 \
  --validate 2>&1 | tee validation_error.txt

# 2. Check schema file
cat schemas/report_v1.json | jq '.required'

# 3. Test minimal valid report
python -c "
import json
from jsonschema import Draft202012Validator

schema = json.load(open('schemas/report_v1.json'))
validator = Draft202012Validator(schema)

# Minimal report
report = {
    'version': 'v1',
    'wallet': '0x' + '1' * 40,
    'window_hours': 24,
    'time': {'from_ts': '2025-10-06T00:00:00Z', 'to_ts': '2025-10-07T00:00:00Z'},
    'totals': {'inbound': 0.0, 'outbound': 0.0},
    'tx_count': 0,
    'transfer_stats': [],
    'top_counterparties': [],
    'meta': {'chain_id': 11155111, 'generated_at': '2025-10-07T00:00:00Z', 'source': 'test'}
}

try:
    validator.validate(report)
    print('Minimal report: VALID')
except Exception as e:
    print(f'Minimal report: INVALID - {e}')
"

# 4. Compare schema versions
git diff HEAD~1 schemas/report_v1.json
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Rollback schema if breaking change
   git checkout HEAD~1 -- schemas/report_v1.json
   
   # Restart API
   pkill -f uvicorn
   uvicorn crypto.service.app:app --reload &
   ```

2. **Short-term:**
   - Fix ReportBuilder to match schema
   - Add/remove fields as needed
   - Update schema with backward compatibility

3. **Long-term:**
   - Schema versioning (`report_v2.json` for breaking changes)
   - API version negotiation (`?version=v1` vs `?version=v2`)
   - Schema-check CI prevents breaking changes
   - Contract tests catch schema drift

**Schema Evolution Strategy:**

```json
// schemas/report_v1.json (stable)
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "version": "v1",
  "required": ["version", "wallet", ...],
  "additionalProperties": false
}

// schemas/report_v1_ext.json (extended, optional fields)
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "version": "v1_ext",
  "allOf": [
    { "$ref": "report_v1.json" },
    {
      "properties": {
        "gas_used": { "type": "number" },  // New optional field
        "usd_value": { "type": "number" }
      }
    }
  ]
}
```

**Verification:**
```bash
# Run contract tests
pytest tests/contract/test_report_schema.py -v

# All should pass

# Test API endpoint
curl -s "http://localhost:8000/wallet/$WALLET/report?hours=24" | \
  python crypto/features/report_validator.py

# Should output: "✅ Report is valid"
```

---

### R10 — API p95 Latency Spike (Sustained)

**Severity:** SEV-2

**Symptoms:**
- `api_p95_ms` consistently > 1000 ms
- Users complaining about slow response
- All metrics healthy except latency

**Root Causes:**
- Low cache hit rate
- Slow database queries
- Single-threaded bottleneck
- External API calls (RPC) in request path

**Diagnosis:**

```bash
# 1. Decompose latency
curl -w "@curl-format.txt" -s "http://localhost:8000/wallet/$WALLET/report?hours=24" > /dev/null

# curl-format.txt:
# time_namelookup:  %{time_namelookup}\n
# time_connect:     %{time_connect}\n
# time_starttransfer: %{time_starttransfer}\n
# time_total:       %{time_total}\n

# 2. Check component latencies from logs
tail -100 /var/log/novadev/api.log | \
  jq '{cache_hit, builder_ms, db_ms, latency_ms}' | \
  awk '{
    if ($2 == "false") cold++; 
    sum+=$6; 
    if ($6>max) max=$6
  } 
  END {
    print "Cache miss rate:", cold/NR
    print "Avg latency:", sum/NR
    print "Max latency:", max
  }'

# 3. Profile specific request
python -m cProfile -o profile.stats crypto/features/report_builder.py

python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# 4. Check worker count
ps aux | grep uvicorn | wc -l
# Should be > 1 for production
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Increase cache TTL and capacity
   export NOVA_CACHE_TTL=300  # 5 minutes
   export NOVA_CACHE_CAPACITY=4096
   
   # Add more workers
   pkill -f uvicorn
   uvicorn crypto.service.app:app --workers 4 --host 0.0.0.0 --port 8000 &
   ```

2. **Short-term:**
   - Optimize slow SQL queries (see R6)
   - Pre-compute popular reports
   - Add query result caching
   - Implement request timeout

3. **Long-term:**
   - Horizontal scaling (multiple API instances)
   - CDN for static reports
   - Async processing for heavy computations
   - Read replicas for database

**Worker Configuration:**

```bash
# Production deployment with Gunicorn
gunicorn crypto.service.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 30 \
  --keep-alive 5 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

**Verification:**
```bash
# Load test
hey -n 500 -c 20 "http://localhost:8000/wallet/$WALLET/report?hours=24"

# Check results:
# p95: < 1000 ms (target)
# p99: < 2000 ms
# Success rate: 100%
```

---

### R11 — Timezone / Time Window Issues

**Severity:** SEV-3

**Symptoms:**
- Reports show wrong time ranges
- Data appears shifted by hours
- Inconsistent timestamps in different components

**Root Causes:**
- Mixed UTC and local timezone usage
- Incorrect timezone conversion
- Server timezone != UTC
- Datetime objects without timezone info

**Diagnosis:**

```bash
# 1. Check server timezone
timedatectl
# Should be UTC

# 2. Check Python timezone
python -c "
from datetime import datetime, timezone
print('System:', datetime.now())
print('UTC:', datetime.now(timezone.utc))
"

# 3. Check database timestamps
duckdb onchain.duckdb -c "
  SELECT 
    block_time,
    timezone('UTC', block_time) AS block_time_utc,
    EXTRACT(timezone FROM block_time) AS tz_offset
  FROM transfers
  LIMIT 5;
"

# 4. Test report time range
python crypto/w0_bootstrap/report_json.py \
  --wallet $WALLET \
  --hours 24 | \
  jq '.time'

# Should show UTC timestamps (Z suffix)
```

**Resolution:**

1. **Immediate:**
   ```bash
   # Set server timezone to UTC
   sudo timedatectl set-timezone UTC
   
   # Restart services
   systemctl restart novadev-api
   systemctl restart novadev-ingest
   ```

2. **Short-term:**
   - Audit all datetime usage
   - Always use `datetime.now(timezone.utc)`
   - Store all timestamps as UTC in database
   - Convert to local timezone only in UI

3. **Long-term:**
   - Add timezone validation in tests
   - Use `pendulum` or `arrow` libraries for timezone handling
   - Document timezone conventions

**Code Standards:**

```python
from datetime import datetime, timezone, timedelta

# ✅ GOOD: Always use timezone-aware datetime
now = datetime.now(timezone.utc)
yesterday = now - timedelta(days=1)

# ✅ GOOD: Parse with timezone
from_ts = datetime.fromisoformat("2025-10-06T00:00:00+00:00")

# ✅ GOOD: Format with explicit UTC
timestamp_str = now.isoformat(timespec='seconds').replace('+00:00', 'Z')

# ❌ BAD: Naive datetime (no timezone)
now = datetime.now()  # Don't use!

# ❌ BAD: Local timezone
import time
local_time = time.localtime()  # Don't use!
```

**Verification:**
```bash
# All timestamps should have Z suffix or +00:00
python crypto/w0_bootstrap/report_json.py --wallet $WALLET --hours 24 | \
  jq -r '.time, .meta.generated_at' | \
  grep -E '(Z|\\+00:00)$' || echo "MISSING TIMEZONE"

# Database check
duckdb onchain.duckdb -c "
  SELECT MIN(block_time) AS earliest, MAX(block_time) AS latest
  FROM transfers;
"
# Verify timestamps make sense (not offset by hours)
```

---

### R12 — Decimal / Precision Errors

**Severity:** SEV-2 (Data correctness)

**Symptoms:**
- Token values off by orders of magnitude
- USDC showing 10^12 instead of 100
- Negative balances
- Sum mismatches

**Root Causes:**
- Missing or wrong `decimals` metadata
- Integer division instead of decimal
- Floating point precision errors
- Raw value not converted to human-readable

**Diagnosis:**

```bash
# 1. Check decimals metadata
duckdb onchain.duckdb -c "
  SELECT 
    token,
    symbol,
    decimals,
    COUNT(*) AS transfer_count,
    SUM(value_unit) AS total_value
  FROM transfers
  GROUP BY token, symbol, decimals
  ORDER BY transfer_count DESC;
"

# Look for:
# - NULL decimals
# - Unusual decimals (> 18 or < 0)
# - Inconsistent decimals for same token

# 2. Check specific wallet totals
python crypto/w0_bootstrap/report_json.py --wallet $WALLET --hours 24 | \
  jq '.totals, .transfer_stats[]'

# Compare with Etherscan

# 3. Check raw vs converted values
duckdb onchain.duckdb -c "
  SELECT 
    tx_hash,
    symbol,
    decimals,
    value_unit AS converted,
    value_unit * POW(10, decimals) AS raw_value
  FROM transfers
  WHERE symbol = 'USDC'
  LIMIT 10;
"
```

**Resolution:**

1. **Immediate:**
   ```python
   # Reprocess with correct decimals
   from decimal import Decimal
   
   def convert_raw_to_unit(raw_value: int, decimals: int) -> Decimal:
       """Convert raw token value to human-readable units"""
       return Decimal(raw_value) / Decimal(10 ** decimals)
   
   # Example: USDC has 6 decimals
   raw = 100000000  # 100 USDC in raw form
   converted = convert_raw_to_unit(raw, 6)
   print(converted)  # 100.000000
   ```

2. **Short-term:**
   - Validate decimals on ingest
   - Use `Decimal` type for precision
   - Add data quality checks
   - Cross-reference with token registry

3. **Long-term:**
   - Token metadata service (decimals, symbol lookup)
   - Automated validation against known tokens
   - Alert on unusual decimal values

**Correct Implementation:**

```python
# crypto/features/report_builder.py

from decimal import Decimal, ROUND_DOWN

class ReportBuilder:
    def _convert_to_unit(self, raw_value: str, decimals: int) -> float:
        """
        Convert raw token value to human-readable units
        
        Args:
            raw_value: Raw token amount (e.g., "100000000" for 100 USDC)
            decimals: Token decimals (e.g., 6 for USDC)
        
        Returns:
            Human-readable value as float
        """
        if decimals is None or decimals < 0:
            # Fallback: assume 18 decimals (ETH standard)
            decimals = 18
        
        # Use Decimal for precision
        raw = Decimal(raw_value)
        divisor = Decimal(10 ** decimals)
        converted = raw / divisor
        
        # Round to reasonable precision
        return float(converted.quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
```

**Verification:**
```bash
# Test known tokens
python -c "
from crypto.features.report_builder import ReportBuilder

# USDC: 6 decimals
assert ReportBuilder()._convert_to_unit('100000000', 6) == 100.0

# WETH: 18 decimals  
assert ReportBuilder()._convert_to_unit('1000000000000000000', 18) == 1.0

print('Decimal conversion: OK')
"

# Check report values reasonable
python crypto/w0_bootstrap/report_json.py --wallet $WALLET --hours 24 | \
  jq '.transfer_stats[] | select(.symbol == "USDC") | .inbound'

# Should be reasonable number (1-10000), not 10^12
```

---

### R13 — API Completely Down (SEV-1)

**Severity:** SEV-1 (CRITICAL)

**Symptoms:**
- All requests return 5xx or timeout
- `/healthz` unreachable
- Process not running or crashed
- Port not listening

**Root Causes:**
- Process crashed (unhandled exception)
- Port already in use
- Database unreachable
- Out of memory (OOM killed)

**Diagnosis:**

```bash
# 1. Check if process running
ps aux | grep uvicorn
# If empty: process crashed

# 2. Check port listening
sudo lsof -i :8000
# If empty: nothing listening

# 3. Check recent logs
tail -100 /var/log/novadev/api.log | grep -E "(ERROR|Exception|Traceback)"

# 4. Check system logs
sudo journalctl -u novadev-api -n 100

# 5. Test port manually
curl -v http://localhost:8000/healthz

# 6. Check disk space
df -h
# If / partition > 90%: might be issue

# 7. Check memory
free -h
# If available memory < 100MB: OOM risk
```

**Resolution:**

1. **Immediate (< 5 min):**
   ```bash
   # Restart service
   systemctl restart novadev-api
   
   # OR manual start
   cd /path/to/novadev-protocol
   source .venv/bin/activate
   export NOVA_DB_PATH=/path/to/onchain.duckdb
   uvicorn crypto.service.app:app --host 0.0.0.0 --port 8000 &
   
   # Verify
   curl http://localhost:8000/healthz
   ```

2. **If restart fails:**
   ```bash
   # Kill process on port
   sudo lsof -t -i:8000 | xargs kill -9
   
   # Check database accessible
   duckdb /path/to/onchain.duckdb -c "SELECT 1;"
   
   # Start with debug logging
   uvicorn crypto.service.app:app --log-level debug &
   
   # Watch logs
   tail -f /var/log/novadev/api.log
   ```

3. **Root cause fix:**
   - Fix unhandled exception in code
   - Add process monitoring (systemd, supervisor)
   - Implement graceful error handling
   - Add health checks and auto-restart

**Systemd Service (Production):**

```ini
# /etc/systemd/system/novadev-api.service

[Unit]
Description=NovaDev Crypto API Service
After=network.target

[Service]
Type=simple
User=novadev
WorkingDirectory=/home/novadev/novadev-protocol
Environment="NOVA_DB_PATH=/var/lib/novadev/onchain.duckdb"
Environment="NOVA_CACHE_TTL=60"
Environment="NOVA_CACHE_CAPACITY=2048"
ExecStart=/home/novadev/novadev-protocol/.venv/bin/uvicorn \
  crypto.service.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
Restart=always
RestartSec=10
StandardOutput=append:/var/log/novadev/api.log
StandardError=append:/var/log/novadev/api.log

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable novadev-api
sudo systemctl start novadev-api

# Check status
sudo systemctl status novadev-api
```

**Verification:**
```bash
# Health check
curl http://localhost:8000/healthz

# Load test
hey -n 100 -c 10 "http://localhost:8000/wallet/$WALLET/report?hours=24"

# Check auto-restart works
sudo systemctl stop novadev-api
sleep 15
curl http://localhost:8000/healthz  # Should auto-restart
```

---

## 4) Data Correctness: Invariants & SQL Checks

### 4.1 Invariants

Essential data quality rules that must **always** hold:

1. **Primary Key Uniqueness**
   ```sql
   -- Test
   SELECT COUNT(*) AS total, 
          COUNT(DISTINCT tx_hash || '#' || log_index) AS unique
   FROM transfers;
   -- Assert: total == unique
   ```

2. **Non-Negative Values**
   ```sql
   -- Test
   SELECT COUNT(*) FROM transfers WHERE value_unit < 0;
   -- Assert: == 0
   ```

3. **No Self-Transfers** (optional business rule)
   ```sql
   -- Test
   SELECT COUNT(*) FROM transfers WHERE from_addr = to_addr;
   -- Assert: == 0 (or low %)
   ```

4. **Time Window Consistency**
   ```sql
   -- Test: All transfers in report window
   WITH report_window AS (
     SELECT 
       TIMESTAMP '2025-10-06 00:00:00' AS from_ts,
       TIMESTAMP '2025-10-07 00:00:00' AS to_ts
   )
   SELECT COUNT(*) 
   FROM transfers, report_window
   WHERE block_time < from_ts OR block_time >= to_ts;
   -- Assert: == 0
   ```

5. **Block Continuity** (no large gaps)
   ```sql
   -- Test
   WITH blocks_range AS (
     SELECT generate_series(
       (SELECT MIN(block_number) FROM transfers),
       (SELECT MAX(block_number) FROM transfers)
     ) AS expected_block
   )
   SELECT COUNT(*) AS missing_blocks
   FROM blocks_range br
   LEFT JOIN (SELECT DISTINCT block_number FROM transfers) t 
     ON br.expected_block = t.block_number
   WHERE t.block_number IS NULL;
   -- Assert: < 100 (some blocks may have no transfers)
   ```

### 4.2 Automated Check Script

```bash
#!/bin/bash
# crypto/scripts/data_quality_check.sh

DB_PATH="${1:-onchain.duckdb}"

echo "=== Data Quality Checks ==="
echo ""

# Check 1: PK Uniqueness
echo "1. Primary Key Uniqueness..."
TOTAL=$(duckdb $DB_PATH -c "SELECT COUNT(*) FROM transfers;" | tail -1)
UNIQUE=$(duckdb $DB_PATH -c "SELECT COUNT(DISTINCT tx_hash||'#'||log_index) FROM transfers;" | tail -1)

if [ "$TOTAL" -eq "$UNIQUE" ]; then
  echo "   ✅ PASS: $TOTAL == $UNIQUE"
else
  echo "   ❌ FAIL: $TOTAL != $UNIQUE (duplicates detected!)"
  exit 1
fi

# Check 2: Non-negative values
echo "2. Non-Negative Values..."
NEG_COUNT=$(duckdb $DB_PATH -c "SELECT COUNT(*) FROM transfers WHERE value_unit < 0;" | tail -1)

if [ "$NEG_COUNT" -eq 0 ]; then
  echo "   ✅ PASS: No negative values"
else
  echo "   ❌ FAIL: $NEG_COUNT negative values found!"
  exit 1
fi

# Check 3: Self-transfers
echo "3. Self-Transfers..."
SELF_COUNT=$(duckdb $DB_PATH -c "SELECT COUNT(*) FROM transfers WHERE from_addr = to_addr;" | tail -1)
SELF_PERCENT=$(awk "BEGIN {print ($SELF_COUNT / $TOTAL) * 100}")

if (( $(echo "$SELF_PERCENT < 1" | bc -l) )); then
  echo "   ✅ PASS: Self-transfers < 1% ($SELF_PERCENT%)"
else
  echo "   ⚠️  WARN: Self-transfers = $SELF_PERCENT%"
fi

# Check 4: NULL metadata
echo "4. NULL Metadata..."
NULL_SYMBOLS=$(duckdb $DB_PATH -c "SELECT COUNT(*) FROM transfers WHERE symbol IS NULL;" | tail -1)
NULL_DECIMALS=$(duckdb $DB_PATH -c "SELECT COUNT(*) FROM transfers WHERE decimals IS NULL;" | tail -1)

if [ "$NULL_SYMBOLS" -eq 0 ] && [ "$NULL_DECIMALS" -eq 0 ]; then
  echo "   ✅ PASS: No NULL metadata"
else
  echo "   ⚠️  WARN: $NULL_SYMBOLS NULL symbols, $NULL_DECIMALS NULL decimals"
fi

# Check 5: Block gaps
echo "5. Block Continuity..."
MIN_BLOCK=$(duckdb $DB_PATH -c "SELECT MIN(block_number) FROM transfers;" | tail -1)
MAX_BLOCK=$(duckdb $DB_PATH -c "SELECT MAX(block_number) FROM transfers;" | tail -1)
UNIQUE_BLOCKS=$(duckdb $DB_PATH -c "SELECT COUNT(DISTINCT block_number) FROM transfers;" | tail -1)
EXPECTED_BLOCKS=$(($MAX_BLOCK - $MIN_BLOCK + 1))
GAP_BLOCKS=$(($EXPECTED_BLOCKS - $UNIQUE_BLOCKS))

if [ "$GAP_BLOCKS" -lt 100 ]; then
  echo "   ✅ PASS: $GAP_BLOCKS gap blocks (< 100)"
else
  echo "   ⚠️  WARN: $GAP_BLOCKS gap blocks detected"
fi

echo ""
echo "=== All Checks Complete ==="
```

```bash
# Run checks
chmod +x crypto/scripts/data_quality_check.sh
./crypto/scripts/data_quality_check.sh onchain.duckdb
```

## 5) Performance Tuning: Optimization Checklist

### 5.1 Query Optimization

- [ ] **Explain plans reviewed** for top 3 slowest queries
- [ ] **Indexes created** on: `block_time`, `to_addr`, `from_addr`
- [ ] **Composite indexes** for common filter combinations
- [ ] **Query rewrite** to avoid full table scans
- [ ] **Predicate pushdown** verified in explain plan
- [ ] **Projection pushdown** (select only needed columns)

### 5.2 Cache Optimization

- [ ] **Hit rate > 70%** (target: 80-90%)
- [ ] **TTL tuned** based on data freshness requirements
- [ ] **Capacity sized** for working set (popular wallets)
- [ ] **Eviction rate < 5%** under normal load
- [ ] **Warmup strategy** for deployment
- [ ] **Single-flight pattern** for thundering herd

### 5.3 Database Optimization

- [ ] **Indexes on hot paths** (transfers, scan_state)
- [ ] **Vacuum + Analyze** run regularly
- [ ] **DuckDB version** up-to-date
- [ ] **Connection pooling** configured
- [ ] **Read-only connections** for API
- [ ] **Write isolation** (single ingest process)

### 5.4 API Optimization

- [ ] **Multi-worker** deployment (4+ workers)
- [ ] **Async I/O** where applicable
- [ ] **Request timeouts** configured
- [ ] **Graceful degradation** on errors
- [ ] **Rate limiting** for expensive endpoints
- [ ] **Response compression** enabled

### 5.5 Infrastructure

- [ ] **SSD storage** for database
- [ ] **Local disk** (not network mount)
- [ ] **Memory adequate** (2GB+ for API)
- [ ] **CPU not saturated** (< 80% avg)
- [ ] **Network latency** to RPC < 100ms
- [ ] **Load balancer** for horizontal scaling

---

## 6) Disaster Recovery: Backup & Restore

### 6.1 Backup Strategy

**Daily Backups:**
```bash
#!/bin/bash
# crypto/scripts/backup.sh

DB_PATH="/var/lib/novadev/onchain.duckdb"
BACKUP_DIR="/var/backups/novadev"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/onchain_$DATE.duckdb.gz"

# Create backup
echo "Starting backup at $(date)..."
gzip -c $DB_PATH > $BACKUP_FILE

# Verify backup
if [ $? -eq 0 ]; then
  echo "✅ Backup successful: $BACKUP_FILE"
  SIZE=$(du -h $BACKUP_FILE | cut -f1)
  echo "   Size: $SIZE"
else
  echo "❌ Backup failed!"
  exit 1
fi

# Keep last 7 days
find $BACKUP_DIR -name "onchain_*.duckdb.gz" -mtime +7 -delete

echo "Backup complete at $(date)"
```

**Automated Schedule (cron):**
```bash
# /etc/cron.d/novadev-backup

# Daily backup at 2 AM
0 2 * * * novadev /home/novadev/novadev-protocol/crypto/scripts/backup.sh >> /var/log/novadev/backup.log 2>&1
```

### 6.2 Restore Procedure

```bash
#!/bin/bash
# crypto/scripts/restore.sh

BACKUP_FILE="$1"
DB_PATH="/var/lib/novadev/onchain.duckdb"

if [ -z "$BACKUP_FILE" ]; then
  echo "Usage: $0 <backup_file.duckdb.gz>"
  exit 1
fi

echo "⚠️  WARNING: This will replace $DB_PATH"
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
  echo "Restore cancelled"
  exit 0
fi

# Stop services
echo "Stopping services..."
systemctl stop novadev-api
systemctl stop novadev-ingest

# Backup current DB (just in case)
mv $DB_PATH ${DB_PATH}.pre-restore

# Restore from backup
echo "Restoring from $BACKUP_FILE..."
gunzip -c $BACKUP_FILE > $DB_PATH

# Verify integrity
echo "Verifying integrity..."
duckdb $DB_PATH -c "PRAGMA integrity_check;"

if [ $? -eq 0 ]; then
  echo "✅ Restore successful"
  rm ${DB_PATH}.pre-restore
else
  echo "❌ Restore failed! Reverting..."
  mv ${DB_PATH}.pre-restore $DB_PATH
  exit 1
fi

# Restart services
echo "Restarting services..."
systemctl start novadev-ingest
systemctl start novadev-api

echo "Restore complete at $(date)"
```

### 6.3 Maintenance Window

**Scheduled maintenance (monthly):**

1. Announce maintenance window (24h notice)
2. Stop ingest process
3. Run full backup
4. Run `VACUUM` and `ANALYZE`
5. Verify database integrity
6. Restart services
7. Monitor for 1 hour

```bash
# crypto/scripts/maintenance.sh

systemctl stop novadev-ingest
./backup.sh
duckdb onchain.duckdb -c "VACUUM; ANALYZE;"
duckdb onchain.duckdb -c "PRAGMA integrity_check;"
systemctl start novadev-ingest
```

---

## 7) Postmortem Template: Blameless Culture

### 7.1 Template

```markdown
# Incident Postmortem: [Brief Title]

**Date:** YYYY-MM-DD
**Duration:** HH:MM - HH:MM (X hours)
**Severity:** SEV-X
**Incident Commander:** @username

## Executive Summary

[2-3 sentence summary of what happened and impact]

## Impact

- **Users Affected:** X% / X wallets
- **Services Affected:** API / Ingest / Both
- **Data Loss:** Yes/No
- **Financial Impact:** $X (if applicable)

## Timeline (all times UTC)

| Time | Event |
|------|-------|
| 14:00 | Initial alert: API p95 > 2s |
| 14:05 | On-call engineer paged |
| 14:10 | Identified root cause: DB locked |
| 14:15 | Mitigation started: Restarted services |
| 14:20 | Services restored, monitoring |
| 14:45 | Confirmed resolution, incident closed |

## Root Cause

[Detailed technical explanation of what went wrong]

**Contributing Factors:**
1. Factor 1
2. Factor 2
3. Factor 3

## Detection

**How was it detected?**
- Automated alert / User report / Proactive monitoring

**Time to detect:** X minutes

**What went well:**
- Quick detection via monitoring

**What could improve:**
- Earlier alerting threshold

## Response

**Actions Taken:**
1. Step 1
2. Step 2
3. Step 3

**Time to mitigate:** X minutes

**What went well:**
- Clear runbooks
- Fast team response

**What could improve:**
- Faster rollback procedure

## Resolution

**Immediate Fix:**
[What was done to restore service]

**Permanent Fix:**
[What was done to prevent recurrence]

## Action Items

| Item | Owner | Due Date | Priority |
|------|-------|----------|----------|
| Add automated DB lock monitoring | @dev | 2025-10-15 | P0 |
| Update runbook R5 | @ops | 2025-10-13 | P1 |
| Implement read-only connection enforcement | @dev | 2025-10-20 | P1 |
| Post-incident review meeting | @lead | 2025-10-10 | P0 |

## Lessons Learned

**What went well:**
- Quick detection (5 minutes)
- Clear communication
- Effective runbook

**What didn't go well:**
- Manual intervention required
- No automated recovery

**Where we got lucky:**
- Happened during business hours
- Backup was recent

## Related Documents

- [Runbook R5: Database Locked](#r5--duckdb-database-is-locked)
- [Monitoring Dashboard](https://grafana.example.com/...)
- [Alert Rules](https://github.com/.../alerts.yml)

---

**Follow-up Date:** YYYY-MM-DD (1 month review)
```

---

## 8) Release Preflight: Go-Live Checklist

### 8.1 Pre-Deployment Checklist

**Code Quality:**
- [ ] All tests passing (unit, integration, contract)
- [ ] Coverage ≥ 70%
- [ ] Ruff lint: 0 errors
- [ ] No `TODO` or `FIXME` in production code
- [ ] Code review approved by 2+ reviewers

**Schema & Contracts:**
- [ ] `report_v1.json` unchanged (or backward compatible)
- [ ] Schema validation tests passing
- [ ] Contract tests passing
- [ ] API versioning strategy clear

**Performance:**
- [ ] Load testing completed (100+ req/s)
- [ ] p95 latency < 1s (target met)
- [ ] Cache hit rate > 70%
- [ ] Database query optimization verified

**Operational:**
- [ ] Backup completed (< 24h old)
- [ ] Rollback plan documented
- [ ] Runbooks updated
- [ ] Monitoring dashboards verified
- [ ] Alerts configured and tested

**Security:**
- [ ] Read-only database connections (API)
- [ ] No private keys in code/config
- [ ] `.env` files not in version control
- [ ] Rate limiting configured
- [ ] Legal disclaimer present

### 8.2 Deployment Checklist

**Pre-Deployment (T-1 hour):**
- [ ] Announce deployment window
- [ ] Verify backup exists
- [ ] Prepare rollback procedure
- [ ] Review incident response plan
- [ ] Confirm on-call availability

**Deployment (T=0):**
- [ ] Deploy code to staging
- [ ] Run smoke tests on staging
- [ ] Deploy to production (canary or blue-green)
- [ ] Verify `/healthz` returns 200
- [ ] Run post-deployment smoke tests

**Post-Deployment (T+15 min):**
- [ ] Monitor error rate (< 0.1%)
- [ ] Monitor p95 latency (< 1s)
- [ ] Monitor cache hit rate (> 70%)
- [ ] Check recent logs for errors
- [ ] Verify sample API requests

**Post-Deployment (T+1 hour):**
- [ ] Confirm metrics stable
- [ ] No new alerts triggered
- [ ] User feedback positive
- [ ] Close deployment issue
- [ ] Update changelog

### 8.3 Rollback Procedure

**If issues detected within 30 minutes:**

```bash
# 1. Revert code
git revert <commit_hash>
git push

# 2. Redeploy previous version
git checkout <previous_tag>
./deploy.sh

# 3. Verify health
curl http://api.novadev.local/healthz

# 4. Monitor for 15 minutes

# 5. Communicate rollback
# Post to incident channel
```

---

## 9) Operations Cheatsheet: Quick Commands

```bash
# === Health Checks ===
curl -s localhost:8000/healthz | jq
hey -n 100 -c 10 "http://localhost:8000/wallet/$W/report?hours=24"

# === Database ===
duckdb onchain.duckdb -c "SELECT COUNT(*) FROM transfers;"
duckdb onchain.duckdb -c "SELECT * FROM scan_state;"
duckdb onchain.duckdb -c "PRAGMA integrity_check;"

# === Logs ===
tail -100 /var/log/novadev/api.log | jq
grep ERROR /var/log/novadev/ingest.log | tail -20
journalctl -u novadev-api -n 100 --no-pager

# === Processes ===
ps aux | grep -E "(uvicorn|python.*capture)"
systemctl status novadev-api
systemctl restart novadev-ingest

# === Metrics ===
# p95 latency (from logs)
tail -1000 /var/log/novadev/api.log | jq -s 'sort_by(.latency_ms) | .[950].latency_ms'

# Cache hit rate
curl -s localhost:8000/healthz | jq '.cache_hit_rate'

# === Data Quality ===
./crypto/scripts/data_quality_check.sh onchain.duckdb

# === Performance ===
# Top 10 slow queries
tail -1000 /var/log/novadev/api.log | jq 'select(.db_ms > 200)' | jq -s 'sort_by(.db_ms) | reverse | .[0:10]'

# === Backup/Restore ===
./crypto/scripts/backup.sh
./crypto/scripts/restore.sh /var/backups/novadev/onchain_20251006.duckdb.gz
```

---

## 10) Quiz + Ödevler

### 📝 Mini Quiz (10 Sorular)

**1. SEV-1 incident için maximum response time nedir?**
- A) 5 dakika
- B) 15 dakika ✅
- C) 1 saat
- D) 4 saat

**2. Golden signal metriklerinden biri DEĞİL?**
- A) api_p95_ms
- B) cache_hit_ratio
- C) cpu_temperature ✅
- D) rpc_429_rate

**3. DuckDB "database is locked" hatası ne zaman oluşur?**
- A) Disk dolu
- B) Birden fazla writer process ✅
- C) Index eksik
- D) Network hatası

**4. Reorg sonrası tail re-scan kaç blok geriye taramalı?**
- A) 1-3 blok
- B) 5-12 blok ✅
- C) 100+ blok
- D) Hiç gerekmez

**5. Cache stampede nasıl önlenir?**
- A) TTL artırma
- B) Single-flight pattern ✅
- C) Database index
- D) Disk upgrade

**6. Data correctness için hangi invariant ZORUNLU?**
- A) Primary key uniqueness ✅
- B) Wallet'lar alfabetik sıralı
- C) Block her zaman ardışık
- D) Symbol her zaman 3 karakter

**7. Postmortem template'de olmaması gereken şey?**
- A) Blameless tone
- B) Timeline
- C) Sorumluyu isimle belirtme ✅
- D) Action items

**8. Release preflight checklist'te kontrol ETMEmeliyiz?**
- A) Test coverage
- B) Schema backward compatibility
- C) Yazılımcının kahve tercihi ✅
- D) Backup existence

**9. p95 latency hedefiniz 1s, ölçüm 1.5s. Ne yapmalısınız?**
- A) Hedefi 1.5s'ye çıkar
- B) Cache optimize et ✅
- C) Metriği gizle
- D) Ignore et

**10. Decimal conversion hatası genelde neden olur?**
- A) Network lag
- B) Missing/wrong decimals metadata ✅
- C) CPU throttling
- D) Disk fragmentation

### 🎯 Pratik Ödevler (6 Ödev)

#### Ödev 1: Triage Drill (15 dk)

**Senaryo:** API'den 429 hatası alıyorsunuz.

**Görev:**
1. Triage matrix kullanarak severity belirle
2. İlk 5 dakika diagnostic checklist uygula
3. Runbook R1'i takip et
4. Çözüm adımlarını dokümante et

**Başarı Kriterleri:**
- Doğru SEV sınıflandırması (SEV-2)
- 5 dakikada root cause bulundu
- Runbook adımları uygulandı
- Çözüm verify edildi

#### Ödev 2: Data Quality Automation (20 dk)

**Görev:** `data_quality_check.sh` scriptini genişlet

Eklenecekler:
- Check 6: Decimal range validation (0-36)
- Check 7: Block timestamp monotonicity
- Check 8: Transfer count per block sanity (< 10K)

**Başarı Kriterleri:**
- Scriptler çalışıyor
- Edge case'ler yakalanıyor
- Exit codes doğru

#### Ödev 3: Postmortem Yazımı (30 dk)

**Senaryo:** Dün 14:00-14:45 arası API down oldu. Sebep: OOM killer.

**Görev:** Template kullanarak complete postmortem yaz

Eklenecekler:
- Timeline (detaylı)
- Root cause analysis
- 3+ action item
- Lessons learned

**Başarı Kriterleri:**
- Blameless tone
- Actionable items
- Clear timeline
- Technical depth

#### Ödev 4: Performance Tuning (25 dk)

**Görev:** Test database'inde slow query bulup optimize et

Steps:
1. `EXPLAIN ANALYZE` ile en yavaş sorguyu bul
2. Missing index tespit et
3. Index ekle
4. Before/after latency ölç

**Başarı Kriterleri:**
- Slow query bulundu
- Index eklendi
- Latency improvement > 50%
- EXPLAIN plan improved (SEQ_SCAN → INDEX_SCAN)

#### Ödev 5: Backup & Restore Drill (20 dk)

**Görev:** Production-like backup/restore test

Steps:
1. Backup script'i çalıştır
2. Test database'i corrupt et
3. Restore script ile geri yükle
4. Data integrity verify et

**Başarı Kriterleri:**
- Backup başarılı
- Restore başarılı
- Zero data loss
- Integrity check pass

#### Ödev 6: Release Preflight (25 dk)

**Görev:** Upcoming release için complete preflight checklist

Steps:
1. Run all tests
2. Check schema compatibility
3. Run load test
4. Verify backup exists
5. Complete all checklist items

**Başarı Kriterleri:**
- All tests green
- Schema backward compatible
- Load test meets SLO
- Checklist 100% complete
- Rollback plan ready

---

## ✅ Definition of Done (DoD)

### Documentation DoD

- [ ] 12+ runbook scenarios documented step-by-step
- [ ] Golden signals & SLOs defined with thresholds
- [ ] Diagnostic toolkit commands provided
- [ ] Data correctness invariants listed with SQL
- [ ] Performance tuning checklist complete
- [ ] DR & backup procedures documented
- [ ] Postmortem template provided
- [ ] Release preflight checklist ready
- [ ] Operations cheatsheet created
- [ ] Quiz + 6 assignments included

### Operational DoD

- [ ] Backup script tested and scheduled
- [ ] Restore procedure verified
- [ ] Data quality checks automated
- [ ] Monitoring dashboards created
- [ ] Alerts configured for golden signals
- [ ] Runbooks tested in staging
- [ ] On-call rotation established
- [ ] Postmortem process agreed upon

### Quality DoD

- [ ] All code examples tested
- [ ] SQL queries verified
- [ ] Shell scripts executable
- [ ] No broken links
- [ ] Consistent formatting
- [ ] Technical accuracy reviewed

---

## 🔗 İlgili Dersler & Kaynaklar

### Önceki Dersler
- **Tahta 01**: EVM Veri Modeli
- **Tahta 02**: JSON-RPC 101
- **Tahta 04**: getLogs + Reorg Strategies
- **Tahta 05**: DuckDB + Idempotent Writes
- **Tahta 06**: State Management & Resume
- **Tahta 07**: JSON Schema & Report
- **Tahta 08**: FastAPI Service
- **Tahta 09**: Quality & CI Automation

### External Resources
- [Google SRE Book - Incident Response](https://sre.google/sre-book/managing-incidents/)
- [Increment: On-Call](https://increment.com/on-call/)
- [Postmortem Culture: Learning from Failure](https://sre.google/sre-book/postmortem-culture/)

---

## 🎓 Özet

```
╔════════════════════════════════════════════════════════════╗
║       TAHTA 10 — TROUBLESHOOTING ÖZET                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  📚 Öğrendiklerimiz:                                      ║
║    • Triage matrix (SEV-1 to SEV-4)                       ║
║    • Golden signals & SLOs                                ║
║    • 13 complete runbooks (RPC, DB, Cache, Schema, etc.)  ║
║    • Data correctness invariants                          ║
║    • Performance tuning strategies                        ║
║    • Disaster recovery procedures                         ║
║    • Postmortem & release processes                       ║
║                                                            ║
║  🎯 Başardıklarımiz:                                      ║
║    • Production-ready incident response                   ║
║    • Step-by-step troubleshooting guides                  ║
║    • Automated quality checks                             ║
║    • Operational excellence foundation                    ║
║                                                            ║
║  🚀 Şimdi Yapabiliyoruz:                                  ║
║    • Diagnose issues in < 5 minutes                       ║
║    • Execute runbooks confidently                         ║
║    • Prevent recurring incidents                          ║
║    • Learn from failures (blameless)                      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**Bugünlük bitti.** Week 0 Complete! 🎉

---

**Versiyon:** 1.0  
**Süre:** 60-75 dakika  
**Seviye:** Production-Ready  
**Status:** Week 0 Complete (100%)

**Son Ders!** 🎓 NovaDev Crypto Week 0 Tahta Serisi tamamlandı!
