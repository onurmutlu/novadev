# 🧑‍🏫 Tahta 04 — `eth_getLogs` Pencere Stratejisi + Reorg Dayanıklılığı (Production Deep-Dive)

> **Amaç:** Log tararken **timeout yemeden**, **rate limit**'e toslamadan, **reorg**'lara dayanıklı kalarak **production-grade** veri toplama hattı kurmak.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Tahta Planı (Genişletilmiş)

1. **Neden pencere?** (Range bölme mantığı + gerçek senaryolar)
2. **Safe range anatomisi:** `safe_latest = latest − CONFIRMATIONS`
3. **Dinamik pencere boyutu** (Adaptive algorithms + rate limiting)
4. **Reorg dayanıklılığı:** Detection + recovery + tail re-scan
5. **State modeli:** Database design + transaction safety
6. **Production implementation** (Complete code + error handling)
7. **Performance optimization:** Benchmarking + tuning
8. **Real-world scenarios:** Mainnet vs Testnet, provider differences
9. **Error taxonomy:** Comprehensive error handling
10. **Testing strategies:** Unit + integration tests
11. **Quiz + ödevler + troubleshooting**

---

## 1) Neden "Pencere" Gerekiyor? (Problem Anatomy)

### 1.1 Naive Approach Problems

**❌ Kötü Yaklaşım:**
```python
# Tüm geçmişi tek seferde tara
logs = eth_getLogs({
    "fromBlock": "0x0",      # Genesis block
    "toBlock": "latest",      # Şu anki block
    "topics": [TRANSFER_SIG]
})
```

**Neden başarısız olur?**

1. **Timeout (Request Timeout)**
   ```
   Sepolia: 6M+ blok × ~0.1ms/blok = 10+ dakika
   → HTTP timeout (30s-120s) → Connection reset
   ```

2. **Rate Limiting (429 Too Many Requests)**
   ```
   Sağlayıcı limitleri:
   - Alchemy: 10,000 log/request
   - Infura: 10,000 log/request
   - QuickNode: 20,000 log/request (plan'a göre)
   
   Büyük aralık → limit aşımı → 429 hatası
   ```

3. **Memory Explosion**
   ```
   10K log × 2KB/log = 20MB JSON
   100K log → 200MB (single response!)
   → Client/server memory issues
   ```

4. **Provider Rejection**
   ```
   Error: "query returned more than 10000 results"
   Error: "block range too large, maximum is 10000 blocks"
   ```

### 1.2 Windowed Scan (Çözüm)

**✅ İyi Yaklaşım:**
```python
# Aralığı küçük parçalara böl
CHUNK_SIZE = 1500  # blocks

for start in range(5_000_000, 6_000_000, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE - 1, 6_000_000)
    
    logs = eth_getLogs({
        "fromBlock": hex(start),
        "toBlock": hex(end),
        "topics": [TRANSFER_SIG]
    })
    
    # Process + store
    # ...
```

**Avantajlar:**

✅ **Predictable latency:** Her request ~500ms-2s  
✅ **Rate limit friendly:** Küçük chunks, spread over time  
✅ **Resumable:** Crash olsa kaldığın yerden devam  
✅ **Memory efficient:** Incremental processing

### 1.3 Görsel: Naive vs Windowed

```
╔═══════════════════════════════════════════════════════════╗
║                    NAIVE APPROACH                         ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  [Block 0 ────────────────────────────── Block 6M]      ║
║         ↓                                                 ║
║     Single massive request                                ║
║         ↓                                                 ║
║     ❌ Timeout / 429 / Memory explosion                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════╗
║                   WINDOWED APPROACH                       ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  [0-1.5K] [1.5K-3K] [3K-4.5K] ... [5.9985M-6M]          ║
║     ↓        ↓          ↓              ↓                  ║
║    Req1     Req2       Req3    ...    Req4000            ║
║     ↓        ↓          ↓              ↓                  ║
║  Process  Process   Process  ...   Process               ║
║     ↓        ↓          ↓              ↓                  ║
║    ✅       ✅         ✅      ...     ✅                  ║
║                                                           ║
║  • Predictable: ~1s/request                              ║
║  • Resumable: State tracking                             ║
║  • Scalable: Handles any range                           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## 2) Safe Range Anatomisi: Reorg Protection

### 2.1 Blockchain Reorganization (Reorg) Nedir?

**Tanım:** Zincirin son blokları **değişmesi** (fork resolution)

**Nasıl oluşur?**

```
Senaryo 1: Normal chain growth
Block N-2 → Block N-1 → Block N
  ✅         ✅          ✅ (all confirmed)

Senaryo 2: Competing blocks (fork)
                    ┌→ Block N (miner A)
Block N-2 → Block N-1 ┤
                    └→ Block N' (miner B)

Senaryo 3: Reorg (N' wins)
Block N-2 → Block N-1 → Block N'  (canonical chain)
                         ↑
                    Block N orphaned!
```

**Consequences:**
- Block N'deki transaction'lar **orphan** oldu
- Block N'deki log'lar artık **geçersiz**
- Yeni Block N''deki transaction'lar farklı olabilir

### 2.2 Safe Latest Calculation

**Formula:**
```python
latest = eth_blockNumber()
safe_latest = latest - CONFIRMATIONS
```

**CONFIRMATIONS değerleri:**

| Network | Typical | Conservative | Risk |
|---------|---------|--------------|------|
| **Sepolia** | 5 | 12 | Low (PoS) |
| **Mainnet** | 12 | 32 | Low (PoS post-merge) |
| **Polygon** | 128 | 256 | Medium (shorter block time) |
| **BSC** | 15 | 50 | Medium |

**PoS (Proof of Stake) sonrası:**
- Finality: ~2 epochs (~13 mins, ~64 blocks)
- Pratik: 12 block yeterli (shallow reorg'lar için)
- Conservative: 32+ block (deep reorg'lara karşı)

### 2.3 Reorg Risk Visualization

```
╔═══════════════════════════════════════════════════════════╗
║              BLOCKCHAIN CONFIRMATION SAFETY               ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Block:  [N-20] [N-15] [N-12] [N-5]  [N-1] [N] (latest) ║
║           │      │      │      │       │     │            ║
║  Safety:  ████████████████████████████▓▓▓▓▓▓░░░          ║
║           │      │      │      │       │     │            ║
║           Safe   ←────  CONFIRMATIONS ──────→ Risky      ║
║                         (buffer zone)                     ║
║                                                           ║
║  Legend:                                                  ║
║    ████ Safe (confirmed)      Reorg probability < 0.01%  ║
║    ▓▓▓▓ Caution zone          Reorg probability < 1%     ║
║    ░░░░ Danger zone           Reorg probability > 1%     ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Scan Policy:
  ✅ DO: Scan [start ... safe_latest]
  ❌ DON'T: Scan [safe_latest+1 ... latest]
  
Rationale:
  • safe_latest blocks are "finalized enough"
  • Reorg risk < threshold
  • Data integrity guaranteed
```

### 2.4 Production Safe Range Implementation

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class BlockRange:
    """Safe block range for scanning"""
    latest: int          # Chain tip
    safe_latest: int     # Latest safe block
    confirmations: int   # Confirmation buffer
    risk_zone: int       # Blocks in risk zone
    
    @property
    def is_safe(self) -> bool:
        """Is there a safe range to scan?"""
        return self.safe_latest > 0
    
    def __repr__(self):
        return (
            f"BlockRange(latest={self.latest:,}, "
            f"safe={self.safe_latest:,}, "
            f"confirmations={self.confirmations}, "
            f"risk_zone={self.risk_zone})"
        )

class SafeRangeCalculator:
    """Calculate safe block ranges with reorg protection"""
    
    def __init__(self, confirmations: int = 12):
        """
        Args:
            confirmations: Number of blocks to wait for safety
                          Sepolia: 5-12
                          Mainnet: 12-32
        """
        self.confirmations = confirmations
    
    def calculate(self, latest_block: int) -> BlockRange:
        """
        Calculate safe scanning range
        
        Args:
            latest_block: Current chain tip (from eth_blockNumber)
        
        Returns:
            BlockRange with safe_latest calculated
        """
        safe_latest = max(0, latest_block - self.confirmations)
        risk_zone = min(latest_block, self.confirmations)
        
        return BlockRange(
            latest=latest_block,
            safe_latest=safe_latest,
            confirmations=self.confirmations,
            risk_zone=risk_zone
        )
    
    def validate_range(self, start: int, end: int, 
                       latest_block: int) -> tuple[bool, Optional[str]]:
        """
        Validate if a proposed range is safe
        
        Returns:
            (is_valid, error_message)
        """
        safe = self.calculate(latest_block).safe_latest
        
        if end > safe:
            return False, (
                f"Range end {end:,} exceeds safe_latest {safe:,}. "
                f"Wait for {end - safe} more blocks."
            )
        
        if start > end:
            return False, f"Invalid range: start {start:,} > end {end:,}"
        
        if start < 0:
            return False, f"Invalid start block: {start}"
        
        return True, None

# Usage
calc = SafeRangeCalculator(confirmations=12)

latest = 6_234_567
range_info = calc.calculate(latest)
print(range_info)
# BlockRange(latest=6,234,567, safe=6,234,555, confirmations=12, risk_zone=12)

# Validate proposed scan
valid, error = calc.validate_range(6_230_000, 6_234_560, latest)
if valid:
    print("✅ Safe to scan")
else:
    print(f"❌ {error}")
```

---

## 3) Dinamik Pencere Boyutu: Adaptive Algorithms

### 3.1 Static vs Dynamic Window

**Static (Basit):**
```python
CHUNK_SIZE = 1500  # Fixed

for start in range(begin, end, CHUNK_SIZE):
    # Always same size
    scan_logs(start, start + CHUNK_SIZE)
```

**Pros:**
- Simple
- Predictable

**Cons:**
- Suboptimal for varying network conditions
- Doesn't adapt to rate limits
- Misses performance opportunities

**Dynamic (Optimal):**
```python
chunk_size = 1500  # Initial

for start in range(begin, end, chunk_size):
    success, latency = scan_logs(start, start + chunk_size)
    
    # Adapt based on feedback
    if success and latency < 1000:
        chunk_size = min(chunk_size * 1.5, MAX_CHUNK)  # Grow
    elif not success:
        chunk_size = max(chunk_size // 2, MIN_CHUNK)   # Shrink
```

**Pros:**
- Adapts to network conditions
- Maximizes throughput
- Handles rate limits gracefully

### 3.2 Adaptive Window Algorithm

**AIMD (Additive Increase, Multiplicative Decrease)**

```python
class AdaptiveWindowManager:
    """
    Adaptive window sizing for getLogs
    
    Strategy: AIMD (TCP-like congestion control)
    - Success → Additive increase (slow growth)
    - Failure → Multiplicative decrease (fast backoff)
    """
    
    def __init__(self, 
                 initial_size: int = 1500,
                 min_size: int = 256,
                 max_size: int = 5000,
                 growth_factor: float = 1.2,
                 shrink_factor: float = 0.5):
        """
        Args:
            initial_size: Starting window size (blocks)
            min_size: Minimum window (safety floor)
            max_size: Maximum window (provider limit)
            growth_factor: Multiplicative increase on success
            shrink_factor: Multiplicative decrease on failure
        """
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        
        # Metrics
        self.success_count = 0
        self.failure_count = 0
        self.total_requests = 0
        self.avg_latency = 0.0
    
    def get_current_size(self) -> int:
        """Get current window size"""
        return int(self.current_size)
    
    def record_success(self, latency_ms: float):
        """
        Record successful request
        
        Strategy:
        - If latency is good (<1s), grow window
        - If latency is high (>2s), don't grow
        - Update moving average latency
        """
        self.success_count += 1
        self.total_requests += 1
        
        # Update moving average (exponential)
        alpha = 0.2
        self.avg_latency = alpha * latency_ms + (1 - alpha) * self.avg_latency
        
        # Grow window if latency is acceptable
        if latency_ms < 1000:  # <1s is good
            self.current_size = min(
                self.current_size * self.growth_factor,
                self.max_size
            )
        elif latency_ms > 2000:  # >2s is slow
            # Don't grow, maybe even shrink slightly
            self.current_size = max(
                self.current_size * 0.9,
                self.min_size
            )
    
    def record_failure(self, error_type: str):
        """
        Record failed request
        
        Strategy:
        - Multiplicative decrease (fast backoff)
        - Different backoff for different errors
        """
        self.failure_count += 1
        self.total_requests += 1
        
        if error_type in ["timeout", "429", "too_many_results"]:
            # Aggressive backoff
            self.current_size = max(
                self.current_size * self.shrink_factor,
                self.min_size
            )
        else:
            # Conservative backoff (might be transient)
            self.current_size = max(
                self.current_size * 0.8,
                self.min_size
            )
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        success_rate = (
            self.success_count / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "current_size": self.get_current_size(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "success_rate": round(success_rate, 3),
            "avg_latency_ms": round(self.avg_latency, 1)
        }

# Usage example
manager = AdaptiveWindowManager(
    initial_size=1500,
    min_size=256,
    max_size=5000
)

for start in range(5_000_000, 5_100_000, manager.get_current_size()):
    end = min(start + manager.get_current_size(), 5_100_000)
    
    try:
        t0 = time.perf_counter()
        logs = fetch_logs(start, end)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        manager.record_success(latency_ms)
        print(f"✅ [{start:,}-{end:,}] {len(logs)} logs, "
              f"{latency_ms:.0f}ms, "
              f"next_window={manager.get_current_size()}")
    
    except TimeoutError:
        manager.record_failure("timeout")
        print(f"❌ Timeout, shrinking to {manager.get_current_size()}")
    
    except RateLimitError:
        manager.record_failure("429")
        time.sleep(2)  # Backoff
        print(f"❌ Rate limit, backing off")

# Final stats
print("\n📊 Performance Summary:")
stats = manager.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")
```

### 3.3 Rate Limiting Strategies

**Token Bucket Algorithm:**

```python
import time
from collections import deque

class TokenBucket:
    """
    Token bucket rate limiter
    
    Classic algorithm for rate limiting:
    - Tokens replenish at fixed rate
    - Each request consumes 1 token
    - Burst capacity = bucket size
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens per second (e.g., 10 = 10 req/s)
            capacity: Bucket size (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> tuple[bool, float]:
        """
        Try to consume tokens
        
        Returns:
            (success, wait_time)
        """
        now = time.time()
        elapsed = now - self.last_update
        
        # Replenish tokens
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0
        else:
            # Calculate wait time
            needed = tokens - self.tokens
            wait = needed / self.rate
            return False, wait
    
    def wait_if_needed(self, tokens: int = 1):
        """Block until tokens available"""
        success, wait = self.consume(tokens)
        if not success:
            time.sleep(wait)
            self.consume(tokens)  # Consume after wait

# Example: 10 requests/second, burst of 20
limiter = TokenBucket(rate=10.0, capacity=20)

for i in range(100):
    limiter.wait_if_needed()  # Blocks if needed
    response = make_rpc_call()
```

**Exponential Backoff:**

```python
class ExponentialBackoff:
    """
    Exponential backoff with jitter
    
    Used for retrying failed requests:
    - First retry: wait 1s
    - Second retry: wait 2s
    - Third retry: wait 4s
    - ...
    - Max wait: 60s
    """
    
    def __init__(self, 
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 multiplier: float = 2.0,
                 jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt = 0
    
    def get_delay(self) -> float:
        """Calculate delay for current attempt"""
        delay = min(
            self.base_delay * (self.multiplier ** self.attempt),
            self.max_delay
        )
        
        # Add jitter (randomness to avoid thundering herd)
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # [0.5, 1.0] * delay
        
        return delay
    
    def wait(self):
        """Wait for current attempt's delay"""
        delay = self.get_delay()
        time.sleep(delay)
        self.attempt += 1
    
    def reset(self):
        """Reset attempt counter after success"""
        self.attempt = 0

# Usage
backoff = ExponentialBackoff()

for attempt in range(5):
    try:
        response = risky_rpc_call()
        backoff.reset()  # Success!
        break
    except RateLimitError:
        print(f"Attempt {attempt+1} failed, backing off...")
        backoff.wait()
```

---

## 4) Reorg Dayanıklılığı: Detection + Recovery

### 4.1 Tail Re-scan Strategy

**Mantık:** Her scan'de son N bloğu **tekrar** tara

```
Scan 1 (initial):
  [5_000_000 ────────────── 5_010_000] (safe_latest)
   └─────────────┬─────────────┘
            Process & store

Scan 2 (next run):
  Overlap zone: [5_009_988 ─ 5_010_000]  (TAIL_RESYNC=12)
                     └────────┬────────┘
                       Re-scan to catch reorg
  New range:    [5_009_988 ──────────── 5_020_000]

Why?
  • If block 5_009_995 reorged between scans
  • Re-scan catches new logs
  • Idempotent insert handles duplicates
```

**Implementation:**

```python
class TailRescanManager:
    """
    Manage tail re-scan for reorg protection
    
    Strategy:
    - Track last scanned block
    - On next run, go back TAIL_RESYNC blocks
    - Re-scan overlap to catch reorg changes
    """
    
    def __init__(self, tail_resync: int = 12):
        """
        Args:
            tail_resync: Number of blocks to re-scan
                        Should equal CONFIRMATIONS
        """
        self.tail_resync = tail_resync
    
    def get_scan_start(self, last_scanned: Optional[int], 
                       absolute_start: int = 0) -> int:
        """
        Calculate where to start next scan
        
        Args:
            last_scanned: Last successfully scanned block
            absolute_start: Absolute minimum (e.g., contract deployment)
        
        Returns:
            start_block for next scan
        """
        if last_scanned is None:
            return absolute_start
        
        # Go back TAIL_RESYNC blocks
        rescan_from = last_scanned - self.tail_resync + 1
        
        # But don't go below absolute start
        return max(rescan_from, absolute_start)
    
    def calculate_overlap(self, last_scanned: int, 
                          new_start: int) -> tuple[int, int]:
        """
        Calculate overlap zone
        
        Returns:
            (overlap_start, overlap_end)
        """
        overlap_start = new_start
        overlap_end = last_scanned
        overlap_size = overlap_end - overlap_start + 1
        
        return overlap_start, overlap_end, overlap_size

# Usage
manager = TailRescanManager(tail_resync=12)

# First run
last_scanned = None
start = manager.get_scan_start(last_scanned, absolute_start=5_000_000)
print(f"Scan 1 start: {start:,}")  # 5,000,000

# Process blocks 5M - 5.01M, save last_scanned=5_010_000

# Second run (next day)
last_scanned = 5_010_000
start = manager.get_scan_start(last_scanned, absolute_start=5_000_000)
print(f"Scan 2 start: {start:,}")  # 5,009,989 (went back 12)

overlap = manager.calculate_overlap(5_010_000, start)
print(f"Overlap: {overlap[0]:,} - {overlap[1]:,} ({overlap[2]} blocks)")
# Overlap: 5,009,989 - 5,010,000 (12 blocks)
```

### 4.2 Reorg Detection

**Method 1: Block Hash Comparison**

```python
def detect_reorg(conn, block_number: int, 
                 current_hash: str) -> bool:
    """
    Detect reorg by comparing block hash
    
    Args:
        conn: Database connection
        block_number: Block to check
        current_hash: Current hash from chain
    
    Returns:
        True if reorg detected
    """
    # Get stored hash
    row = conn.execute("""
        SELECT block_hash 
        FROM blocks 
        WHERE number = ?
    """, [block_number]).fetchone()
    
    if not row:
        return False  # No previous record
    
    stored_hash = row[0]
    
    if stored_hash != current_hash.lower():
        print(f"⚠️  REORG detected at block {block_number:,}")
        print(f"   Stored:  {stored_hash[:16]}...")
        print(f"   Current: {current_hash[:16]}...")
        return True
    
    return False

# Usage in scan loop
for block_num in range(start, end):
    block = eth_getBlockByNumber(block_num)
    current_hash = block["hash"]
    
    if detect_reorg(conn, block_num, current_hash):
        # Reorg detected! Re-process this block
        delete_logs_from_block(conn, block_num)
        # Continue with new data...
```

**Method 2: Parent Hash Chain Validation**

```python
def validate_chain_continuity(conn, start: int, end: int) -> list[int]:
    """
    Validate parent hash chain
    
    Returns:
        List of block numbers where chain breaks (reorg points)
    """
    reorg_points = []
    
    for block_num in range(start + 1, end + 1):
        # Get current block
        current = eth_getBlockByNumber(block_num)
        parent_hash = current["parentHash"]
        
        # Get previous block from DB
        prev = conn.execute("""
            SELECT block_hash 
            FROM blocks 
            WHERE number = ?
        """, [block_num - 1]).fetchone()
        
        if prev and prev[0] != parent_hash.lower():
            reorg_points.append(block_num)
            print(f"⚠️  Chain break at {block_num:,}")
            print(f"   Expected parent: {prev[0][:16]}...")
            print(f"   Actual parent:   {parent_hash[:16]}...")
    
    return reorg_points
```

### 4.3 Reorg Recovery Strategy

```python
class ReorgRecovery:
    """
    Handle blockchain reorganizations
    
    Strategy:
    1. Detect reorg (hash mismatch)
    2. Find reorg depth (how far back?)
    3. Delete affected data
    4. Re-scan from reorg point
    """
    
    def __init__(self, conn, rpc_url: str):
        self.conn = conn
        self.rpc_url = rpc_url
    
    def find_reorg_depth(self, suspected_block: int, 
                         max_depth: int = 100) -> Optional[int]:
        """
        Binary search to find reorg point
        
        Returns:
            Block number where reorg started, or None
        """
        # Check backwards to find where chain matches
        for depth in range(1, max_depth + 1):
            check_block = suspected_block - depth
            if check_block < 0:
                return None
            
            # Get hashes
            current_hash = self._get_current_hash(check_block)
            stored_hash = self._get_stored_hash(check_block)
            
            if current_hash == stored_hash:
                # Found matching block!
                return check_block + 1  # Reorg starts at next block
        
        return None  # Reorg deeper than max_depth
    
    def _get_current_hash(self, block_num: int) -> str:
        """Fetch current hash from chain"""
        block = eth_getBlockByNumber(self.rpc_url, block_num)
        return block["hash"].lower()
    
    def _get_stored_hash(self, block_num: int) -> Optional[str]:
        """Get stored hash from DB"""
        row = self.conn.execute("""
            SELECT block_hash FROM blocks WHERE number = ?
        """, [block_num]).fetchone()
        return row[0] if row else None
    
    def recover(self, reorg_start: int):
        """
        Recover from reorg
        
        1. Delete data from reorg_start onwards
        2. Update scan state to reorg_start - 1
        3. Ready to re-scan
        """
        print(f"🔄 Recovering from reorg at block {reorg_start:,}")
        
        # Delete affected logs
        deleted_logs = self.conn.execute("""
            DELETE FROM transfers 
            WHERE block_number >= ?
            RETURNING COUNT(*)
        """, [reorg_start]).fetchone()[0]
        
        print(f"   Deleted {deleted_logs:,} logs")
        
        # Delete affected blocks
        deleted_blocks = self.conn.execute("""
            DELETE FROM blocks 
            WHERE number >= ?
            RETURNING COUNT(*)
        """, [reorg_start]).fetchone()[0]
        
        print(f"   Deleted {deleted_blocks:,} blocks")
        
        # Update scan state
        self.conn.execute("""
            UPDATE scan_state 
            SET last_scanned_block = ?
            WHERE key = 'transfers_v1'
        """, [reorg_start - 1])
        
        self.conn.commit()
        print(f"   ✅ Reset to block {reorg_start - 1:,}")
        print(f"   Ready to re-scan from {reorg_start:,}")

# Usage
recovery = ReorgRecovery(conn, rpc_url)

# Detect reorg
if detect_reorg(conn, 5_010_000, current_hash):
    # Find depth
    reorg_start = recovery.find_reorg_depth(5_010_000)
    
    if reorg_start:
        # Recover
        recovery.recover(reorg_start)
        
        # Continue scanning from reorg_start
        scan_logs(reorg_start, safe_latest)
```

---

## 5) State Modeli: Database Design + Transaction Safety

### 5.1 State Table Schema

```sql
-- Scan state tracking
CREATE TABLE IF NOT EXISTS scan_state (
    key TEXT PRIMARY KEY,              -- State identifier (e.g., "transfers_v1")
    last_scanned_block BIGINT NOT NULL, -- Last successfully scanned block
    last_scanned_hash TEXT,            -- Block hash (for reorg detection)
    updated_at TIMESTAMP NOT NULL,     -- Last update time
    created_at TIMESTAMP NOT NULL,     -- First scan time
    metadata JSON                      -- Additional metadata (optional)
);

-- Block metadata (for reorg detection)
CREATE TABLE IF NOT EXISTS blocks (
    number BIGINT PRIMARY KEY,         -- Block number
    block_hash TEXT NOT NULL UNIQUE,   -- Block hash
    parent_hash TEXT NOT NULL,         -- Parent block hash
    timestamp BIGINT NOT NULL,         -- Unix timestamp
    tx_count INTEGER,                  -- Transaction count
    log_count INTEGER,                 -- Log count (our scope)
    processed_at TIMESTAMP NOT NULL    -- When we processed this block
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(block_hash);
CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks(timestamp);
```

### 5.2 Transactional State Updates

**Problem:** Crash между scan и state update → data loss

**Solution:** Atomic transactions

```python
def scan_and_update_atomically(conn, start: int, end: int):
    """
    Scan logs and update state in single transaction
    
    Either both succeed or both fail (atomicity)
    """
    try:
        # Start transaction
        conn.execute("BEGIN TRANSACTION")
        
        # 1. Fetch logs from chain
        logs = fetch_logs(start, end)
        
        # 2. Parse and prepare data
        parsed = [parse_log(log) for log in logs]
        
        # 3. Insert logs (idempotent)
        insert_logs_idempotent(conn, parsed)
        
        # 4. Update blocks table
        for block_num in range(start, end + 1):
            block = get_block_info(block_num)
            conn.execute("""
                INSERT OR REPLACE INTO blocks 
                (number, block_hash, parent_hash, timestamp, processed_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                block_num,
                block["hash"],
                block["parentHash"],
                block["timestamp"]
            ])
        
        # 5. Update scan state (critical!)
        conn.execute("""
            INSERT OR REPLACE INTO scan_state 
            (key, last_scanned_block, last_scanned_hash, updated_at, created_at)
            VALUES (
                'transfers_v1', 
                ?, 
                ?,
                CURRENT_TIMESTAMP,
                COALESCE(
                    (SELECT created_at FROM scan_state WHERE key = 'transfers_v1'),
                    CURRENT_TIMESTAMP
                )
            )
        """, [end, get_block_hash(end)])
        
        # Commit transaction
        conn.commit()
        
        print(f"✅ Scanned and saved {start:,} - {end:,} ({len(logs)} logs)")
        return True
        
    except Exception as e:
        # Rollback on any error
        conn.rollback()
        print(f"❌ Transaction failed: {e}")
        print(f"   Rolled back, state unchanged")
        return False
```

### 5.3 State Recovery After Crash

```python
def recover_state(conn) -> dict:
    """
    Recover scan state after crash/restart
    
    Returns:
        {
            'last_scanned_block': int,
            'last_scanned_hash': str,
            'need_rescan': bool,
            'rescan_from': int
        }
    """
    # Get last state
    row = conn.execute("""
        SELECT last_scanned_block, last_scanned_hash, updated_at
        FROM scan_state
        WHERE key = 'transfers_v1'
    """).fetchone()
    
    if not row:
        return {
            'last_scanned_block': None,
            'last_scanned_hash': None,
            'need_rescan': False,
            'rescan_from': 0
        }
    
    last_block, last_hash, updated_at = row
    
    # Verify hash (detect reorg during downtime)
    current_hash = get_current_block_hash(last_block)
    
    if current_hash != last_hash:
        print(f"⚠️  State hash mismatch (reorg during downtime)")
        print(f"   Stored:  {last_hash[:16]}...")
        print(f"   Current: {current_hash[:16]}...")
        
        # Find reorg point
        recovery = ReorgRecovery(conn, rpc_url)
        reorg_start = recovery.find_reorg_depth(last_block)
        
        return {
            'last_scanned_block': last_block,
            'last_scanned_hash': last_hash,
            'need_rescan': True,
            'rescan_from': reorg_start or 0
        }
    
    return {
        'last_scanned_block': last_block,
        'last_scanned_hash': last_hash,
        'need_rescan': False,
        'rescan_from': last_block + 1
    }

# Usage at startup
state = recover_state(conn)

if state['need_rescan']:
    print(f"🔄 Reorg detected, re-scanning from {state['rescan_from']:,}")
    start_block = state['rescan_from']
else:
    if state['last_scanned_block']:
        print(f"✅ Resuming from {state['last_scanned_block']:,}")
        start_block = state['last_scanned_block'] + 1
    else:
        print(f"🆕 Starting fresh scan")
        start_block = GENESIS_BLOCK
```

---

## 6) Production Implementation: Complete Pipeline

```python
#!/usr/bin/env python3
"""
Production-grade getLogs scanner with:
- Adaptive window sizing
- Reorg protection
- State tracking
- Error recovery
"""

import os
import time
import requests
from typing import Optional, List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv
import duckdb

# Configuration
load_dotenv()
RPC_URL = os.getenv("RPC_URL")
DB_PATH = "onchain.duckdb"
TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
CONFIRMATIONS = 12
INITIAL_CHUNK = 1500
MIN_CHUNK = 256
MAX_CHUNK = 5000
TAIL_RESYNC = 12

@dataclass
class ScanMetrics:
    """Scan performance metrics"""
    total_blocks: int = 0
    total_logs: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    start_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0
    
    def print_summary(self):
        print("\n" + "="*60)
        print("📊 Scan Performance Summary")
        print("="*60)
        print(f"Blocks scanned:     {self.total_blocks:,}")
        print(f"Logs collected:     {self.total_logs:,}")
        print(f"Requests made:      {self.total_requests:,}")
        print(f"Success rate:       {self.success_rate:.1%}")
        print(f"Avg latency:        {self.avg_latency_ms:.0f}ms")
        print(f"Total time:         {self.elapsed_seconds:.1f}s")
        if self.elapsed_seconds > 0:
            blocks_per_sec = self.total_blocks / self.elapsed_seconds
            print(f"Throughput:         {blocks_per_sec:.1f} blocks/s")
        print("="*60)

class ProductionScanner:
    """
    Production-grade blockchain log scanner
    
    Features:
    - Adaptive window sizing
    - Reorg protection (tail re-scan)
    - State tracking (resume capability)
    - Error recovery (exponential backoff)
    - Performance metrics
    """
    
    def __init__(self, rpc_url: str, db_path: str):
        self.rpc_url = rpc_url
        self.db_path = db_path
        self.conn = self._init_db()
        
        # Adaptive window
        self.chunk_size = INITIAL_CHUNK
        self.min_chunk = MIN_CHUNK
        self.max_chunk = MAX_CHUNK
        
        # Metrics
        self.metrics = ScanMetrics()
        
        # Backoff
        self.backoff_delay = 1.0
        self.max_backoff = 60.0
    
    def _init_db(self):
        """Initialize database schema"""
        conn = duckdb.connect(self.db_path)
        
        # Transfers table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transfers (
                block_number BIGINT,
                block_time TIMESTAMP,
                tx_hash TEXT,
                log_index INTEGER,
                token TEXT,
                from_addr TEXT,
                to_addr TEXT,
                raw_value DECIMAL(38,0),
                value_unit DOUBLE,
                UNIQUE(tx_hash, log_index)
            )
        """)
        
        # Blocks table (reorg detection)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                number BIGINT PRIMARY KEY,
                block_hash TEXT NOT NULL,
                parent_hash TEXT,
                timestamp BIGINT,
                processed_at TIMESTAMP
            )
        """)
        
        # State table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_state (
                key TEXT PRIMARY KEY,
                last_scanned_block BIGINT,
                last_scanned_hash TEXT,
                updated_at TIMESTAMP
            )
        """)
        
        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_block ON transfers(block_number)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_from ON transfers(from_addr)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_to ON transfers(to_addr)")
        
        return conn
    
    def rpc_call(self, method: str, params: list, timeout: int = 30) -> dict:
        """Make RPC call with error handling"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        r = requests.post(self.rpc_url, json=payload, timeout=timeout)
        r.raise_for_status()
        
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"RPC error: {data['error']}")
        
        return data["result"]
    
    def get_latest_block(self) -> int:
        """Get latest block number"""
        hex_num = self.rpc_call("eth_blockNumber", [])
        return int(hex_num, 16)
    
    def get_safe_latest(self) -> int:
        """Get safe block number (with confirmation buffer)"""
        latest = self.get_latest_block()
        return max(0, latest - CONFIRMATIONS)
    
    def get_last_scanned(self) -> Optional[int]:
        """Get last scanned block from state"""
        row = self.conn.execute("""
            SELECT last_scanned_block 
            FROM scan_state 
            WHERE key = 'transfers_v1'
        """).fetchone()
        
        return row[0] if row else None
    
    def set_last_scanned(self, block: int, block_hash: str):
        """Update last scanned block"""
        self.conn.execute("""
            INSERT OR REPLACE INTO scan_state 
            (key, last_scanned_block, last_scanned_hash, updated_at)
            VALUES ('transfers_v1', ?, ?, CURRENT_TIMESTAMP)
        """, [block, block_hash])
        self.conn.commit()
    
    def fetch_logs(self, start: int, end: int) -> tuple[List[dict], float]:
        """
        Fetch logs for block range
        
        Returns:
            (logs, latency_ms)
        """
        t0 = time.perf_counter()
        
        logs = self.rpc_call("eth_getLogs", [{
            "fromBlock": hex(start),
            "toBlock": hex(end),
            "topics": [TRANSFER_SIG]
        }])
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return logs, latency_ms
    
    def parse_log(self, log: dict) -> dict:
        """Parse Transfer log"""
        topics = log["topics"]
        
        from_addr = "0x" + topics[1][-40:]
        to_addr = "0x" + topics[2][-40:]
        raw_value = int(log["data"], 16)
        value_unit = raw_value / 1e18  # Assuming 18 decimals
        
        return {
            "block_number": int(log["blockNumber"], 16),
            "tx_hash": log["transactionHash"].lower(),
            "log_index": int(log["logIndex"], 16),
            "token": log["address"].lower(),
            "from_addr": from_addr.lower(),
            "to_addr": to_addr.lower(),
            "raw_value": raw_value,
            "value_unit": value_unit
        }
    
    def insert_logs_idempotent(self, logs: List[dict]):
        """Insert logs with idempotency (anti-join)"""
        if not logs:
            return
        
        # Create staging table
        self.conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS _staging AS 
            SELECT * FROM transfers WHERE 1=0
        """)
        
        # Clear staging
        self.conn.execute("DELETE FROM _staging")
        
        # Insert to staging
        for log in logs:
            # Get block timestamp (cache this in production!)
            block_hex = hex(log["block_number"])
            block = self.rpc_call("eth_getBlockByNumber", [block_hex, False])
            ts = int(block["timestamp"], 16)
            block_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
            
            self.conn.execute("""
                INSERT INTO _staging VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                log["block_number"],
                block_time,
                log["tx_hash"],
                log["log_index"],
                log["token"],
                log["from_addr"],
                log["to_addr"],
                log["raw_value"],
                log["value_unit"]
            ])
        
        # Anti-join insert
        inserted = self.conn.execute("""
            INSERT INTO transfers
            SELECT s.*
            FROM _staging s
            LEFT JOIN transfers t 
                ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
            WHERE t.tx_hash IS NULL
            RETURNING COUNT(*)
        """).fetchone()[0]
        
        self.conn.commit()
        
        return inserted
    
    def adapt_chunk_size(self, success: bool, latency_ms: float):
        """Adapt chunk size based on feedback"""
        if success:
            if latency_ms < 1000:  # Fast response
                self.chunk_size = min(
                    int(self.chunk_size * 1.2),
                    self.max_chunk
                )
                self.backoff_delay = 1.0  # Reset backoff
            elif latency_ms > 2000:  # Slow response
                self.chunk_size = max(
                    int(self.chunk_size * 0.9),
                    self.min_chunk
                )
        else:  # Failure
            self.chunk_size = max(
                int(self.chunk_size * 0.5),
                self.min_chunk
            )
            # Exponential backoff
            self.backoff_delay = min(
                self.backoff_delay * 2,
                self.max_backoff
            )
    
    def scan(self, start_block: Optional[int] = None, 
             end_block: Optional[int] = None):
        """
        Main scan loop
        
        Args:
            start_block: Override start (default: resume from state)
            end_block: Override end (default: safe_latest)
        """
        # Initialize metrics
        self.metrics = ScanMetrics()
        self.metrics.start_time = time.time()
        
        # Determine range
        if start_block is None:
            last = self.get_last_scanned()
            start_block = (last - TAIL_RESYNC + 1) if last else 5_000_000
            start_block = max(0, start_block)
        
        if end_block is None:
            end_block = self.get_safe_latest()
        
        print(f"\n🔍 Starting scan: {start_block:,} → {end_block:,}")
        print(f"   Window: {self.chunk_size} blocks")
        print(f"   Confirmations: {CONFIRMATIONS}")
        print(f"   Tail resync: {TAIL_RESYNC}\n")
        
        current = start_block
        
        while current <= end_block:
            chunk_end = min(current + self.chunk_size - 1, end_block)
            
            try:
                # Fetch logs
                logs, latency_ms = self.fetch_logs(current, chunk_end)
                
                # Parse
                parsed = [self.parse_log(log) for log in logs]
                
                # Insert (idempotent)
                inserted = self.insert_logs_idempotent(parsed)
                
                # Update metrics
                self.metrics.total_blocks += (chunk_end - current + 1)
                self.metrics.total_logs += len(parsed)
                self.metrics.total_requests += 1
                self.metrics.successful_requests += 1
                self.metrics.total_latency_ms += latency_ms
                
                # Update state
                block_hash = self.rpc_call("eth_getBlockByNumber", 
                                           [hex(chunk_end), False])["hash"]
                self.set_last_scanned(chunk_end, block_hash)
                
                # Adapt window
                self.adapt_chunk_size(True, latency_ms)
                
                # Progress
                progress = (chunk_end - start_block) / (end_block - start_block) * 100
                print(f"✅ [{current:,}-{chunk_end:,}] "
                      f"{len(parsed):>4} logs | "
                      f"{latency_ms:>5.0f}ms | "
                      f"window={self.chunk_size:>4} | "
                      f"{progress:>5.1f}%")
                
                # Move forward
                current = chunk_end + 1
                
            except requests.exceptions.Timeout:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.adapt_chunk_size(False, 0)
                
                print(f"❌ Timeout [{current:,}-{chunk_end:,}], "
                      f"shrinking to {self.chunk_size}, "
                      f"backing off {self.backoff_delay:.1f}s")
                
                time.sleep(self.backoff_delay)
            
            except Exception as e:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.adapt_chunk_size(False, 0)
                
                print(f"❌ Error [{current:,}-{chunk_end:,}]: {e}")
                print(f"   Backing off {self.backoff_delay:.1f}s...")
                
                time.sleep(self.backoff_delay)
            
            # Rate limiting (be nice to provider)
            time.sleep(0.05)  # 50ms between requests
        
        # Final summary
        self.metrics.print_summary()

# Main execution
if __name__ == "__main__":
    scanner = ProductionScanner(RPC_URL, DB_PATH)
    
    # Scan from last state to current safe block
    scanner.scan()
```

---

## 7) Performance Optimization: Benchmarking + Tuning

### 7.1 Chunk Size Benchmarking

**Experiment:** Test different chunk sizes

```python
def benchmark_chunk_sizes(rpc_url: str, 
                          start: int, 
                          end: int,
                          chunk_sizes: List[int]) -> dict:
    """
    Benchmark different chunk sizes
    
    Returns:
        {chunk_size: {latency_ms, requests, logs, errors}}
    """
    results = {}
    
    for chunk_size in chunk_sizes:
        print(f"\n🧪 Testing chunk_size={chunk_size}")
        
        total_requests = 0
        total_latency = 0.0
        total_logs = 0
        errors = 0
        
        current = start
        while current <= end:
            chunk_end = min(current + chunk_size - 1, end)
            
            try:
                t0 = time.perf_counter()
                logs = fetch_logs(rpc_url, current, chunk_end)
                latency = (time.perf_counter() - t0) * 1000
                
                total_requests += 1
                total_latency += latency
                total_logs += len(logs)
                
                print(f"  [{current:,}-{chunk_end:,}] "
                      f"{len(logs)} logs, {latency:.0f}ms")
                
            except Exception as e:
                errors += 1
                print(f"  ❌ Error: {e}")
            
            current = chunk_end + 1
            time.sleep(0.1)  # Rate limit
        
        avg_latency = total_latency / total_requests if total_requests > 0 else 0
        
        results[chunk_size] = {
            'requests': total_requests,
            'avg_latency_ms': round(avg_latency, 1),
            'total_logs': total_logs,
            'errors': errors
        }
    
    return results

# Run benchmark
results = benchmark_chunk_sizes(
    rpc_url=RPC_URL,
    start=5_000_000,
    end=5_010_000,  # 10K blocks
    chunk_sizes=[500, 1000, 1500, 2000, 3000]
)

# Print results
print("\n" + "="*70)
print("📊 Chunk Size Benchmark Results")
print("="*70)
print(f"{'Chunk':<8} {'Requests':<10} {'Avg Latency':<15} {'Logs':<10} {'Errors'}")
print("-"*70)

for chunk_size, stats in sorted(results.items()):
    print(f"{chunk_size:<8} {stats['requests']:<10} "
          f"{stats['avg_latency_ms']:<15.0f} "
          f"{stats['total_logs']:<10} {stats['errors']}")

print("="*70)

# Recommend optimal
optimal = min(results.items(), 
              key=lambda x: x[1]['avg_latency_ms'] if x[1]['errors'] == 0 else float('inf'))

print(f"\n✅ Recommended chunk_size: {optimal[0]}")
print(f"   Avg latency: {optimal[1]['avg_latency_ms']:.0f}ms")
print(f"   Total requests: {optimal[1]['requests']}")
```

**Example output:**

```
======================================================================
📊 Chunk Size Benchmark Results
======================================================================
Chunk    Requests   Avg Latency     Logs       Errors
----------------------------------------------------------------------
500      20         450.3           12450      0
1000     10         720.5           12450      0
1500     7          950.2           12450      0
2000     5          1250.8          12450      0
3000     4          1820.5          12450      2
======================================================================

✅ Recommended chunk_size: 1500
   Avg latency: 950.2ms
   Total requests: 7
```

### 7.2 Network Latency Impact

```python
def analyze_latency_distribution(conn) -> dict:
    """
    Analyze latency distribution from logs
    
    Assumes you've been logging request metrics
    """
    # Query latency data
    rows = conn.execute("""
        SELECT 
            request_latency_ms,
            chunk_size,
            log_count,
            success
        FROM request_log
        WHERE timestamp > NOW() - INTERVAL 1 HOUR
        ORDER BY request_latency_ms
    """).fetchall()
    
    latencies = [r[0] for r in rows if r[3]]  # Success only
    
    if not latencies:
        return {}
    
    # Calculate percentiles
    import statistics
    
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
    
    return {
        'count': len(latencies),
        'min_ms': min(latencies),
        'p50_ms': p50,
        'p95_ms': p95,
        'p99_ms': p99,
        'max_ms': max(latencies),
        'mean_ms': statistics.mean(latencies),
        'stdev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }

# Example usage
stats = analyze_latency_distribution(conn)

print("📊 Latency Distribution (last 1 hour)")
print(f"  Requests: {stats['count']}")
print(f"  Min:      {stats['min_ms']:.0f}ms")
print(f"  p50:      {stats['p50_ms']:.0f}ms")
print(f"  p95:      {stats['p95_ms']:.0f}ms ⭐")
print(f"  p99:      {stats['p99_ms']:.0f}ms")
print(f"  Max:      {stats['max_ms']:.0f}ms")
print(f"  Mean:     {stats['mean_ms']:.0f}ms ± {stats['stdev_ms']:.0f}ms")
```

---

## 8) Real-World Scenarios

### 8.1 Mainnet vs Testnet

| Aspect | Mainnet | Sepolia Testnet |
|--------|---------|-----------------|
| **Block time** | ~12s | ~12s |
| **Traffic** | High (100K+ tx/block) | Low (~100 tx/block) |
| **Reorg risk** | Very low (PoS) | Low (PoS) |
| **Confirmations** | 12-32 | 5-12 |
| **RPC costs** | $$$ (compute units) | Free/cheap |
| **Chunk size** | 500-1000 (high traffic) | 1500-3000 (low traffic) |
| **Rate limits** | Strict | Lenient |

### 8.2 Provider Comparison

**Alchemy:**
- Max results: 10,000 logs
- Max block range: 10,000 blocks
- Rate limit: 300 CU/s (varies by plan)
- Best for: Production (reliable, fast)

**Infura:**
- Max results: 10,000 logs
- Max block range: No hard limit (but timeout)
- Rate limit: 100K req/day (free), unlimited (paid)
- Best for: Development, testing

**QuickNode:**
- Max results: 20,000 logs (plan-dependent)
- Max block range: Flexible
- Rate limit: Plan-dependent
- Best for: High-performance needs

**Public nodes:**
- Max results: Varies (often 1,000-5,000)
- Rate limit: Aggressive (IP-based)
- Best for: Prototyping only

### 8.3 Production Deployment Checklist

```markdown
## Pre-Production Checklist

### Infrastructure
- [ ] Dedicated RPC endpoint (not public node)
- [ ] Database backup strategy
- [ ] Monitoring + alerting (Prometheus/Grafana)
- [ ] Log aggregation (Loki/CloudWatch)

### Configuration
- [ ] CONFIRMATIONS tuned for network
- [ ] Chunk size benchmarked for provider
- [ ] Rate limits configured (token bucket)
- [ ] Backoff parameters tested

### Reliability
- [ ] State persistence tested (crash recovery)
- [ ] Reorg detection validated
- [ ] Idempotency verified (duplicate runs)
- [ ] Error handling comprehensive

### Performance
- [ ] Latency p95 < 2s
- [ ] Success rate > 99%
- [ ] Throughput: >500 blocks/min
- [ ] Memory usage stable (<1GB)

### Observability
- [ ] Metrics exported (blocks/s, logs/s, errors)
- [ ] Dashboards created
- [ ] Alerts configured (high error rate, stuck state)
- [ ] Logs searchable
```

---

## 9) Error Taxonomy + Handling

### 9.1 Common Errors

| Error Type | Cause | Solution |
|------------|-------|----------|
| **Timeout** | Chunk too large, network slow | Shrink chunk, retry |
| **429 (Rate Limit)** | Too many requests | Backoff, slow down |
| **"query too large"** | Result set > provider limit | Shrink chunk |
| **"invalid range"** | fromBlock > toBlock | Fix logic |
| **Connection reset** | Network instability | Retry with backoff |
| **Invalid block** | Block not yet available | Wait, retry |
| **Reorg detected** | Chain reorganization | Delete affected data, re-scan |

### 9.2 Error Recovery Patterns

```python
def robust_fetch_logs(rpc_url: str, start: int, end: int, 
                      max_retries: int = 3) -> List[dict]:
    """
    Fetch logs with comprehensive error handling
    
    Returns:
        List of logs (may be empty on failure)
    """
    backoff = ExponentialBackoff(base_delay=1.0, max_delay=60.0)
    
    for attempt in range(max_retries):
        try:
            logs = fetch_logs(rpc_url, start, end)
            return logs
        
        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                backoff.wait()
            else:
                print(f"❌ Max retries reached, giving up")
                return []
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                print(f"⚠️  Rate limited (attempt {attempt+1}/{max_retries})")
                backoff.wait()
            elif e.response.status_code >= 500:  # Server error
                print(f"⚠️  Server error {e.response.status_code}")
                backoff.wait()
            else:
                print(f"❌ HTTP error {e.response.status_code}, giving up")
                return []
        
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            if "query returned more than" in error_msg:
                print(f"❌ Query too large, need smaller chunk")
                return []  # Caller should shrink chunk
            
            elif "invalid" in error_msg:
                print(f"❌ Invalid request: {e}")
                return []  # Don't retry invalid requests
            
            else:
                print(f"⚠️  Runtime error: {e}")
                if attempt < max_retries - 1:
                    backoff.wait()
                else:
                    return []
        
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {e}")
            return []
    
    return []
```

---

## 10) Testing Strategies

### 10.1 Unit Tests

```python
import unittest
from unittest.mock import Mock, patch

class TestSafeRangeCalculator(unittest.TestCase):
    def test_basic_calculation(self):
        calc = SafeRangeCalculator(confirmations=12)
        range_info = calc.calculate(latest_block=1000)
        
        self.assertEqual(range_info.latest, 1000)
        self.assertEqual(range_info.safe_latest, 988)
        self.assertEqual(range_info.confirmations, 12)
    
    def test_zero_block(self):
        calc = SafeRangeCalculator(confirmations=12)
        range_info = calc.calculate(latest_block=5)
        
        # Should not go negative
        self.assertEqual(range_info.safe_latest, 0)
    
    def test_validate_safe_range(self):
        calc = SafeRangeCalculator(confirmations=12)
        
        # Safe range
        valid, error = calc.validate_range(900, 988, latest_block=1000)
        self.assertTrue(valid)
        self.assertIsNone(error)
        
        # Unsafe range (beyond safe_latest)
        valid, error = calc.validate_range(900, 995, latest_block=1000)
        self.assertFalse(valid)
        self.assertIn("exceeds safe_latest", error)

class TestAdaptiveWindow(unittest.TestCase):
    def test_growth_on_success(self):
        manager = AdaptiveWindowManager(initial_size=1000)
        initial = manager.get_current_size()
        
        # Fast response → grow
        manager.record_success(latency_ms=500)
        
        self.assertGreater(manager.get_current_size(), initial)
    
    def test_shrink_on_failure(self):
        manager = AdaptiveWindowManager(initial_size=1000)
        initial = manager.get_current_size()
        
        # Timeout → shrink
        manager.record_failure("timeout")
        
        self.assertLess(manager.get_current_size(), initial)
    
    def test_min_max_bounds(self):
        manager = AdaptiveWindowManager(
            initial_size=1000,
            min_size=256,
            max_size=5000
        )
        
        # Many failures → should not go below min
        for _ in range(10):
            manager.record_failure("timeout")
        
        self.assertEqual(manager.get_current_size(), 256)
        
        # Many successes → should not exceed max
        for _ in range(20):
            manager.record_success(latency_ms=100)
        
        self.assertEqual(manager.get_current_size(), 5000)

if __name__ == "__main__":
    unittest.main()
```

### 10.2 Integration Tests

```python
def test_full_scan_pipeline():
    """
    Integration test: Full scan pipeline
    
    Tests:
    - State persistence
    - Idempotency
    - Resume capability
    """
    # Clean slate
    if os.path.exists("test.duckdb"):
        os.remove("test.duckdb")
    
    # First scan (blocks 5M - 5M+1K)
    scanner1 = ProductionScanner(RPC_URL, "test.duckdb")
    scanner1.scan(start_block=5_000_000, end_block=5_001_000)
    
    # Check state
    last = scanner1.get_last_scanned()
    assert last == 5_001_000, f"Expected 5001000, got {last}"
    
    # Count logs
    count1 = scanner1.conn.execute(
        "SELECT COUNT(*) FROM transfers"
    ).fetchone()[0]
    
    print(f"✅ First scan: {count1} logs")
    
    # Second scan (should resume and go further)
    scanner2 = ProductionScanner(RPC_URL, "test.duckdb")
    scanner2.scan(end_block=5_002_000)  # No start → resumes
    
    # Check state updated
    last2 = scanner2.get_last_scanned()
    assert last2 == 5_002_000, f"Expected 5002000, got {last2}"
    
    # Count should increase
    count2 = scanner2.conn.execute(
        "SELECT COUNT(*) FROM transfers"
    ).fetchone()[0]
    
    assert count2 >= count1, "Count should not decrease"
    
    print(f"✅ Second scan: {count2} logs (+{count2 - count1})")
    
    # Third scan (re-scan same range → idempotency)
    scanner3 = ProductionScanner(RPC_URL, "test.duckdb")
    scanner3.scan(start_block=5_000_000, end_block=5_001_000)
    
    # Count should be same (idempotent)
    count3 = scanner3.conn.execute(
        "SELECT COUNT(*) FROM transfers"
    ).fetchone()[0]
    
    assert count3 == count2, f"Idempotency broken: {count2} → {count3}"
    
    print(f"✅ Re-scan: {count3} logs (idempotent ✓)")
    
    # Cleanup
    os.remove("test.duckdb")
    
    print("\n✅ All integration tests passed!")

# Run
test_full_scan_pipeline()
```

---

## 11) Quiz + Ödevler + Troubleshooting

### Mini Quiz (10 Soru)

1. `safe_latest` neden `latest - CONFIRMATIONS` şeklinde hesaplanır?
2. Adaptive window'da latency yüksek olunca ne olur?
3. Tail re-scan'in amacı nedir?
4. İdempotency key neden `(tx_hash, log_index)` çifti olmalı?
5. Rate limiting için hangi iki algoritma kullanılır?
6. Reorg detection için hangi iki yöntem vardır?
7. Chunk size çok büyük olursa ne tür hatalar alırsın?
8. Exponential backoff'ta jitter neden kullanılır?
9. State update neden transaction içinde yapılmalı?
10. p95 latency nedir ve neden önemlidir?

### Cevap Anahtarı

1. Son N blok reorg riski taşır, confirmed block'lara odaklanmak için
2. Window shrink olmaz (stable kalır), growth durur
3. Reorg olmuş block'ları tekrar tarayıp güncellemek
4. Bir tx'te birden fazla log olabilir, log_index tekliği sağlar
5. Token bucket (burst capacity) ve exponential backoff (retry)
6. (a) Block hash comparison, (b) Parent hash chain validation
7. Timeout, 429 Rate Limit, "query too large" error
8. Thundering herd'den kaçınmak (many clients retry aynı anda)
9. Crash olursa data/state inconsistency olmasın (atomicity)
10. 95th percentile latency; tipik kullanıcı deneyimini gösterir

### Ödevler (6 Pratik)

#### Ödev 1: Chunk Size Benchmark
```python
# Kendi RPC provider'ında 5 farklı chunk size test et
# [500, 1000, 1500, 2000, 3000]
# Her biri için:
#   - Avg latency
#   - Success rate
#   - Error count
# Tablolaştır, optimal'i bul
```

#### Ödev 2: Reorg Simulation
```python
# Manuel reorg simülasyonu:
# 1. Bir range tara (e.g., 5M-5.001M)
# 2. DB'de son block'un hash'ini değiştir (fake reorg)
# 3. Tail re-scan ile tekrar tara
# 4. Reorg detection tetikleniyor mu?
# 5. Recovery çalışıyor mu?
```

#### Ödev 3: Adaptive Window Tuning
```python
# AdaptiveWindowManager parametrelerini test et:
# - growth_factor: [1.1, 1.2, 1.5]
# - shrink_factor: [0.3, 0.5, 0.7]
# Hangisi en stabil throughput veriyor?
```

#### Ödev 4: Error Recovery Test
```python
# RPC provider'ı kasıtlı olarak rate limit'e çek:
# - Çok hızlı request gönder (no sleep)
# - 429 alınca exponential backoff çalışıyor mu?
# - Recovery time ne kadar?
# - Final success rate?
```

#### Ödev 5: State Persistence Test
```python
# Crash simulation:
# 1. Scan başlat (5M-5.1M)
# 2. %50'de kill et (Ctrl+C)
# 3. Restart et
# 4. Kaldığı yerden devam ediyor mu?
# 5. Duplicate log var mı? (idempotency check)
```

#### Ödev 6: Performance Dashboard
```python
# Basit monitoring dashboard yap:
# - Real-time: blocks/s, logs/s
# - Latency: p50, p95, p99
# - Errors: rate, types
# - Window: current size, trend
# Plotly/Streamlit ile görselleştir
```

---

## 12) Troubleshooting Rehberi

### Problem 1: "Query returned more than 10000 results"

**Belirtiler:**
```
RuntimeError: query returned more than 10000 results
```

**Çözüm:**
```python
# Chunk size'ı küçült
CHUNK_SIZE = 1000  # 1500'den düşür

# Veya spesifik token filtrele
filter_obj["address"] = "0xTokenAddress"  # Daha az log
```

### Problem 2: Scan Yavaşladı (Throughput Düştü)

**Belirtiler:**
- Başta 1000 block/min, sonra 100 block/min
- Latency yükseliyor

**Debug:**
```python
# Metrics'i analiz et
stats = manager.get_stats()
print(stats)

# Latency dağılımına bak
# p95 > 2000ms ise provider problemi
```

**Çözüm:**
- RPC provider değiştir
- Chunk size küçült
- Rate limiting ekle (token bucket)

### Problem 3: Çift Kayıt (Duplicate Logs)

**Belirtiler:**
```sql
SELECT tx_hash, log_index, COUNT(*) 
FROM transfers 
GROUP BY tx_hash, log_index 
HAVING COUNT(*) > 1;
-- Returns duplicates
```

**Çözüm:**
```sql
-- UNIQUE constraint ekle
ALTER TABLE transfers 
ADD CONSTRAINT unique_log UNIQUE(tx_hash, log_index);

-- Veya anti-join kullan (yukarıdaki kod)
```

### Problem 4: State Stuck (İlerlemiyor)

**Belirtiler:**
- `last_scanned_block` değişmiyor
- Scan loop dönüyor ama state update edilmiyor

**Debug:**
```python
# Transaction commit ediliyor mu?
# Log ekle
print(f"Updating state to {block_num}")
conn.execute("UPDATE scan_state SET last_scanned_block = ?", [block_num])
conn.commit()  # ← Bu var mı?
print(f"State updated!")
```

### Problem 5: Memory Leak

**Belirtiler:**
- Uzun scan'lerde memory kullanımı artıyor
- Process kill ediliyor (OOM)

**Çözüm:**
```python
# Batch processing ile memory temizle
for start in range(begin, end, CHUNK_SIZE):
    logs = fetch_logs(start, start + CHUNK_SIZE)
    process_logs(logs)
    
    logs = None  # Explicit cleanup
    gc.collect()  # Force garbage collection
    
    # Periyodik DB vacuum
    if start % 100_000 == 0:
        conn.execute("VACUUM")
```

---

## 13) Terimler Sözlüğü (Genişletilmiş)

| Terim | Tanım |
|-------|-------|
| **Window/Chunk** | getLogs aralığını parçalayan blok genişliği |
| **Reorg** | Blockchain reorganization; son blokların değişmesi |
| **Confirmations** | Blok güvenlik tamponu (N adet bekle) |
| **Tail re-scan** | Son N bloğu her koşuda tekrar tarama |
| **Idempotent** | Tekrarlanan işlem aynı sonucu verir |
| **Safe latest** | `latest - CONFIRMATIONS`; reorg-safe block |
| **AIMD** | Additive Increase, Multiplicative Decrease |
| **Token bucket** | Rate limiting algoritması (burst capacity) |
| **Exponential backoff** | Retry delay'i katlanarak artırma |
| **Anti-join** | SQL pattern: yeni kayıtları filtreleme |
| **State tracking** | İlerleme kaydetme (resume capability) |
| **p95 latency** | 95th percentile response time |

---

## 14) Definition of Done (Tahta 04)

**Öğrenme Hedefleri:**

- [ ] `safe_latest` kavramını açıklayabiliyorum
- [ ] Adaptive window algoritmasını kodlayabiliyorum
- [ ] Tail re-scan stratejisini uygulayabiliyorum
- [ ] Reorg detection/recovery yapabiliyorum
- [ ] State persistence'ı transaction-safe implement edebiliyorum
- [ ] Rate limiting (token bucket) kullanabiliyorum
- [ ] Exponential backoff ile retry logic yazabiliyorum
- [ ] Production scanner'ı deploy edebiliyorum
- [ ] Performance metrics'i topluyorum ve analiz edebiliyorum
- [ ] Common errors'ı tanıyıp çözebiliyorum

**Pratik Çıktılar:**

- [ ] Chunk size benchmark tamamlandı
- [ ] Reorg simulation test edildi
- [ ] Full scan pipeline (5M-5.1M blocks) başarılı
- [ ] Crash recovery test edildi (idempotency ✓)
- [ ] Production code repo'da çalışır halde

---

## 🔗 İlgili Dersler

- **← Tahta 03:** [Transfer Anatomisi](03_tahta_transfer_anatomi.md) (topics/data parsing)
- **→ Tahta 05:** [DuckDB + İdempotent Yazma](05_tahta_duckdb_idempotent.md) (Coming)
- **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

## 🛡️ Güvenlik / Etik

- **Read-only:** Özel anahtar yok, imza yok, custody yok
- **`.env` hygiene:** API keys asla commit etme
- **Testnet-first:** Sepolia ile başla
- **Rate limiting:** Provider'a saygılı ol (DoS yok)
- **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 📌 Navigasyon

- **→ Sonraki:** [05 - DuckDB + İdempotent Yazma](05_tahta_duckdb_idempotent.md) (Coming)
- **← Önceki:** [03 - Transfer Anatomisi](03_tahta_transfer_anatomi.md)
- **↑ Ana Sayfa:** [Week 0 Bootstrap](../../../crypto/w0_bootstrap/README.md)

---

**Tahta 04 — getLogs Pencere Stratejisi + Reorg Dayanıklılığı**  
*Format: Production Deep-Dive*  
*Süre: 60-75 dk*  
*Prerequisite: Tahta 01-03*  
*Versiyon: 2.0 (Complete Expansion)*
