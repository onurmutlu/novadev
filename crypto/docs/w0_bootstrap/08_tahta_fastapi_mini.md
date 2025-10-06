# 🧑‍🏫 Tahta 08 — FastAPI Mini Servis: Production-Grade HTTP API

> **Amaç:** `report_v1` JSON'unu HTTP üstünden servis etmek: **/healthz** ve **/wallet/{address}/report** endpoints, p95<1s hedefi, LRU+TTL cache, concurrency safety, observability.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (Genişletilmiş Tahta)

1. **FastAPI architecture** (Why FastAPI? vs Flask/Django)
2. **Service structure** (Layered design, separation of concerns)
3. **Dependency injection** (DuckDB connection pool, thread-safety)
4. **Pydantic models** (Request validation, response typing)
5. **Caching strategies** (LRU+TTL, cache key design, eviction)
6. **Endpoints implementation** (/healthz, /wallet/{addr}/report)
7. **Performance optimization** (p95<1s, benchmarking)
8. **Concurrency patterns** (Thread-local connections, async/sync)
9. **Error handling** (4xx/5xx, detailed messages, logging)
10. **Observability** (Metrics, timing, structured logging)
11. **Testing strategies** (Unit, integration, contract, performance)
12. **Deployment** (Uvicorn, Docker, process management)
13. **Troubleshooting guide** (10 common issues)
14. **Quiz + ödevler**

---

## 1) FastAPI Architecture: Why FastAPI?

### 1.1 Framework Comparison

```
╔════════════════════════════════════════════════════════════╗
║           PYTHON WEB FRAMEWORK COMPARISON                  ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Flask (Micro-framework):                                 ║
║    ✅ Simple, flexible                                     ║
║    ✅ Large ecosystem                                      ║
║    ❌ No built-in validation                               ║
║    ❌ Manual async handling                                ║
║    ❌ No automatic OpenAPI docs                            ║
║    Use case: Small projects, prototypes                   ║
║                                                            ║
║  Django (Full-stack):                                     ║
║    ✅ Batteries included (ORM, admin, auth)               ║
║    ✅ Mature ecosystem                                     ║
║    ❌ Heavy for APIs                                       ║
║    ❌ Django-specific patterns                             ║
║    Use case: Full web apps with DB/admin                  ║
║                                                            ║
║  FastAPI ⭐ RECOMMENDED:                                  ║
║    ✅ Modern Python 3.7+ (type hints)                      ║
║    ✅ Automatic validation (Pydantic)                      ║
║    ✅ Automatic OpenAPI/Swagger docs                       ║
║    ✅ Async support (native)                               ║
║    ✅ High performance (Starlette + Uvicorn)               ║
║    ✅ Dependency injection                                 ║
║    Use case: Modern REST APIs ⭐                           ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 1.2 FastAPI Core Concepts

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI()

# 1. Automatic validation (Pydantic)
class Item(BaseModel):
    name: str
    price: float

# 2. Dependency injection
def get_db():
    return Database()

# 3. Type hints → OpenAPI schema
@app.post("/items/")
def create_item(item: Item, db = Depends(get_db)):
    return {"item": item, "db": db}

# Result:
# - Automatic request validation
# - Automatic response serialization
# - Automatic OpenAPI docs at /docs
# - Type safety
```

### 1.3 Our Architecture

```
╔════════════════════════════════════════════════════════════╗
║              NOVADEV API ARCHITECTURE                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Client (HTTP)                                            ║
║    ↓                                                       ║
║  ┌────────────────────────────────────────────────────┐   ║
║  │ FastAPI (Uvicorn ASGI server)                      │   ║
║  └─────────────────┬──────────────────────────────────┘   ║
║                    ↓                                       ║
║  ┌────────────────────────────────────────────────────┐   ║
║  │ Middleware Layer                                   │   ║
║  │  • Timing (request duration)                       │   ║
║  │  • Error handling (500 → structured)               │   ║
║  │  • CORS (optional)                                 │   ║
║  │  • Rate limiting (future)                          │   ║
║  └─────────────────┬──────────────────────────────────┘   ║
║                    ↓                                       ║
║  ┌────────────────────────────────────────────────────┐   ║
║  │ Routes                                             │   ║
║  │  • /healthz                                        │   ║
║  │  • /wallet/{addr}/report                           │   ║
║  └─────────────────┬──────────────────────────────────┘   ║
║                    ↓                                       ║
║  ┌────────────────────────────────────────────────────┐   ║
║  │ Dependencies (Injected)                            │   ║
║  │  • DuckDB connection pool (thread-local)           │   ║
║  │  • Cache (LRU+TTL, thread-safe)                    │   ║
║  │  • Config (environment)                            │   ║
║  └─────────────────┬──────────────────────────────────┘   ║
║                    ↓                                       ║
║  ┌────────────────────────────────────────────────────┐   ║
║  │ Business Logic                                     │   ║
║  │  • ReportBuilder (Tahta 07)                        │   ║
║  │  • ReportValidator (Tahta 07)                      │   ║
║  └─────────────────┬──────────────────────────────────┘   ║
║                    ↓                                       ║
║  ┌────────────────────────────────────────────────────┐   ║
║  │ Data Layer                                         │   ║
║  │  • DuckDB (read-only)                              │   ║
║  │  • onchain.duckdb                                  │   ║
║  └────────────────────────────────────────────────────┘   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 2) Service Structure: Layered Design

### 2.1 Directory Layout

```
crypto/service/
├── __init__.py
├── app.py                  # FastAPI app initialization
├── config.py               # Settings (Pydantic BaseSettings)
├── deps.py                 # Dependency injection (DB pool, cache)
├── models.py               # Pydantic models (request/response)
├── cache.py                # LRU+TTL cache implementation
├── middleware.py           # Custom middleware (timing, etc.)
├── monitoring.py           # Metrics, structured logging
├── routes/
│   ├── __init__.py
│   ├── health.py           # GET /healthz
│   └── wallet.py           # GET /wallet/{addr}/report
└── exceptions.py           # Custom exceptions
```

### 2.2 Layered Design Principles

```
╔════════════════════════════════════════════════════════════╗
║            LAYERED ARCHITECTURE PRINCIPLES                 ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Layer 1: HTTP (Routes)                                   ║
║    Responsibility: Request/response handling              ║
║    Dependencies: Pydantic models, dependencies            ║
║    Rules:                                                 ║
║      - Thin layer (no business logic)                     ║
║      - Validation via Pydantic                            ║
║      - Error handling via FastAPI                         ║
║                                                            ║
║  Layer 2: Business Logic (Services)                       ║
║    Responsibility: Core functionality                     ║
║    Dependencies: Data layer (DB, cache)                   ║
║    Rules:                                                 ║
║      - Framework-agnostic (no FastAPI imports)            ║
║      - Testable in isolation                              ║
║      - Pure Python                                        ║
║                                                            ║
║  Layer 3: Data (Repository)                               ║
║    Responsibility: Data access                            ║
║    Dependencies: DuckDB, cache                            ║
║    Rules:                                                 ║
║      - Single source of truth                             ║
║      - Connection management                              ║
║      - Thread-safety                                      ║
║                                                            ║
║  Cross-cutting: Middleware, Monitoring                    ║
║    Responsibility: Observability, security                ║
║    Applied: All layers                                    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 3) Dependency Injection: DuckDB Connection Pool

### 3.1 Thread-Local Connection Pattern

```python
"""
Connection pool using thread-local storage

Why thread-local?
- DuckDB connections are NOT thread-safe
- Each thread needs its own connection
- FastAPI runs sync endpoints in thread pool
- Solution: Thread-local storage

Pattern:
1. Each thread gets one connection (lazy init)
2. Connection persists for thread lifetime
3. Read-only mode (no write conflicts)
"""

import duckdb
import threading
from typing import Optional

class DuckDBConnectionPool:
    """
    Thread-local connection pool for DuckDB
    
    Features:
    - One connection per thread
    - Lazy initialization
    - Read-only mode
    - Thread-safe
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False
    
    def init(self):
        """Initialize pool (call once at startup)"""
        with self._lock:
            if self._initialized:
                return
            
            # Test connection
            try:
                conn = duckdb.connect(self.db_path, read_only=True)
                conn.execute("SELECT 1").fetchone()
                conn.close()
                self._initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to initialize DB pool: {e}")
    
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get connection for current thread
        
        Returns:
            DuckDB connection (thread-local)
        
        Note:
            Creates new connection on first call per thread
        """
        if not self._initialized:
            raise RuntimeError("Pool not initialized. Call init() first.")
        
        # Get or create thread-local connection
        conn = getattr(self.local, 'conn', None)
        
        if conn is None:
            conn = duckdb.connect(self.db_path, read_only=True)
            self.local.conn = conn
        
        return conn
    
    def close_all(self):
        """Close all connections (call at shutdown)"""
        # Note: Thread-local connections will be GC'd
        # This is mostly for explicit cleanup
        pass


# Global pool instance
_pool: Optional[DuckDBConnectionPool] = None


def init_db_pool(db_path: str):
    """Initialize global pool (call at startup)"""
    global _pool
    _pool = DuckDBConnectionPool(db_path)
    _pool.init()


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """
    FastAPI dependency: Get DB connection
    
    Usage:
        @app.get("/...")
        def endpoint(conn = Depends(get_db_connection)):
            result = conn.execute("SELECT ...").fetchall()
    """
    if _pool is None:
        raise RuntimeError("DB pool not initialized")
    
    return _pool.get_connection()
```

### 3.2 Alternative: Connection-Per-Request

```python
"""
Alternative pattern: New connection per request

Pros:
- Simple (no threading concerns)
- Isolated (each request independent)

Cons:
- Overhead (connection creation)
- Slower (~10-50ms per connection)

Use when:
- Very light traffic
- Connection pooling not critical
"""

def get_db_connection_per_request():
    """Create new connection per request"""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()

# Usage
@app.get("/...")
def endpoint(conn = Depends(get_db_connection_per_request)):
    # conn is fresh, will be closed after request
    pass
```

---

## 4) Pydantic Models: Type-Safe API

### 4.1 Request Models

```python
"""
Pydantic models for request validation

Benefits:
- Automatic validation
- Type hints → OpenAPI schema
- Detailed error messages
- Easy to test
"""

from pydantic import BaseModel, Field, validator
import re

# Regex for Ethereum address
ETHEREUM_ADDRESS_PATTERN = re.compile(r'^0x[a-fA-F0-9]{40}$')


class WalletReportQuery(BaseModel):
    """Query parameters for /wallet/{address}/report"""
    
    hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Time window in hours (1-720, default: 24)"
    )
    
    @validator('hours')
    def validate_hours(cls, v):
        """Additional validation (optional)"""
        if v not in [1, 6, 12, 24, 48, 168, 720]:
            # Warning: not error, just logging
            import logging
            logging.warning(f"Non-standard hours value: {v}")
        return v


class WalletAddressPath(BaseModel):
    """Path parameter validation for wallet address"""
    
    address: str = Field(
        ...,
        regex=r'^0x[a-fA-F0-9]{40}$',
        description="Ethereum wallet address (0x...)"
    )
    
    def normalize(self) -> str:
        """Return lowercase normalized address"""
        return self.address.lower()


class HealthResponse(BaseModel):
    """Response model for /healthz"""
    
    status: str = Field(..., description="Health status (ok/degraded/down)")
    uptime_seconds: float = Field(..., description="Process uptime in seconds")
    db_status: str = Field(default="ok", description="Database connection status")
    cache_size: int = Field(default=0, description="Cache entry count")


# Note: Report response uses dict (validated by ReportValidator)
# Alternative: Define full Pydantic model mirroring JSON schema
```

### 4.2 Full Report Response Model (Optional)

```python
"""
Optional: Full Pydantic model for report response

Pros:
- Type-safe response
- Automatic OpenAPI schema
- Client SDK generation

Cons:
- Duplication with JSON schema
- Maintenance burden
"""

class TransferStat(BaseModel):
    token: str
    symbol: str
    decimals: int
    inbound: float
    outbound: float
    tx_count: int

class Counterparty(BaseModel):
    address: str
    count: int

class ReportTime(BaseModel):
    from_ts: str
    to_ts: str

class ReportTotals(BaseModel):
    inbound: float
    outbound: float

class ReportMeta(BaseModel):
    chain_id: int
    generated_at: str
    source: str
    notes: str = ""

class WalletReportResponse(BaseModel):
    """Full report response (mirrors schemas/report_v1.json)"""
    version: str = "v1"
    wallet: str
    window_hours: int
    time: ReportTime
    totals: ReportTotals
    tx_count: int
    transfer_stats: list[TransferStat]
    top_counterparties: list[Counterparty]
    meta: ReportMeta
```

---

## 5) Caching Strategies: LRU+TTL

### 5.1 Cache Design

```
╔════════════════════════════════════════════════════════════╗
║                 CACHE ARCHITECTURE                         ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Cache Key Design:                                        ║
║    key = f"{wallet.lower()}|{hours}"                      ║
║    Example: "0xabc...def|24"                              ║
║                                                            ║
║  Cache Entry:                                             ║
║    {                                                       ║
║      "data": {...},  # Report dict                        ║
║      "timestamp": 1696680000.0,  # Unix timestamp         ║
║      "hits": 3  # Cache hit count (optional)              ║
║    }                                                       ║
║                                                            ║
║  Eviction Policy:                                         ║
║    1. TTL: Entry expires after N seconds                  ║
║    2. LRU: When capacity reached, evict least recent      ║
║                                                            ║
║  Thread Safety:                                           ║
║    • threading.Lock on all operations                     ║
║    • OrderedDict for LRU (move_to_end)                    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 5.2 Production Cache Implementation

```python
"""
Production-grade LRU+TTL cache

Features:
- Thread-safe (Lock)
- TTL-based expiration
- LRU eviction
- Hit/miss tracking
- Size management
"""

import time
import threading
from collections import OrderedDict
from typing import Optional, Any, Dict
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """Single cache entry"""
    data: Any
    timestamp: float
    hits: int = 0


class TTLLRUCache:
    """
    Thread-safe cache with TTL and LRU eviction
    
    Usage:
        cache = TTLLRUCache(capacity=1024, ttl_seconds=60)
        
        # Set
        cache.set("key", {"data": "value"})
        
        # Get
        value = cache.get("key")  # Returns dict or None
        
        # Stats
        stats = cache.stats()
    """
    
    def __init__(self, capacity: int = 1024, ttl_seconds: int = 60):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _now(self) -> float:
        """Current timestamp"""
        return time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            age = self._now() - entry.timestamp
            if age > self.ttl_seconds:
                # Expired
                self._cache.pop(key)
                self._misses += 1
                return None
            
            # Hit!
            entry.hits += 1
            self._hits += 1
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            
            return entry.data
    
    def set(self, key: str, data: Any):
        """
        Set value in cache
        
        Args:
            key: Cache key
            data: Data to cache (any JSON-serializable object)
        """
        with self._lock:
            # Create entry
            entry = CacheEntry(
                data=data,
                timestamp=self._now(),
                hits=0
            )
            
            # Update or insert
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if over capacity
            if len(self._cache) > self.capacity:
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
    
    def invalidate(self, key: str) -> bool:
        """
        Remove key from cache
        
        Returns:
            True if key was present
        """
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            {
                'size': int,
                'capacity': int,
                'hits': int,
                'misses': int,
                'hit_rate': float,
                'evictions': int
            }
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'capacity': self.capacity,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions
            }
    
    def size(self) -> int:
        """Current cache size"""
        with self._lock:
            return len(self._cache)


# Global cache instance
_cache: Optional[TTLLRUCache] = None


def init_cache(capacity: int = 2048, ttl_seconds: int = 60):
    """Initialize global cache"""
    global _cache
    _cache = TTLLRUCache(capacity, ttl_seconds)


def get_cache() -> TTLLRUCache:
    """FastAPI dependency: Get cache"""
    if _cache is None:
        raise RuntimeError("Cache not initialized")
    return _cache
```

### 5.3 Cache Key Strategy

```python
"""
Cache key design patterns

Key format: "{wallet}|{hours}|{version}"

Examples:
- "0xabc...def|24|v1"
- "0x123...456|168|v1"

Benefits:
- Unique per (wallet, timeframe, schema)
- Lowercase wallet (normalized)
- Version prefix for schema changes
"""

def make_cache_key(wallet: str, hours: int, version: str = "v1") -> str:
    """Generate cache key"""
    return f"{wallet.lower()}|{hours}|{version}"

def parse_cache_key(key: str) -> tuple[str, int, str]:
    """Parse cache key (for debugging)"""
    wallet, hours, version = key.split("|")
    return wallet, int(hours), version
```

---

## 6) Endpoints Implementation

### 6.1 Health Endpoint

```python
"""
GET /healthz - Health check endpoint

Purpose:
- Process liveness check (uptime > 0)
- Database connectivity (can query)
- Cache status (size, hit rate)

Returns:
- 200 OK: System healthy
- 503 Service Unavailable: Degraded/down
"""

from fastapi import APIRouter, Depends, status
from ..models import HealthResponse
from ..deps import get_db_connection, get_cache
import time

router = APIRouter(tags=["health"])

# Track startup time
_start_time = time.time()


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API and database health"
)
async def healthz(
    conn = Depends(get_db_connection),
    cache = Depends(get_cache)
):
    """
    Health check endpoint
    
    Checks:
    1. Process uptime (> 0)
    2. Database connection (SELECT 1)
    3. Cache availability
    
    Returns:
        HealthResponse with status and metrics
    """
    uptime = time.time() - _start_time
    
    # Check database
    db_status = "ok"
    try:
        result = conn.execute("SELECT 1").fetchone()
        if result[0] != 1:
            db_status = "error"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    
    # Overall status
    overall_status = "ok" if db_status == "ok" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        uptime_seconds=round(uptime, 2),
        db_status=db_status,
        cache_size=cache.size()
    )
```

### 6.2 Wallet Report Endpoint

```python
"""
GET /wallet/{address}/report?hours=24 - Wallet activity report

Features:
- Path validation (address format)
- Query validation (hours range)
- Cache integration (LRU+TTL)
- Schema validation (report_v1)
- Error handling (422 on invalid input)
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from typing import Dict, Any
import re

from ..models import WalletReportQuery
from ..deps import get_db_connection, get_cache
from ...features.report_builder import ReportBuilder, ReportConfig
from ...features.report_validator import ReportValidator

router = APIRouter(tags=["wallet"])

# Address validation regex
ETHEREUM_ADDRESS_RE = re.compile(r'^0x[a-fA-F0-9]{40}$')

# Initialize builder and validator (singleton)
_builder = None
_validator = None


def get_builder() -> ReportBuilder:
    """Get global report builder"""
    global _builder
    if _builder is None:
        # Note: Builder will use injected DB connection
        _builder = ReportBuilder("dummy.db")  # Path not used (injected)
    return _builder


def get_validator() -> ReportValidator:
    """Get global report validator"""
    global _validator
    if _validator is None:
        _validator = ReportValidator()
    return _validator


@router.get(
    "/wallet/{address}/report",
    summary="Get wallet activity report",
    description="Generate wallet activity report for specified time window",
    response_description="Wallet activity report (schema v1 compliant)"
)
async def wallet_report(
    address: str = Path(
        ...,
        regex=r'^0x[a-fA-F0-9]{40}$',
        description="Ethereum wallet address",
        example="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    ),
    query: WalletReportQuery = Depends(),
    cache = Depends(get_cache),
    conn = Depends(get_db_connection)
) -> Dict[str, Any]:
    """
    Get wallet activity report
    
    Args:
        address: Wallet address (0x...)
        query: Query parameters (hours)
        cache: Cache instance (injected)
        conn: Database connection (injected)
    
    Returns:
        Report dict (schema v1 compliant)
    
    Raises:
        HTTPException(422): Invalid input or report generation failed
    """
    # Normalize address
    address_lower = address.lower()
    
    # Cache key
    cache_key = f"{address_lower}|{query.hours}"
    
    # Check cache
    cached_report = cache.get(cache_key)
    if cached_report is not None:
        return cached_report
    
    # Generate report
    try:
        builder = get_builder()
        
        # Override connection (use injected, not builder's)
        builder.conn = conn
        
        report = builder.build(
            wallet=address_lower,
            window_hours=query.hours
        )
        
        # Validate report
        validator = get_validator()
        validator.validate(report)
        
        # Cache result
        cache.set(cache_key, report)
        
        return report
        
    except ValueError as e:
        # Input validation error
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        # Other errors
        import logging
        logging.error(f"Report generation failed: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail="Report generation failed. Check logs for details."
        )
```

---

## 7) Performance Optimization

### 7.1 Performance Budget

```
╔════════════════════════════════════════════════════════════╗
║              PERFORMANCE BUDGET (p95)                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Target: p95 < 1000ms (API response)                      ║
║                                                            ║
║  Breakdown:                                               ║
║  ┌────────────────────────────────────────────────────┐   ║
║  │ Network (client → server)      ~50ms  (5%)        │   ║
║  ├────────────────────────────────────────────────────┤   ║
║  │ FastAPI routing/validation     ~10ms  (1%)        │   ║
║  ├────────────────────────────────────────────────────┤   ║
║  │ Cache lookup (hit)             ~5ms   (0.5%)      │   ║
║  ├────────────────────────────────────────────────────┤   ║
║  │ Report builder (miss):                            │   ║
║  │   • SQL queries (3)            ~200ms (20%)       │   ║
║  │   • JSON construction          ~30ms  (3%)        │   ║
║  │   • Validation                 ~10ms  (1%)        │   ║
║  ├────────────────────────────────────────────────────┤   ║
║  │ Response serialization         ~20ms  (2%)        │   ║
║  ├────────────────────────────────────────────────────┤   ║
║  │ Network (server → client)      ~50ms  (5%)        │   ║
║  └────────────────────────────────────────────────────┘   ║
║                                                            ║
║  Cache Hit: ~150ms  (network + routing + cache)           ║
║  Cache Miss: ~400ms (above + builder)                     ║
║                                                            ║
║  With 80% cache hit rate:                                 ║
║    p95 = 0.8 × 150ms + 0.2 × 400ms = 200ms ✅             ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 7.2 Benchmarking

```bash
# Install hey (HTTP load testing)
brew install hey

# Benchmark (single wallet, cache cold)
hey -n 100 -c 1 \
  "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24"

# Expected output (cache cold):
# Summary:
#   Total:  42.1234 secs
#   Slowest:  0.4521 secs
#   Fastest:  0.3821 secs
#   Average:  0.4212 secs
#   p95:  0.4485 secs ✅

# Benchmark (cache warm, high concurrency)
hey -n 1000 -c 50 \
  "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24"

# Expected output (cache warm):
# Summary:
#   Total:  3.5678 secs
#   Slowest:  0.0521 secs
#   Fastest:  0.0021 secs
#   Average:  0.0178 secs
#   p95:  0.0395 secs ✅ (100x faster!)
```

### 7.3 Optimization Checklist

```python
"""
Performance optimization checklist

✅ Database:
  - Indexes on block_time, from_addr, to_addr
  - Read-only connection (no lock contention)
  - Efficient SQL (no SELECT *, use WHERE)

✅ Caching:
  - LRU+TTL cache (60s default)
  - Normalized cache keys (lowercase)
  - Thread-safe cache operations

✅ JSON Serialization:
  - Orjson for fast JSON (optional)
  - Pre-validated data (skip redundant checks)

✅ Connection Pooling:
  - Thread-local connections (reuse)
  - No connection overhead per request

✅ Async/Await:
  - Use async def for I/O-bound ops (optional)
  - Sync builder OK (CPU-bound)

❌ Avoid:
  - N+1 queries (use JOINs)
  - Large result sets (LIMIT)
  - Synchronous blocking in async context
  - Global locks (use thread-local)
"""

# Optional: Use orjson for faster JSON
try:
    import orjson
    
    def serialize_json(data):
        return orjson.dumps(data).decode('utf-8')
except ImportError:
    import json
    
    def serialize_json(data):
        return json.dumps(data)
```

---

## 8) Concurrency Patterns

### 8.1 Async vs Sync

```python
"""
FastAPI concurrency patterns

Sync endpoints (def):
- Run in thread pool (via asyncio.to_thread)
- Good for: CPU-bound, blocking I/O
- Example: Our case (DuckDB queries)

Async endpoints (async def):
- Run in event loop
- Good for: I/O-bound, async libraries
- Example: HTTP requests, async DB drivers

Our choice: Sync (def)
Reason: DuckDB is sync, builder is CPU-bound
"""

# ❌ DON'T: Mix sync and async incorrectly
@app.get("/report")
async def report():
    # BAD: Blocking call in async function
    result = conn.execute("SELECT ...").fetchall()  # Blocks event loop!
    return result

# ✅ DO: Use sync endpoint for sync operations
@app.get("/report")
def report():  # sync def, not async
    # GOOD: FastAPI runs in thread pool automatically
    result = conn.execute("SELECT ...").fetchall()
    return result

# ✅ ALTERNATIVE: Explicit async wrapper
@app.get("/report")
async def report():
    import asyncio
    
    def _build_report():
        return conn.execute("SELECT ...").fetchall()
    
    # Run sync function in thread pool
    result = await asyncio.to_thread(_build_report)
    return result
```

### 8.2 Thread Safety Checklist

```
╔════════════════════════════════════════════════════════════╗
║              THREAD SAFETY CHECKLIST                       ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  ✅ Thread-Safe:                                          ║
║    • Thread-local connections (DuckDB)                    ║
║    • Cache with Lock (TTLLRUCache)                        ║
║    • Immutable config (Pydantic Settings)                 ║
║    • Stateless endpoints (no shared state)                ║
║                                                            ║
║  ❌ NOT Thread-Safe (avoid):                              ║
║    • Global mutable state (counters without lock)         ║
║    • Shared DuckDB connection (race conditions)           ║
║    • Non-atomic cache operations                          ║
║                                                            ║
║  Testing:                                                 ║
║    • Stress test: hey -n 1000 -c 50                       ║
║    • Check for race conditions (cache inconsistency)      ║
║    • Monitor errors under load                            ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 9) Error Handling

### 9.1 Error Response Format

```python
"""
Standardized error responses

FastAPI automatically formats exceptions:

422 Unprocessable Entity (validation error):
{
  "detail": [
    {
      "loc": ["query", "hours"],
      "msg": "ensure this value is less than or equal to 720",
      "type": "value_error.number.not_le"
    }
  ]
}

Custom errors:
{
  "detail": "Invalid wallet address format"
}
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom validation error handler
    
    Formats validation errors consistently
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler
    
    Prevents stack traces leaking to clients
    """
    import logging
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )
```

### 9.2 Error Categories

```
╔════════════════════════════════════════════════════════════╗
║                  ERROR CATEGORIES                          ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  4xx Client Errors:                                       ║
║    400 Bad Request: Malformed request                     ║
║    422 Unprocessable Entity: Validation failed            ║
║    429 Too Many Requests: Rate limit exceeded             ║
║    404 Not Found: Invalid endpoint                        ║
║                                                            ║
║  5xx Server Errors:                                       ║
║    500 Internal Server Error: Unexpected error            ║
║    503 Service Unavailable: Database down                 ║
║    504 Gateway Timeout: Request timeout                   ║
║                                                            ║
║  Our Mapping:                                             ║
║    Invalid address format → 422                           ║
║    Invalid hours range → 422                              ║
║    Report generation failed → 500                         ║
║    Database unreachable → 503                             ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 10) Observability: Metrics & Logging

### 10.1 Structured Logging

```python
"""
Structured logging with JSON format

Benefits:
- Machine-readable
- Easy to parse/aggregate
- Searchable in log aggregators
"""

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno', 
                          'module', 'msecs', 'message', 'pathname', 'process', 
                          'processName', 'relativeCreated', 'thread', 'threadName',
                          'exc_info', 'exc_text', 'stack_info']:
                log_obj[key] = value
        
        return json.dumps(log_obj)


# Setup logging
def setup_logging():
    """Configure structured JSON logging"""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
```

### 10.2 Request Timing Middleware

```python
"""
Middleware to track request timing

Measures:
- Total request duration
- Path and method
- Status code
- Logs to JSONL for analysis
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class TimingMiddleware(BaseHTTPMiddleware):
    """Track request timing"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log timing
        logging.info(
            "Request completed",
            extra={
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'duration_ms': round(duration_ms, 2),
                'cache_hit': getattr(request.state, 'cache_hit', None)
            }
        )
        
        # Add timing header
        response.headers['X-Response-Time'] = f"{duration_ms:.2f}ms"
        
        return response


# Add to app
app.add_middleware(TimingMiddleware)
```

### 10.3 Metrics Collection

```python
"""
Metrics collection and export

Metrics:
- Request count (by endpoint, status)
- Request duration (p50, p95, p99)
- Cache hit rate
- Error rate
"""

from dataclasses import dataclass, asdict
from typing import List
import statistics

@dataclass
class RequestMetric:
    timestamp: str
    method: str
    path: str
    status_code: int
    duration_ms: float
    cache_hit: bool

class MetricsCollector:
    """Collect and export metrics"""
    
    def __init__(self):
        self._metrics: List[RequestMetric] = []
        self._lock = threading.Lock()
    
    def record(self, metric: RequestMetric):
        """Record a metric"""
        with self._lock:
            self._metrics.append(metric)
            
            # Keep only last 10k metrics
            if len(self._metrics) > 10000:
                self._metrics = self._metrics[-10000:]
    
    def get_stats(self, path: str = None) -> dict:
        """Get statistics"""
        with self._lock:
            metrics = self._metrics
            
            if path:
                metrics = [m for m in metrics if m.path == path]
            
            if not metrics:
                return {}
            
            durations = [m.duration_ms for m in metrics]
            cache_hits = sum(1 for m in metrics if m.cache_hit)
            errors = sum(1 for m in metrics if m.status_code >= 400)
            
            return {
                'count': len(metrics),
                'duration_p50': statistics.median(durations),
                'duration_p95': statistics.quantiles(durations, n=20)[18],  # 95th percentile
                'duration_p99': statistics.quantiles(durations, n=100)[98],
                'cache_hit_rate': cache_hits / len(metrics) if metrics else 0,
                'error_rate': errors / len(metrics) if metrics else 0
            }

# Global collector
_metrics = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    return _metrics
```

---

## 11) Testing Strategies

### 11.1 Test Pyramid

```
╔════════════════════════════════════════════════════════════╗
║              TESTING PYRAMID (FastAPI)                     ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║              ▲                                            ║
║             ╱ ╲    E2E Tests (5%)                         ║
║            ╱   ╲   • Full stack                           ║
║           ╱─────╲  • Real DB                              ║
║          ╱       ╲                                        ║
║         ╱  Contract (10%)                                 ║
║        ╱    • Schema validation                           ║
║       ╱     • OpenAPI compliance                          ║
║      ╱───────────╲                                        ║
║     ╱             ╲                                       ║
║    ╱  Integration  ╲ (25%)                               ║
║   ╱  • TestClient                                         ║
║  ╱   • Mock DB                                            ║
║ ╱    • Route tests                                        ║
║╱─────────────────────╲                                    ║
║                       ╲                                   ║
║  Unit Tests (60%)                                         ║
║  • Cache logic                                            ║
║  • Models                                                 ║
║  • Utilities                                              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 11.2 Unit Tests

```python
"""
Unit tests for cache, models, utilities

Fast, isolated, no external dependencies
"""

import pytest
from crypto.service.cache import TTLLRUCache
import time

def test_cache_basic():
    """Test basic cache operations"""
    cache = TTLLRUCache(capacity=2, ttl_seconds=60)
    
    # Set and get
    cache.set("key1", {"data": "value1"})
    assert cache.get("key1") == {"data": "value1"}
    
    # Miss
    assert cache.get("key2") is None

def test_cache_ttl():
    """Test TTL expiration"""
    cache = TTLLRUCache(capacity=10, ttl_seconds=1)
    
    cache.set("key1", {"data": "value1"})
    assert cache.get("key1") == {"data": "value1"}
    
    # Wait for expiration
    time.sleep(1.5)
    assert cache.get("key1") is None

def test_cache_lru_eviction():
    """Test LRU eviction"""
    cache = TTLLRUCache(capacity=2, ttl_seconds=60)
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")  # Should evict key1
    
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"

def test_pydantic_validation():
    """Test Pydantic model validation"""
    from crypto.service.models import WalletReportQuery
    
    # Valid
    query = WalletReportQuery(hours=24)
    assert query.hours == 24
    
    # Invalid (out of range)
    with pytest.raises(ValueError):
        WalletReportQuery(hours=1000)
```

### 11.3 Integration Tests

```python
"""
Integration tests using TestClient

Tests full request/response cycle
"""

from fastapi.testclient import TestClient
from crypto.service.app import app
import pytest

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

def test_healthz(client):
    """Test health endpoint"""
    response = client.get("/healthz")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["uptime_seconds"] >= 0

def test_wallet_report_valid(client):
    """Test wallet report with valid input"""
    response = client.get(
        "/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report",
        params={"hours": 24}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Schema validation
    assert data["version"] == "v1"
    assert data["wallet"] == "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
    assert data["window_hours"] == 24
    assert "totals" in data
    assert "transfer_stats" in data

def test_wallet_report_invalid_address(client):
    """Test with invalid address"""
    response = client.get(
        "/wallet/invalid_address/report",
        params={"hours": 24}
    )
    
    assert response.status_code == 422

def test_wallet_report_invalid_hours(client):
    """Test with out-of-range hours"""
    response = client.get(
        "/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report",
        params={"hours": 1000}
    )
    
    assert response.status_code == 422

def test_cache_hit(client):
    """Test cache hit on second request"""
    wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    # First request (cold cache)
    response1 = client.get(f"/wallet/{wallet}/report?hours=24")
    assert response1.status_code == 200
    time1 = float(response1.headers.get('X-Response-Time', '0ms').rstrip('ms'))
    
    # Second request (warm cache)
    response2 = client.get(f"/wallet/{wallet}/report?hours=24")
    assert response2.status_code == 200
    time2 = float(response2.headers.get('X-Response-Time', '0ms').rstrip('ms'))
    
    # Cache should be faster
    assert time2 < time1
    
    # Data should be identical
    assert response1.json() == response2.json()
```

### 11.4 Contract Tests

```python
"""
Contract tests: Validate against JSON schema

Ensures API contract compliance
"""

import json
from jsonschema import Draft202012Validator

def test_report_schema_compliance(client):
    """Test that report response complies with schema"""
    # Get report
    response = client.get(
        "/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24"
    )
    assert response.status_code == 200
    report = response.json()
    
    # Load schema
    with open("schemas/report_v1.json") as f:
        schema = json.load(f)
    
    # Validate
    validator = Draft202012Validator(schema)
    validator.validate(report)  # Raises if invalid
```

### 11.5 Performance Tests

```python
"""
Performance tests: Ensure p95 < 1s

Uses pytest-benchmark or manual timing
"""

import pytest
import time

def test_report_performance(client, benchmark):
    """Benchmark report generation"""
    
    def get_report():
        response = client.get(
            "/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24"
        )
        assert response.status_code == 200
        return response
    
    # Run benchmark
    result = benchmark(get_report)
    
    # Assert p95 < 1s
    # Note: benchmark.stats.percentiles[95] available with pytest-benchmark
    assert benchmark.stats.median < 1.0  # 1 second

def test_concurrent_requests(client):
    """Test concurrent request handling"""
    import concurrent.futures
    
    def make_request():
        response = client.get(
            "/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24"
        )
        return response.status_code == 200
    
    # 50 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [f.result() for f in futures]
    
    # All should succeed
    assert all(results)
```

---

## 12) Deployment: Uvicorn & Docker

### 12.1 Uvicorn Configuration

```bash
# Development (auto-reload)
uvicorn crypto.service.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info

# Production (multi-worker)
uvicorn crypto.service.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level warning \
  --access-log \
  --no-use-colors

# With Gunicorn (production)
gunicorn crypto.service.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### 12.2 Dockerfile

```dockerfile
# Multi-stage build for smaller image

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[crypto]" uvicorn[standard]

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY crypto/ crypto/
COPY schemas/ schemas/

# Create non-root user
RUN useradd -m -u 1000 novadev && chown -R novadev:novadev /app
USER novadev

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import requests; requests.get('http://localhost:8000/healthz').raise_for_status()"

# Run
CMD ["uvicorn", "crypto.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 12.3 Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NOVA_DB_PATH=/data/onchain.duckdb
      - NOVA_CACHE_TTL=60
      - NOVA_MAX_WORKERS=4
    volumes:
      - ./data:/data:ro  # Read-only DB mount
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 3s
      retries: 3
```

---

## 13) Troubleshooting Guide

### Problem 1: 422 Invalid Address Format

**Symptoms:**
```json
{
  "detail": "Invalid input: Invalid wallet address: 0xZZZ..."
}
```

**Causes:**
1. Wrong address format (not 0x + 40 hex chars)
2. Mixed case issues (should normalize)
3. ENS name (not supported)

**Solutions:**
```python
# Validate address
import re
ADDR_RE = re.compile(r'^0x[a-fA-F0-9]{40}$')
if not ADDR_RE.match(address):
    raise ValueError("Invalid address")

# Normalize (lowercase)
address = address.lower()
```

---

### Problem 2: 503 Database Unavailable

**Symptoms:**
```json
{
  "status": "degraded",
  "db_status": "error: unable to open database"
}
```

**Causes:**
1. DB file not found
2. Permission denied
3. DB file corrupted

**Solutions:**
```bash
# Check file exists
ls -l $NOVA_DB_PATH

# Check permissions
chmod 644 $NOVA_DB_PATH

# Test connection
python -c "import duckdb; duckdb.connect('$NOVA_DB_PATH', read_only=True).execute('SELECT 1')"
```

---

### Problem 3: High p95 Latency (> 1s)

**Symptoms:**
- API responses slow
- `X-Response-Time` header > 1000ms

**Causes:**
1. Cache disabled/not working
2. Missing indexes
3. Large result sets

**Solutions:**
```python
# Check cache stats
curl http://localhost:8000/healthz
# Look at cache_size

# Verify indexes
SELECT * FROM duckdb_indexes();

# Limit result sets
SELECT ... LIMIT 50
```

---

### Problem 4: Cache Not Working

**Symptoms:**
- All requests slow (no speedup on repeats)
- Cache hit rate = 0%

**Causes:**
1. TTL too short
2. Cache key mismatch
3. Cache not initialized

**Solutions:**
```python
# Check cache initialization
# In app.py startup:
init_cache(capacity=2048, ttl_seconds=60)

# Debug cache keys
print(f"Cache key: {wallet.lower()}|{hours}")

# Increase TTL
export NOVA_CACHE_TTL=300  # 5 minutes
```

---

### Problem 5: Thread Safety Issues

**Symptoms:**
- Random crashes under load
- Database lock errors
- Inconsistent cache

**Causes:**
1. Shared DuckDB connection
2. Cache without lock
3. Global mutable state

**Solutions:**
```python
# Use thread-local connections
class DuckDBConnectionPool:
    def __init__(self, db_path: str):
        self.local = threading.local()
    
    def get_connection(self):
        conn = getattr(self.local, 'conn', None)
        if conn is None:
            conn = duckdb.connect(self.db_path, read_only=True)
            self.local.conn = conn
        return conn

# Use locks in cache
class TTLLRUCache:
    def __init__(self):
        self._lock = threading.Lock()
    
    def get(self, key):
        with self._lock:
            # Thread-safe operation
            pass
```

---

## 14) Mini Quiz (10 Soru)

1. FastAPI'nin Flask'a göre temel avantajları nelerdir?
2. Thread-local connection neden gereklidir?
3. LRU+TTL cache'de eviction stratejisi nedir?
4. p95 < 1s hedefine nasıl ulaşırız?
5. Async vs sync endpoint ne zaman kullanılır?
6. 422 vs 500 error kodları ne zaman döner?
7. Structured logging'in faydaları nelerdir?
8. TestClient ile integration test nasıl yazılır?
9. Uvicorn'da multi-worker mode ne zaman kullanılır?
10. Cache key normalizasyonu neden önemlidir?

### Cevap Anahtarı

1. Otomatik validation (Pydantic), OpenAPI docs, async support, type hints
2. DuckDB connections thread-safe değil, her thread kendi connection'ını almalı
3. TTL: Time-based expiration; LRU: Capacity-based eviction (least recently used)
4. Cache (80%+ hit rate), indexes, optimized queries, read-only connections
5. Async: I/O-bound (HTTP, async DB); Sync: CPU-bound, blocking I/O (DuckDB)
6. 422: Client error (validation); 500: Server error (unexpected)
7. Machine-readable, searchable, aggregatable, parseable (JSON)
8. `client = TestClient(app); response = client.get(...); assert response.status_code == 200`
9. Production, CPU-bound workload, horizontal scaling
10. Case-insensitive matching (0xABC... = 0xabc...), consistent cache hits

---

## 15) Ödevler (6 Praktik)

### Ödev 1: Complete Service Implementation
```
Task: Implement all service files
- crypto/service/app.py
- crypto/service/config.py
- crypto/service/deps.py
- crypto/service/models.py
- crypto/service/cache.py
- crypto/service/routes/health.py
- crypto/service/routes/wallet.py
Test: uvicorn runs, /healthz responds
```

### Ödev 2: Performance Benchmark
```
Task: Benchmark API performance
- Install hey
- Run 100 requests (cold cache)
- Run 100 requests (warm cache)
- Calculate p95 latency
- Document results
Target: p95 < 1s
```

### Ödev 3: Cache Optimization
```
Task: Tune cache parameters
- Test TTL: 30s, 60s, 300s
- Test capacity: 512, 1024, 2048
- Measure hit rate
- Find optimal config
Document: Trade-offs
```

### Ödev 4: Integration Test Suite
```
Task: Write comprehensive tests
- test_healthz
- test_wallet_report_valid
- test_wallet_report_invalid
- test_cache_hit
- test_concurrent_requests (50 threads)
Coverage: > 80%
```

### Ödev 5: Docker Deployment
```
Task: Containerize service
- Write Dockerfile
- Build image
- Run container
- Test endpoints
- Check health check
Deliverable: Working container
```

### Ödev 6: Monitoring Dashboard
```
Task: Build metrics dashboard
- Collect timing metrics (JSONL)
- Plot p95 latency over time
- Plot cache hit rate
- Plot error rate
Tool: Grafana or matplotlib
```

---

## 16) Definition of Done (Tahta 08)

### Learning Objectives
- [ ] FastAPI architecture understanding
- [ ] Dependency injection pattern
- [ ] Thread-safe connection pooling
- [ ] Caching strategies (LRU+TTL)
- [ ] Performance optimization (p95<1s)
- [ ] Concurrency patterns
- [ ] Error handling (4xx/5xx)
- [ ] Observability (metrics, logging)
- [ ] Testing strategies (unit, integration, contract)
- [ ] Deployment (Uvicorn, Docker)

### Practical Outputs
- [ ] Service structure complete (8 files)
- [ ] /healthz endpoint working
- [ ] /wallet/{addr}/report endpoint working
- [ ] Schema v1 validation passing
- [ ] Cache hit rate > 50%
- [ ] p95 latency < 1s (benchmark proof)
- [ ] Integration tests passing (5+ tests)
- [ ] Docker container running
- [ ] OpenAPI docs accessible (/docs)

---

## 🔗 İlgili Dersler

- **← Tahta 07:** [JSON Report + Schema](07_tahta_rapor_json_schema.md)
- **→ Tahta 09:** Quality & CI (Coming)
- **↑ Ana Sayfa:** [Week 0 Bootstrap](../../../crypto/w0_bootstrap/README.md)

---

## 🛡️ Güvenlik / Etik

- **Read-only:** Database connections read-only
- **Input validation:** All inputs validated (Pydantic)
- **Rate limiting:** Consider adding (reverse proxy)
- **CORS:** Disabled by default (enable carefully)
- **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 📌 Navigasyon

- **→ Sonraki:** [09 - Quality & CI](09_tahta_kalite_ci.md) (Coming)
- **← Önceki:** [07 - JSON Report + Schema](07_tahta_rapor_json_schema.md)
- **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

**Tahta 08 — FastAPI Mini Servis: Production-Grade HTTP API**  
*Format: Production Deep-Dive + Complete Code*  
*Süre: 60-75 dk*  
*Prerequisite: Tahta 01-07*  
*Versiyon: 2.0 (Complete Expansion)*  
*Code Examples: 2,000+ lines (docs + implementation)*

