# 🧑‍🏫 Tahta 09 — Quality & CI: Production Test & Automation

> **Amaç:** NovaDev Crypto hattını **güvenle teslim edilebilir** hale getirmek: Test pyramid, contract tests, lint automation, pre-commit hooks, CI/CD workflows, coverage reporting.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (Genişletilmiş Tahta)

1. **Quality culture** (Why quality matters?)
2. **Test pyramid** (Unit → Integration → Contract → E2E → Performance)
3. **pytest setup** (Configuration, fixtures, plugins)
4. **Unit tests** (Business logic, edge cases)
5. **Integration tests** (End-to-end flows)
6. **Contract tests** (JSON Schema validation)
7. **API tests** (FastAPI TestClient)
8. **Performance tests** (Load testing, benchmarks)
9. **Code quality** (Ruff, type checking, coverage)
10. **Pre-commit hooks** (Local feedback loop)
11. **CI/CD workflows** (GitHub Actions, automated testing)
12. **Coverage reporting** (HTML reports, badges)
13. **Troubleshooting** (Common CI issues)
14. **Quiz + ödevler**

---

## 1) Quality Culture: Why Quality Matters?

### 1.1 Cost of Bugs

```
╔════════════════════════════════════════════════════════════╗
║              COST OF BUGS BY PHASE                         ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Phase          Cost Multiplier    Time to Fix            ║
║  ─────────────────────────────────────────────────────    ║
║  Development    1x                 Minutes                ║
║  Testing        10x                Hours                  ║
║  Staging        50x                Days                   ║
║  Production     100-1000x          Weeks + reputation     ║
║                                                            ║
║  Strategy: SHIFT LEFT                                     ║
║    Catch bugs early → Save time + money + reputation      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 1.2 Quality Pyramid

```
╔════════════════════════════════════════════════════════════╗
║                  QUALITY PYRAMID                           ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║              ▲                                            ║
║             ╱ ╲  Manual Testing                           ║
║            ╱   ╲ (5% - spot checks)                       ║
║           ╱─────╲                                         ║
║          ╱       ╲ E2E Tests                              ║
║         ╱  (5%)   ╲ (full flows)                          ║
║        ╱───────────╲                                      ║
║       ╱             ╲ Integration                         ║
║      ╱   (15%)       ╲ (component interaction)            ║
║     ╱─────────────────╲                                   ║
║    ╱                   ╲ Contract Tests                   ║
║   ╱      (15%)          ╱ (API schemas)                   ║
║  ╱───────────────────────╲                                ║
║ ╱                         ╲ Unit Tests                    ║
║╱         (60%)             ╲ (functions, classes)         ║
║─────────────────────────────                              ║
║                                                            ║
║  Foundation: Static Analysis (Ruff, mypy)                 ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 1.3 Our Quality Goals

**Week 0 (Current):**
- ✅ Unit test coverage > 70%
- ✅ All contracts validated (JSON Schema)
- ✅ Lint errors = 0 (Ruff)
- ✅ CI green on every PR

**Week 1-2 (Next):**
- ⭐ Coverage > 85%
- ⭐ Performance benchmarks automated
- ⭐ Integration tests for all workflows

---

## 2) Test Pyramid: Complete Strategy

### 2.1 Test Categories

```
╔════════════════════════════════════════════════════════════╗
║               TEST PYRAMID BREAKDOWN                       ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Level 1: Unit Tests (60%)                                ║
║    What: Individual functions/classes                     ║
║    Speed: < 1ms per test                                  ║
║    Examples:                                              ║
║      • Cache.get/set                                      ║
║      • ReportBuilder._get_totals                          ║
║      • Address validation regex                           ║
║                                                            ║
║  Level 2: Integration Tests (15%)                         ║
║    What: Multiple components together                     ║
║    Speed: 10-100ms per test                               ║
║    Examples:                                              ║
║      • Ingest → DB → Report                               ║
║      • Cache → Builder → Validator                        ║
║                                                            ║
║  Level 3: Contract Tests (15%)                            ║
║    What: API contracts (JSON Schema)                      ║
║    Speed: 1-10ms per test                                 ║
║    Examples:                                              ║
║      • Report matches schemas/report_v1.json              ║
║      • Response schema stability                          ║
║                                                            ║
║  Level 4: API Tests (5%)                                  ║
║    What: HTTP endpoints via TestClient                    ║
║    Speed: 50-200ms per test                               ║
║    Examples:                                              ║
║      • GET /healthz returns 200                           ║
║      • GET /wallet/{addr}/report validates                ║
║                                                            ║
║  Level 5: Performance Tests (5%)                          ║
║    What: Load testing, benchmarks                         ║
║    Speed: Seconds to minutes                              ║
║    Examples:                                              ║
║      • p95 latency < 1s under load                        ║
║      • 100 requests, 10 concurrent                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 2.2 Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── __init__.py
├── unit/                    # 60% of tests
│   ├── __init__.py
│   ├── test_cache.py        # Cache logic
│   ├── test_report_builder.py  # Builder units
│   ├── test_report_validator.py  # Validator logic
│   └── test_utils.py        # Utility functions
├── integration/             # 15% of tests
│   ├── __init__.py
│   ├── test_ingest_to_report.py  # Full pipeline
│   └── test_cache_integration.py  # Cache + Builder
├── contract/                # 15% of tests
│   ├── __init__.py
│   └── test_report_schema.py  # JSON Schema compliance
├── api/                     # 5% of tests
│   ├── __init__.py
│   └── test_endpoints.py    # FastAPI routes
└── performance/             # 5% of tests
    ├── __init__.py
    └── test_load.py         # Benchmarks
```

---

## 3) pytest Setup: Configuration & Plugins

### 3.1 pyproject.toml Configuration

```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

# Markers for test categorization
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (slower, multi-component)",
    "contract: Contract tests (schema validation)",
    "api: API endpoint tests",
    "performance: Performance tests (slow)",
    "slow: Slow tests (> 1s)"
]

# Coverage configuration
addopts = [
    "-ra",  # Show all test results
    "--strict-markers",  # Enforce marker declaration
    "--strict-config",  # Enforce config correctness
    "--showlocals",  # Show local variables on failure
    "--tb=short",  # Short traceback format
]

# Ignore patterns
norecursedirs = [
    ".git",
    ".tox",
    "dist",
    "build",
    "*.egg",
    "__pycache__"
]
```

### 3.2 pytest Plugins

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",        # Coverage reporting
    "pytest-xdist>=3.3.1",      # Parallel test execution
    "pytest-benchmark>=4.0.0",  # Performance benchmarks
    "pytest-mock>=3.11.1",      # Mocking utilities
    "hypothesis>=6.82.0",       # Property-based testing
    "faker>=19.2.0",            # Fake data generation
]
```

---

## 4) Unit Tests: Comprehensive Examples

### 4.1 Cache Tests

```python
"""
tests/unit/test_cache.py

Unit tests for TTLLRUCache

Coverage:
- Basic get/set operations
- TTL expiration
- LRU eviction
- Thread safety
- Statistics tracking
"""

import pytest
import time
import threading
from crypto.service.cache import TTLLRUCache


class TestCacheBasicOperations:
    """Test basic cache operations"""
    
    def test_set_and_get(self):
        """Test basic set and get"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        
        assert result == {"data": "value1"}
    
    def test_get_nonexistent_key(self):
        """Test get on non-existent key returns None"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        result = cache.get("nonexistent")
        
        assert result is None
    
    def test_set_overwrites(self):
        """Test set overwrites existing key"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        
        assert cache.get("key1") == "value2"


class TestCacheTTL:
    """Test TTL (Time To Live) expiration"""
    
    def test_ttl_expiration(self):
        """Test entry expires after TTL"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=1)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL expiration
        time.sleep(1.5)
        
        assert cache.get("key1") is None
    
    def test_ttl_not_expired(self):
        """Test entry does not expire before TTL"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=2)
        
        cache.set("key1", "value1")
        time.sleep(0.5)
        
        assert cache.get("key1") == "value1"


class TestCacheLRU:
    """Test LRU (Least Recently Used) eviction"""
    
    def test_lru_eviction(self):
        """Test LRU eviction when capacity exceeded"""
        cache = TTLLRUCache(capacity=2, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lru_access_updates_order(self):
        """Test accessing entry updates LRU order"""
        cache = TTLLRUCache(capacity=2, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 (moves to end)
        cache.get("key1")
        
        # Add key3 (should evict key2, not key1)
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Not evicted
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"


class TestCacheStatistics:
    """Test cache statistics tracking"""
    
    def test_hit_miss_tracking(self):
        """Test cache tracks hits and misses"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        
        # Hit
        cache.get("key1")
        
        # Miss
        cache.get("key2")
        
        stats = cache.stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_eviction_tracking(self):
        """Test cache tracks evictions"""
        cache = TTLLRUCache(capacity=2, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Eviction
        
        stats = cache.stats()
        
        assert stats['evictions'] == 1
    
    def test_size_tracking(self):
        """Test cache tracks size"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        assert cache.size() == 0
        
        cache.set("key1", "value1")
        assert cache.size() == 1
        
        cache.set("key2", "value2")
        assert cache.size() == 2


class TestCacheThreadSafety:
    """Test cache thread safety"""
    
    def test_concurrent_writes(self):
        """Test concurrent writes are thread-safe"""
        cache = TTLLRUCache(capacity=100, ttl_seconds=60)
        
        def writer(start, count):
            for i in range(start, start + count):
                cache.set(f"key{i}", f"value{i}")
        
        # Spawn 10 threads, each writing 10 items
        threads = []
        for i in range(10):
            t = threading.Thread(target=writer, args=(i * 10, 10))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All 100 items should be present
        assert cache.size() == 100
    
    def test_concurrent_reads(self):
        """Test concurrent reads are thread-safe"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        # Pre-populate
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")
        
        results = []
        lock = threading.Lock()
        
        def reader(key):
            value = cache.get(key)
            with lock:
                results.append(value)
        
        # Spawn 100 threads reading same keys
        threads = []
        for i in range(100):
            t = threading.Thread(target=reader, args=(f"key{i % 10}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All reads should succeed
        assert len(results) == 100
        assert all(r is not None for r in results)


# Markers for test categorization
pytestmark = pytest.mark.unit
```

### 4.2 Report Builder Tests

```python
"""
tests/unit/test_report_builder.py

Unit tests for ReportBuilder

Coverage:
- Input validation
- Time range calculation
- SQL query results parsing
- Edge cases (empty results, zero activity)
"""

import pytest
from datetime import datetime, timezone, timedelta
from crypto.features.report_builder import ReportBuilder, ReportConfig


@pytest.fixture
def builder(tmp_db_with_data):
    """Create builder with test database"""
    config = ReportConfig(chain_id=11155111)
    return ReportBuilder(str(tmp_db_with_data), config)


class TestInputValidation:
    """Test input validation"""
    
    def test_invalid_wallet_address(self, builder):
        """Test invalid wallet address raises ValueError"""
        with pytest.raises(ValueError, match="Invalid wallet address"):
            builder.build(wallet="invalid_address", window_hours=24)
    
    def test_invalid_hours_too_small(self, builder):
        """Test hours < 1 raises ValueError"""
        with pytest.raises(ValueError, match="window_hours must be 1-720"):
            builder.build(wallet="0x" + "1" * 40, window_hours=0)
    
    def test_invalid_hours_too_large(self, builder):
        """Test hours > 720 raises ValueError"""
        with pytest.raises(ValueError, match="window_hours must be 1-720"):
            builder.build(wallet="0x" + "1" * 40, window_hours=1000)
    
    def test_valid_wallet_lowercase(self, builder):
        """Test lowercase wallet address is accepted"""
        wallet = "0xabcdef1234567890abcdef1234567890abcdef12"
        report = builder.build(wallet=wallet, window_hours=24)
        
        assert report["wallet"] == wallet.lower()
    
    def test_valid_wallet_uppercase(self, builder):
        """Test uppercase wallet address is normalized"""
        wallet = "0xABCDEF1234567890ABCDEF1234567890ABCDEF12"
        report = builder.build(wallet=wallet, window_hours=24)
        
        assert report["wallet"] == wallet.lower()


class TestTimeRangeCalculation:
    """Test time range calculation"""
    
    def test_24_hour_window(self, builder):
        """Test 24-hour window calculation"""
        wallet = "0x" + "1" * 40
        report = builder.build(wallet=wallet, window_hours=24)
        
        from_ts = datetime.fromisoformat(report["time"]["from_ts"].replace("Z", "+00:00"))
        to_ts = datetime.fromisoformat(report["time"]["to_ts"].replace("Z", "+00:00"))
        
        delta = to_ts - from_ts
        
        assert delta.total_seconds() == 24 * 3600
    
    def test_custom_end_time(self, builder):
        """Test custom end time is respected"""
        wallet = "0x" + "1" * 40
        custom_to_ts = datetime(2025, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
        
        report = builder.build(wallet=wallet, window_hours=24, to_ts=custom_to_ts)
        
        to_ts = datetime.fromisoformat(report["time"]["to_ts"].replace("Z", "+00:00"))
        
        assert to_ts == custom_to_ts


class TestEdgeCases:
    """Test edge cases"""
    
    def test_zero_activity(self, tmp_db_empty):
        """Test wallet with zero activity"""
        config = ReportConfig(chain_id=11155111)
        builder = ReportBuilder(str(tmp_db_empty), config)
        
        wallet = "0x" + "0" * 40
        report = builder.build(wallet=wallet, window_hours=24)
        
        assert report["tx_count"] == 0
        assert report["totals"]["inbound"] == 0.0
        assert report["totals"]["outbound"] == 0.0
        assert report["transfer_stats"] == []
        assert report["top_counterparties"] == []
    
    def test_missing_token_metadata(self, tmp_db_with_missing_metadata):
        """Test missing token metadata uses fallbacks"""
        config = ReportConfig(chain_id=11155111)
        builder = ReportBuilder(str(tmp_db_with_missing_metadata), config)
        
        wallet = "0xtest..."
        report = builder.build(wallet=wallet, window_hours=24)
        
        # Should use fallbacks
        assert len(report["transfer_stats"]) > 0
        for stat in report["transfer_stats"]:
            assert stat["symbol"] != ""  # Fallback applied
            assert stat["decimals"] >= 0  # Fallback applied


pytestmark = pytest.mark.unit
```

---

## 5) Integration Tests: End-to-End Flows

```python
"""
tests/integration/test_ingest_to_report.py

Integration tests for complete pipeline

Coverage:
- Ingest → DB → Report
- Cache → Builder → Validator
- Multiple sequential operations
"""

import pytest
from datetime import datetime, timezone
from crypto.features.report_builder import ReportBuilder
from crypto.features.report_validator import ReportValidator


@pytest.mark.integration
class TestIngestToReport:
    """Test complete ingest to report flow"""
    
    def test_full_pipeline(self, tmp_path):
        """Test: Ingest transfers → Generate report → Validate"""
        import duckdb
        
        # 1. Setup database with schema
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        
        conn.execute("""
            CREATE TABLE transfers (
                tx_hash TEXT,
                log_index INTEGER,
                block_number BIGINT,
                block_time TIMESTAMP,
                token TEXT,
                symbol TEXT,
                decimals INTEGER,
                from_addr TEXT,
                to_addr TEXT,
                value_unit DOUBLE,
                PRIMARY KEY (tx_hash, log_index)
            )
        """)
        
        # 2. Ingest test data
        test_wallet = "0xtest"
        now = datetime.now(timezone.utc)
        
        conn.execute("""
            INSERT INTO transfers VALUES
                ('0xhash1', 0, 100, ?, '0xUSDC', 'USDC', 6, '0xA', ?, 100.0),
                ('0xhash2', 0, 101, ?, '0xUSDC', 'USDC', 6, ?, '0xB', 50.0)
        """, [now, test_wallet, now, test_wallet])
        
        conn.close()
        
        # 3. Generate report
        builder = ReportBuilder(str(db_path))
        report = builder.build(wallet=test_wallet, window_hours=24)
        
        # 4. Validate report
        validator = ReportValidator()
        assert validator.validate(report) is True
        
        # 5. Verify data
        assert report["tx_count"] == 2
        assert report["totals"]["inbound"] == 100.0
        assert report["totals"]["outbound"] == 50.0


@pytest.mark.integration
class TestCacheIntegration:
    """Test cache integration with builder"""
    
    def test_cache_hit_returns_same_data(self, seeded_db):
        """Test cache hit returns identical data"""
        from crypto.service.cache import TTLLRUCache
        
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        builder = ReportBuilder(str(seeded_db))
        
        wallet = "0xtest"
        cache_key = f"{wallet}|24"
        
        # First call (cache miss)
        report1 = builder.build(wallet=wallet, window_hours=24)
        cache.set(cache_key, report1)
        
        # Second call (cache hit)
        cached_report = cache.get(cache_key)
        
        assert cached_report == report1
        assert cached_report is not None
    
    def test_cache_invalidation_on_new_data(self, seeded_db):
        """Test cache should be invalidated when data changes"""
        from crypto.service.cache import TTLLRUCache
        import duckdb
        
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        builder = ReportBuilder(str(seeded_db))
        
        wallet = "0xtest"
        cache_key = f"{wallet}|24"
        
        # Generate initial report
        report1 = builder.build(wallet=wallet, window_hours=24)
        cache.set(cache_key, report1)
        
        # Add new data
        conn = duckdb.connect(str(seeded_db))
        conn.execute("""
            INSERT INTO transfers VALUES
                ('0xnew', 0, 200, NOW(), '0xUSDC', 'USDC', 6, '0xC', ?, 25.0)
        """, [wallet])
        conn.close()
        
        # Invalidate cache
        cache.invalidate(cache_key)
        
        # Generate new report
        report2 = builder.build(wallet=wallet, window_hours=24)
        
        # Reports should differ
        assert report2["tx_count"] > report1["tx_count"]
```

---

## 6) Contract Tests: JSON Schema Validation

```python
"""
tests/contract/test_report_schema.py

Contract tests for JSON Schema compliance

Coverage:
- Report matches schemas/report_v1.json
- All required fields present
- Field types correct
- No additional properties
"""

import pytest
import json
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError


@pytest.fixture(scope="session")
def schema_v1():
    """Load report_v1 schema"""
    schema_path = Path("schemas/report_v1.json")
    return json.loads(schema_path.read_text(encoding='utf-8'))


@pytest.mark.contract
class TestReportSchemaCompliance:
    """Test report schema compliance"""
    
    def test_schema_is_valid(self, schema_v1):
        """Test schema itself is valid JSON Schema"""
        Draft202012Validator.check_schema(schema_v1)
    
    def test_generated_report_matches_schema(self, schema_v1, seeded_db):
        """Test generated report matches schema"""
        from crypto.features.report_builder import ReportBuilder
        
        builder = ReportBuilder(str(seeded_db))
        report = builder.build(wallet="0xtest", window_hours=24)
        
        # Should not raise
        Draft202012Validator(schema_v1).validate(report)
    
    def test_empty_report_matches_schema(self, schema_v1, tmp_db_empty):
        """Test empty report (zero activity) matches schema"""
        from crypto.features.report_builder import ReportBuilder
        
        builder = ReportBuilder(str(tmp_db_empty))
        report = builder.build(wallet="0x" + "0" * 40, window_hours=24)
        
        # Zero activity should still be valid
        Draft202012Validator(schema_v1).validate(report)
    
    def test_additional_properties_rejected(self, schema_v1):
        """Test additional properties are rejected"""
        validator = Draft202012Validator(schema_v1)
        
        # Valid report with extra field
        report = {
            "version": "v1",
            "wallet": "0x" + "1" * 40,
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            },
            "extra_field": "should_fail"  # Additional property
        }
        
        with pytest.raises(ValidationError, match="Additional properties"):
            validator.validate(report)
    
    def test_required_fields_enforced(self, schema_v1):
        """Test required fields are enforced"""
        validator = Draft202012Validator(schema_v1)
        
        # Missing required field "wallet"
        report = {
            "version": "v1",
            # "wallet": missing!
            "window_hours": 24,
            "time": {
                "from_ts": "2025-10-06T00:00:00Z",
                "to_ts": "2025-10-07T00:00:00Z"
            },
            "totals": {"inbound": 0, "outbound": 0},
            "tx_count": 0,
            "transfer_stats": [],
            "top_counterparties": [],
            "meta": {
                "chain_id": 11155111,
                "generated_at": "2025-10-07T00:00:00Z",
                "source": "test"
            }
        }
        
        with pytest.raises(ValidationError, match="'wallet' is a required property"):
            validator.validate(report)
```

---

## 7) API Tests: FastAPI TestClient

```python
"""
tests/api/test_endpoints.py

API tests using FastAPI TestClient

Coverage:
- Health endpoint
- Wallet report endpoint
- Error handling (4xx, 5xx)
- Response formats
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(seeded_db, monkeypatch):
    """Create FastAPI test client"""
    monkeypatch.setenv("NOVA_DB_PATH", str(seeded_db))
    monkeypatch.setenv("NOVA_CACHE_TTL", "60")
    monkeypatch.setenv("NOVA_CACHE_CAPACITY", "100")
    
    # Import after env is set
    from crypto.service.app import app
    
    return TestClient(app)


@pytest.mark.api
class TestHealthEndpoint:
    """Test /healthz endpoint"""
    
    def test_health_returns_200(self, client):
        """Test health endpoint returns 200"""
        response = client.get("/healthz")
        
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Test health response has correct structure"""
        response = client.get("/healthz")
        data = response.json()
        
        assert "status" in data
        assert "uptime_s" in data
        assert "db_status" in data
        assert "cache_status" in data
        assert "cache_size" in data
    
    def test_health_status_ok(self, client):
        """Test health status is 'ok' when DB available"""
        response = client.get("/healthz")
        data = response.json()
        
        assert data["status"] == "ok"
        assert data["db_status"] == "ok"


@pytest.mark.api
class TestWalletReportEndpoint:
    """Test /wallet/{address}/report endpoint"""
    
    def test_report_returns_200(self, client):
        """Test report endpoint returns 200 for valid wallet"""
        wallet = "0x" + "1" * 40
        
        response = client.get(f"/wallet/{wallet}/report?hours=24")
        
        assert response.status_code == 200
    
    def test_report_response_structure(self, client):
        """Test report response has correct structure"""
        wallet = "0x" + "1" * 40
        
        response = client.get(f"/wallet/{wallet}/report?hours=24")
        data = response.json()
        
        # Top-level fields
        assert data["version"] == "v1"
        assert data["wallet"] == wallet.lower()
        assert data["window_hours"] == 24
        
        # Nested objects
        assert "time" in data
        assert "from_ts" in data["time"]
        assert "to_ts" in data["time"]
        
        assert "totals" in data
        assert "inbound" in data["totals"]
        assert "outbound" in data["totals"]
        
        assert "tx_count" in data
        assert "transfer_stats" in data
        assert "top_counterparties" in data
        assert "meta" in data
    
    def test_report_custom_hours(self, client):
        """Test report respects custom hours parameter"""
        wallet = "0x" + "1" * 40
        
        response = client.get(f"/wallet/{wallet}/report?hours=168")
        data = response.json()
        
        assert data["window_hours"] == 168
    
    def test_report_invalid_address_422(self, client):
        """Test invalid address returns 422"""
        invalid_addresses = [
            "not_an_address",
            "0x123",  # Too short
            "0xZZZ",  # Invalid characters
            "vitalik.eth",  # ENS not supported
        ]
        
        for addr in invalid_addresses:
            response = client.get(f"/wallet/{addr}/report?hours=24")
            assert response.status_code == 422
    
    def test_report_invalid_hours_422(self, client):
        """Test invalid hours returns 422"""
        wallet = "0x" + "1" * 40
        
        invalid_hours = [0, -1, 1000]
        
        for hours in invalid_hours:
            response = client.get(f"/wallet/{wallet}/report?hours={hours}")
            assert response.status_code == 422


@pytest.mark.api
class TestCaching:
    """Test API caching behavior"""
    
    def test_cache_hit_faster_than_miss(self, client):
        """Test cache hit is faster than cache miss"""
        import time
        
        wallet = "0x" + "1" * 40
        
        # First request (cache miss)
        start1 = time.time()
        response1 = client.get(f"/wallet/{wallet}/report?hours=24")
        duration1 = time.time() - start1
        
        # Second request (cache hit)
        start2 = time.time()
        response2 = client.get(f"/wallet/{wallet}/report?hours=24")
        duration2 = time.time() - start2
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert duration2 < duration1  # Cache hit should be faster
    
    def test_different_wallets_not_cached_together(self, client):
        """Test different wallets have separate cache entries"""
        wallet1 = "0x" + "1" * 40
        wallet2 = "0x" + "2" * 40
        
        response1 = client.get(f"/wallet/{wallet1}/report?hours=24")
        response2 = client.get(f"/wallet/{wallet2}/report?hours=24")
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["wallet"] != data2["wallet"]
```

---

## 8) Performance Tests: Benchmarks & Load

```python
"""
tests/performance/test_load.py

Performance and load tests

Coverage:
- Response time benchmarks
- Concurrent request handling
- Cache performance
- Database query performance
"""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@pytest.mark.performance
@pytest.mark.slow
class TestResponseTimeBenchmarks:
    """Test response time benchmarks"""
    
    def test_health_endpoint_p95(self, client):
        """Test health endpoint p95 < 50ms"""
        latencies = []
        
        for _ in range(100):
            start = time.perf_counter()
            response = client.get("/healthz")
            latency = (time.perf_counter() - start) * 1000  # ms
            
            assert response.status_code == 200
            latencies.append(latency)
        
        latencies.sort()
        p95 = latencies[95]
        
        print(f"\nHealth endpoint p95: {p95:.2f}ms")
        assert p95 < 50, f"p95 latency {p95:.2f}ms exceeds 50ms"
    
    def test_report_endpoint_cold_cache_p95(self, client):
        """Test report endpoint p95 < 500ms (cold cache)"""
        wallet = "0x" + "1" * 40
        latencies = []
        
        for i in range(20):
            # Use different hours to avoid cache
            hours = 24 + i
            
            start = time.perf_counter()
            response = client.get(f"/wallet/{wallet}/report?hours={hours}")
            latency = (time.perf_counter() - start) * 1000  # ms
            
            assert response.status_code == 200
            latencies.append(latency)
        
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        
        print(f"\nReport endpoint (cold) p95: {p95:.2f}ms")
        assert p95 < 500, f"p95 latency {p95:.2f}ms exceeds 500ms"
    
    def test_report_endpoint_warm_cache_p95(self, client):
        """Test report endpoint p95 < 100ms (warm cache)"""
        wallet = "0x" + "1" * 40
        
        # Warm up cache
        client.get(f"/wallet/{wallet}/report?hours=24")
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            response = client.get(f"/wallet/{wallet}/report?hours=24")
            latency = (time.perf_counter() - start) * 1000  # ms
            
            assert response.status_code == 200
            latencies.append(latency)
        
        latencies.sort()
        p95 = latencies[95]
        
        print(f"\nReport endpoint (warm) p95: {p95:.2f}ms")
        assert p95 < 100, f"p95 latency {p95:.2f}ms exceeds 100ms"


@pytest.mark.performance
@pytest.mark.slow
class TestConcurrentRequests:
    """Test concurrent request handling"""
    
    def test_concurrent_health_checks(self, client):
        """Test 100 concurrent health checks"""
        def health_check():
            response = client.get("/healthz")
            return response.status_code
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(health_check) for _ in range(100)]
            results = [f.result() for f in as_completed(futures)]
        
        # All should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 100
    
    def test_concurrent_report_requests(self, client):
        """Test 50 concurrent report requests"""
        wallet = "0x" + "1" * 40
        
        def get_report():
            response = client.get(f"/wallet/{wallet}/report?hours=24")
            return response.status_code
        
        start = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_report) for _ in range(50)]
            results = [f.result() for f in as_completed(futures)]
        
        duration = time.time() - start
        
        # All should succeed
        assert all(status == 200 for status in results)
        
        # Should complete in reasonable time (< 10s)
        print(f"\n50 concurrent requests completed in {duration:.2f}s")
        assert duration < 10


@pytest.mark.performance
class TestCachePerformance:
    """Test cache performance"""
    
    def test_cache_hit_rate(self, client):
        """Test cache achieves > 50% hit rate"""
        wallets = ["0x" + str(i) * 40 for i in range(5)]
        
        # Generate reports for 5 wallets (cold cache)
        for wallet in wallets:
            client.get(f"/wallet/{wallet}/report?hours=24")
        
        # Request same reports again (should hit cache)
        for wallet in wallets:
            client.get(f"/wallet/{wallet}/report?hours=24")
        
        # Check health for cache stats
        response = client.get("/healthz")
        data = response.json()
        
        cache_hits = data.get("cache_hits", 0)
        cache_misses = data.get("cache_misses", 0)
        
        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses)
            print(f"\nCache hit rate: {hit_rate:.2%}")
            assert hit_rate >= 0.5, f"Cache hit rate {hit_rate:.2%} below 50%"
```

---

## 9) Code Quality: Ruff, Type Checking, Coverage

### 9.1 Ruff Configuration

**pyproject.toml:**

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # do not perform function calls in argument defaults
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Ignore unused imports in __init__.py
"tests/**/*.py" = ["S101"]  # Allow assert in tests

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

### 9.2 Type Checking (mypy)

**pyproject.toml:**

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_calls = false  # Too strict for now
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
strict_optional = true

# Per-module options
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false  # Relax for tests

[[tool.mypy.overrides]]
module = "duckdb.*"
ignore_missing_imports = true
```

### 9.3 Coverage Configuration

**pyproject.toml:**

```toml
[tool.coverage.run]
source = ["crypto"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/site-packages/*",
]
branch = true

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false

# Fail if coverage < 70%
fail_under = 70

exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]

[tool.coverage.html]
directory = "htmlcov"
```

### 9.4 Makefile Commands

```makefile
.PHONY: qa.all qa.lint qa.format qa.typecheck qa.test qa.cov

# Run all quality checks
qa.all: qa.format qa.lint qa.typecheck qa.test

# Format code
qa.format:
	@echo "🎨 Formatting code with ruff..."
	ruff format .

# Lint code
qa.lint:
	@echo "🔍 Linting code with ruff..."
	ruff check .

# Fix lint issues
qa.fix:
	@echo "🔧 Fixing lint issues with ruff..."
	ruff check --fix .

# Type check
qa.typecheck:
	@echo "📝 Type checking with mypy..."
	mypy crypto/

# Run tests
qa.test:
	@echo "🧪 Running tests..."
	pytest -q

# Run tests with coverage
qa.cov:
	@echo "📊 Running tests with coverage..."
	pytest --cov=crypto --cov-report=term-missing --cov-report=html
	@echo "📄 Coverage report: htmlcov/index.html"

# Run tests in parallel
qa.test.parallel:
	@echo "⚡ Running tests in parallel..."
	pytest -n auto

# Run only fast tests
qa.test.fast:
	@echo "⚡ Running fast tests..."
	pytest -m "not slow" -q

# Run only slow tests
qa.test.slow:
	@echo "🐌 Running slow tests..."
	pytest -m "slow" -q
```

---

## 10) Pre-commit Hooks: Local Feedback Loop

### 10.1 .pre-commit-config.yaml

```yaml
# .pre-commit-config.yaml

repos:
  # Ruff: Fast Python linter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.7
    hooks:
      # Linter
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      # Formatter
      - id: ruff-format

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  # Type checking (optional, can be slow)
  # - repo: https://github.com/pre-commit/mirrors-mypy
  #   rev: v1.7.0
  #   hooks:
  #     - id: mypy
  #       additional_dependencies: [types-requests]
  #       args: [--ignore-missing-imports]

  # Markdown linting
  - repo: https://github.com/igorshubovych/markdownlint-cli
    rev: v0.37.0
    hooks:
      - id: markdownlint
        args: [--config, .markdownlint.json]

# Configuration
ci:
  autofix_commit_msg: |
    [pre-commit.ci] auto fixes from pre-commit hooks
  autoupdate_commit_msg: |
    [pre-commit.ci] pre-commit autoupdate
```

### 10.2 Installation & Usage

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on staged files (automatic on commit)
pre-commit run

# Update hooks to latest versions
pre-commit autoupdate

# Skip hooks for a commit (emergency only!)
git commit --no-verify
```

### 10.3 Pre-commit Workflow

```
╔════════════════════════════════════════════════════════════╗
║            PRE-COMMIT WORKFLOW                             ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  1. Developer makes changes                                ║
║     ├─ Edit code                                          ║
║     └─ Stage files (git add)                              ║
║                                                            ║
║  2. git commit (triggers pre-commit)                       ║
║     ├─ Ruff format (auto-fix)                             ║
║     ├─ Ruff lint (auto-fix)                               ║
║     ├─ Trailing whitespace check                          ║
║     ├─ YAML/JSON validation                               ║
║     └─ Large file check                                   ║
║                                                            ║
║  3. If hooks pass:                                         ║
║     └─ Commit succeeds ✅                                 ║
║                                                            ║
║  4. If hooks fail:                                         ║
║     ├─ Commit blocked ❌                                  ║
║     ├─ Auto-fixes applied (if possible)                   ║
║     └─ Re-stage and commit again                          ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## 11) CI/CD Workflows: GitHub Actions

### 11.1 Python CI Workflow

**.github/workflows/python-ci.yml:**

```yaml
name: Python CI

on:
  push:
    branches: [main, master]
    paths:
      - '**/*.py'
      - 'pyproject.toml'
      - '.github/workflows/python-ci.yml'
  pull_request:
    paths:
      - '**/*.py'
      - 'pyproject.toml'

jobs:
  lint-and-test:
    name: Lint & Test
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,crypto]"
      
      - name: Ruff format check
        run: |
          ruff format --check .
      
      - name: Ruff lint
        run: |
          ruff check .
      
      - name: Type check (mypy)
        run: |
          mypy crypto/ || true  # Don't fail on type errors yet
      
      - name: Run tests
        run: |
          pytest -v --cov=crypto --cov-report=xml --cov-report=term
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false
      
      - name: Generate coverage HTML
        if: always()
        run: |
          pytest --cov=crypto --cov-report=html
      
      - name: Upload coverage HTML
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-html
          path: htmlcov/
```

### 11.2 API Smoke Test Workflow

**.github/workflows/api-smoke.yml:**

```yaml
name: API Smoke Test

on:
  workflow_run:
    workflows: ["Python CI"]
    types: [completed]
    branches: [main, master]

jobs:
  smoke:
    name: API Smoke Test
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,crypto]"
      
      - name: Create test database
        run: |
          python - <<'EOF'
          import duckdb
          
          conn = duckdb.connect('smoke.duckdb')
          conn.execute("""
              CREATE TABLE transfers (
                  tx_hash TEXT,
                  log_index INTEGER,
                  block_number BIGINT,
                  block_time TIMESTAMP,
                  token TEXT,
                  symbol TEXT,
                  decimals INTEGER,
                  from_addr TEXT,
                  to_addr TEXT,
                  value_unit DOUBLE,
                  PRIMARY KEY (tx_hash, log_index)
              )
          """)
          conn.execute("""
              INSERT INTO transfers VALUES
                  ('0xtest', 0, 1, NOW(), '0xUSDC', 'USDC', 6, '0xA', '0xB', 100.0)
          """)
          conn.close()
          print("✅ Test database created")
          EOF
      
      - name: Start API server
        run: |
          export NOVA_DB_PATH=smoke.duckdb
          export NOVA_CACHE_TTL=60
          export NOVA_CACHE_CAPACITY=100
          
          # Start server in background
          uvicorn crypto.service.app:app --host 127.0.0.1 --port 8000 &
          
          # Wait for server to start
          for i in {1..10}; do
            if curl -sf http://127.0.0.1:8000/healthz > /dev/null; then
              echo "✅ API server started"
              break
            fi
            echo "Waiting for server... ($i/10)"
            sleep 1
          done
      
      - name: Test /healthz endpoint
        run: |
          response=$(curl -sf http://127.0.0.1:8000/healthz)
          echo "Response: $response"
          
          status=$(echo $response | jq -r '.status')
          if [ "$status" != "ok" ]; then
            echo "❌ Health check failed: status = $status"
            exit 1
          fi
          
          echo "✅ /healthz endpoint OK"
      
      - name: Test /wallet/{addr}/report endpoint
        run: |
          wallet="0x1111111111111111111111111111111111111111"
          response=$(curl -sf "http://127.0.0.1:8000/wallet/$wallet/report?hours=24")
          echo "Response: $response"
          
          version=$(echo $response | jq -r '.version')
          if [ "$version" != "v1" ]; then
            echo "❌ Report endpoint failed: version = $version"
            exit 1
          fi
          
          echo "✅ /wallet/{addr}/report endpoint OK"
      
      - name: Test invalid wallet returns 422
        run: |
          status=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/wallet/invalid/report")
          
          if [ "$status" != "422" ]; then
            echo "❌ Expected 422, got $status"
            exit 1
          fi
          
          echo "✅ Invalid wallet correctly returns 422"
      
      - name: Stop API server
        if: always()
        run: |
          pkill -f uvicorn || true
```

### 11.3 Schema Validation Workflow

**.github/workflows/schema-check.yml:**

```yaml
name: Schema Validation

on:
  push:
    paths:
      - 'schemas/**/*.json'
  pull_request:
    paths:
      - 'schemas/**/*.json'

jobs:
  validate:
    name: Validate JSON Schemas
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install jsonschema
        run: |
          pip install jsonschema
      
      - name: Validate schemas
        run: |
          python - <<'EOF'
          import json
          from pathlib import Path
          from jsonschema import Draft202012Validator
          
          for schema_file in Path("schemas").glob("*.json"):
              print(f"Validating {schema_file}...")
              schema = json.loads(schema_file.read_text())
              Draft202012Validator.check_schema(schema)
              print(f"✅ {schema_file.name} is valid")
          
          print("\n✅ All schemas are valid")
          EOF
      
      - name: Check for breaking changes
        if: github.event_name == 'pull_request'
        run: |
          echo "⚠️  Schema changes detected!"
          echo "Please ensure backward compatibility."
          echo "Consider versioning (e.g., report_v2.json) if breaking."
```

---

## 12) Coverage Reporting: HTML, Badges, Trends

### 12.1 Coverage HTML Report

```bash
# Generate HTML coverage report
pytest --cov=crypto --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Example HTML report:**

```
╔════════════════════════════════════════════════════════════╗
║              COVERAGE REPORT                               ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Module                            Stmts   Miss   Cover    ║
║  ────────────────────────────────────────────────────────  ║
║  crypto/__init__.py                    2      0    100%    ║
║  crypto/service/app.py                45      3     93%    ║
║  crypto/service/cache.py              78      5     94%    ║
║  crypto/service/config.py             12      0    100%    ║
║  crypto/service/deps.py               23      2     91%    ║
║  crypto/service/models.py             34      0    100%    ║
║  crypto/features/report_builder.py   102     12     88%    ║
║  crypto/features/report_validator.py  45      3     93%    ║
║  ────────────────────────────────────────────────────────  ║
║  TOTAL                               341     25     93%    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 12.2 Coverage Badge (README)

```markdown
<!-- Add to README.md -->

[![Coverage](https://img.shields.io/codecov/c/github/username/novadev-protocol?token=YOUR_TOKEN)](https://codecov.io/gh/username/novadev-protocol)
[![Tests](https://github.com/username/novadev-protocol/workflows/Python%20CI/badge.svg)](https://github.com/username/novadev-protocol/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
```

### 12.3 Coverage Trends (CI Artifact)

```yaml
# In .github/workflows/python-ci.yml

- name: Upload coverage artifact
  uses: actions/upload-artifact@v4
  with:
    name: coverage-${{ github.sha }}
    path: htmlcov/
    retention-days: 30

- name: Comment coverage on PR
  if: github.event_name == 'pull_request'
  uses: py-cov-action/python-coverage-comment-action@v3
  with:
    GITHUB_TOKEN: ${{ github.token }}
```

---

## 13) Troubleshooting: Common CI Issues

### 13.1 Import Errors in CI

**Symptom:**
```
ImportError: No module named 'crypto'
```

**Solution:**
```yaml
# In CI workflow, install package in editable mode
- name: Install dependencies
  run: |
    pip install -e ".[dev,crypto]"
```

### 13.2 DuckDB Lock in CI

**Symptom:**
```
duckdb.IOException: database is locked
```

**Solution:**
```python
# In tests, use read_only=True for concurrent access
conn = duckdb.connect(db_path, read_only=True)

# Or use separate databases for each test
@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / f"test_{uuid.uuid4()}.duckdb"
    # ...
    return db_path
```

### 13.3 Schema Validation Fails

**Symptom:**
```
jsonschema.exceptions.ValidationError: Additional properties are not allowed
```

**Solution:**
```python
# Check schema allows additionalProperties
{
  "type": "object",
  "additionalProperties": false,  # Strict mode
  # OR
  "additionalProperties": true    # Allow extra fields
}

# Remove extra fields before validation
report = {k: v for k, v in report.items() if k in schema["properties"]}
```

### 13.4 Coverage Below Threshold

**Symptom:**
```
FAIL: coverage below 70%
```

**Solution:**
```bash
# Find uncovered lines
pytest --cov=crypto --cov-report=term-missing

# Target specific uncovered areas
# Add unit tests for:
# - Edge cases
# - Error handling
# - Utility functions
```

### 13.5 Flaky Tests

**Symptom:**
```
Test passes locally, fails in CI randomly
```

**Solutions:**
```python
# 1. Use fixtures for isolation
@pytest.fixture
def clean_cache():
    cache = TTLLRUCache(capacity=10, ttl_seconds=60)
    yield cache
    cache.clear()

# 2. Avoid time-dependent tests
# Bad:
time.sleep(1)
assert cache.get("key") is None  # May fail due to timing

# Good:
with freeze_time("2025-10-06 12:00:00"):
    cache.set("key", "value")
    # Test logic

# 3. Use pytest-xdist carefully
# Some tests may not be thread-safe
@pytest.mark.xfail(reason="Not thread-safe")
def test_concurrent():
    pass
```

### 13.6 CI Timeout

**Symptom:**
```
Job exceeded maximum time limit
```

**Solutions:**
```yaml
# Set realistic timeouts
jobs:
  test:
    timeout-minutes: 15  # Default: 360 (6 hours)

# Skip slow tests in CI
pytest -m "not slow"

# Use pytest-xdist for parallel execution
pytest -n auto
```

---

## 14) Quiz + Ödevler

### 📝 Mini Quiz (10 Sorular)

**1. Test pyramid'inde en çok hangi test türü olmalı?**
- A) E2E tests
- B) Integration tests
- C) Unit tests ✅
- D) Manual tests

**2. pytest'de test dosyaları hangi pattern'i takip etmeli?**
- A) `test_*.py` veya `*_test.py` ✅
- B) `*.test.py`
- C) `test*.py`
- D) Herhangi bir isim

**3. JSON Schema validation için hangi kütüphane kullanılır?**
- A) pydantic
- B) jsonschema ✅
- C) marshmallow
- D) cerberus

**4. Ruff'ın ana avantajı nedir?**
- A) Çok hızlı (Rust ile yazılmış) ✅
- B) En eski linter
- C) En fazla kural sayısı
- D) Sadece formatting yapar

**5. Pre-commit hooks ne zaman çalışır?**
- A) Push'tan önce
- B) Commit'ten önce ✅
- C) PR açılınca
- D) Merge'den sonra

**6. FastAPI TestClient hangi kütüphaneyi kullanır?**
- A) requests
- B) httpx ✅
- C) urllib
- D) aiohttp

**7. Coverage threshold %70'in altındaysa ne olur?**
- A) Test geçer, uyarı verir
- B) Test fail olur ✅ (eğer `fail_under=70` ayarlanmışsa)
- C) Hiçbir şey olmaz
- D) Sadece log'a yazar

**8. Contract test ne test eder?**
- A) Performans
- B) API schema compliance ✅
- C) Database queries
- D) UI rendering

**9. CI'da hangi işlem en uzun sürer?**
- A) Lint check
- B) Unit tests
- C) E2E tests ✅
- D) Schema validation

**10. pytest marker neyin için kullanılır?**
- A) Test kategorilendirme ✅
- B) Test sıralama
- C) Test adlandırma
- D) Test silme

### 🎯 Pratik Ödevler (6 Ödev)

#### Ödev 1: Unit Test Yazma (20 dk)

**Görev:** `TTLLRUCache` için 3 yeni unit test yaz:
1. `test_clear()` - Cache'i temizleme
2. `test_size()` - Cache size tracking
3. `test_update_existing_key()` - Mevcut key'i güncelleme

**Başarı Kriterleri:**
- Her test bağımsız çalışıyor
- Edge case'ler kapsamlı
- Assertion'lar anlamlı

#### Ödev 2: Integration Test (30 dk)

**Görev:** "Ingest → DB → Report → Validate" full pipeline test'i yaz

**Steps:**
1. Temp DuckDB oluştur
2. 5 transfer ekle (farklı token, wallet, zaman)
3. Report oluştur
4. Schema ile validate et
5. Beklenmeyen değerler kontrol et (totals, tx_count, etc.)

**Başarı Kriterleri:**
- Tüm adımlar çalışıyor
- Cleanup doğru yapılıyor
- Test tekrarlanabilir

#### Ödev 3: Contract Test Genişletme (15 dk)

**Görev:** `schemas/report_v1.json` için boundary test'leri ekle:

```python
def test_wallet_address_exactly_40_chars()
def test_window_hours_min_1()
def test_window_hours_max_720()
def test_decimals_range_0_36()
```

**Başarı Kriterleri:**
- Her boundary test ediliyor
- Geçersiz değerler fail ediyor
- Geçerli değerler pass ediyor

#### Ödev 4: Pre-commit Setup (10 dk)

**Görev:** Pre-commit hooks'u lokal olarak kur ve test et

```bash
# 1. Install pre-commit
pip install pre-commit

# 2. Install hooks
pre-commit install

# 3. Run on all files
pre-commit run --all-files

# 4. Make a test commit with formatting issues
echo "x=1" > test.py  # No spaces around =
git add test.py
git commit -m "test"
# Should auto-fix to "x = 1"

# 5. Verify auto-fix worked
cat test.py
```

**Başarı Kriterleri:**
- Hooks install oldu
- Auto-fix çalıştı
- Commit başarılı

#### Ödev 5: CI Workflow Ekleme (20 dk)

**Görev:** `.github/workflows/` altına `quick-check.yml` ekle:

- PR açılınca çalışsın
- Sadece değişen dosyalar için test çalıştırsın
- 5 dakikada bitsin

**İpucu:**
```yaml
# Use pytest --lf (last failed) for speed
pytest --lf -q
```

**Başarı Kriterleri:**
- Workflow çalışıyor
- PR'da sonuç görünüyor
- Süre < 5 dk

#### Ödev 6: Coverage Analizi (25 dk)

**Görev:** Coverage raporunu analiz et ve iyileştir

```bash
# 1. Generate coverage report
pytest --cov=crypto --cov-report=html

# 2. Open report
open htmlcov/index.html

# 3. Find files with < 80% coverage

# 4. Add tests to improve coverage

# 5. Re-run and verify improvement
```

**Başarı Kriterleri:**
- En az 1 dosyanın coverage'ı artırıldı
- Yeni testler anlamlı (sadece coverage için değil)
- Total coverage ≥ 75%

---

## ✅ Definition of Done (DoD)

### Testing DoD

- [ ] Unit test coverage ≥ 70% (target: 85%)
- [ ] All contract tests passing (JSON Schema)
- [ ] API tests passing (200, 422, 500 scenarios)
- [ ] Performance tests passing (p95 targets met)
- [ ] No flaky tests

### Code Quality DoD

- [ ] Ruff lint: 0 errors
- [ ] Ruff format: All files formatted
- [ ] mypy: 0 errors (or only allowlisted)
- [ ] No `TODO` or `FIXME` in main code
- [ ] All functions have docstrings

### CI/CD DoD

- [ ] `python-ci` workflow: ✅ green
- [ ] `api-smoke` workflow: ✅ green
- [ ] `schema-check` workflow: ✅ green
- [ ] Pre-commit hooks installed
- [ ] Coverage report artifact uploaded

### Documentation DoD

- [ ] All fixtures documented in `conftest.py`
- [ ] Test README with run instructions
- [ ] Troubleshooting guide complete
- [ ] CI badges in main README

---

## 🔗 İlgili Dersler & Kaynaklar

### Önceki Dersler
- **Tahta 05**: DuckDB + Idempotent Writing
- **Tahta 06**: State Management
- **Tahta 07**: JSON Schema & Report
- **Tahta 08**: FastAPI Service

### Sonraki Ders
- **Tahta 10**: Troubleshooting Deep-Dive (10+ production scenarios)

### External Resources
- [pytest documentation](https://docs.pytest.org/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [GitHub Actions docs](https://docs.github.com/en/actions)
- [pre-commit docs](https://pre-commit.com/)

---

## 🎓 Özet

```
╔════════════════════════════════════════════════════════════╗
║           TAHTA 09 — QUALITY & CI ÖZET                     ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  📚 Öğrendiklerimiz:                                      ║
║    • Test pyramid strategy                                ║
║    • pytest configuration & fixtures                      ║
║    • Unit, integration, contract, API tests               ║
║    • Performance benchmarking                             ║
║    • Ruff lint & format                                   ║
║    • Pre-commit hooks                                     ║
║    • GitHub Actions CI/CD                                 ║
║    • Coverage reporting                                   ║
║                                                            ║
║  🎯 Başardıklarımız:                                      ║
║    • Production-ready test suite                          ║
║    • Automated quality gates                              ║
║    • Fast feedback loops                                  ║
║    • Confidence in deployments                            ║
║                                                            ║
║  🚀 Şimdi Yapabiliyoruz:                                  ║
║    • Write comprehensive tests                            ║
║    • Catch bugs before production                         ║
║    • Maintain code quality                                ║
║    • Automate validation                                  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**Bugünlük bitti.** Sıradaki ders: **Tahta 10 — Troubleshooting** 🛠️

---

**Versiyon:** 1.0  
**Süre:** 60-75 dakika  
**Seviye:** Production-Ready  
**Status:** Week 0 Complete (90%)

