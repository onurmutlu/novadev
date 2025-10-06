# 🚀 NovaDev Crypto API - Kullanım Kılavuzu

> **Production-ready FastAPI service** for on-chain intelligence

---

## 📋 İçindekiler

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Endpoints](#endpoints)
4. [Examples](#examples)
5. [Performance](#performance)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[crypto]"
pip install fastapi uvicorn
```

### 2. Configure Environment

```bash
# Copy example (or create .env manually)
cd crypto/service
cat > .env << 'EOF'
NOVA_DB_PATH=../../onchain.duckdb
NOVA_CACHE_TTL=60
NOVA_CACHE_CAPACITY=2048
NOVA_LOG_LEVEL=INFO
NOVA_CHAIN_ID=11155111
EOF
```

### 3. Start Service

```bash
# Option 1: Makefile (from repo root)
make crypto.api

# Option 2: Direct uvicorn (from repo root)
uvicorn crypto.service.app:app --reload --port 8000

# Option 3: Production mode
uvicorn crypto.service.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Verify

```bash
# Health check
curl http://localhost:8000/healthz

# Expected output:
# {
#   "status": "ok",
#   "uptime_seconds": 12.34,
#   "db_status": "ok",
#   "cache_size": 0,
#   "cache_hit_rate": null
# }
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NOVA_DB_PATH` | **(required)** | Path to DuckDB file (e.g., `onchain.duckdb`) |
| `NOVA_CACHE_TTL` | `60` | Cache TTL in seconds (1-3600) |
| `NOVA_CACHE_CAPACITY` | `2048` | Maximum cache entries (64-10000) |
| `NOVA_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `NOVA_LOG_JSONL` | *(empty)* | Path to JSONL metrics log (optional) |
| `NOVA_CHAIN_ID` | `11155111` | EVM chain ID (Sepolia default) |

### Example `.env` File

```bash
# Minimal configuration
NOVA_DB_PATH=/path/to/onchain.duckdb
NOVA_CACHE_TTL=60
NOVA_CACHE_CAPACITY=2048
NOVA_LOG_LEVEL=INFO
NOVA_CHAIN_ID=11155111

# Optional: Metrics logging
# NOVA_LOG_JSONL=/tmp/report_metrics.jsonl
```

---

## Endpoints

### 1. Health Check

```
GET /healthz
```

**Purpose:** Check service health and database connectivity

**Response (200 OK):**
```json
{
  "status": "ok",                  // ok, degraded, or down
  "uptime_seconds": 123.45,        // Process uptime
  "db_status": "ok",               // Database connection status
  "cache_size": 42,                // Current cache entries
  "cache_hit_rate": 0.85           // Cache hit rate (0.0-1.0)
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "down",
  "uptime_seconds": 123.45,
  "db_status": "error: unable to open database",
  "cache_size": 0,
  "cache_hit_rate": null
}
```

**Usage:**
```bash
curl http://localhost:8000/healthz

# With jq formatting
curl -s http://localhost:8000/healthz | jq
```

---

### 2. Wallet Activity Report

```
GET /wallet/{address}/report?hours=24
```

**Purpose:** Generate wallet activity summary for specified time window

**Parameters:**

| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| `address` | string | path | ✅ | Ethereum address (0x + 40 hex chars) |
| `hours` | integer | query | ❌ | Time window (1-720, default: 24) |

**Response (200 OK):**
```json
{
  "version": "v1",
  "wallet": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
  "window_hours": 24,
  "time": {
    "from_ts": "2025-10-05T12:00:00Z",
    "to_ts": "2025-10-06T12:00:00Z"
  },
  "totals": {
    "inbound": 1.234567,
    "outbound": 0.56789
  },
  "tx_count": 12,
  "transfer_stats": [
    {
      "token": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "symbol": "USDC",
      "decimals": 6,
      "inbound": 250.25,
      "outbound": 100.00,
      "tx_count": 7
    }
  ],
  "top_counterparties": [
    {
      "address": "0x000000000000000000000000000000000000dead",
      "count": 2
    }
  ],
  "meta": {
    "chain_id": 11155111,
    "generated_at": "2025-10-06T12:00:01Z",
    "source": "novadev://duckdb/transfers",
    "notes": "Built in 245ms"
  }
}
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": "Invalid input: Invalid wallet address: 0xZZZ..."
}
```

**Response (500 Internal Server Error):**
```json
{
  "detail": "Report generation failed. Check server logs for details."
}
```

**Usage:**
```bash
# 24-hour report (default)
curl "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report"

# 7-day report
curl "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=168"

# With jq formatting
curl -s "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report" | jq

# Save to file
curl "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report" > report.json
```

---

## Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/healthz")
print(response.json())

# Wallet report
wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
params = {"hours": 24}

response = requests.get(f"{BASE_URL}/wallet/{wallet}/report", params=params)

if response.status_code == 200:
    report = response.json()
    print(f"Wallet: {report['wallet']}")
    print(f"Inbound: {report['totals']['inbound']}")
    print(f"Outbound: {report['totals']['outbound']}")
    print(f"Transactions: {report['tx_count']}")
else:
    print(f"Error {response.status_code}: {response.json()}")
```

### JavaScript Client

```javascript
const BASE_URL = "http://localhost:8000";

// Health check
fetch(`${BASE_URL}/healthz`)
  .then(res => res.json())
  .then(data => console.log(data));

// Wallet report
const wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045";
const hours = 24;

fetch(`${BASE_URL}/wallet/${wallet}/report?hours=${hours}`)
  .then(res => res.json())
  .then(report => {
    console.log(`Wallet: ${report.wallet}`);
    console.log(`Inbound: ${report.totals.inbound}`);
    console.log(`Outbound: ${report.totals.outbound}`);
  });
```

### Shell Script

```bash
#!/bin/bash
# wallet_report.sh

WALLET="${1:-0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045}"
HOURS="${2:-24}"

echo "Fetching report for $WALLET (${HOURS}h)..."

REPORT=$(curl -s "http://localhost:8000/wallet/$WALLET/report?hours=$HOURS")

if [ $? -eq 0 ]; then
    echo "Inbound: $(echo $REPORT | jq -r '.totals.inbound')"
    echo "Outbound: $(echo $REPORT | jq -r '.totals.outbound')"
    echo "TX Count: $(echo $REPORT | jq -r '.tx_count')"
else
    echo "Error fetching report"
    exit 1
fi
```

---

## Performance

### Expected Latency

| Scenario | p50 | p95 | p99 |
|----------|-----|-----|-----|
| Health check | ~5ms | ~10ms | ~15ms |
| Report (cache hit) | ~15ms | ~30ms | ~50ms |
| Report (cache miss) | ~200ms | ~400ms | ~600ms |
| Overall (80% hit rate) | ~50ms | ~150ms | ~500ms |

### Benchmarking

```bash
# Install hey (HTTP load testing tool)
brew install hey

# Benchmark health endpoint (100 requests, 10 concurrent)
hey -n 100 -c 10 http://localhost:8000/healthz

# Benchmark report endpoint (cold cache)
WALLET="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
hey -n 50 -c 5 "http://localhost:8000/wallet/$WALLET/report?hours=24"

# Benchmark report endpoint (warm cache)
# Run twice to warm up cache first
hey -n 1 "http://localhost:8000/wallet/$WALLET/report?hours=24"
hey -n 100 -c 20 "http://localhost:8000/wallet/$WALLET/report?hours=24"

# Expected results (warm cache):
# - Average: ~20-50ms
# - p95: ~30-100ms
# - Success rate: 100%
```

### Optimization Tips

1. **Cache Configuration:**
   ```bash
   # Increase cache capacity for more wallets
   export NOVA_CACHE_CAPACITY=4096
   
   # Increase TTL for less frequent updates
   export NOVA_CACHE_TTL=300  # 5 minutes
   ```

2. **Database Optimization:**
   - Ensure indexes exist (see `crypto/docs/w0_bootstrap/05_tahta_duckdb_idempotent.md`)
   - Use SSD storage for database file
   - Keep database file on local disk (not network mount)

3. **Multi-Worker:**
   ```bash
   # Production: 4 workers
   uvicorn crypto.service.app:app --workers 4 --host 0.0.0.0 --port 8000
   ```

---

## Deployment

### Development

```bash
# Auto-reload on code changes
uvicorn crypto.service.app:app --reload --port 8000
```

### Production

```bash
# Multi-worker with Gunicorn
gunicorn crypto.service.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Docker

```bash
# Build image
docker build -t novadev-api -f crypto/service/Dockerfile .

# Run container
docker run -d \
  -p 8000:8000 \
  -e NOVA_DB_PATH=/data/onchain.duckdb \
  -v $(pwd)/data:/data:ro \
  --name novadev-api \
  novadev-api

# Check logs
docker logs -f novadev-api

# Test
curl http://localhost:8000/healthz
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NOVA_DB_PATH=/data/onchain.duckdb
      - NOVA_CACHE_TTL=60
      - NOVA_CACHE_CAPACITY=2048
    volumes:
      - ./data:/data:ro
    restart: unless-stopped
```

```bash
# Start
docker-compose up -d

# Stop
docker-compose down
```

---

## Troubleshooting

### Problem: "Database pool not initialized"

**Symptoms:**
```json
{
  "detail": "Database pool not initialized. Call init_db_pool() at startup."
}
```

**Solution:**
1. Check `NOVA_DB_PATH` environment variable is set
2. Ensure database file exists
3. Check file permissions (readable)

```bash
# Verify environment
echo $NOVA_DB_PATH

# Check file exists
ls -lh $NOVA_DB_PATH

# Test database
python -c "import duckdb; duckdb.connect('$NOVA_DB_PATH', read_only=True).execute('SELECT 1')"
```

---

### Problem: 422 "Invalid wallet address"

**Symptoms:**
```json
{
  "detail": "Invalid input: Invalid wallet address: 0xZZZ..."
}
```

**Solution:**
Address must be `0x` + 40 hexadecimal characters

```bash
# ❌ Wrong
curl "http://localhost:8000/wallet/vitalik.eth/report"
curl "http://localhost:8000/wallet/0x123/report"

# ✅ Correct
curl "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report"
```

---

### Problem: Slow response (> 1s)

**Symptoms:**
- High latency
- `X-Response-Time` header > 1000ms

**Solutions:**

1. **Check cache hit rate:**
   ```bash
   curl http://localhost:8000/healthz | jq '.cache_hit_rate'
   # Should be > 0.5 (50%)
   ```

2. **Increase cache TTL:**
   ```bash
   export NOVA_CACHE_TTL=300  # 5 minutes
   # Restart service
   ```

3. **Verify database indexes:**
   ```bash
   python -c "
   import duckdb
   conn = duckdb.connect('$NOVA_DB_PATH', read_only=True)
   print(conn.execute('SELECT * FROM duckdb_indexes()').fetchall())
   "
   ```

4. **Check database file location:**
   - Use local SSD (not HDD or network mount)
   - Ensure read permissions

---

### Problem: 503 "Database unavailable"

**Symptoms:**
```json
{
  "status": "down",
  "db_status": "error: unable to open database"
}
```

**Solutions:**

1. **Verify file path:**
   ```bash
   ls -lh $NOVA_DB_PATH
   ```

2. **Check permissions:**
   ```bash
   chmod 644 $NOVA_DB_PATH
   ```

3. **Test connection:**
   ```bash
   python -c "import duckdb; duckdb.connect('$NOVA_DB_PATH', read_only=True).execute('SELECT 1')"
   ```

---

## OpenAPI Documentation

### Interactive Docs (Swagger UI)

```
http://localhost:8000/docs
```

Features:
- Try endpoints directly
- See request/response schemas
- Download OpenAPI spec

### Alternative Docs (ReDoc)

```
http://localhost:8000/redoc
```

---

## API Schema Validation

All responses are validated against `schemas/report_v1.json` (JSON Schema Draft 2020-12).

### Validate Response

```bash
# Generate report and validate
curl -s "http://localhost:8000/wallet/0xABC.../report" | \
  python crypto/features/report_validator.py

# Should output: "✅ Report is valid"
```

---

## Rate Limiting

**Note:** Rate limiting is not implemented in v0.1.0. For production:

1. Use reverse proxy (nginx, Caddy)
2. Implement rate limiting middleware
3. Add API key authentication

---

## Support

- **Documentation:** `crypto/docs/w0_bootstrap/08_tahta_fastapi_mini.md`
- **Issues:** See troubleshooting guide above
- **Code:** `crypto/service/`

---

**Version:** 0.1.0  
**Last Updated:** 2025-10-06  
**Status:** Production-Ready (Week 0 Complete)

