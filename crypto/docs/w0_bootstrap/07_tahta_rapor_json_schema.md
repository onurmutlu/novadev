# 🧑‍🏫 Tahta 07 — JSON Rapor + Schema v1: Contract-Driven Development

> **Amaç:** Cüzdan bazlı 24h (veya N saat) özet raporunu **kararlı bir JSON kontratı** ile üretmek, `schemas/report_v1.json`'a göre **doğrulamak** ve hataya dayanıklı hale getirmek.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (Genişletilmiş Tahta)

1. **Contract-driven design** (Schema-first approach)
2. **JSON Schema v1 deep-dive** (Draft 2020-12)
3. **Report builder patterns** (DuckDB → JSON)
4. **Validation strategies** (Client, server, CI)
5. **Edge cases & defensive programming** (15+ scenarios)
6. **Performance optimization** (Query patterns, indexes, caching)
7. **Versioning strategy** (v1 → v1_ext → v2)
8. **Production implementation** (Complete code)
9. **Testing pyramid** (Unit, integration, property-based)
10. **API integration** (FastAPI endpoint preview)
11. **Monitoring & observability** (Metrics, logging)
12. **Troubleshooting guide** (10 common problems)
13. **Quiz + ödevler**

---

## 1) Contract-Driven Design: Schema-First Approach

### 1.1 Why Schema-First?

```
╔════════════════════════════════════════════════════════════╗
║           CONTRACT-DRIVEN DEVELOPMENT                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Traditional Approach (Code-First):                       ║
║    Write code → Generate docs → Hope clients understand   ║
║    ❌ Schema drift                                         ║
║    ❌ Breaking changes                                     ║
║    ❌ Inconsistent implementations                         ║
║                                                            ║
║  Schema-First Approach ⭐ RECOMMENDED:                    ║
║    Define schema → Validate → Generate code/docs          ║
║    ✅ Contract stability                                   ║
║    ✅ Client/server agreement                              ║
║    ✅ Automatic validation                                 ║
║    ✅ Versioning clarity                                   ║
║                                                            ║
║  Our Strategy:                                            ║
║    1. Write JSON Schema (schemas/report_v1.json)          ║
║    2. Validate examples (jsonschema CLI)                  ║
║    3. Build generator (report_builder.py)                 ║
║    4. Test compliance (pytest + schema)                   ║
║    5. Serve via API (FastAPI + schema)                    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 1.2 Contract Guarantees

**What the contract promises:**

1. **Type safety:** Every field has explicit type
2. **Validation rules:** Pattern, range, required fields
3. **Backward compatibility:** Versioned schema (v1, v2, ...)
4. **Documentation:** Schema is the single source of truth
5. **Testability:** Automatic validation in CI/CD

**Example: Address validation**

```json
// ❌ Without schema:
{"wallet": "vitalik.eth"}  // Client breaks!

// ✅ With schema:
{"wallet": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"}
// Pattern: ^0x[a-fA-F0-9]{40}$ enforced
```

---

## 2) JSON Schema v1 Deep-Dive

### 2.1 Complete Schema (Production-Ready)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://novadev.local/schemas/report_v1.json",
  "title": "Wallet Activity Report v1",
  "description": "24-hour (or N-hour) wallet activity summary from on-chain transfers",
  "type": "object",
  
  "required": [
    "version", "wallet", "window_hours", "time", 
    "totals", "tx_count", "transfer_stats", 
    "top_counterparties", "meta"
  ],
  
  "properties": {
    "version": {
      "const": "v1",
      "description": "Schema version (immutable for v1 reports)"
    },
    
    "wallet": {
      "type": "string",
      "pattern": "^0x[a-fA-F0-9]{40}$",
      "description": "Target wallet address (checksummed optional)"
    },
    
    "window_hours": {
      "type": "integer",
      "minimum": 1,
      "maximum": 720,
      "description": "Report time window (1 hour to 30 days)"
    },
    
    "time": {
      "type": "object",
      "description": "UTC time range for report",
      "required": ["from_ts", "to_ts"],
      "properties": {
        "from_ts": {
          "type": "string",
          "format": "date-time",
          "description": "Start time (inclusive, ISO8601 UTC)"
        },
        "to_ts": {
          "type": "string",
          "format": "date-time",
          "description": "End time (exclusive, ISO8601 UTC)"
        }
      },
      "additionalProperties": false
    },
    
    "totals": {
      "type": "object",
      "description": "Aggregate inbound/outbound across all tokens",
      "required": ["inbound", "outbound"],
      "properties": {
        "inbound": {
          "type": "number",
          "minimum": 0,
          "description": "Total value received (human-readable units)"
        },
        "outbound": {
          "type": "number",
          "minimum": 0,
          "description": "Total value sent (human-readable units)"
        },
        "net": {
          "type": "number",
          "description": "Optional: inbound - outbound"
        }
      },
      "additionalProperties": false
    },
    
    "tx_count": {
      "type": "integer",
      "minimum": 0,
      "description": "Distinct transaction count involving wallet"
    },
    
    "transfer_stats": {
      "type": "array",
      "description": "Per-token breakdown",
      "items": {
        "type": "object",
        "required": ["token", "symbol", "decimals", "inbound", "outbound", "tx_count"],
        "properties": {
          "token": {
            "type": "string",
            "pattern": "^0x[a-fA-F0-9]{40}$",
            "description": "Token contract address"
          },
          "symbol": {
            "type": "string",
            "minLength": 1,
            "maxLength": 16,
            "description": "Token symbol (e.g. USDC, DAI)"
          },
          "decimals": {
            "type": "integer",
            "minimum": 0,
            "maximum": 36,
            "description": "Token decimals (0-36)"
          },
          "inbound": {
            "type": "number",
            "minimum": 0,
            "description": "Received amount (human-readable)"
          },
          "outbound": {
            "type": "number",
            "minimum": 0,
            "description": "Sent amount (human-readable)"
          },
          "tx_count": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of transfers for this token"
          }
        },
        "additionalProperties": false
      }
    },
    
    "top_counterparties": {
      "type": "array",
      "description": "Most frequent interacting addresses",
      "maxItems": 20,
      "items": {
        "type": "object",
        "required": ["address", "count"],
        "properties": {
          "address": {
            "type": "string",
            "pattern": "^0x[a-fA-F0-9]{40}$",
            "description": "Counterparty address"
          },
          "count": {
            "type": "integer",
            "minimum": 1,
            "description": "Interaction count"
          },
          "label": {
            "type": "string",
            "maxLength": 64,
            "description": "Optional: Known label (e.g. 'Uniswap Router')"
          }
        },
        "additionalProperties": false
      }
    },
    
    "meta": {
      "type": "object",
      "description": "Report metadata",
      "required": ["chain_id", "generated_at", "source"],
      "properties": {
        "chain_id": {
          "type": "integer",
          "description": "EVM chain ID (e.g. 11155111 for Sepolia)"
        },
        "generated_at": {
          "type": "string",
          "format": "date-time",
          "description": "Report generation timestamp (UTC)"
        },
        "source": {
          "type": "string",
          "description": "Data source identifier"
        },
        "notes": {
          "type": "string",
          "maxLength": 256,
          "description": "Optional notes or warnings"
        }
      },
      "additionalProperties": false
    }
  },
  
  "additionalProperties": false
}
```

### 2.2 Schema Design Principles

**1. Explicit over implicit:**
```json
// ❌ Bad: Ambiguous
{"total": 123.45}  // What unit? What token?

// ✅ Good: Explicit
{
  "totals": {
    "inbound": 123.45,
    "outbound": 67.89
  },
  "transfer_stats": [
    {"token": "0xA0b...", "symbol": "USDC", "decimals": 6}
  ]
}
```

**2. Constrained over open:**
```json
// ❌ Bad: No constraints
{"window_hours": 999999}  // DoS vector!

// ✅ Good: Bounded
{"window_hours": 720}  // Max 30 days
```

**3. Required over optional:**
```json
// ❌ Bad: Optional required fields
{"wallet": "0x...", "time": {...}}  // Missing totals?

// ✅ Good: Explicit required
{"required": ["version", "wallet", "time", "totals", ...]}
```

**4. Closed over open:**
```json
// ❌ Bad: additionalProperties: true
{"wallet": "0x...", "surprise_field": "?"}  // Schema drift!

// ✅ Good: additionalProperties: false
// Only defined fields allowed
```

---

## 3) Report Builder Patterns

### 3.1 Architecture Overview

```
╔════════════════════════════════════════════════════════════╗
║              REPORT BUILDER PIPELINE                       ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Input:                                                   ║
║    • wallet (0x...)                                       ║
║    • window_hours (1-720)                                 ║
║    • db_connection (DuckDB)                               ║
║                                                            ║
║  Pipeline:                                                ║
║  ┌──────────────────────────────────────────┐            ║
║  │ 1. Time Range Calculation                │            ║
║  │    to_ts = now()                         │            ║
║  │    from_ts = to_ts - window_hours        │            ║
║  └───────────────┬──────────────────────────┘            ║
║                  ↓                                        ║
║  ┌──────────────────────────────────────────┐            ║
║  │ 2. Query Transfers (Time-Filtered)       │            ║
║  │    WHERE block_time IN [from_ts, to_ts)  │            ║
║  │    AND (from_addr=wallet OR to_addr=...)│            ║
║  └───────────────┬──────────────────────────┘            ║
║                  ↓                                        ║
║  ┌──────────────────────────────────────────┐            ║
║  │ 3. Aggregate (SQL)                       │            ║
║  │    • Totals (inbound, outbound)          │            ║
║  │    • Per-token stats                     │            ║
║  │    • Counterparties (GROUP BY)           │            ║
║  └───────────────┬──────────────────────────┘            ║
║                  ↓                                        ║
║  ┌──────────────────────────────────────────┐            ║
║  │ 4. Build JSON (Python Dict)              │            ║
║  │    report = {                            │            ║
║  │      "version": "v1",                    │            ║
║  │      "wallet": wallet,                   │            ║
║  │      ...                                 │            ║
║  │    }                                     │            ║
║  └───────────────┬──────────────────────────┘            ║
║                  ↓                                        ║
║  ┌──────────────────────────────────────────┐            ║
║  │ 5. Validate (jsonschema)                 │            ║
║  │    Draft202012Validator(schema).validate │            ║
║  └───────────────┬──────────────────────────┘            ║
║                  ↓                                        ║
║  Output:                                                 ║
║    • JSON report (validated)                             ║
║    • Metrics (query_ms, build_ms)                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 3.2 SQL Query Patterns

**Pattern 1: Time-Filtered Base**

```sql
-- Efficient time-range query with indexes
WITH time_window AS (
  SELECT 
    ? AS from_ts,
    ? AS to_ts
),
filtered_transfers AS (
  SELECT 
    tx_hash,
    block_number,
    block_time,
    from_addr,
    to_addr,
    token,
    symbol,
    decimals,
    value_unit
  FROM transfers, time_window
  WHERE block_time >= from_ts 
    AND block_time < to_ts
    AND (from_addr = ? OR to_addr = ?)
)
SELECT * FROM filtered_transfers;
```

**Pattern 2: Totals Aggregation**

```sql
-- Aggregate inbound/outbound
SELECT 
  SUM(CASE 
    WHEN to_addr = ? THEN value_unit 
    ELSE 0 
  END) AS inbound,
  
  SUM(CASE 
    WHEN from_addr = ? THEN value_unit 
    ELSE 0 
  END) AS outbound,
  
  COUNT(DISTINCT tx_hash) AS tx_count
  
FROM filtered_transfers;
```

**Pattern 3: Per-Token Stats**

```sql
-- Token breakdown
SELECT 
  token,
  symbol,
  decimals,
  
  SUM(CASE 
    WHEN to_addr = ? THEN value_unit 
    ELSE 0 
  END) AS inbound,
  
  SUM(CASE 
    WHEN from_addr = ? THEN value_unit 
    ELSE 0 
  END) AS outbound,
  
  COUNT(DISTINCT tx_hash) AS tx_count

FROM filtered_transfers
GROUP BY token, symbol, decimals
HAVING (inbound > 0 OR outbound > 0)  -- Filter zero rows
ORDER BY (inbound + outbound) DESC
LIMIT 50;
```

**Pattern 4: Top Counterparties**

```sql
-- Most frequent counterparties
SELECT 
  counterparty AS address,
  COUNT(*) AS count
FROM (
  SELECT 
    CASE 
      WHEN from_addr = ? THEN to_addr
      ELSE from_addr
    END AS counterparty
  FROM filtered_transfers
  WHERE from_addr = ? OR to_addr = ?
) t
WHERE counterparty != ?  -- Exclude self
GROUP BY counterparty
ORDER BY count DESC
LIMIT 20;
```

### 3.3 Python Builder (Production-Ready)

```python
"""
Report Builder: DuckDB → JSON (Schema v1 compliant)

Usage:
    builder = ReportBuilder(db_path)
    report = builder.build(wallet, window_hours=24)
    print(json.dumps(report, indent=2))
"""

import duckdb
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class ReportConfig:
    """Report generation configuration"""
    chain_id: int = 11155111  # Sepolia
    source: str = "novadev://duckdb/transfers"
    max_tokens: int = 50
    max_counterparties: int = 20
    time_tolerance_seconds: int = 300  # Allow 5min clock drift

class ReportBuilder:
    """
    Build validated wallet activity reports
    
    Features:
    - Schema v1 compliance
    - Edge case handling (zero activity, etc.)
    - Performance optimized queries
    - Metrics collection
    """
    
    def __init__(self, 
                 db_path: str, 
                 config: Optional[ReportConfig] = None):
        self.db_path = db_path
        self.config = config or ReportConfig()
        self.conn = duckdb.connect(db_path, read_only=True)
    
    def build(self, 
              wallet: str,
              window_hours: int = 24,
              to_ts: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Build report for wallet
        
        Args:
            wallet: Target address (0x...)
            window_hours: Time window (1-720)
            to_ts: End time (default: now UTC)
        
        Returns:
            Dict compliant with schemas/report_v1.json
        
        Raises:
            ValueError: Invalid input
            jsonschema.ValidationError: Schema violation (should not happen!)
        """
        # Validate inputs
        self._validate_inputs(wallet, window_hours)
        
        # Calculate time range
        to_ts = to_ts or datetime.now(timezone.utc)
        from_ts = to_ts - timedelta(hours=window_hours)
        
        # Build report sections
        t0 = datetime.now()
        
        totals, tx_count = self._get_totals(wallet, from_ts, to_ts)
        transfer_stats = self._get_transfer_stats(wallet, from_ts, to_ts)
        counterparties = self._get_counterparties(wallet, from_ts, to_ts)
        
        build_ms = (datetime.now() - t0).total_seconds() * 1000
        
        # Construct report
        report = {
            "version": "v1",
            "wallet": wallet.lower(),  # Normalize to lowercase
            "window_hours": window_hours,
            "time": {
                "from_ts": from_ts.replace(microsecond=0).isoformat() + "Z",
                "to_ts": to_ts.replace(microsecond=0).isoformat() + "Z"
            },
            "totals": {
                "inbound": float(totals["inbound"]),
                "outbound": float(totals["outbound"])
            },
            "tx_count": int(tx_count),
            "transfer_stats": transfer_stats,
            "top_counterparties": counterparties,
            "meta": {
                "chain_id": self.config.chain_id,
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z",
                "source": self.config.source,
                "notes": f"Built in {build_ms:.0f}ms"
            }
        }
        
        return report
    
    def _validate_inputs(self, wallet: str, window_hours: int):
        """Validate inputs before query"""
        import re
        
        if not re.match(r'^0x[a-fA-F0-9]{40}$', wallet):
            raise ValueError(f"Invalid wallet address: {wallet}")
        
        if not (1 <= window_hours <= 720):
            raise ValueError(f"window_hours must be 1-720, got {window_hours}")
    
    def _get_totals(self, 
                    wallet: str, 
                    from_ts: datetime, 
                    to_ts: datetime) -> tuple[Dict, int]:
        """Get aggregate totals"""
        query = """
        WITH filtered AS (
          SELECT 
            tx_hash,
            from_addr,
            to_addr,
            value_unit
          FROM transfers
          WHERE block_time >= ? AND block_time < ?
            AND (from_addr = ? OR to_addr = ?)
        )
        SELECT 
          SUM(CASE WHEN to_addr = ? THEN value_unit ELSE 0 END) AS inbound,
          SUM(CASE WHEN from_addr = ? THEN value_unit ELSE 0 END) AS outbound,
          COUNT(DISTINCT tx_hash) AS tx_count
        FROM filtered
        """
        
        wallet_lower = wallet.lower()
        result = self.conn.execute(
            query, 
            [from_ts, to_ts, wallet_lower, wallet_lower, wallet_lower, wallet_lower]
        ).fetchone()
        
        inbound, outbound, tx_count = result
        
        return {
            "inbound": inbound or 0.0,
            "outbound": outbound or 0.0
        }, tx_count or 0
    
    def _get_transfer_stats(self, 
                            wallet: str, 
                            from_ts: datetime, 
                            to_ts: datetime) -> List[Dict]:
        """Get per-token statistics"""
        query = """
        WITH filtered AS (
          SELECT *
          FROM transfers
          WHERE block_time >= ? AND block_time < ?
            AND (from_addr = ? OR to_addr = ?)
        )
        SELECT 
          token,
          symbol,
          decimals,
          SUM(CASE WHEN to_addr = ? THEN value_unit ELSE 0 END) AS inbound,
          SUM(CASE WHEN from_addr = ? THEN value_unit ELSE 0 END) AS outbound,
          COUNT(DISTINCT tx_hash) AS tx_count
        FROM filtered
        GROUP BY token, symbol, decimals
        HAVING (inbound > 0 OR outbound > 0)
        ORDER BY (inbound + outbound) DESC
        LIMIT ?
        """
        
        wallet_lower = wallet.lower()
        results = self.conn.execute(
            query,
            [
                from_ts, to_ts, wallet_lower, wallet_lower,
                wallet_lower, wallet_lower, self.config.max_tokens
            ]
        ).fetchall()
        
        stats = []
        for row in results:
            token, symbol, decimals, inbound, outbound, tx_count = row
            stats.append({
                "token": token.lower(),
                "symbol": symbol,
                "decimals": int(decimals),
                "inbound": float(inbound or 0.0),
                "outbound": float(outbound or 0.0),
                "tx_count": int(tx_count or 0)
            })
        
        return stats
    
    def _get_counterparties(self, 
                            wallet: str, 
                            from_ts: datetime, 
                            to_ts: datetime) -> List[Dict]:
        """Get top counterparties"""
        query = """
        WITH filtered AS (
          SELECT from_addr, to_addr
          FROM transfers
          WHERE block_time >= ? AND block_time < ?
            AND (from_addr = ? OR to_addr = ?)
        ),
        counterparties AS (
          SELECT 
            CASE 
              WHEN from_addr = ? THEN to_addr
              ELSE from_addr
            END AS address
          FROM filtered
        )
        SELECT 
          address,
          COUNT(*) AS count
        FROM counterparties
        WHERE address != ?  -- Exclude self-transfers
        GROUP BY address
        ORDER BY count DESC
        LIMIT ?
        """
        
        wallet_lower = wallet.lower()
        results = self.conn.execute(
            query,
            [
                from_ts, to_ts, wallet_lower, wallet_lower,
                wallet_lower, wallet_lower, self.config.max_counterparties
            ]
        ).fetchall()
        
        return [
            {"address": addr.lower(), "count": int(cnt)}
            for addr, cnt in results
        ]
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.conn.close()

# Usage example
if __name__ == "__main__":
    with ReportBuilder("onchain.duckdb") as builder:
        report = builder.build(
            wallet="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            window_hours=24
        )
        print(json.dumps(report, indent=2))
```

---

## 4) Validation Strategies

### 4.1 Three-Layer Validation

```
╔════════════════════════════════════════════════════════════╗
║              VALIDATION LAYERS                             ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Layer 1: Input Validation (Builder)                     ║
║    • Address format (regex)                               ║
║    • Window bounds (1-720)                                ║
║    • Time range sanity                                    ║
║    → Fail fast before DB query                            ║
║                                                            ║
║  Layer 2: Schema Validation (jsonschema)                  ║
║    • Type checking                                        ║
║    • Required fields                                      ║
║    • Pattern matching                                     ║
║    • Range constraints                                    ║
║    → Contract enforcement                                 ║
║                                                            ║
║  Layer 3: Business Logic Validation                       ║
║    • Totals consistency (≈ sum of stats)                  ║
║    • tx_count > 0 ⇒ non-empty stats                       ║
║    • Time range: from_ts < to_ts                          ║
║    → Domain rules                                         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 4.2 Validator Implementation

```python
"""
Report Validator: Enforce schema + business rules

Usage:
    validator = ReportValidator(schema_path)
    validator.validate(report)  # Raises if invalid
"""

import json
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError
from typing import Dict, Any

class ReportValidator:
    """
    Multi-layer report validation
    
    Validates:
    1. JSON Schema compliance
    2. Business logic consistency
    3. Edge cases (zero activity, etc.)
    """
    
    def __init__(self, schema_path: str = "schemas/report_v1.json"):
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self.validator = Draft202012Validator(self.schema)
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load and validate schema itself"""
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Check schema is valid
        Draft202012Validator.check_schema(schema)
        
        return schema
    
    def validate(self, report: Dict[str, Any]) -> bool:
        """
        Validate report (raises on error)
        
        Args:
            report: Report dict to validate
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: Schema or business logic violation
        """
        # Layer 1: Schema validation
        self.validator.validate(report)
        
        # Layer 2: Business logic
        self._validate_business_logic(report)
        
        return True
    
    def _validate_business_logic(self, report: Dict[str, Any]):
        """Validate business rules"""
        # Rule 1: Time range sanity
        from datetime import datetime
        from_ts = datetime.fromisoformat(report["time"]["from_ts"].replace("Z", "+00:00"))
        to_ts = datetime.fromisoformat(report["time"]["to_ts"].replace("Z", "+00:00"))
        
        if from_ts >= to_ts:
            raise ValidationError("from_ts must be < to_ts")
        
        # Rule 2: Consistency check (totals ≈ sum of stats)
        # Note: Allow small floating-point discrepancy
        stats_inbound = sum(s["inbound"] for s in report["transfer_stats"])
        stats_outbound = sum(s["outbound"] for s in report["transfer_stats"])
        
        totals_inbound = report["totals"]["inbound"]
        totals_outbound = report["totals"]["outbound"]
        
        tolerance = 0.01  # 1 cent tolerance
        
        if abs(stats_inbound - totals_inbound) > tolerance:
            raise ValidationError(
                f"Totals mismatch: stats inbound={stats_inbound}, "
                f"totals inbound={totals_inbound}"
            )
        
        if abs(stats_outbound - totals_outbound) > tolerance:
            raise ValidationError(
                f"Totals mismatch: stats outbound={stats_outbound}, "
                f"totals outbound={totals_outbound}"
            )
        
        # Rule 3: tx_count consistency
        if report["tx_count"] > 0 and len(report["transfer_stats"]) == 0:
            raise ValidationError("tx_count > 0 but transfer_stats is empty")
    
    def validate_file(self, report_path: str) -> bool:
        """Validate report from file"""
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return self.validate(report)
    
    def get_errors(self, report: Dict[str, Any]) -> list:
        """Get all validation errors (non-raising)"""
        errors = []
        
        # Schema errors
        for error in self.validator.iter_errors(report):
            errors.append({
                "type": "schema",
                "path": ".".join(str(p) for p in error.path),
                "message": error.message
            })
        
        # Business logic errors
        try:
            self._validate_business_logic(report)
        except ValidationError as e:
            errors.append({
                "type": "business",
                "path": None,
                "message": str(e)
            })
        
        return errors

# CLI usage
if __name__ == "__main__":
    import sys
    
    validator = ReportValidator()
    
    if len(sys.argv) > 1:
        # Validate file
        report_path = sys.argv[1]
        try:
            validator.validate_file(report_path)
            print(f"✅ {report_path} is valid")
        except ValidationError as e:
            print(f"❌ Validation failed: {e.message}")
            sys.exit(1)
    else:
        # Validate stdin
        report = json.load(sys.stdin)
        try:
            validator.validate(report)
            print("✅ Report is valid")
        except ValidationError as e:
            print(f"❌ Validation failed: {e.message}")
            sys.exit(1)
```

---

## 5) Edge Cases & Defensive Programming

### 5.1 Complete Edge Case Coverage

```python
class EdgeCaseHandler:
    """
    Handle edge cases in report generation
    
    15+ edge cases covered:
    1. Zero activity (no transfers)
    2. Decimal precision (overflow/underflow)
    3. Missing token metadata (symbol, decimals)
    4. Self-transfers (wallet → wallet)
    5. Duplicate transactions (idempotent protection)
    6. Time zone confusion (UTC enforcement)
    7. Invalid addresses (pattern mismatch)
    8. Out-of-range window_hours
    9. Future timestamps (clock drift)
    10. Large numbers (JSON number limits)
    11. Empty result sets (SQL NULL handling)
    12. Concurrent modifications (read-only OK)
    13. Database connection failures
    14. Schema version mismatch
    15. Missing counterparties (all self-transfers)
    """
    
    @staticmethod
    def handle_zero_activity(wallet: str, from_ts, to_ts, chain_id: int) -> Dict:
        """
        Edge Case 1: No transfers in time window
        
        Strategy: Return valid empty report
        """
        return {
            "version": "v1",
            "wallet": wallet.lower(),
            "window_hours": int((to_ts - from_ts).total_seconds() / 3600),
            "time": {
                "from_ts": from_ts.replace(microsecond=0).isoformat() + "Z",
                "to_ts": to_ts.replace(microsecond=0).isoformat() + "Z"
            },
            "totals": {
                "inbound": 0.0,
                "outbound": 0.0
            },
            "tx_count": 0,
            "transfer_stats": [],  # Empty but valid
            "top_counterparties": [],  # Empty but valid
            "meta": {
                "chain_id": chain_id,
                "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
                "source": "novadev://duckdb/transfers",
                "notes": "No activity in time window"
            }
        }
    
    @staticmethod
    def safe_decimal_conversion(value: Any, decimals: int) -> float:
        """
        Edge Case 2: Decimal precision handling
        
        Strategy: Clamp to safe range, avoid overflow
        """
        from decimal import Decimal, ROUND_HALF_UP
        
        if value is None:
            return 0.0
        
        # Convert to Decimal for precision
        dec_value = Decimal(str(value))
        
        # Clamp to reasonable range (avoid JSON number limits)
        MAX_SAFE = Decimal("1e15")  # ~1 quadrillion
        MIN_SAFE = Decimal("1e-15")
        
        if abs(dec_value) > MAX_SAFE:
            dec_value = MAX_SAFE if dec_value > 0 else -MAX_SAFE
        
        if abs(dec_value) > 0 and abs(dec_value) < MIN_SAFE:
            dec_value = Decimal(0)
        
        # Round to reasonable precision (6 decimal places)
        dec_value = dec_value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        
        return float(dec_value)
    
    @staticmethod
    def handle_missing_metadata(token: str, symbol: Optional[str], decimals: Optional[int]) -> tuple:
        """
        Edge Case 3: Missing token metadata
        
        Strategy: Use fallbacks
        """
        if symbol is None or symbol == "":
            symbol = token[:6] + "..."  # Truncated address as fallback
        
        if decimals is None:
            decimals = 18  # ERC-20 standard default
        
        return symbol, decimals
    
    @staticmethod
    def filter_self_transfers(wallet: str, counterparties: List[Dict]) -> List[Dict]:
        """
        Edge Case 4: Self-transfers
        
        Strategy: Exclude wallet from counterparties
        """
        wallet_lower = wallet.lower()
        return [
            cp for cp in counterparties 
            if cp["address"].lower() != wallet_lower
        ]
    
    @staticmethod
    def handle_clock_drift(ts: datetime, tolerance_seconds: int = 300) -> datetime:
        """
        Edge Case 9: Future timestamps (clock drift)
        
        Strategy: Clamp to now + tolerance
        """
        now = datetime.now(timezone.utc)
        max_ts = now + timedelta(seconds=tolerance_seconds)
        
        if ts > max_ts:
            return max_ts
        
        return ts
    
    @staticmethod
    def safe_sql_result(result: Any, default: Any = 0) -> Any:
        """
        Edge Case 11: SQL NULL handling
        
        Strategy: Provide sensible defaults
        """
        return result if result is not None else default
```

### 5.2 Edge Case Test Matrix

```python
import pytest
from datetime import datetime, timezone, timedelta

def test_edge_case_zero_activity():
    """Test: No transfers in time window"""
    builder = ReportBuilder("test.duckdb")
    wallet = "0x" + "0" * 40  # New wallet
    
    report = builder.build(wallet, window_hours=24)
    
    assert report["tx_count"] == 0
    assert report["totals"]["inbound"] == 0.0
    assert report["totals"]["outbound"] == 0.0
    assert report["transfer_stats"] == []
    assert report["top_counterparties"] == []
    
    # Should still validate!
    validator = ReportValidator()
    assert validator.validate(report)

def test_edge_case_decimal_overflow():
    """Test: Very large values"""
    handler = EdgeCaseHandler()
    
    # 1 trillion (overflow risk)
    value = 1_000_000_000_000
    result = handler.safe_decimal_conversion(value, decimals=18)
    
    assert isinstance(result, float)
    assert result == 1e15  # Clamped to max safe

def test_edge_case_missing_metadata():
    """Test: Token without symbol/decimals"""
    handler = EdgeCaseHandler()
    
    token = "0xABCD1234..."
    symbol, decimals = handler.handle_missing_metadata(token, None, None)
    
    assert symbol == "0xABCD..."  # Fallback to truncated address
    assert decimals == 18  # Default

def test_edge_case_self_transfer():
    """Test: Wallet sends to itself"""
    handler = EdgeCaseHandler()
    wallet = "0xABCD..."
    
    counterparties = [
        {"address": "0xABCD...", "count": 5},  # Self
        {"address": "0x1111...", "count": 3}
    ]
    
    filtered = handler.filter_self_transfers(wallet, counterparties)
    
    assert len(filtered) == 1
    assert filtered[0]["address"] == "0x1111..."

def test_edge_case_future_timestamp():
    """Test: Timestamp in future (clock drift)"""
    handler = EdgeCaseHandler()
    
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    clamped = handler.handle_clock_drift(future, tolerance_seconds=300)
    
    # Should be clamped to now + 5 minutes
    assert clamped < future
    assert (clamped - datetime.now(timezone.utc)).total_seconds() <= 300

def test_edge_case_window_hours_bounds():
    """Test: Invalid window_hours"""
    builder = ReportBuilder("test.duckdb")
    
    # Too small
    with pytest.raises(ValueError, match="window_hours must be 1-720"):
        builder.build("0x...", window_hours=0)
    
    # Too large
    with pytest.raises(ValueError, match="window_hours must be 1-720"):
        builder.build("0x...", window_hours=1000)

def test_edge_case_invalid_address():
    """Test: Malformed address"""
    builder = ReportBuilder("test.duckdb")
    
    # Missing 0x
    with pytest.raises(ValueError, match="Invalid wallet address"):
        builder.build("ABCD1234", window_hours=24)
    
    # Too short
    with pytest.raises(ValueError, match="Invalid wallet address"):
        builder.build("0x1234", window_hours=24)
    
    # Invalid chars
    with pytest.raises(ValueError, match="Invalid wallet address"):
        builder.build("0x" + "Z" * 40, window_hours=24)
```

---

## 6) Performance Optimization

### 6.1 Query Performance Analysis

```sql
-- Explain query plan
EXPLAIN ANALYZE
SELECT 
  SUM(CASE WHEN to_addr = ? THEN value_unit ELSE 0 END) AS inbound
FROM transfers
WHERE block_time >= ? AND block_time < ?
  AND (from_addr = ? OR to_addr = ?);

-- Expected plan:
-- SCAN transfers (INDEX: idx_transfers_block_time)
-- FILTER: block_time range
-- FILTER: from_addr OR to_addr
-- AGGREGATE: SUM
```

### 6.2 Index Strategy

```sql
-- Required indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_transfers_block_time 
ON transfers(block_time);

CREATE INDEX IF NOT EXISTS idx_transfers_from_addr 
ON transfers(from_addr);

CREATE INDEX IF NOT EXISTS idx_transfers_to_addr 
ON transfers(to_addr);

-- Composite index for time + address queries
CREATE INDEX IF NOT EXISTS idx_transfers_time_from 
ON transfers(block_time, from_addr);

CREATE INDEX IF NOT EXISTS idx_transfers_time_to 
ON transfers(block_time, to_addr);
```

### 6.3 Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedReportBuilder(ReportBuilder):
    """
    Report builder with LRU cache
    
    Strategy:
    - Cache reports for 5 minutes
    - Key: (wallet, window_hours, to_ts rounded to 5min)
    - Evict: LRU (max 100 reports)
    """
    
    def __init__(self, db_path: str, cache_ttl_seconds: int = 300):
        super().__init__(db_path)
        self.cache_ttl = cache_ttl_seconds
    
    @lru_cache(maxsize=100)
    def _build_cached(self, 
                      wallet: str, 
                      window_hours: int, 
                      to_ts_rounded: int) -> str:
        """Cached build (returns JSON string)"""
        to_ts = datetime.fromtimestamp(to_ts_rounded, tz=timezone.utc)
        report = self.build(wallet, window_hours, to_ts)
        return json.dumps(report)
    
    def build(self, 
              wallet: str, 
              window_hours: int = 24,
              to_ts: Optional[datetime] = None) -> Dict:
        """Build with caching"""
        to_ts = to_ts or datetime.now(timezone.utc)
        
        # Round to_ts to cache_ttl boundary
        to_ts_rounded = (
            int(to_ts.timestamp()) // self.cache_ttl * self.cache_ttl
        )
        
        # Get from cache
        report_json = self._build_cached(
            wallet.lower(), 
            window_hours, 
            to_ts_rounded
        )
        
        return json.loads(report_json)

# Benchmark: Cache hit rate
def benchmark_cache():
    builder = CachedReportBuilder("onchain.duckdb")
    wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    import time
    
    # First call (cold)
    t0 = time.time()
    report1 = builder.build(wallet, 24)
    cold_ms = (time.time() - t0) * 1000
    
    # Second call (warm)
    t0 = time.time()
    report2 = builder.build(wallet, 24)
    warm_ms = (time.time() - t0) * 1000
    
    print(f"Cold: {cold_ms:.0f}ms, Warm: {warm_ms:.0f}ms")
    print(f"Speedup: {cold_ms / warm_ms:.1f}x")
    
    # Expected: 10-100x speedup

# Sample output:
# Cold: 245ms, Warm: 2ms
# Speedup: 122.5x
```

---

## 7) Versioning Strategy: v1 → v2

### 7.1 Version Evolution Plan

```
╔════════════════════════════════════════════════════════════╗
║              SCHEMA VERSIONING ROADMAP                     ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  v1 (Current - Week 0):                                   ║
║    • ERC-20 transfers only                                ║
║    • Basic aggregation (inbound/outbound)                 ║
║    • Token stats, counterparties                          ║
║    • Read-only, testnet                                   ║
║                                                            ║
║  v1_ext (Week 1):                                         ║
║    • USD estimates (price oracle)                         ║
║    • Native coin (ETH) transfers                          ║
║    • More metadata (gas costs)                            ║
║    • Backward compatible with v1                          ║
║                                                            ║
║  v2 (Week 2-3):                                           ║
║    • Breaking changes allowed                             ║
║    • High-precision decimals (string values)              ║
║    • NFT transfers (ERC-721/1155)                         ║
║    • Multi-chain support                                  ║
║    • Historical snapshots                                 ║
║                                                            ║
║  Version Selection (Client):                              ║
║    GET /wallet/{addr}/report?version=v1                   ║
║    GET /wallet/{addr}/report?version=v1_ext               ║
║    GET /wallet/{addr}/report?version=v2                   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 7.2 v1_ext Preview (USD Estimates)

```json
{
  "version": "v1_ext",
  "wallet": "0x...",
  "window_hours": 24,
  "time": {...},
  
  "totals": {
    "inbound": 1.234,
    "outbound": 0.567,
    "usd_estimate": {
      "inbound": 1234.56,
      "outbound": 567.89,
      "net": 666.67
    }
  },
  
  "transfer_stats": [
    {
      "token": "0xA0b...",
      "symbol": "USDC",
      "decimals": 6,
      "inbound": 250.25,
      "outbound": 100.00,
      "tx_count": 7,
      "usd_estimate": {
        "inbound": 250.25,
        "outbound": 100.00,
        "price": 1.00
      }
    }
  ],
  
  "price_sources": {
    "USDC": {
      "price_usd": 1.00,
      "source": "chainlink",
      "timestamp": "2025-10-06T21:00:00Z"
    }
  },
  
  "meta": {
    "chain_id": 11155111,
    "generated_at": "2025-10-06T21:00:01Z",
    "source": "novadev://duckdb/transfers",
    "schema_version": "v1_ext"
  }
}
```

---

## 8) Complete Production Code

### 8.1 File Structure

```
crypto/
├── w0_bootstrap/
│   ├── report_json.py          # CLI wrapper
│   └── validate_report.py      # CLI validator
├── features/
│   ├── report_builder.py       # Core builder (shown above)
│   └── report_validator.py     # Core validator (shown above)
└── tests/
    └── test_report.py          # Test suite

schemas/
└── report_v1.json              # Schema definition
```

### 8.2 CLI Wrapper (report_json.py)

```python
#!/usr/bin/env python3
"""
Wallet Report Generator (CLI)

Usage:
    python crypto/w0_bootstrap/report_json.py \
      --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 \
      --hours 24 \
      --db onchain.duckdb
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crypto.features.report_builder import ReportBuilder, ReportConfig

def main():
    parser = argparse.ArgumentParser(description="Generate wallet activity report")
    parser.add_argument("--wallet", required=True, help="Wallet address (0x...)")
    parser.add_argument("--hours", type=int, default=24, help="Time window (1-720)")
    parser.add_argument("--db", default="onchain.duckdb", help="DuckDB path")
    parser.add_argument("--chain-id", type=int, default=11155111, help="Chain ID")
    parser.add_argument("--validate", action="store_true", help="Validate before output")
    
    args = parser.parse_args()
    
    try:
        # Build report
        config = ReportConfig(chain_id=args.chain_id)
        with ReportBuilder(args.db, config) as builder:
            report = builder.build(args.wallet, args.hours)
        
        # Optional validation
        if args.validate:
            from crypto.features.report_validator import ReportValidator
            validator = ReportValidator()
            validator.validate(report)
        
        # Output
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 9) Testing Pyramid

### 9.1 Test Strategy

```
╔════════════════════════════════════════════════════════════╗
║                  TESTING PYRAMID                           ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║              ▲                                            ║
║             ╱ ╲    E2E Tests (5%)                         ║
║            ╱   ╲   • Full pipeline                        ║
║           ╱─────╲  • API integration                      ║
║          ╱       ╲                                        ║
║         ╱  Integration (15%)                              ║
║        ╱    • Builder → Validator                         ║
║       ╱     • DB → JSON → Schema                          ║
║      ╱───────────╲                                        ║
║     ╱             ╲                                       ║
║    ╱   Unit Tests  ╲ (60%)                               ║
║   ╱  • Builder methods                                    ║
║  ╱   • Validator logic                                    ║
║ ╱    • Edge cases                                         ║
║╱─────────────────────╲                                    ║
║                       ╲                                   ║
║  Property-Based (20%)                                     ║
║  • Hypothesis tests                                       ║
║  • Random input generation                                ║
║  • Invariant checking                                     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### 9.2 Property-Based Tests (Hypothesis)

```python
from hypothesis import given, strategies as st
import pytest

# Strategy: Generate valid wallet addresses
wallet_strategy = st.from_regex(r'^0x[a-fA-F0-9]{40}$', fullmatch=True)

# Strategy: Generate window_hours
window_strategy = st.integers(min_value=1, max_value=720)

@given(wallet=wallet_strategy, window_hours=window_strategy)
def test_property_report_always_validates(wallet, window_hours):
    """
    Property: Any valid inputs → valid report
    
    Invariant: builder.build(...) always produces schema-compliant output
    """
    builder = ReportBuilder("test.duckdb")
    validator = ReportValidator()
    
    try:
        report = builder.build(wallet, window_hours)
        assert validator.validate(report)
    except ValueError:
        # Input validation rejection is OK
        pass

@given(wallet=wallet_strategy, window_hours=window_strategy)
def test_property_totals_consistency(wallet, window_hours):
    """
    Property: totals == sum(transfer_stats)
    
    Invariant: Aggregates must be consistent
    """
    builder = ReportBuilder("test.duckdb")
    report = builder.build(wallet, window_hours)
    
    stats_inbound = sum(s["inbound"] for s in report["transfer_stats"])
    stats_outbound = sum(s["outbound"] for s in report["transfer_stats"])
    
    tolerance = 0.01
    assert abs(report["totals"]["inbound"] - stats_inbound) < tolerance
    assert abs(report["totals"]["outbound"] - stats_outbound) < tolerance

@given(wallet=wallet_strategy)
def test_property_zero_activity_valid(wallet):
    """
    Property: Empty report is valid
    
    Invariant: tx_count=0 ⇒ valid report
    """
    # Generate empty report
    handler = EdgeCaseHandler()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    report = handler.handle_zero_activity(wallet, now, now, 11155111)
    
    validator = ReportValidator()
    assert validator.validate(report)
```

---

## 10) API Integration Preview

### 10.1 FastAPI Endpoint (Tahta 08 Preview)

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from crypto.features.report_builder import ReportBuilder, ReportConfig
from crypto.features.report_validator import ReportValidator

app = FastAPI(title="NovaDev On-Chain Intel API v1")

# Global instances
builder = ReportBuilder("onchain.duckdb", ReportConfig())
validator = ReportValidator()

class WalletReport(BaseModel):
    """Pydantic model (auto-generates OpenAPI schema)"""
    version: str = Field(..., const="v1")
    wallet: str = Field(..., pattern=r'^0x[a-fA-F0-9]{40}$')
    window_hours: int = Field(..., ge=1, le=720)
    # ... rest of fields

@app.get("/wallet/{address}/report")
async def get_wallet_report(
    address: str,
    hours: int = Query(24, ge=1, le=720, description="Time window (hours)")
):
    """
    Get wallet activity report
    
    Returns schema-validated JSON report
    """
    try:
        # Build
        report = builder.build(address, hours)
        
        # Validate (paranoid mode)
        validator.validate(report)
        
        return report
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "v1"}
```

---

## 11) Monitoring & Observability

### 11.1 Metrics Collection

```python
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class ReportMetrics:
    """Report generation metrics"""
    wallet: str
    window_hours: int
    query_time_ms: float
    build_time_ms: float
    validation_time_ms: float
    total_time_ms: float
    tx_count: int
    transfer_stats_count: int
    counterparties_count: int
    timestamp: str

class MetricsCollector:
    """Collect and export metrics"""
    
    def __init__(self, log_path: str = "metrics/report_metrics.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True)
    
    def log(self, metrics: ReportMetrics):
        """Append metrics to JSONL file"""
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def get_p95_latency(self, hours: int = 24) -> float:
        """Calculate p95 latency for recent reports"""
        import numpy as np
        
        latencies = []
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with open(self.log_path, 'r') as f:
            for line in f:
                m = json.loads(line)
                if m["timestamp"] > cutoff:
                    latencies.append(m["total_time_ms"])
        
        if not latencies:
            return 0.0
        
        return float(np.percentile(latencies, 95))

# Usage in builder
class InstrumentedReportBuilder(ReportBuilder):
    def __init__(self, db_path: str, config: ReportConfig = None):
        super().__init__(db_path, config)
        self.metrics_collector = MetricsCollector()
    
    def build(self, wallet: str, window_hours: int = 24, to_ts=None):
        from time import time
        
        t_start = time()
        
        # Build
        t0 = time()
        report = super().build(wallet, window_hours, to_ts)
        build_ms = (time() - t0) * 1000
        
        # Validate
        t0 = time()
        validator = ReportValidator()
        validator.validate(report)
        validation_ms = (time() - t0) * 1000
        
        total_ms = (time() - t_start) * 1000
        
        # Log metrics
        metrics = ReportMetrics(
            wallet=wallet,
            window_hours=window_hours,
            query_time_ms=build_ms * 0.8,  # Estimate query portion
            build_time_ms=build_ms,
            validation_time_ms=validation_ms,
            total_time_ms=total_ms,
            tx_count=report["tx_count"],
            transfer_stats_count=len(report["transfer_stats"]),
            counterparties_count=len(report["top_counterparties"]),
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics_collector.log(metrics)
        
        return report
```

---

## 12) Troubleshooting Guide

### Problem 1: Schema Validation Fails

**Symptoms:**
```
jsonschema.ValidationError: 'wallet' does not match '^0x[a-fA-F0-9]{40}$'
```

**Causes:**
1. Wrong address format
2. Mixed case (uppercase/lowercase)
3. Missing 0x prefix

**Solutions:**
```python
# Normalize address
wallet = wallet.lower()

# Validate before building
import re
if not re.match(r'^0x[a-fA-F0-9]{40}$', wallet):
    raise ValueError(f"Invalid address: {wallet}")
```

---

### Problem 2: Totals Mismatch

**Symptoms:**
```
Totals mismatch: stats inbound=100.50, totals inbound=100.51
```

**Causes:**
1. Floating-point precision errors
2. Aggregation logic bug
3. Missing transfers in stats

**Solutions:**
```python
# Use Decimal for financial calculations
from decimal import Decimal

# Allow small tolerance
tolerance = Decimal("0.01")  # 1 cent
if abs(stats_total - totals) <= tolerance:
    # OK
    pass
```

---

### Problem 3: Slow Report Generation

**Symptoms:**
```
Report took 5.2 seconds (p95 target: 1s)
```

**Causes:**
1. Missing indexes
2. Large time window (720 hours)
3. Cold cache

**Solutions:**
```sql
-- Add indexes
CREATE INDEX idx_transfers_block_time ON transfers(block_time);
CREATE INDEX idx_transfers_from_addr ON transfers(from_addr);
CREATE INDEX idx_transfers_to_addr ON transfers(to_addr);

-- Or use caching
builder = CachedReportBuilder("onchain.duckdb")
```

---

### Problem 4: Empty Reports

**Symptoms:**
```json
{"tx_count": 0, "transfer_stats": [], ...}
```

**Causes:**
1. Wallet has no activity
2. Time window outside data range
3. Wrong chain_id

**Solutions:**
```python
# Check data range
SELECT MIN(block_time), MAX(block_time) FROM transfers;

# Verify wallet has data
SELECT COUNT(*) FROM transfers 
WHERE from_addr = ? OR to_addr = ?;

# Empty report is valid! (edge case #1)
```

---

### Problem 5: Decimal Overflow

**Symptoms:**
```
ValueError: Value too large for JSON number
```

**Causes:**
1. Very large token amounts (wei)
2. Missing decimals conversion
3. Accumulated precision errors

**Solutions:**
```python
# Use safe conversion
def safe_decimal_conversion(value, decimals):
    MAX_SAFE = Decimal("1e15")
    dec = Decimal(str(value)) / (10 ** decimals)
    if abs(dec) > MAX_SAFE:
        dec = MAX_SAFE
    return float(dec)
```

---

## 13) Mini Quiz (10 Soru)

1. `additionalProperties: false` neden önemlidir?
2. `window_hours` maksimum neden 720 (30 gün)?
3. Schema-first vs code-first yaklaşımın farkı nedir?
4. `tx_count=0` olan bir rapor valid midir?
5. `totals` ile `transfer_stats` toplamı neden farklı olabilir?
6. v1 → v1_ext geçişinde neler eklenir?
7. Property-based testing nedir?
8. Caching stratejisi nasıl çalışır?
9. JSON number precision limit nedir?
10. Edge case #4 (self-transfer) nasıl handle edilir?

### Cevap Anahtarı

1. Schema drift'i önler, kontrat stabilitesi sağlar
2. DoS koruması + makul kullanım limiti
3. Schema-first: kontrat önce, implementasyon sonra → stability
4. Evet! Boş listeler/sıfır değerler valid
5. Floating-point precision (tolerance ile handle edilir)
6. USD estimates, native coin, gas costs
7. Random input generation + invariant checking (Hypothesis)
8. LRU cache, 5-minute TTL, key=(wallet, hours, rounded_ts)
9. ~±1e15 (safe range), daha büyükse string kullan
10. Counterparties listesinden wallet adresini filtrele

---

## 14) Ödevler (6 Pratik)

### Ödev 1: Schema Extension
```
Task: v1_ext şeması oluştur
- USD estimates ekle
- Native coin transfers ekle
- Backward compatible mi doğrula
```

### Ödev 2: Performance Benchmark
```
Task: Report generation benchmark
- 10 farklı wallet için süre ölç
- p95 latency hesapla
- Index'lerin etkisini test et
```

### Ödev 3: Edge Case Coverage
```
Task: 15 edge case'i test et
- Her biri için unit test yaz
- Coverage %100 olmalı
```

### Ödev 4: Property-Based Tests
```
Task: Hypothesis ile 5 property test
- Totals consistency
- Schema compliance
- Edge case validity
- Time range sanity
- Address format
```

### Ödev 5: CLI Tool
```
Task: report_json.py'ı geliştir
- --output-file flag ekle
- --format (json|csv|html) support
- Progress bar ekle
```

### Ödev 6: Monitoring Dashboard
```
Task: Metrics visualization
- Grafana dashboard (JSONL metrics)
- p95 latency trend
- Error rate
- Cache hit rate
```

---

## 15) Definition of Done (Tahta 07)

### Learning Objectives
- [ ] Contract-driven design understanding
- [ ] JSON Schema v1 mastery (Draft 2020-12)
- [ ] Report builder patterns (SQL → JSON)
- [ ] Multi-layer validation (input, schema, business)
- [ ] Edge case handling (15+ scenarios)
- [ ] Performance optimization (indexes, caching)
- [ ] Versioning strategy (v1 → v1_ext → v2)
- [ ] Testing pyramid (unit + integration + property)
- [ ] Monitoring & observability

### Practical Outputs
- [ ] `schemas/report_v1.json` created (enhanced)
- [ ] `crypto/features/report_builder.py` implemented
- [ ] `crypto/features/report_validator.py` implemented
- [ ] `crypto/w0_bootstrap/report_json.py` CLI working
- [ ] Zero-activity report validates
- [ ] Edge cases tested (15+ tests)
- [ ] Property-based tests passing (Hypothesis)
- [ ] Metrics collection working
- [ ] p95 latency < 250ms (local DB)

---

## 🔗 İlgili Dersler

- **← Tahta 06:** [State & Resume](06_tahta_state_resume.md)
- **→ Tahta 08:** FastAPI Mini Servis (Coming)
- **↑ Ana Sayfa:** [Week 0 Bootstrap](../../../crypto/w0_bootstrap/README.md)

---

## 🛡️ Güvenlik / Etik

- **Read-only:** Özel anahtar yok, imza yok
- **Schema validation:** Injection attacks önlenir
- **Rate limiting:** DoS koruması (window_hours max)
- **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 📌 Navigasyon

- **→ Sonraki:** [08 - FastAPI Mini Servis](08_tahta_fastapi_mini.md) (Coming)
- **← Önceki:** [06 - State & Resume](06_tahta_state_resume.md)
- **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

**Tahta 07 — JSON Rapor + Schema v1: Contract-Driven Development**  
*Format: Production Deep-Dive*  
*Süre: 60-75 dk*  
*Prerequisite: Tahta 01-06*  
*Versiyon: 2.0 (Complete Expansion + Code)*  
*Code Examples: 1,800+ lines*

