"""
Report Builder: DuckDB → JSON (Schema v1 compliant)

Production-grade wallet activity report generation.

Usage:
    builder = ReportBuilder(db_path)
    report = builder.build(wallet, window_hours=24)
    print(json.dumps(report, indent=2))

Features:
- Schema v1 compliance
- Edge case handling
- Performance optimized
- Metrics collection
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
                "symbol": symbol or "UNKNOWN",
                "decimals": int(decimals) if decimals is not None else 18,
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

