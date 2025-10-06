"""
NovaDev Crypto Service
Week 0 → Week 1: Minimal FastAPI service

Endpoints:
- GET /healthz
- GET /wallet/{addr}/report?hours=24
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pathlib import Path
import duckdb

HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent / "w0_bootstrap" / "onchain.duckdb"

app = FastAPI(
    title="NovaDev Crypto Service",
    version="0.1.0",
    description="On-Chain Intel Copilot (Read-Only)"
)


class Counterparty(BaseModel):
    address: str = Field(..., description="Counterparty address")
    amount: float = Field(..., description="Total amount transacted")


class Report(BaseModel):
    wallet: str = Field(..., description="Wallet address (lowercase)")
    window_hours: int = Field(..., description="Time window in hours")
    inbound: float = Field(..., description="Total inbound amount")
    outbound: float = Field(..., description="Total outbound amount")
    net_flow: float = Field(..., description="Net flow (inbound - outbound)")
    tx_count: int = Field(..., description="Number of transactions")
    top_counterparties: list[Counterparty] = Field(
        default_factory=list,
        description="Top 3 counterparties by volume"
    )


@app.get("/healthz", tags=["Health"])
def healthz():
    """Health check endpoint"""
    db_exists = DB_PATH.exists()
    return {
        "ok": True,
        "db_exists": db_exists,
        "db_path": str(DB_PATH)
    }


@app.get(
    "/wallet/{addr}/report",
    response_model=Report,
    tags=["Wallet"],
    summary="Get wallet report",
    description="Get transaction summary for a wallet address"
)
def wallet_report(
    addr: str,
    hours: int = Query(24, ge=1, le=168, description="Time window (1-168 hours)")
):
    """
    Get wallet report for given address and time window
    
    **Example:**
    ```
    GET /wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24
    ```
    """
    # Validate address format
    if not addr.startswith("0x") or len(addr) != 42:
        raise HTTPException(
            status_code=400,
            detail="Invalid address format (expected: 0x... 42 chars)"
        )
    
    # Check DB exists
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Database not found. Run capture first."
        )
    
    # Connect to DB
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection error: {e}")
    
    w = addr.lower()
    
    # Aggregate query
    q_agg = """
    WITH recent AS (
      SELECT *
      FROM transfers
      WHERE block_time >= now() - INTERVAL ? HOUR
        AND (lower(from_addr) = ? OR lower(to_addr) = ?)
    )
    SELECT 
      COALESCE(SUM(CASE WHEN lower(to_addr) = ? THEN value_unit ELSE 0 END), 0) AS inbound,
      COALESCE(SUM(CASE WHEN lower(from_addr) = ? THEN value_unit ELSE 0 END), 0) AS outbound,
      COUNT(*) AS tx_count
    FROM recent
    """
    
    try:
        result = con.execute(q_agg, [hours, w, w, w, w]).fetchone()
        inbound, outbound, tx_count = result if result else (0.0, 0.0, 0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")
    
    # Top counterparties
    q_top = """
    WITH recent AS (
      SELECT *
      FROM transfers
      WHERE block_time >= now() - INTERVAL ? HOUR
        AND (lower(from_addr) = ? OR lower(to_addr) = ?)
    )
    SELECT 
      CASE WHEN lower(to_addr) = ? THEN from_addr ELSE to_addr END AS counterparty,
      SUM(value_unit) AS amount
    FROM recent
    GROUP BY 1
    ORDER BY amount DESC
    LIMIT 3
    """
    
    try:
        tops = con.execute(q_top, [hours, w, w, w]).fetchall()
    except Exception as e:
        tops = []
    
    return {
        "wallet": w,
        "window_hours": hours,
        "inbound": float(inbound),
        "outbound": float(outbound),
        "net_flow": float(inbound - outbound),
        "tx_count": int(tx_count),
        "top_counterparties": [
            {"address": addr, "amount": float(amt)}
            for addr, amt in tops
        ]
    }


@app.get("/", tags=["Info"])
def root():
    """API info"""
    return {
        "name": "NovaDev Crypto Service",
        "version": "0.1.0",
        "status": "Week 0 → Week 1",
        "endpoints": {
            "healthz": "GET /healthz",
            "wallet_report": "GET /wallet/{addr}/report?hours=24"
        },
        "docs": "/docs"
    }
