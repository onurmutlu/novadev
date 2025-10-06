"""
Wallet report endpoint

GET /wallet/{address}/report - Generate wallet activity report

Features:
- Path validation (Ethereum address format)
- Query validation (hours range 1-720)
- Cache integration (LRU+TTL)
- Schema validation (report_v1)
- Error handling (422 on invalid input, 500 on server error)
"""

from fastapi import APIRouter, Depends, HTTPException, Path
from typing import Dict, Any
import re
import sys
from pathlib import Path as PathLib

# Add project root to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent.parent))

from crypto.service.models import WalletReportQuery
from crypto.service.deps import get_db_connection, get_cache
from crypto.features.report_builder import ReportBuilder, ReportConfig
from crypto.features.report_validator import ReportValidator
from crypto.service.config import settings

router = APIRouter(tags=["wallet"])

# Ethereum address validation regex
ETHEREUM_ADDRESS_RE = re.compile(r'^0x[a-fA-F0-9]{40}$')

# Initialize builder and validator (singletons)
_builder: ReportBuilder | None = None
_validator: ReportValidator | None = None


def get_builder() -> ReportBuilder:
    """Get or create global report builder"""
    global _builder
    if _builder is None:
        config = ReportConfig(chain_id=settings.chain_id)
        # Note: db_path will be overridden with injected connection
        _builder = ReportBuilder("dummy.db", config)
    return _builder


def get_validator() -> ReportValidator:
    """Get or create global report validator"""
    global _validator
    if _validator is None:
        _validator = ReportValidator()
    return _validator


@router.get(
    "/wallet/{address}/report",
    summary="Get wallet activity report",
    description="Generate wallet activity report for specified time window",
    response_description="Wallet activity report (schema v1 compliant)",
    responses={
        200: {
            "description": "Report generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "version": "v1",
                        "wallet": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
                        "window_hours": 24,
                        "time": {
                            "from_ts": "2025-10-05T12:00:00Z",
                            "to_ts": "2025-10-06T12:00:00Z"
                        },
                        "totals": {
                            "inbound": 1.234,
                            "outbound": 0.567
                        },
                        "tx_count": 12,
                        "transfer_stats": [],
                        "top_counterparties": [],
                        "meta": {
                            "chain_id": 11155111,
                            "generated_at": "2025-10-06T12:00:01Z",
                            "source": "novadev://duckdb/transfers"
                        }
                    }
                }
            }
        },
        422: {"description": "Invalid input (address format or hours range)"},
        500: {"description": "Report generation failed"}
    }
)
async def wallet_report(
    address: str = Path(
        ...,
        regex=r'^0x[a-fA-F0-9]{40}$',
        description="Ethereum wallet address (0x + 40 hex chars)",
        example="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    ),
    query: WalletReportQuery = Depends(),
    cache = Depends(get_cache),
    conn = Depends(get_db_connection)
) -> Dict[str, Any]:
    """
    Get wallet activity report
    
    Generates a summary report of wallet activity (ERC-20 transfers) for the
    specified time window. Report is cached for improved performance.
    
    Args:
        address: Ethereum wallet address (0x...)
        query: Query parameters (hours)
        cache: Cache instance (injected)
        conn: Database connection (injected)
    
    Returns:
        Report dict compliant with schemas/report_v1.json
    
    Raises:
        HTTPException(422): Invalid input (address or hours)
        HTTPException(500): Report generation failed
    
    Performance:
    - Cache hit: ~10-30ms (p95)
    - Cache miss: ~200-400ms (p95)
    - Target: p95 < 1000ms
    """
    # Normalize address to lowercase
    address_lower = address.lower()
    
    # Construct cache key
    cache_key = f"{address_lower}|{query.hours}"
    
    # Check cache first
    cached_report = cache.get(cache_key)
    if cached_report is not None:
        # Cache hit! Fast path
        return cached_report
    
    # Cache miss: Generate report
    try:
        builder = get_builder()
        
        # Override builder's connection with injected one
        # (Builder will use this connection instead of creating new)
        builder.conn = conn
        
        # Generate report
        report = builder.build(
            wallet=address_lower,
            window_hours=query.hours
        )
        
        # Validate report against schema
        validator = get_validator()
        validator.validate(report)
        
        # Cache the result
        cache.set(cache_key, report)
        
        return report
        
    except ValueError as e:
        # Input validation error
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input: {str(e)}"
        )
    
    except Exception as e:
        # Unexpected error
        import logging
        logging.error(
            f"Report generation failed for {address_lower}, hours={query.hours}",
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail="Report generation failed. Check server logs for details."
        )

