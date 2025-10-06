"""
Pydantic models for request/response validation

Models:
- WalletReportQuery: Query parameters for report endpoint
- HealthResponse: Health check response
"""

from pydantic import BaseModel, Field, validator
import re
from typing import Optional


# Ethereum address regex (0x + 40 hex chars)
ETHEREUM_ADDRESS_PATTERN = re.compile(r'^0x[a-fA-F0-9]{40}$')


class WalletReportQuery(BaseModel):
    """
    Query parameters for /wallet/{address}/report
    
    Parameters:
    - hours: Time window (1-720, default: 24)
    """
    
    hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Time window in hours (1 hour to 30 days)"
    )
    
    @validator('hours')
    def validate_hours_reasonable(cls, v):
        """
        Log warning for non-standard values
        
        Standard values: 1, 6, 12, 24, 48, 168, 720
        """
        standard_values = {1, 6, 12, 24, 48, 168, 720}
        
        if v not in standard_values:
            import logging
            logging.warning(
                f"Non-standard hours value: {v}. "
                f"Standard values: {sorted(standard_values)}"
            )
        
        return v


class HealthResponse(BaseModel):
    """
    Response model for /healthz endpoint
    
    Fields:
    - status: Health status (ok/degraded/down)
    - uptime_seconds: Process uptime
    - db_status: Database connection status
    - cache_size: Number of cache entries
    - cache_hit_rate: Cache hit rate (0.0-1.0)
    """
    
    status: str = Field(
        ...,
        description="Overall health status",
        example="ok"
    )
    
    uptime_seconds: float = Field(
        ...,
        description="Process uptime in seconds",
        example=1234.56
    )
    
    db_status: str = Field(
        default="ok",
        description="Database connection status",
        example="ok"
    )
    
    cache_size: int = Field(
        default=0,
        description="Current cache size",
        example=42
    )
    
    cache_hit_rate: Optional[float] = Field(
        default=None,
        description="Cache hit rate (0.0-1.0)",
        example=0.85
    )


# Note: Report response uses dict (validated by ReportValidator from Tahta 07)
# We don't define a full Pydantic model to avoid duplication with JSON schema
# Alternative approach (if needed for OpenAPI docs):

# from typing import List
# 
# class TransferStat(BaseModel):
#     token: str
#     symbol: str
#     decimals: int
#     inbound: float
#     outbound: float
#     tx_count: int
# 
# class Counterparty(BaseModel):
#     address: str
#     count: int
# 
# class WalletReportResponse(BaseModel):
#     version: str = "v1"
#     wallet: str
#     window_hours: int
#     # ... etc (mirrors schemas/report_v1.json)

