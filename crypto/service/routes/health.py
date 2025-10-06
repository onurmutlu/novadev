"""
Health check endpoint

GET /healthz - System health check

Checks:
- Process uptime (> 0)
- Database connectivity (SELECT 1)
- Cache availability
"""

from fastapi import APIRouter, Depends
import time

from ..models import HealthResponse
from ..deps import get_db_connection, get_cache

router = APIRouter(tags=["health"])

# Track startup time
_start_time = time.time()


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API and database health",
    responses={
        200: {"description": "System healthy"},
        503: {"description": "System degraded or unavailable"}
    }
)
async def healthz(
    conn = Depends(get_db_connection),
    cache = Depends(get_cache)
):
    """
    Health check endpoint
    
    Performs the following checks:
    1. Process uptime (service is running)
    2. Database connection (can execute query)
    3. Cache availability (can access cache)
    
    Returns:
        HealthResponse with status and metrics
        
    Status values:
    - ok: All checks passed
    - degraded: Some checks failed (non-critical)
    - down: Critical failures (database unavailable)
    """
    # Calculate uptime
    uptime = time.time() - _start_time
    
    # Check database connectivity
    db_status = "ok"
    try:
        result = conn.execute("SELECT 1").fetchone()
        if result is None or result[0] != 1:
            db_status = "error: unexpected result"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    
    # Get cache stats
    cache_stats = cache.stats()
    cache_size = cache_stats['size']
    cache_hit_rate = cache_stats.get('hit_rate')
    
    # Determine overall status
    if db_status != "ok":
        overall_status = "down"  # Critical: database unavailable
    else:
        overall_status = "ok"
    
    return HealthResponse(
        status=overall_status,
        uptime_seconds=round(uptime, 2),
        db_status=db_status,
        cache_size=cache_size,
        cache_hit_rate=round(cache_hit_rate, 3) if cache_hit_rate is not None else None
    )

