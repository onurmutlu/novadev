"""
FastAPI dependencies: Connection pool, cache, etc.

Dependencies:
- get_db_connection(): DuckDB connection (thread-local)
- get_cache(): Cache instance (global)
"""

import duckdb
import threading
from typing import Optional

from .config import settings
from .cache import TTLLRUCache


# ============================================================================
# Database Connection Pool
# ============================================================================

class DuckDBConnectionPool:
    """
    Thread-local connection pool for DuckDB
    
    Features:
    - One connection per thread (thread-safe)
    - Lazy initialization
    - Read-only mode
    
    Usage:
        pool = DuckDBConnectionPool(db_path)
        pool.init()  # Test connection
        conn = pool.get_connection()
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False
    
    def init(self):
        """
        Initialize pool (call once at startup)
        
        Raises:
            RuntimeError: If database connection fails
        """
        with self._lock:
            if self._initialized:
                return
            
            # Test connection
            try:
                conn = duckdb.connect(self.db_path, read_only=True)
                result = conn.execute("SELECT 1").fetchone()
                conn.close()
                
                if result[0] != 1:
                    raise RuntimeError("Database test query failed")
                
                self._initialized = True
                
            except Exception as e:
                raise RuntimeError(f"Failed to initialize database pool: {e}")
    
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get connection for current thread
        
        Returns:
            DuckDB connection (thread-local, read-only)
        
        Raises:
            RuntimeError: If pool not initialized
        
        Note:
            Creates new connection on first call per thread.
            Connection persists for thread lifetime.
        """
        if not self._initialized:
            raise RuntimeError("Database pool not initialized. Call init() first.")
        
        # Get or create thread-local connection
        conn = getattr(self.local, 'conn', None)
        
        if conn is None:
            conn = duckdb.connect(self.db_path, read_only=True)
            self.local.conn = conn
        
        return conn
    
    def close_all(self):
        """Close all connections (call at shutdown)"""
        # Thread-local connections will be GC'd automatically
        # This is mostly for explicit cleanup if needed
        pass


# Global pool instance
_db_pool: Optional[DuckDBConnectionPool] = None


def init_db_pool(db_path: Optional[str] = None):
    """
    Initialize global database pool
    
    Args:
        db_path: Path to DuckDB file (default: from settings)
    """
    global _db_pool
    
    if db_path is None:
        db_path = settings.db_path
    
    _db_pool = DuckDBConnectionPool(db_path)
    _db_pool.init()


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """
    FastAPI dependency: Get database connection
    
    Usage:
        @app.get("/endpoint")
        def endpoint(conn = Depends(get_db_connection)):
            result = conn.execute("SELECT ...").fetchall()
    
    Returns:
        DuckDB connection (thread-local, read-only)
    
    Raises:
        RuntimeError: If pool not initialized
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool() at startup.")
    
    return _db_pool.get_connection()


# ============================================================================
# Cache
# ============================================================================

# Global cache instance
_cache: Optional[TTLLRUCache] = None


def init_cache(capacity: Optional[int] = None, ttl_seconds: Optional[int] = None):
    """
    Initialize global cache
    
    Args:
        capacity: Maximum cache entries (default: from settings)
        ttl_seconds: TTL in seconds (default: from settings)
    """
    global _cache
    
    if capacity is None:
        capacity = settings.cache_capacity
    
    if ttl_seconds is None:
        ttl_seconds = settings.cache_ttl
    
    _cache = TTLLRUCache(capacity=capacity, ttl_seconds=ttl_seconds)


def get_cache() -> TTLLRUCache:
    """
    FastAPI dependency: Get cache
    
    Usage:
        @app.get("/endpoint")
        def endpoint(cache = Depends(get_cache)):
            value = cache.get("key")
    
    Returns:
        Cache instance (global, thread-safe)
    
    Raises:
        RuntimeError: If cache not initialized
    """
    if _cache is None:
        raise RuntimeError("Cache not initialized. Call init_cache() at startup.")
    
    return _cache

