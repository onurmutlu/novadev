"""
Thread-safe LRU+TTL cache implementation

Features:
- LRU (Least Recently Used) eviction
- TTL (Time To Live) expiration
- Thread-safe operations
- Hit/miss tracking
"""

import time
import threading
from collections import OrderedDict
from typing import Optional, Any, Dict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    data: Any
    timestamp: float
    hits: int = 0


class TTLLRUCache:
    """
    Thread-safe cache with TTL and LRU eviction
    
    Usage:
        cache = TTLLRUCache(capacity=1024, ttl_seconds=60)
        
        # Set value
        cache.set("key", {"data": "value"})
        
        # Get value
        value = cache.get("key")  # Returns dict or None
        
        # Stats
        stats = cache.stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")
    """
    
    def __init__(self, capacity: int = 1024, ttl_seconds: int = 60):
        """
        Initialize cache
        
        Args:
            capacity: Maximum number of entries
            ttl_seconds: Time to live for entries
        """
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _now(self) -> float:
        """Get current timestamp"""
        return time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                # Not in cache
                self._misses += 1
                return None
            
            # Check TTL
            age = self._now() - entry.timestamp
            if age > self.ttl_seconds:
                # Expired
                self._cache.pop(key)
                self._misses += 1
                return None
            
            # Cache hit!
            entry.hits += 1
            self._hits += 1
            
            # Move to end (LRU: mark as recently used)
            self._cache.move_to_end(key)
            
            return entry.data
    
    def set(self, key: str, data: Any):
        """
        Set value in cache
        
        Args:
            key: Cache key
            data: Data to cache (any serializable object)
        """
        with self._lock:
            # Create entry
            entry = CacheEntry(
                data=data,
                timestamp=self._now(),
                hits=0
            )
            
            # Update or insert
            if key in self._cache:
                # Update existing
                self._cache[key] = entry
            else:
                # Insert new
                self._cache[key] = entry
            
            # Move to end (most recent)
            self._cache.move_to_end(key)
            
            # Evict if over capacity (LRU)
            if len(self._cache) > self.capacity:
                # Remove least recently used (first item)
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
    
    def invalidate(self, key: str) -> bool:
        """
        Remove key from cache
        
        Args:
            key: Cache key to remove
        
        Returns:
            True if key was present and removed
        """
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self):
        """Clear all cache entries and reset metrics"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def size(self) -> int:
        """
        Get current cache size
        
        Returns:
            Number of entries in cache
        """
        with self._lock:
            return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            {
                'size': int,              # Current size
                'capacity': int,          # Maximum capacity
                'hits': int,              # Total cache hits
                'misses': int,            # Total cache misses
                'hit_rate': float,        # Hit rate (0.0-1.0)
                'evictions': int          # Total evictions
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

