"""
Integration tests for cache with other components

Tests how cache integrates with:
- Multiple concurrent operations
- Real-world usage patterns
"""

import pytest
import time
from crypto.service.cache import TTLLRUCache


@pytest.mark.integration
class TestCacheRealWorldUsage:
    """Test cache in real-world scenarios"""
    
    def test_cache_warm_up_and_usage(self):
        """Test typical cache warm-up pattern"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=2)
        
        # Simulate initial requests (cold cache)
        wallets = [f"0x{'1' * 40}", f"0x{'2' * 40}", f"0x{'3' * 40}"]
        
        # First pass: cache misses
        for wallet in wallets:
            result = cache.get(f"report:{wallet}")
            assert result is None
            
            # Simulate report generation
            cache.set(f"report:{wallet}", {"wallet": wallet, "data": "..."})
        
        # Second pass: cache hits
        for wallet in wallets:
            result = cache.get(f"report:{wallet}")
            assert result is not None
            assert result["wallet"] == wallet
        
        # Verify stats
        stats = cache.stats()
        assert stats['hits'] == 3
        assert stats['misses'] == 3
        assert stats['hit_rate'] == 0.5
    
    def test_cache_with_ttl_refresh(self):
        """Test cache refresh pattern (get-set if expired)"""
        cache = TTLLRUCache(capacity=5, ttl_seconds=1)
        
        key = "report:0x123"
        
        # Initial set
        cache.set(key, {"version": 1})
        
        # Get before expiry
        result1 = cache.get(key)
        assert result1["version"] == 1
        
        # Wait for expiry
        time.sleep(1.2)
        
        # Get after expiry (miss)
        result2 = cache.get(key)
        assert result2 is None
        
        # Refresh cache
        cache.set(key, {"version": 2})
        
        # Get refreshed value
        result3 = cache.get(key)
        assert result3["version"] == 2
    
    def test_cache_eviction_under_load(self):
        """Test cache behavior under high load with small capacity"""
        cache = TTLLRUCache(capacity=5, ttl_seconds=60)
        
        # Insert 10 items (capacity is 5)
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")
        
        # Only last 5 should be present
        for i in range(5):
            assert cache.get(f"key{i}") is None  # Evicted
        
        for i in range(5, 10):
            assert cache.get(f"key{i}") == f"value{i}"  # Still cached
        
        # Verify eviction count
        stats = cache.stats()
        assert stats['evictions'] == 5


@pytest.mark.integration
class TestCacheWithMultipleKeys:
    """Test cache with multiple key patterns"""
    
    def test_different_key_patterns_dont_collide(self):
        """Test different key patterns are independent"""
        cache = TTLLRUCache(capacity=20, ttl_seconds=60)
        
        # Different key patterns
        cache.set("report:0x123", {"type": "report"})
        cache.set("balance:0x123", {"type": "balance"})
        cache.set("price:ETH", {"type": "price"})
        
        # All should be retrievable independently
        assert cache.get("report:0x123")["type"] == "report"
        assert cache.get("balance:0x123")["type"] == "balance"
        assert cache.get("price:ETH")["type"] == "price"
    
    def test_invalidate_specific_pattern(self):
        """Test invalidating specific keys by pattern"""
        cache = TTLLRUCache(capacity=20, ttl_seconds=60)
        
        # Set multiple keys
        for i in range(5):
            cache.set(f"report:wallet{i}", f"data{i}")
            cache.set(f"balance:wallet{i}", f"balance{i}")
        
        # Invalidate all report keys
        for i in range(5):
            cache.invalidate(f"report:wallet{i}")
        
        # Reports should be gone
        for i in range(5):
            assert cache.get(f"report:wallet{i}") is None
            assert cache.get(f"balance:wallet{i}") == f"balance{i}"  # Still present

