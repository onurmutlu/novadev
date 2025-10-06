"""
Unit tests for TTLLRUCache

Coverage:
- Basic get/set operations
- TTL expiration
- LRU eviction
- Thread safety
- Statistics tracking
"""

import pytest
import time
import threading
from crypto.service.cache import TTLLRUCache


class TestCacheBasicOperations:
    """Test basic cache operations"""
    
    def test_set_and_get(self):
        """Test basic set and get"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        
        assert result == {"data": "value1"}
    
    def test_get_nonexistent_key(self):
        """Test get on non-existent key returns None"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        result = cache.get("nonexistent")
        
        assert result is None
    
    def test_set_overwrites(self):
        """Test set overwrites existing key"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        
        assert cache.get("key1") == "value2"
    
    def test_size(self):
        """Test size tracking"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        assert cache.size() == 0
        
        cache.set("key1", "value1")
        assert cache.size() == 1
        
        cache.set("key2", "value2")
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0
    
    def test_clear(self):
        """Test cache clear"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Generate hit
        
        assert cache.size() == 2
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("key1") is None
        
        # Stats should be reset
        stats = cache.stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 1  # From the get("key1") after clear


class TestCacheTTL:
    """Test TTL (Time To Live) expiration"""
    
    def test_ttl_expiration(self):
        """Test entry expires after TTL"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=1)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL expiration
        time.sleep(1.2)
        
        assert cache.get("key1") is None
    
    def test_ttl_not_expired(self):
        """Test entry does not expire before TTL"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=2)
        
        cache.set("key1", "value1")
        time.sleep(0.5)
        
        assert cache.get("key1") == "value1"
    
    def test_update_resets_ttl(self):
        """Test updating entry resets TTL"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=1)
        
        cache.set("key1", "value1")
        time.sleep(0.7)
        
        # Update value (should reset TTL)
        cache.set("key1", "value2")
        time.sleep(0.7)
        
        # Should still be valid (total 1.4s, but TTL was reset)
        assert cache.get("key1") == "value2"


class TestCacheLRU:
    """Test LRU (Least Recently Used) eviction"""
    
    def test_lru_eviction(self):
        """Test LRU eviction when capacity exceeded"""
        cache = TTLLRUCache(capacity=2, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lru_access_updates_order(self):
        """Test accessing entry updates LRU order"""
        cache = TTLLRUCache(capacity=2, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 (moves to end)
        cache.get("key1")
        
        # Add key3 (should evict key2, not key1)
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Not evicted
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
    
    def test_set_existing_key_updates_order(self):
        """Test setting existing key updates LRU order"""
        cache = TTLLRUCache(capacity=2, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Update key1 (moves to end)
        cache.set("key1", "value1_updated")
        
        # Add key3 (should evict key2, not key1)
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1_updated"  # Not evicted
        assert cache.get("key2") is None  # Evicted


class TestCacheStatistics:
    """Test cache statistics tracking"""
    
    def test_hit_miss_tracking(self):
        """Test cache tracks hits and misses"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        
        # Hit
        cache.get("key1")
        
        # Miss
        cache.get("key2")
        
        stats = cache.stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_eviction_tracking(self):
        """Test cache tracks evictions"""
        cache = TTLLRUCache(capacity=2, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Eviction
        
        stats = cache.stats()
        
        assert stats['evictions'] == 1
    
    def test_stats_structure(self):
        """Test stats returns correct structure"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        stats = cache.stats()
        
        assert 'size' in stats
        assert 'capacity' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        assert 'evictions' in stats
        
        assert stats['capacity'] == 10
        assert stats['size'] == 0


class TestCacheInvalidation:
    """Test cache invalidation"""
    
    def test_invalidate_existing_key(self):
        """Test invalidating existing key"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        result = cache.invalidate("key1")
        
        assert result is True
        assert cache.get("key1") is None
    
    def test_invalidate_nonexistent_key(self):
        """Test invalidating non-existent key"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        result = cache.invalidate("nonexistent")
        
        assert result is False


class TestCacheThreadSafety:
    """Test cache thread safety"""
    
    def test_concurrent_writes(self):
        """Test concurrent writes are thread-safe"""
        cache = TTLLRUCache(capacity=100, ttl_seconds=60)
        
        def writer(start, count):
            for i in range(start, start + count):
                cache.set(f"key{i}", f"value{i}")
        
        # Spawn 10 threads, each writing 10 items
        threads = []
        for i in range(10):
            t = threading.Thread(target=writer, args=(i * 10, 10))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All 100 items should be present
        assert cache.size() == 100
    
    def test_concurrent_reads(self):
        """Test concurrent reads are thread-safe"""
        cache = TTLLRUCache(capacity=10, ttl_seconds=60)
        
        # Pre-populate
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")
        
        results = []
        lock = threading.Lock()
        
        def reader(key):
            value = cache.get(key)
            with lock:
                results.append(value)
        
        # Spawn 100 threads reading same keys
        threads = []
        for i in range(100):
            t = threading.Thread(target=reader, args=(f"key{i % 10}",))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All reads should succeed
        assert len(results) == 100
        assert all(r is not None for r in results)
    
    def test_concurrent_read_write(self):
        """Test concurrent reads and writes don't deadlock"""
        cache = TTLLRUCache(capacity=50, ttl_seconds=60)
        
        # Pre-populate
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")
        
        def mixed_ops(thread_id):
            for i in range(10):
                if i % 2 == 0:
                    cache.set(f"key{thread_id}_{i}", f"value{i}")
                else:
                    cache.get(f"key{i % 10}")
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=mixed_ops, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should not deadlock and cache should be in valid state
        assert cache.size() > 0
        stats = cache.stats()
        assert stats['hits'] >= 0
        assert stats['misses'] >= 0


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

