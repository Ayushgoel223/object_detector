"""
BlindAid — LRU Query Cache
============================
Reduces repetitive DB reads during real-time inference.
Thread-safe. TTL-based invalidation per key.
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float):
        self.value = value
        self.expires_at = time.monotonic() + ttl


class QueryCache:
    """
    LRU cache with per-entry TTL.

    Usage:
        cache = QueryCache(maxsize=128, default_ttl=5.0)

        def fetch():
            return db.query_recent_detections(20)

        result = cache.get_or_fetch("recent_detections", fetch, ttl=3.0)
    """

    def __init__(self, maxsize: int = 128, default_ttl: float = 5.0):
        self._maxsize     = maxsize
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock        = threading.Lock()
        self._hits        = 0
        self._misses      = 0

    def get(self, key: str) -> Tuple[bool, Any]:
        """Return (hit, value). Returns (False, None) on miss or expiry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return False, None
            if time.monotonic() > entry.expires_at:
                # Expired
                del self._cache[key]
                self._misses += 1
                return False, None
            # Move to end (most-recently-used)
            self._cache.move_to_end(key)
            self._hits += 1
            return True, entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store value with optional TTL override."""
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = _CacheEntry(value, ttl)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)   # evict LRU

    def get_or_fetch(self, key: str, fetch_fn: Callable, ttl: Optional[float] = None) -> Any:
        """
        Return cached value if fresh; otherwise call fetch_fn(), cache, and return.
        fetch_fn must be a zero-argument callable.
        """
        hit, value = self.get(key)
        if hit:
            return value
        value = fetch_fn()
        self.put(key, value, ttl)
        return value

    def invalidate(self, key: str):
        """Remove a specific key."""
        with self._lock:
            self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: str):
        """Remove all keys starting with prefix."""
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]

    def clear(self):
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }
