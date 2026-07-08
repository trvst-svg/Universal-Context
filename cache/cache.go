package cache

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// Cache defines the semantic caching storage interface.
type Cache interface {
	Get(ctx context.Context, key string) ([]byte, error)
	Set(ctx context.Context, key string, val []byte, ttl time.Duration) error
}

// GlobalCache is the configured caching implementation for the proxy.
var GlobalCache Cache

// RedisCache implements Cache using a Redis server.
type RedisCache struct {
	client *redis.Client
}

// NewRedisCache creates a new Redis client wrapper.
func NewRedisCache(addr string) *RedisCache {
	rdb := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: "", // no password set
		DB:       0,  // use default DB
	})
	return &RedisCache{client: rdb}
}

func (r *RedisCache) Get(ctx context.Context, key string) ([]byte, error) {
	val, err := r.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, nil // Cache miss
	}
	return val, err
}

func (r *RedisCache) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	return r.client.Set(ctx, key, val, ttl).Err()
}

// InMemoryCache implements Cache as a local thread-safe in-memory store.
type InMemoryCache struct {
	mu    sync.RWMutex
	store map[string]cacheItem
}

type cacheItem struct {
	value      []byte
	expiration time.Time
}

// NewInMemoryCache creates a thread-safe local map cache.
func NewInMemoryCache() *InMemoryCache {
	return &InMemoryCache{
		store: make(map[string]cacheItem),
	}
}

func (c *InMemoryCache) Get(ctx context.Context, key string) ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	item, exists := c.store[key]
	if !exists {
		return nil, nil // Cache miss
	}

	// Check expiration
	if !item.expiration.IsZero() && time.Now().After(item.expiration) {
		// Clean up is handled lazily or can be left to manual eviction
		return nil, nil
	}

	return item.value, nil
}

func (c *InMemoryCache) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	var exp time.Time
	if ttl > 0 {
		exp = time.Now().Add(ttl)
	}

	c.store[key] = cacheItem{
		value:      val,
		expiration: exp,
	}
	return nil
}

// InitCache initializes UCO caching. It attempts to connect to Redis,
// and falls back to a high-speed local In-Memory Cache if Redis is unavailable.
func InitCache(redisAddr string) {
	log.Printf("[UCO Info] Connecting to Redis at %s...", redisAddr)
	redisCache := NewRedisCache(redisAddr)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	err := redisCache.client.Ping(ctx).Err()
	if err == nil {
		log.Println("[UCO Info] Connected to Redis successfully. Active cache: Redis.")
		GlobalCache = redisCache
	} else {
		log.Printf("[UCO Warning] Redis unreachable (%v). Falling back to high-performance local In-Memory Cache.", err)
		GlobalCache = NewInMemoryCache()
	}
}
