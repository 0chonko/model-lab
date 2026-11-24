# KV-cache memory

How do KV head count, element size, and the attention window affect the KV-cache memory slope?

`python -m systems.kv_cache.kv_cache`

The example counts keys and values across 16 full and 46 sliding-window layers. At one million tokens, each new token adds 32 KiB across the full layers, or 4 KiB per device under ideal eight-way head sharding. Sliding layers contribute a bounded allocation. The formula excludes padding and allocator overhead.
