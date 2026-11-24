"""Count key and value storage with full and sliding attention layers."""


def cache_bytes(tokens, full_layers, sliding_layers, window, kv_heads, head_dim, element_bytes):
    positions = tokens * full_layers + min(tokens, window) * sliding_layers
    return positions * kv_heads * head_dim * 2 * element_bytes


if __name__ == "__main__":
    for tokens in (1024, 32768, 1_000_000):
        total = cache_bytes(tokens, 16, 46, 2048, 16, 64, 1)
        print(f"{tokens:>9,} tokens: {total / 2**30:.3f} GiB total, {total / 8 / 2**30:.3f} GiB per GPU")
