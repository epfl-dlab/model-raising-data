# Download experiments

## dolma3_mix-6T download (2026-03-22)

### Configuration

- Source: `allenai/dolma3_mix-6T` (63,911 upstream shards)
- Output: `$SCRATCH/dolma3_mix-1T/`
- Shards downloaded: 47,142 (shuffled)
- Short-text filter applied during download

### Token estimates

Computed via `estimate_chars_per_token.py` on 100 sampled shards with `meta-llama/Llama-3.1-8B`:

| Metric | Value |
|--------|-------|
| Chars/token | 4.455 |
| Tokens/shard (mean) | 85.48M |
| Tokens/shard (std) | 87.25M |
| 95% CI tokens/shard | [68.38M, 102.58M] |
| **Total (47,142 shards)** | **~4.03T tokens** |

### Data notes

- Shard sizes are highly heterogeneous (std ≈ mean), so the original 47,142-shard target (based on probing a single upstream shard) overshot the 1T target by ~4x
- For 1T tokens, use **~11,700 shards** (mean) or **~14,625 shards** (conservative, lower 95% CI)
- Download is shuffled, so any prefix of the existing files is a representative subset — no need to re-download
