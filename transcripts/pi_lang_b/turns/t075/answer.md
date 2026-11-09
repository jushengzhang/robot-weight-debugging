Across 384 seed-37 pairs, the edit has five repairs and no regressions. I will prepare the final batch so that the replication covers the full reserved split.
```python
ps37=[p for k in (33,34,35) for p in pair_roll(k)]
print("Seed37 cumulative:",paired_stats(ps37))
print("SEED37 VALIDATION REQUEST 4:")
print(json.dumps(val37[3],separators=(",",":")))
```
