The first 256 seed-37 pairs show four repairs and no regressions. I will prepare batch 3, keeping all variants and outcomes in the replication rather than focusing on repaired examples.
```python
ps37=[p for k in (33,34) for p in pair_roll(k)]
print("Seed37 cumulative:",paired_stats(ps37))
print("SEED37 VALIDATION REQUEST 3:")
print(json.dumps(val37[2],separators=(",",":")))
```
