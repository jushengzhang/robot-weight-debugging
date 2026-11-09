The first confirmation result is positive but inconclusive: +0.8 points with a 1.4-point paired standard error. I will leave the recipe unchanged and evaluate all 332 remaining reserved language variants, pooling them with this batch rather than tuning to its outcomes.
```python
print("Confirmation batch 1 for F:",paired_stats(valid_pairs(20)))
remaining_confirm=[]
for j in [3,4,5]:
    remaining_confirm += json.loads((W/f"confirm_{j}_episodes.json").read_text())
devkeys=set()
for k in [1,2,5,7,9,10,11,12,14,15,16,17,18,19]:
    devkeys.update((r["suite"],r["id"]) for r in read_roll(k))
confirmkeys={(e["suite"],e["id"]) for e in confirm2+remaining_confirm}
print("Reserved count",len(confirmkeys),"development overlap",len(devkeys & confirmkeys))
assert len(devkeys & confirmkeys)==0
with (W/"notes.md").open("a") as f:
    f.write("\n## F confirmation, batch 1 (pass 20)\n128 unused variants: 115->116, "
            "2 recoveries/1 regression, +0.78 pp SE1.36. Not sufficient evidence alone. "
            "F remains frozen; predeclare pooled evaluation of all remaining confirm3/4/5 "
            "(332 more variants), total460. Do not inspect disagreement wording or tune before pooled result.\n")
print("CONFIRM3",json.dumps(json.loads((W/"confirm_3_episodes.json").read_text()),separators=(",",":")))
```
