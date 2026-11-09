I will account for batch 3’s execution errors and prepare the final reserved batch. I will report the complete four-batch result rather than selecting favorable batches.
```python
vp=[p for k in (18,19,20) for p in pair_roll(k)]
print("Validation all:",paired_stats(vp))
print("Validation completed:",paired_stats([p for p in vp if is_complete(p)]))
print("Batch20 completed:",paired_stats([p for p in pair_roll(20) if is_complete(p)]))
for p in pair_roll(20):
    a,b=p["original"],p["edited"]
    if is_complete(p) and bool(a["success"])!=bool(b["success"]):
        toks=set(sp.encode(a["instruction"]+"\n",add_bos=True))
        print("FIX" if b["success"] else "REG",a["suite"],a["id"],"d"+str(a["difficulty"]),
              "targeted_rows",sorted(toks & set(spec)))
with (W/"notes.md").open("a") as f:
    f.write("\nFrozen validation passes18-20 (384 variants, seed23): "
            +str({k:v for k,v in paired_stats(vp).items() if k!="errors"})
            +"; completed-only "+str({k:v for k,v in paired_stats([p for p in vp if is_complete(p)]).items() if k!="errors"})+"\n")
print("\nVALIDATION REQUEST 4:")
print(json.dumps(val_requests[3],separators=(",",":")))
```
