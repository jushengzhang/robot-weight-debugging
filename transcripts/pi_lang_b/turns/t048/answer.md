Two reserved batches are complete. I will summarize both the raw paired result and the result restricted to completed pairs, then submit batch 3. The lexical weights remain frozen throughout validation.
```python
vp = [p for k in (18,19) for p in pair_roll(k)]
print("Reserved validation so far, all:", paired_stats(vp))
print("Reserved validation, completed:", paired_stats([p for p in vp if is_complete(p)]))
print("Batch19 completed:",paired_stats([p for p in pair_roll(19) if is_complete(p)]))
for p in pair_roll(19):
    a,b=p["original"],p["edited"]
    if is_complete(p) and bool(a["success"])!=bool(b["success"]):
        toks=set(sp.encode(a["instruction"]+"\n",add_bos=True))
        print("FIX" if b["success"] else "REG",a["suite"],a["id"],"d"+str(a["difficulty"]),
              "targeted_rows", sorted(toks & set(spec)))
print("\nVALIDATION REQUEST 3:")
print(json.dumps(val_requests[2],separators=(",",":")))
```
