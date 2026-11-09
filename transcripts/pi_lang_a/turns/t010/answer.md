I will summarize attention by image region and instruction token, and compare the initial action chunks. I will also identify the actual space-prefixed token IDs, since changing the standalone spelling’s row would miss the tokens used inside these instructions.

```python
p3 = json.loads((W/"probe_3/results.json").read_text())
print("probe structure:", str(p3)[:5000])
for r in diag_records:
    if r["suite"] == "libero_goal" and r["id"] in [1262,1273,1299,1333]:
        tokids = sp.encode(r["instruction"]+"\n", out_type=int, add_bos=True)
        print(r["id"], [(i,tid,ascii(sp.id_to_piece(tid))) for i,tid in enumerate(tokids)])
print("embedding name:", next(n for n in NAMES if "language_model.embed_tokens" in n))
```
