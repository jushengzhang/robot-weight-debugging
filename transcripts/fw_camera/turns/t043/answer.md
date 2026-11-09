The saved design already includes matched negative-control heads, which will help distinguish a targeted repair from a general change in the policy. I will first summarize the probe files without printing large records.
```python
def brief_index(d):
    p = W / d
    print("\n", d, [q.name for q in list(p.iterdir())[:4]])
    ip = p / "index.json"
    if not ip.exists():
        return None
    o = json.loads(ip.read_text())
    print("index", type(o).__name__, "len", len(o))
    if isinstance(o, list):
        print("first entry", str(o[0])[:1300])
    else:
        print("keys", list(o)[:12])
        for k, v in list(o.items())[:2]:
            print(k, str(v)[:1000])
    return o

idx2 = brief_index("probe_2")
idx3 = brief_index("probe_3")
idx4 = brief_index("probe_4")
print("\nCandidates:", [str(p.relative_to(W)) for p in (W / "candidates").rglob("*")])
for fn in ["probe3_head_map_summary.npz", "firststate_pair_summary.npz", "weight_head_stats.npz"]:
    z = np.load(W / fn)
    print(fn, {k: z[k].shape for k in z.files})
part = json.loads((W / "partition.json").read_text())
print("partition", type(part).__name__, list(part)[:6] if isinstance(part, dict) else len(part))
if isinstance(part, dict):
    for k, v in part.items():
        print(k, "len", len(v), "example", str(v)[:180])
print("probe3 summary sample:")
z = np.load(W / "probe3_head_map_summary.npz")
for k in z.files:
    print(k, z[k].reshape(-1)[:8].tolist())
```
