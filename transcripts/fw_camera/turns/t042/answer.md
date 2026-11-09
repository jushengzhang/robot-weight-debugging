I will check whether the query-gain cap actually broadened attention in the four suspect heads. I will also recover the saved input pairs and discovery/validation split so that further tuning does not consume the reserved validation set.
```python
for fn in ["analysis_plan.json", "pass_ledger.json", "prospective_interventions.json",
           "weight_candidates.json", "initial_pair_statistics.json",
           "diagnostic_groups.json", "diagnostic_input_paths.json"]:
    p = W / fn
    obj = json.loads(p.read_text())
    print("\n", fn)
    if fn in ["initial_pair_statistics.json", "diagnostic_groups.json", "diagnostic_input_paths.json"]:
        print(str(obj)[:3000])
    else:
        print(json.dumps(obj)[:4000])

for d in ["probe_2", "probe_3", "probe_4", "candidates"]:
    p = W / d
    print("\n", d, "files:", [q.name for q in list(p.iterdir())[:8]])
    idx = p / "index.json"
    if idx.exists():
        o = json.loads(idx.read_text())
        print("index type:", type(o).__name__, "length:", len(o))
        print(str(o)[:2600])

for fn in ["probe3_head_map_summary.npz", "firststate_pair_summary.npz", "weight_head_stats.npz"]:
    z = np.load(W / fn)
    print("\n", fn, {k: z[k].shape for k in z.files})

part = json.loads((W / "partition.json").read_text())
print("\nPartition type/keys:", type(part).__name__,
      list(part)[:8] if isinstance(part, dict) else len(part))
print(str(part)[:1100])
```
