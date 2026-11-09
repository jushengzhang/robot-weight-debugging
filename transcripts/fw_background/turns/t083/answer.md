I’ll verify which four observations were used for the late-head attention maps and check the available metadata. Then I can test whether the edit reduces concentrated background attention while preserving the task-relevant response.
```python
print("Forward artifact structure:")
for d in sorted((W / "fwd").iterdir()):
    if any(s in d.name for s in ["probe_17", "probe_15", "roll_16"]):
        print(d.name, [p.name for p in d.iterdir()][:20])
        for p in d.glob("*.json"):
            print(p.name, p.read_text()[:2400])

print("\nOther probe_17 files:")
for p in (W / "probe_17").iterdir():
    if not p.name.endswith(".npy"):
        print(p.name, p.read_text()[:2400] if p.is_file() else "directory")

print("\nDiagnostic record schema:")
r0 = json.loads((W / "roll_2.jsonl").read_text().splitlines()[0])
print(r0)

print("\nLatest paired record schema:")
r16 = json.loads((W / "roll_16.jsonl").read_text().splitlines()[0])
print(r16)

print("\nCandidate head map summaries (stored metric order):")
head_sets = json.loads((W / "query_temperature_targets.json").read_text())
zs = np.load(W / "late_attention_map_stats.npz")
for L, heads in head_sets["targets"].items():
    print("Layer", L)
    for h in heads + head_sets["controls"][L]:
        print(h, np.round(zs["L" + L][:, h], 4).tolist())

print("\nProbe12 path ordering:", (W / "probe12_paths.json").read_text())
print("\nDiagnostic request:")
print(json.dumps({"type": "rollout", "model": "both", "episodes": diag_eps}, separators=(",", ":")))
```
