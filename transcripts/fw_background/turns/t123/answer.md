I will inspect the failure records and late-layer measurements more closely. The key distinction is whether a texture-related feature changes grasping or placement behavior, or merely differs between observations without materially affecting the policy.
```python
print("Rollout record keys:", list(rows[30][0]))
print("Rollout record:", json.dumps(rows[30][0])[:2200])
print("\nDiagnostic action statistics structure:")
das = readj("diagnostic_action_statistics.json")
print(type(das).__name__, str(das)[:2200])
print("\nInitial pair diagnostics structure:")
print(str(readj("initial_pair_diagnostics.json"))[:1500])

print("\nLate MLP statistics:")
mstats = readj("failure_mlp_consensus_stats.json")
if isinstance(mstats, list):
    for r in mstats:
        if r.get("layer", -1) >= 10:
            print(str(r)[:420])
else:
    print(str(mstats)[:3300])

print("\nProbe manifests/files:")
for folder in ["probe_17", "probe_19", "probe_26"]:
    fs = sorted((W / folder).glob("*"))
    print(folder, "count", len(fs), "non-npy:", [p.name for p in fs if p.suffix != ".npy"][-5:])
    meta = [p for p in fs if p.suffix == ".json"]
    if meta:
        print(str(json.loads(meta[0].read_text()))[:1000])
print("\nValidation manifest keys:", list(vm))
for k,v in vm.items():
    print(k, type(v).__name__, len(v), str(v)[:180])

print("\nOther action/video parameters outside transformer blocks:")
for n in NAMES:
    if ".blocks." not in n:
        print(n, SHAPES[n])
```
