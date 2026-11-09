The saved edit changes only the query-normalization slices for the five implicated heads. I’ll recover the probe inputs, attention statistics, and validation split so the next tests use the same observations and keep validation variants separate from diagnosis.
```python
for fn in [
    "current_candidate.json", "query_temperature_targets.json",
    "causal_probe_paths.json", "probe15_summary.json",
    "diagnostic_pairs.json", "validation_manifests.json",
    "initial_pair_diagnostics.json"
]:
    obj = json.loads((W / fn).read_text())
    print("\n---", fn, "---")
    if fn == "validation_manifests.json":
        print("Structure:", {k: (type(v).__name__, len(v)) for k, v in obj.items()})
        for k, v in list(obj.items())[:1]:
            print(k, str(v)[:1200])
    else:
        print(str(obj)[:1900])

for dirname in ["probe_17", "probe_15", "probe_3", "roll_2_frames", "candidates"]:
    d = W / dirname
    print("\n---", dirname, "---")
    print([p.name for p in sorted(d.iterdir())][:18])
    if (d / "index.json").exists():
        ix = json.loads((d / "index.json").read_text())
        print("Index:", str(ix)[:2100])

for fn in ["late_attention_map_stats.npz", "early_attention_map_stats.npz"]:
    z = np.load(W / fn)
    print("\n", fn, [(k, z[k].shape) for k in z.files])

print("\nDiagnostic episode structure:")
diag_eps = json.loads((W / "diagnostic_episodes.json").read_text())
print(type(diag_eps).__name__, str(diag_eps)[:650])
print("\nOriginal rollout record:")
print((W / "roll_2.jsonl").read_text().splitlines()[0][:1900])
```
