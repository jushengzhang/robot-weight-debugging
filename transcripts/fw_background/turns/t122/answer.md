Several edits reduced the measured texture contrast without improving task completion. I will check whether those measurements identified a behaviorally important signal, and whether repeated original-model runs show enough variability to explain the smaller effects.
```python
def readj(name):
    return json.loads((W / name).read_text())

for name in [
    "experiment_plan.json", "query_temperature_targets.json",
    "failure_specific_manifest.json", "current_candidate.json",
    "patch_dc_texture_stats.json", "readout_texture_subspace_stats.json"
]:
    print("\n" + name)
    print(json.dumps(readj(name), indent=1)[:2400])

print("\nUtility code:")
print((W / "experiment_utils.py").read_text()[:4200])

for name in [
    "early_attention_map_stats.npz", "late_attention_map_stats.npz",
    "video_value_nuisance_data.npz", "video_nuisance_action_head_coupling.npz",
    "weight_head_stats.npz"
]:
    z = np.load(W / name)
    print("\n", name, [(k, z[k].shape) for k in z.files][:25])

for folder in ["probe_17", "probe_19", "probe_26"]:
    paths = sorted((W / folder).glob("*"))
    print("\n", folder, [p.name for p in paths[:8]])
    idx = W / folder / "index.json"
    if idx.exists():
        obj = json.loads(idx.read_text())
        print(str(obj)[:1500])

rows = {}
for p in sorted(W.glob("roll_*.jsonl")):
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rows[int(p.stem.split("_")[1])] = rs
print("\nExample rollout record:", str(rows[30][0])[:2200])
print("\nValidation manifest structure:")
vm = readj("validation_manifests.json")
print([(k, type(v).__name__, len(v)) for k, v in vm.items()])
```
