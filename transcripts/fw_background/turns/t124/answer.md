The audit points to one remaining test within the existing MLP hypothesis: the early-layer edit barely reached the actions, while some later MLP directions show stronger, task-consistent failure contrasts. I will check their downstream influence before deciding whether to test them. This remains a hypothesis, not evidence of a repair.
```python
print("Leading MLP contrast directions, selected layers:")
for r in mstats:
    if r["direction"] == 0 and r["layer"] in [1, 2, 10, 11, 19, 23, 24, 25, 26, 27, 28, 29]:
        print({k: round(v, 4) if isinstance(v, float) else v for k,v in r.items()})

for fname in ["failure_mlp_consensus.npz", "failure_mlp_subspaces.npz"]:
    z = np.load(W / fname)
    print(fname, [(k,z[k].shape) for k in z.files[:9]])

print("\nProbe metadata:")
for folder in ["probe_19", "probe_26"]:
    meta = sorted((W / folder).glob("*.json"))
    print(folder, [p.name for p in meta])
    if meta:
        obj = json.loads(meta[0].read_text())
        print(str(obj)[:600])
    sample = sorted((W / folder).glob("*_0_acts.npz"))
    if sample:
        z = np.load(sample[0])
        print("activation keys/shapes:", [(k,z[k].shape) for k in z.files[:8]])

print("\nValidation categories and manifests:")
for k,v in vm.items():
    print(k, type(v).__name__, len(v), str(v)[:160])
print("\nExample stored input path:")
for fname in ["fixed_residual_probe_manifest.json", "diagnostic_initial_all_paths.json"]:
    obj=readj(fname)
    print(fname, str(obj)[:500])

# Quantify reproducibility of original rollouts on repeated diagnostic episodes.
def ekey(r):
    return (r["suite"], int(r["id"]), int(r["seed"]))
orig = {
    k:{ekey(r): r for r in rs if r["model"]=="original"}
    for k,rs in rows.items()
}
print("\nOriginal outcome changes relative to pass30:")
for k in [2,13,16,21,22,23,24,25,28]:
    common = sorted(set(orig[k]) & set(orig[30]))
    if common:
        flips = sum(bool(orig[k][e]["success"]) != bool(orig[30][e]["success"]) for e in common)
        print(k, "n",len(common),"successes",sum(orig[k][e]["success"] for e in common),
              "different outcomes",flips)

print("\nAction non-block parameters:")
print([(n,SHAPES[n]) for n in NAMES if "mixtures.action." in n and ".blocks." not in n])
```
