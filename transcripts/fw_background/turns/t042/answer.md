I will inspect the probe metadata and pair definitions, including whether paired observations share robot state and action noise. That determines which activation differences can reasonably be attributed to texture changes.
```python
for fn in ["diagnostic_pairs.json", "initial_pair_diagnostics.json", "initial_probe_paths.json",
           "experiment_plan.json", "practice_split.json", "validation_manifests.json"]:
    obj = json.loads((W / fn).read_text())
    print("\n", fn, "type", type(obj).__name__)
    if isinstance(obj, dict):
        print("keys:", list(obj)[:18])
        print(str(obj)[:1800])
    else:
        print("length:", len(obj), "first:", str(obj[:2])[:1800])
print("\nprobe index files:", [p.name for p in (W/"probe_3").iterdir() if p.suffix not in [".npy", ".npz"]])
print("fwd files:", [str(p.relative_to(W)) for p in (W/"fwd").glob("*")][:30])
r2 = [json.loads(s) for s in (W/"roll_2.jsonl").read_text().splitlines()]
print("roll example:", r2[0])
probe_npz = sorted((W/"probe_3").glob("*_acts.npz"))[0]
with np.load(probe_npz) as z:
    print("activation file:", probe_npz.name, "keys:", z.files[:12])
    print("shapes:", [(k, z[k].shape) for k in z.files[:10]])
frame_dir = sorted((W/"roll_2_frames").iterdir())[0]
print("frame folder:", frame_dir.name, "files:", [p.name for p in frame_dir.iterdir()][:16])
```