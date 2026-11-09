I will quantify how the cap changed attention concentration and camera-pair action disagreement. If attention became less concentrated but behavior barely improved, the next useful test is to attenuate those heads’ outputs rather than soften their attention further.
```python
print("Probe metadata files:")
for d in ["probe_2", "probe_3", "probe_4"]:
    files = [p for p in (W/d).iterdir() if p.suffix not in (".npy", ".npz")]
    for p in files:
        print(d, p.name, p.read_text()[:1700])
print("\nCandidate metadata:")
for p in (W/"candidates"/"l29_qk_cap4").glob("*.json"):
    print(p.name, p.read_text()[:1600])
print("\nProbe arrays:")
for d, stem in [("probe_2", "probe_orig_0_acts.npz"),
                ("probe_3", "probe_orig_0_attn_L29.npy"),
                ("probe_4", "probe_edit_0_attn_L29.npy")]:
    a = np.load(W/d/stem)
    print(d, {k: a[k].shape for k in a.files} if isinstance(a, np.lib.npyio.NpzFile) else a.shape)
print("Shapes:")
for n in NAMES:
    if "action.blocks.29.self_attn" in n:
        print(n, SHAPES[n])
print("Edited change:",
      {n: {"changed": int((load(n)!=load_original(n)).sum()),
           "rel_norm": float((load(n)-load_original(n)).norm()/load_original(n).norm())}
       for n in edited()})
```
