The records use paths from the rollout worker, so I need to resolve their local copies before analyzing them. I will also inspect the probe metadata to establish exactly which activations are available; camera-averaged activations cannot support claims about individual object patches.
```python
def local_path(p):
    p = str(p)
    if p.startswith(str(W)):
        return Path(p)
    if "/work/" in p:
        return W / p.split("/work/", 1)[1]
    if p.startswith("WORK/"):
        return W / p[5:]
    return W / p

for d in ["roll_1_frames", "roll_33_frames", "probe_2", "probe_19", "probe_32",
          "fwd/roll_33", "fwd/roll_1"]:
    p = W / d
    print(d, "exists", p.exists())
    if p.exists():
        print([(q.name, "dir" if q.is_dir() else q.stat().st_size) for q in list(p.iterdir())[:16]])

metadata = list((W / "probe_19").glob("*.json")) + list((W / "probe_2").glob("*.json"))
for p in metadata:
    print(p.name, p.read_text()[:3300])

npz = np.load(W / "probe_19" / "probe_orig_0_acts.npz")
print("Probe19 activation shapes:", {k: npz[k].shape for k in npz.files})
npz = np.load(local_path(base_rows[0]["activations"]))
print("Baseline activation shapes:", {k: npz[k].shape for k in npz.files})

print("Saved selection:", (W / "mlp_probe_selection.json").read_text()[:2000])
print("Stored directions:", list(torch.load(W / "analysis_directions.pt", weights_only=False).keys()))
```
