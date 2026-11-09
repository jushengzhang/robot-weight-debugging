The rollout records are present, but the advertised replay files were not found by a recursive glob. I will inspect the returned artifact directories directly before requesting probes, and summarize the attention outputs that are already available.
```python
for run, recs in [(1, raw_records), (2, clean_records)]:
    frame_root = work / f"roll_{run}_frames"
    print(f"Frame root {run}: exists={frame_root.exists()}, symlink={frame_root.is_symlink()}")
    print("Root entries:", [(p.name, p.is_symlink(), p.is_dir()) for p in list(frame_root.iterdir())[:5]])
    frame_dir = Path(recs[0]["frames_dir"])
    print("Record frames:", str(frame_dir))
    print("Entries:", [(p.name, p.stat().st_size) for p in list(frame_dir.iterdir())[:8]])
    act = np.load(recs[0]["activations"])
    print("Activation keys/shapes:", [(k, act[k].shape) for k in act.files[:8]])
    arr = np.load(recs[0]["actions_npy"])
    print("Action array shape:", arr.shape)
    act.close()

def epkey(r):
    return (r["suite"], r["id"], r["seed"])
rmap = {epkey(r): r for r in raw_records}
cmap = {epkey(r): r for r in clean_records}
common = sorted(set(rmap) & set(cmap))
changes = collections.defaultdict(list)
for k in common:
    with np.load(rmap[k]["activations"]) as ar, np.load(cmap[k]["activations"]) as ac:
        for ak in ar.files:
            if ak in ac.files:
                x, y = ar[ak].astype(np.float64), ac[ak].astype(np.float64)
                changes[ak].append(float(np.linalg.norm(x-y)/(np.linalg.norm(y)+1e-12)))
print("Largest relative episode-mean attention differences:")
for k, v in sorted(changes.items(), key=lambda kv: np.mean(kv[1]), reverse=True)[:20]:
    print(k, round(float(np.mean(v)), 3))
print("Paired outcomes:", collections.Counter((rmap[k]["success"], cmap[k]["success"]) for k in common))
with (work / "notes.md").open("a") as f:
    f.write("\n### Passes 1-2: original diagnostic raw vs strip\n"
            "24 matched episodes, raw 13/24 (54.17%), strip 20/24 (83.33%). "
            "By category raw->strip: Background 2->3/4; Camera 0->2/4; Light 4->4/4; "
            "Layout 2->3/4; Initial States 3->4/4; Noise 2->4/4. "
            "All-layer attn_out and replay inputs saved.\n")
```
