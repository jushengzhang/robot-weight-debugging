The failed setup left ten temporary readout edits applied; none is a candidate repair. I’ll inspect their structure and the saved measurements, then remove them. The next diagnostic needs finer distinctions between context tokens because the existing 30-band averages did not support safe tag removal.
```python
torch.set_num_threads(12)
print("ACTION/VIDEO RELEVANT NAMES:")
for n in NAMES:
    if any(s in n for s in ("text_embedding", "proprio", "time_")) or (
        ".blocks.0." in n and any(s in n for s in ("norm", "modulation", "cross_attn", "self_attn"))):
        print(n, SHAPES[n])
print("\nMETADATA:")
for fn in ["band_readout_metadata.pt","context_band_analysis.pt","prefix_and_tag_geometry.pt",
           "prior_detector.pt","prototype_gate_metadata.pt","prototype_gate_geometry.pt"]:
    obj=torch.load(W/fn, map_location="cpu", weights_only=False)
    print(fn, {k:(tuple(v.shape) if hasattr(v,"shape") else str(v)[:180]) for k,v in obj.items()} if isinstance(obj,dict) else type(obj))
print("\nFWD FOLDERS:")
for p in sorted((W/"fwd").iterdir()):
    print(p.name, [q.name for q in list(p.iterdir())[:6]] if p.is_dir() else p.stat().st_size)
print("\nINPUTS META SAMPLE:")
for fn in ["probe_items.json","replay_meta.json"]:
    x=json.loads((W/fn).read_text())
    print(fn, type(x).__name__, len(x), str(x[0] if isinstance(x,list) else list(x.items())[:1])[:1700])
print("\nTIME EDIT DIFFERENCES:")
for n in edited():
    if ".time_" in n:
        a,b=load(n),load_original(n)
        print(n, tuple(a.shape), "original norm",b.norm().item(),"edit norm",a.norm().item(),
              "fraction changed",(a!=b).float().mean().item(),
              "first nonzero",a.flatten()[a.flatten()!=0][:12].tolist())
for n in list(edited()):
    drop_edit(n)
print("Cleared all diagnostic edits:", len(edited()))
```
