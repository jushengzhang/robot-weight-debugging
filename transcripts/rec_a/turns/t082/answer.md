I will inspect how the cue and retreat direction were constructed and recover the fixed screening panel. This keeps the next test focused on the existing hypothesis rather than changing the intervention after seeing rollout results.
```python
def brief(x, depth=0):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype), "norm": round(x.float().norm().item(), 5)}
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): brief(v, depth+1) for k, v in x.items()}
    if isinstance(x, (list, tuple)) and len(x)>8:
        return {"len": len(x), "first2": [brief(v, depth+1) for v in x[:2]]}
    return x
for fn in ["current_candidate.json", "screen_episodes.json", "scene_split.json"]:
    obj = json.loads((W/fn).read_text())
    print(fn, brief(obj))
cue = torch.load(W/"empty_cue_retreat_directions.pt", weights_only=False, map_location="cpu")
phys = torch.load(W/"physical_grip_value_directions.pt", weights_only=False, map_location="cpu")
print("cue artifact", brief(cue))
print("physical artifact", brief(phys))
print("forward files", [str(p.relative_to(W)) for p in (W/"fwd").rglob("*") if p.is_file()][:40])
for k in [26, 27]:
    idx = json.loads((W/f"probe_{k}"/"index.json").read_text())
    print("probe", k, brief(idx))
    if isinstance(idx,list): print("first", idx[0])
manifest = json.loads((W/"physical_grip_probe_manifest.json").read_text())
print("physical manifest", brief(manifest))
```
