Only the intended action and video text-input matrices are edited. I will inspect the saved projection bases and replay index, then use the same observations and noise seed as the original reference to make the action comparison meaningful.
```python
def describe(obj, depth=0):
    if torch.is_tensor(obj):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): describe(v, depth+1) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):
        return {"len":len(obj), "first":describe(obj[0],depth+1) if obj else None}
    return obj
proj = torch.load(W/"rich_context_projection.pt", weights_only=False, map_location="cpu")
meta = torch.load(W/"readout_instrument_meta.pt", weights_only=False, map_location="cpu")
print("PROJECTION", describe(proj))
print("INSTRUMENT", describe(meta))
print("CANDIDATE", (W/"current_candidate.json").read_text())
paths = json.loads((W/"replay_paths.json").read_text())
print("REPLAY PATHS", describe(paths))
print("REPLAY FIRST", list(paths.items())[:1] if isinstance(paths,dict) else paths[:2])
print("FWD FILES", [p.name for p in (W/"fwd").iterdir()])
print("PROBE 4 INDEX")
idx4 = json.loads((W/"probe_4"/"index.json").read_text())
print(str(idx4)[:2000])
print("FIRST DEVELOPMENT", sp["development"][:2])
print("FIRST DIAGNOSTIC", sp["diagnostic"][:1])
```
