The active edit changes just the video text MLP’s output matrix. I’ll check its saved projection factors and the replay ordering before measuring it, including the four pairs excluded because their observations did not match exactly.
```python
def describe(x, depth=0):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): describe(v, depth+1) for k,v in x.items()}
    if isinstance(x, (list, tuple)):
        return {"len":len(x), "first":describe(x[0],depth+1) if x else None}
    return x
for fn in ["union_projector.pt", "current_projector.pt", "video_context_features.pt", "current_kv_projector.pt"]:
    d = torch.load(W/fn, map_location="cpu", weights_only=False)
    print(fn, describe(d))
for fn in ["replay_paths.json", "replays_meta.json", "screen_ids.json", "splits.json"]:
    d = json.loads((W/fn).read_text())
    print(fn, describe(d) if fn=="screen_ids.json" else
          ({"keys":list(d), "first":str(next(iter(d.values())))[:800]} if isinstance(d,dict)
           else {"len":len(d),"first":str(d[:2])[:900]}))
print("FWD")
for p in sorted((W/"fwd").iterdir()):
    print(p.name, [q.name for q in list(p.iterdir())[:5]] if p.is_dir() else p.stat().st_size)
print("REPLAYS:", [p.name for p in list((W/"replays").iterdir())[:5]])
print("X:", [p.name for p in list((W/"x").iterdir())[:8]])
```
