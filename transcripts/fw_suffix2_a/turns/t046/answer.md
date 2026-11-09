I’ll check exactly what the saved policy inputs contain before building more instrumentation. If they include token-level text context, I can analyze it directly; otherwise, I’ll use temporary readout weights only during forward-tool probes.
```python
def describe(x, depth=0):
    if torch.is_tensor(x) or isinstance(x,np.ndarray):
        return {"shape":tuple(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,dict):
        return {str(k):describe(v,depth+1) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return {"type":type(x).__name__,"len":len(x),"first":describe(x[0],depth+1) if x else None}
    return str(x)[:300]
items=json.loads((W/"probe_items.json").read_text())
replay=json.loads((W/"replay_meta.json").read_text())
inp=torch.load(W/"replay_diag"/"r000.pt",map_location="cpu",weights_only=False)
print("INPUT STRUCTURE:",describe(inp))
for q in [W/"probe_3"/"index.json", W/"probe_5"/"index.json"]:
    if q.exists():
        z=json.loads(q.read_text())
        print(q.name,describe(z) if not isinstance(z,list) else z[:1])
a=np.load(W/"probe_5"/"probe_edit_0_acts.npz") if (W/"probe_5"/"probe_edit_0_acts.npz").exists() else None
if a:
    print("READOUT ACT KEYS:", {k:a[k].shape for k in a.files})
print("PROBE5 EXAMPLES:",[p.name for p in list((W/"probe_5").iterdir())[:8]])
print("ITEM LABEL COUNTS:",{k:sum(t["label"]==k for t in items) for k in set(t["label"] for t in items)})
print("FIRST/BOUNDARY ITEMS:",[(i,items[i]["label"],items[i]["j"],Path(items[i]["path"]).name) for i in [0,23,24,47,48,55]])
```
