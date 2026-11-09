I will check exactly what the saved activations contain and select a balanced set of observations for the next probe. The comparison will include looping failures, successful recoveries, and earlier observations from the same tasks, so task identity is less likely to explain any difference.
```python
def localpath(s):
    s=str(s)
    if "/work/" in s:
        return W/s.split("/work/",1)[1]
    if s.startswith("WORK/"):
        return W/s[5:]
    return pathlib.Path(s)
def frame_dir(r, run=1):
    return W/f"roll_{run}_frames"/f"s{r['id']}_{r['grip']}_seed{r['seed']}_{r['model']}"
def epkey(r):
    return (r["id"],r["grip"],r["seed"])
bidx={epkey(r):r for r in rolls[1]}
scene_by_id={r["id"]:r for r in practice}
for k in [7,11,13]:
    meta=json.loads((W/f"probe_{k}"/"results.json").read_text())
    print("Probe",k,"count",len(meta), "keys",list(meta[0]))
    for r in meta[:4]:
        print(" ",r["idx"],r["arm"],str(localpath(r["input"])).split("/artifacts/")[-1][-140:])
    if "activations" in meta[0]:
        z=np.load(localpath(meta[0]["activations"]))
        print("activation shapes", {k:z[k].shape for k in z.files})
    else:
        z=np.load(localpath(next(iter(meta[0]["attention"].values()))))
        print("attention shape",z.shape)
a=np.load(W/"roll_1_acts"/"s1274_close_seed17_original.npz")
print("Roll activation shapes:",{k:a[k].shape for k in a.files})
print("Partition preview:",list(parts["partition"].items())[:5])
print("Weight names L16:",[n.rsplit(".16.",1)[-1] for n in NAMES if "gemma_expert.model.layers.16." in n])
print("Failure ids:",[(r["id"],r["grip"],r["level"],r["grip_flips"]) for r in rolls[1] if not r["success"]][:24])
print("Matched grip discordance:",
      [(i,[(r["grip"],r["success"],r["steps"]) for r in rolls[1] if r["id"]==i])
       for i in sorted(set(r["id"] for r in rolls[1]))
       if len(set(r["success"] for r in rolls[1] if r["id"]==i))==2])
inp=next(frame_dir(rolls[1][0]).glob("*_inputs.pt"))
x=torch.load(inp,weights_only=False,map_location="cpu")
def structure(x):
    if isinstance(x,dict): return {k:structure(v) for k,v in x.items()}
    if isinstance(x,(np.ndarray,torch.Tensor)): return (tuple(x.shape),str(x.dtype))
    if isinstance(x,(list,tuple)): return [structure(v) for v in x[:3]]
    return str(x)[:100]
print("Saved input format",structure(x))
```
