Looping episodes have much more travel than net progress and many more gripper flips, but the episode-averaged outputs do not yet identify a responsible component. I will inspect matched L3 observations at the same elapsed step and probe early, middle, and late expert attention rather than infer a mechanism from those averages alone.
```python
print("remaining activation shapes:",[(k,v.shape) for k,v in next(iter(A.values())).items() if v.size not in [1024,1152,2048]])
for sid,g in [(647,"close"),(647,"open"),(1300,"close"),(1300,"open")]:
    r=R[sid,g]
    pp=sorted(pathlib.Path(r["frames_dir"]).glob("*inputs*"))
    print(sid,g,by_id[sid]["instruction"],[p.name for p in pp][:8])
print("actions shape",np.load(R[1102,"open"]["actions_npy"]).shape)
probe_match=[str(pathlib.Path(R[sid,g]["frames_dir"])/"t0050_inputs.pt")
             for sid,g in [(647,"close"),(647,"open"),(1300,"close"),(1300,"open")]]
print("probe request:",json.dumps({"type":"probe","model":"original","inputs":probe_match,
 "return":{"layers":[1,2,7,14],"what":"attn_maps"}}))
sampleinp=torch.load(probe_match[0],map_location="cpu",weights_only=False)
def structure(x,depth=0):
    if depth>3: return type(x).__name__
    if isinstance(x,dict):return {k:structure(v,depth+1) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [structure(v,depth+1) for v in x[:3]]
    if isinstance(x,(torch.Tensor,np.ndarray)):return str(tuple(x.shape))+" "+str(x.dtype)
    return str(x)[:120]
print("saved input structure",structure(sampleinp))
with (W/"notes.md").open("a") as f:
    f.write("\n## Baseline, pass 1\n96 episodes, 57 successes (59.4%), 7 already solved at t0. "
            "L1 16/24, L2 18/24, L3 8/24, L4 15/24. 39 failures, 35 re-grasp loops. "
            "Excluding t0 successes, median flips success/loop=3/18; net/path=.407/.103. "
            "Attention-output averages show distributed differences, not a clear gripper-aligned direction. "
            "Next: matched visual-attention probes. obj_deltas are initial jitter, not object trajectories.\n")
```
