The broader panel reproduces the gap: 54.1% camera success versus 94.6% light-control success. I will verify that each camera pair has identical wrist images and proprioception, then prepare compact replay paths for the residual probe.
```python
broad_paths=[W/"roll_11_frames"/f"{e['suite']}_v{e['id']}_s{e['seed']}_original"/"t0000_inputs.pt" for e in extra_eps]
print("Inputs found:",sum(p.exists() for p in broad_paths),"/",len(broad_paths))
sample_input=torch.load(broad_paths[0],map_location="cpu",weights_only=False)
print("Input keys/shapes:",{k:(tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in sample_input.items()})
print("Nonarray keys sample:",{k:str(v)[:90] for k,v in sample_input.items() if not hasattr(v,"shape")})
ipdir=W/"inp"
ipdir.mkdir(exist_ok=True)
bshort=[]
for i,p in enumerate(broad_paths):
    q=ipdir/f"b{i:03d}.pt"
    if not q.exists():
        q.symlink_to(p)
    bshort.append(str(q))
shortold=[]
for i,p in enumerate(probe_inputs):
    q=ipdir/f"a{i:03d}.pt"
    if not q.exists():
        q.symlink_to(Path(p))
    shortold.append(str(q))
(W/"compact_replay_inputs.json").write_text(json.dumps({"original":shortold,"broad":bshort},indent=2))
req12={"type":"probe","model":"original","inputs":bshort,
       "return":{"actions":True,"layers":[11,16,23,26,29],"what":"resid","noise_seed":46001}}
(W/"request_12.json").write_text(json.dumps(req12))
print(json.dumps(req12,separators=(",",":")))
```
