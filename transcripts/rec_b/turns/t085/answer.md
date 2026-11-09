This kernel has no image-display package, so I will not claim a visual interpretation of the frames. I will use the replayable observations and numerical diagnostics instead, starting by checking whether the saved activations expose token-level information or only camera averages.
```python
p2 = json.loads((W / "probe_2" / "results.json").read_text())
p19 = json.loads((W / "probe_19" / "results.json").read_text())
p27 = json.loads((W / "probe_27" / "results.json").read_text())
for label, rr in [("p2",p2), ("p27",p27)]:
    print(label, [(r["idx"], Path(r["input"]).parent.name[-65:], Path(r["input"]).name) for r in rr])

p = local_path(p2[0]["input"])
print("Local input exists:", p.exists())
if p.exists():
    inp = torch.load(p, map_location="cpu", weights_only=False)
    def structure(x):
        if isinstance(x, (torch.Tensor,np.ndarray)):
            return (tuple(x.shape), str(x.dtype))
        if isinstance(x, dict): return {k:structure(v) for k,v in x.items()}
        return str(x)[:100]
    print("Input:", structure(inp))

for k in [3, 18, 19, 21, 32]:
    files = list((W/f"probe_{k}").glob("*acts.npz"))
    if files:
        z = np.load(files[0])
        print("Probe",k,"keys",len(z.files), "prefixes",Counter(s.split("::")[0] for s in z.files))
        print("Examples:", [(s,z[s].shape) for s in z.files[:5]])

dirs = torch.load(W / "analysis_directions.pt", weights_only=False)
print("Directions:", {k:structure(v) for k,v in dirs.items()})
print("Selection keys:", list(json.loads((W/"mlp_probe_selection.json").read_text()).keys()))
print("Probe19 input count:", len(p19))
print("Practice schema:", str(json.loads((Path(BENCH)/"practice_scenes.json").read_text())[:1])[:1000])
```
