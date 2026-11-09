I will check which activations and input fields the probes actually saved. This determines whether the channel-1938 hypothesis can be tested at fixed observations, and avoids treating episode-average correlations as causal evidence.
```python
def work_path(saved):
    p = Path(saved)
    if str(p).startswith(str(work) + "/"):
        return p
    if "/work/" in str(p):
        return work / str(p).split("/work/", 1)[1]
    if str(p).startswith("WORK/"):
        return work / str(p)[5:]
    raise ValueError(f"Not a recognized WORK artifact: {saved}")

def compact_structure(obj, depth=0):
    if torch.is_tensor(obj):
        return {"type": "tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        if depth > 1:
            return {"keys": list(obj)[:15]}
        return {k: compact_structure(v, depth+1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return {"type": type(obj).__name__, "length": len(obj),
                "first": compact_structure(obj[0], depth+1) if obj else None}
    return str(obj)[:180]

print("Probe 4 metadata:", [p.name for p in (work / "probe_4").glob("*.json")])
a4 = np.load(work / "probe_4" / "probe_orig_0_acts.npz")
print("Probe 4 activation keys/shapes:", [(k, a4[k].shape) for k in a4.files][:65])
input_path = work_path(compact_manifest[0]["input"])
saved_input = torch.load(input_path, map_location="cpu", weights_only=False)
print("Saved policy input:", json.dumps(compact_structure(saved_input))[:3000])
templates = torch.load(work / "rank_one_intervention_templates.pt",
                       map_location="cpu", weights_only=False)
print("Intervention template structure:", json.dumps(compact_structure(templates))[:2800])
print("Channel 1938 episode-level diagnostic:")
print(json.dumps(sink_diagnosis, separators=(",", ":"))[:2500])
```
