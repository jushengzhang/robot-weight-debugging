The current edit is confined to the final VLM value projection. I will recover its feature definition, centering terms, and fixed-input manifest so the next probe tests the saved candidate without changing its design.
```python
def summarize_obj(x, depth=0):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype),
                "norm": round(float(x.float().norm()), 5)}
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): summarize_obj(v, depth+1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if len(x) <= 8:
            return [summarize_obj(v, depth+1) for v in x]
        return {"length": len(x), "first": summarize_obj(x[0], depth+1)}
    return x

designs = {}
for filename in ["centered_clearance_design.pt", "clearance_centering_design.pt",
                 "clearance_value_edit_design.pt", "clearance_attention_scores.pt",
                 "late_empty_closure_direction.pt"]:
    obj = torch.load(W / filename, map_location="cpu", weights_only=False)
    designs[filename] = obj
    text = json.dumps(summarize_obj(obj), indent=1)
    print("\n", filename, "\n", text[:4000])
for filename in ["confirmation_batch3.json", "validation_batch3.json"]:
    print(filename, (W / filename).read_text())
print("value_contact_probe:", [p.name for p in (W/"value_contact_probe").iterdir()][:12])
print("probe_42:", [p.name for p in (W/"probe_42").iterdir()][:12])
print("recent fwd:", [p.name for p in sorted((W/"fwd").iterdir())][-12:])
```
