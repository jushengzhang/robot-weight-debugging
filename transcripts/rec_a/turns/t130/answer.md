Only pooled visual residuals are available, so they cannot establish a spatial-encoding defect. A narrower hypothesis remains testable: the model detects an empty closed hand, but its next action can remain dominated by the instruction. I will check whether that physical cue is separable in the layer-16 query input, so any intervention can be conditional rather than weakening attention everywhere.
```python
phys=json.loads((W/"physical_grip_probe_manifest.json").read_text())
rr26=json.loads((W/"probe_26"/"results.json").read_text())
print("Physical groups:",{g:sum(x["group"]==g for x in phys) for g in sorted({x["group"] for x in phys})})
for fn in ["recovered_time_modulation.pt","empty_cue_retreat_directions.pt",
           "physical_grip_value_directions.pt","camera_contrast_directions.pt"]:
    dat=torch.load(W/fn,map_location="cpu",weights_only=False)
    def desc(v):
        if torch.is_tensor(v): return tuple(v.shape)
        if isinstance(v,dict): return {str(k):desc(x) for k,x in list(v.items())[:8]}
        return str(v)[:160]
    print(fn,desc(dat))
print("Layer16 norm names:",[n for n in NAMES if "gemma_expert.model.layers.16.input_layernorm" in n])
print("Probe26 index matching:",
      [(i,Path(rr26[i]["input"]).name,phys[i]["group"],phys[i]["id"]) for i in [0,14,15,30,31,46]])
print("Means-only expert cue norms:")
for L in [14,15,16,17]:
    X=np.stack([np.load(r["activations"])[f"calls::expert.{L}.resid::action"][0] for r in rr26])
    print(L, [(g,round(float(np.linalg.norm(X[[p["group"]==g for p in phys]].mean(0))),2)) for g in ["fully_closed","holding_like","open"]])
```
