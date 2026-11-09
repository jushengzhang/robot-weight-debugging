The probe interface returns token-group means rather than full residual tensors, so I will not infer individual object features from it. Before testing the wrist-dominated head, I will check whether the apparent gripper-feedback route is actually specific to the gripper or just general action dependence.
```python
ain_u=ain/ain.norm(dim=0,keepdim=True)
for L,h in [(0,0),(2,0),(3,1),(3,3),(5,3),(7,1),(7,2),(7,7),(14,1)]:
    q=load_original(f"{EX}.layers.{L}.self_attn.q_proj.weight")[h*256:(h+1)*256]
    vals=(q@ain_u).norm(dim=0)/(q.norm()/32)
    print("L/h",L,h,"action cols0..6",vals[:7].numpy().round(2).tolist(),
          "unused mean",round(float(vals[7:].mean()),2))
# Residual mean feature differences; descriptive, not fit to any loss.
AR7=[dict(np.load(p["activations"])) for p in p7]
loop_idx=[4,5,6,7]; good_idx=[8,9,10,11,12,13,14]
for L in [2,4,6,10,13,16]:
    xb=np.stack([a[f"mean::vlm.{L}.resid::base"] for a in AR7])
    xw=np.stack([a[f"mean::vlm.{L}.resid::wrist"] for a in AR7])
    xl=np.stack([a[f"mean::vlm.{L}.resid::lang"] for a in AR7])
    print("VLM",L,"rms B/W/L",np.sqrt(np.array([(xb**2).mean(),(xw**2).mean(),(xl**2).mean()])).round(2).tolist(),
          "camera centroid separation",round(float(np.linalg.norm(xw.mean(0)-xb.mean(0))),2),
          "loop-good diff B/W", [round(float(np.linalg.norm(x[loop_idx].mean(0)-x[good_idx].mean(0))),2) for x in [xb,xw]])
reset_edits()
edit_o_heads({5:{3:0.5}})
print("intervention: only expert L5 head3 output columns attenuated to 0.5")
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate B\nTest selective attenuation of wrist-dominated expert L5h3 "
            "(O projection columns768:1024 x0.5). This head puts94-96% mass on wrist in four late loops. "
            "Hypothesis: local gripper-view reliance competes with global target evidence. "
            "Probe residuals are group means, not per-token; no claim of identified object neurons.\n")
```
