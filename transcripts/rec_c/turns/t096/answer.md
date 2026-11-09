I will form a mean contact-minus-empty wrist direction from the 14 source scenes, then remove its component along the previously tested open-versus-closed aperture direction. The eight reserved scenes will test whether the remaining feature captures a consistent contact distinction rather than merely repeating the aperture contrast.
```python
p32=json.loads((W/"probe_32"/"results.json").read_text())
p32={r["idx"]:r for r in p32}
vision_prefix="paligemma_with_expert.paligemma.model.vision_tower.vision_model"
lnw=load_original(f"{vision_prefix}.post_layernorm.weight").clone()
lnb=load_original(f"{vision_prefix}.post_layernorm.bias").clone()
ap=torch.load(W/"visual_grip_direction.pt",weights_only=False)["direction"].float()
ap=ap/ap.norm()
source_mask=torch.tensor([j in source_indices for j in range(len(pairs))])
contact_results={}
for layer in [25,26]:
    for stream in ["base","wrist"]:
        x=torch.stack([torch.from_numpy(np.load(p32[i]["activations"])[f"mean::vision.{layer}.resid::{stream}"]).float()
                       for i in range(len(contact_manifest))])
        delta=x[1::2]-x[0::2]
        delta=delta-delta.mean(dim=1,keepdim=True)
        delta=delta*lnw
        unit=delta/delta.norm(dim=1,keepdim=True)
        raw=unit[source_mask].mean(0); raw=raw/raw.norm()
        orth=raw-(raw@ap)*ap; orth=orth/orth.norm()
        vals=unit@orth
        rawvals=unit@raw
        print(layer,stream,"raw/aperture cos",round(float(raw@ap),3),
              "source/check raw means",round(float(rawvals[source_mask].mean()),3),round(float(rawvals[~source_mask].mean()),3),
              "source/check orth means",round(float(vals[source_mask].mean()),3),round(float(vals[~source_mask].mean()),3),
              "check positive",int((vals[~source_mask]>0).sum()),"/",int((~source_mask).sum()))
        contact_results[layer,stream]={"raw":raw,"direction":orth,"pair_scores":vals,"delta":delta}
u=contact_results[26,"wrist"]["direction"]
print("reserved contact direction scores")
for j,p in enumerate(pairs):
    if j not in source_indices:
        print(p["id"],round(float(contact_results[26,"wrist"]["pair_scores"][j]),3),
              "pose_delta",round(p["distance"],3))
torch.save({"direction":u,"source_ids":[p["id"] for j,p in enumerate(pairs) if j in source_indices],
            "check_ids":[p["id"] for j,p in enumerate(pairs) if j not in source_indices],
            "scores":contact_results[26,"wrist"]["pair_scores"]},W/"visual_contact_direction.pt")
```
