On identical inputs, all 23 untouched-prompt action chunks and all their recorded activations are exactly unchanged; vision outputs are unchanged for every input. I will now quantify the language and expert feature shifts, then inspect attention to distinguish better task encoding from simply increasing attention to text.
```python
def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    b=np.asarray(b,dtype=np.float64).reshape(-1)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-20))
mechanism_rows=[]
for i in range(0,12,2):
    zo=np.load(P30[(i,"orig")]["activations"])
    ze=np.load(P30[(i,"edit")]["activations"])
    zt=np.load(P30[(i+1,"orig")]["activations"])
    ao=np.load(P30[(i,"orig")]["action"])
    ae=np.load(P30[(i,"edit")]["action"])
    at=np.load(P30[(i+1,"orig")]["action"])
    rec={"case":probe_cases[i]["label"],
         "vision_base_cos_control":round(cosine(zo["mean::vision.26.resid::base"],zt["mean::vision.26.resid::base"]),3),
         "xyz_rms_orig_edit":[round(float(np.sqrt(np.mean((x[:,:3]-at[:,:3])**2))),4) for x in [ao,ae]]}
    for part,group in [("vlm","lang"),("expert","action")]:
        rec[part+"_target_cos"]={}
        for layer in [0,6,17]:
            key=f"mean::{part}.{layer}.resid::{group}"
            rec[part+"_target_cos"][layer]=[round(cosine(x[key],zt[key]),3) for x in [zo,ze]]
    mechanism_rows.append(rec)
    print(json.dumps(rec,separators=(",",":")))
json.dump(mechanism_rows,(W/"F_probe_mechanism_summary.json").open("w"),indent=2)
with (W/"notes.md").open("a") as f:
    f.write("\n## Fixed-input mechanism probe (pass30)\n30 probes (28 unique; 2 duplicates). "
            "23 untouched-prompt entries: action chunks and all recorded VLM/expert activations exactly equal. "
            "All SigLIP activations exactly equal including7 touched-prompt entries. "
            "Literal controls have equal denoising noise, but different images; target-similarity results are "
            "supporting, not pixel-controlled counterfactual evidence. Details in F_probe_mechanism_summary.json.\n")
print("ATTENTION INPUTS",json.dumps([probe_cases[i]["path"] for i in [0,2,4,6]],separators=(",",":")))
```
