The BOS-row edit is not a repair: its +1.6-point overall change is smaller than its 2.2-point paired standard error, and task performance worsens. I will revert it and inspect whether the dominant residual coordinates enter the shared VLM keys, where a more localized edit could reduce sink competition without altering the token embeddings.
```python
clear_edits()
r6=read_jsonl(W/"roll_6.jsonl")
print("instruction_changed counts:",dict(collections.Counter((e["ptype"],e["instruction_changed"]) for e in eps)))
print("dominant coordinates in residual means and key projections")
outlier_dims=[604,674,738]
for prev,l in [(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]:
    print("layer",l)
    nw=load_original(f"{LP}.layers.{l}.input_layernorm.weight")
    kw=load_original(f"{LP}.layers.{l}.self_attn.k_proj.weight")
    cn=kw.norm(dim=0)
    print(" input norm weights:",[round(float(nw[d]),3) for d in outlier_dims],
          "key colnorms:",[round(float(cn[d]),3) for d in outlier_dims],"median",round(float(cn.median()),3))
    for m in ["base","wrist","lang"]:
        xx=np.stack([np.load(p["activations"])[f"mean::vlm.{prev}.resid::{m}"] for p in p3])
        mu=xx.mean(0)
        print(" ",m,"norm",round(float(np.linalg.norm(mu)),1),
              "coords",np.round(mu[outlier_dims],1).tolist(),
              "energy%",round(float(np.square(xx[:,outlier_dims]).sum()/np.square(xx).sum()*100),1))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 5-6: reject candidate A (BOS row x0.5)\nProbe: middle-layer BOS attention down modestly; content mass rises only slightly. Paired development n128 seed2401: 71->73 successes, +1.6pp SE2.2; task8->6/23, old-goal failures11->13; env14->17/30, object18->19/24, position4->4/24, semantic27->27/27. Reverted; insufficient evidence and task regression.\n")
```
