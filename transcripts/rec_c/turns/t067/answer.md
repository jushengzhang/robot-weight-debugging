Halving the padded inputs gave an inconclusive 1.6-point gain and increased mean gripper flips from 14.6 to 16.3, so it does not support the proposed loop mechanism. I will remove it. Before another intervention, I will check whether a few large residual channels dominate the VLM keys and obscure smaller visual changes.
```python
for n in list(edited()):drop_edit(n)
# Inspect stored VLM representations and static LayerNorm weights only.
r15=json.loads((W/"probe_15/results.json").read_text())
z15=[np.load(r["activations"]) for r in r15]
for L in [2,8,15]:
    arr=np.stack([z[f"mean::vlm.{L}.resid::{mod}"] for z in z15 for mod in ["base","wrist","lang"]])
    energy=np.mean(arr.astype(float)**2,axis=0)
    inds=np.argsort(energy)[-8:][::-1]
    singular=np.linalg.svd(arr.astype(float),compute_uv=False)
    print("VLM",L,"top8dims",inds.tolist(),"energy fractions",np.round(energy[inds]/energy.sum(),3).tolist(),
          "uncentered PC1 fraction",round(float(singular[0]**2/(singular@singular)),3))
for L in [0,2,8,14,15,16,17]:
    n=f"paligemma_with_expert.paligemma.model.language_model.layers.{L}.input_layernorm.weight"
    if n not in NAMES:
        n=next(n for n in NAMES if f"language_model" in n and n.endswith(f"layers.{L}.input_layernorm.weight"))
    a=load_original(n)+1.0
    vals,inds=a.abs().topk(5)
    print("VLM norm",L,"median/max",round(float(a.abs().median()),3),round(float(vals[0]),3),
          "topdims",inds.tolist())
for z in z15:z.close()
# More direct diagnosis: first-chunk grip invariance under the padded-input perturbation.
first=[]
for key,a,b in paired_rows(25):
    pa=Path(a.get("frames_dir",""))/"t0000_inputs.pt"
    pb=Path(b.get("frames_dir",""))/"t0000_inputs.pt"
    if pa.exists() and pb.exists():
        da,db=torch.load(pa,weights_only=False),torch.load(pb,weights_only=False)
        aa,bb=np.asarray(da["action_chunk"]),np.asarray(db["action_chunk"])
        first.append([np.sqrt(np.mean((bb[:,:3]-aa[:,:3])**2)),np.mean((bb[:,6]>0)!=(aa[:,6]>0))])
print("padded intervention first-query XYZ RMS and grip sign disagreement",np.round(np.mean(first,0),4).tolist())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass25: halve unused action-input columns7:31\n"
            "72 vs70/126 (+1.6+/-2.8pp),7 wins/5 losses; loops44 vs46 but mean flips16.26 vs14.59 "
            "and slower successes156.8 vs142.9 steps. Mechanism unsupported; removed.\n")
```
