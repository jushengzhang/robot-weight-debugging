I’ll quantify attention mass by camera and instruction, identify heads that consistently ignore language, and check whether the matched images really are identical. I’ll also examine how strongly the stored residuals distinguish new instructions from paraphrases, while treating episode-averaged correlations as exploratory only.

```python
pr2 = json.loads((W/"probe_2"/"results.json").read_text())
print("probe structure:", str(pr2)[:4500])
# Compare initial observations without running any network locally.
def initial_input(eid):
    return torch.load(W/"roll_1_frames"/f"e{eid}_seed17_original"/"t0000_inputs.pt",
                      map_location="cpu", weights_only=False)
for a,b in ((1545,1522),(3553,3531),(1545,1496),(1545,1571)):
    xa,xb = initial_input(a),initial_input(b)
    diffs = {}
    for k in ("observation/image","observation/wrist_image"):
        ia=np.asarray(xa["element"][k]).astype(float)
        ib=np.asarray(xb["element"][k]).astype(float)
        diffs[k.split("/")[-1]] = round(float(np.abs(ia-ib).mean()),3)
    print("pixel MAD",a,b,diffs)
# Lightweight residual summaries.
acts1 = {r["id"]:dict(np.load(workpath(r["activations"]))) for r in r1}
for l in (0,3,6,9,12,15,17):
    ratios=[]
    for g in diag_groups:
        ids={p:e["id"] for p,e in groups[g].items()}
        lang=f"mean::vlm.{l}.resid::lang"
        base=f"mean::vlm.{l}.resid::base"
        dlang=np.linalg.norm(acts1[ids["task"]][lang]-acts1[ids["semantic"]][lang])
        dbase=np.linalg.norm(acts1[ids["task"]][base]-acts1[ids["semantic"]][base])
        ratios.append((dlang,dbase))
    print("layer",l,"mean task-semantic residual distances [lang,base]",
          np.round(np.mean(ratios,axis=0),2).tolist())
```
