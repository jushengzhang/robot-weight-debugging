Weakening the gripper-specific query direction made performance worse, with three regressions and no rescues. That argues against treating this route as the cause of the loop. I will restore it, inspect the probe effects, and investigate the unusually large late-VLM camera residuals before making another intervention.
```python
p9=json.loads((W/"probe_9/results.json").read_text())
for idx,d in paired_probe(p9).items():
    print("input",idx)
    for arm,p in d.items():
        m,gr,ent=attn_stats(np.load(p["attention"]["2"]))
        a=np.load(p["action"])
        print(arm,"L2h0 B/W/L/self",np.round(np.r_[gr[0,[0,1,3]],1-gr[0].sum()],3).tolist(),
              "grip10",a[:,6].round(2).tolist(),
              "xyz5",a[:5,:3].mean(0).round(3).tolist())
reset_edits()
print("Late VLM residual concentration:")
for L in [2,4,6,10,13,16]:
    for group in ["base","wrist","lang"]:
        X=np.stack([a[f"mean::vlm.{L}.resid::{group}"] for a in AR7])
        pow=(X**2).mean(0); inds=np.argsort(pow)[-5:][::-1]
        frac=float(pow[inds].sum()/pow.sum())
        print(L,group,"top5 energy fraction",round(frac,3),
              "dims",inds.tolist(),"mean values",X.mean(0)[inds].round(1).tolist(),
              "rms loop/good",round(float(np.sqrt((X[loop_idx]**2).mean())),1),
              round(float(np.sqrt((X[good_idx]**2).mean())),1))
with (W/"notes.md").open("a") as f:
    f.write("\n## Rejected candidate C, passes 9-10\nRank-one reduction of gripper direction in L2h0 Q "
            "(beta .5):26/48->23/48, -6.2 +/-3.5 pp, 0 rescues/3 regressions; loops21->22. "
            "This route appears useful; do not suppress it. Reset.\n")
```
