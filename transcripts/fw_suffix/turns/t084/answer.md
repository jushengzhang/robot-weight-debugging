The existing readout contains several possible suffix detectors, not just the one tested. I’ll compare their responses on tagged, clean, and paraphrased inputs to see whether combining a few adds coverage without lowering the threshold that previously hurt performance.
```python
clean_idx = list(range(1,48,2)) + list(range(48,80))
raw_idx = list(range(0,48,2))
cats = [r["category"] for r in cm["paired"]]
print("Safe thresholds: max clean/paraphrase response <= 1e-6")
safe_detectors=[]
for di,label in enumerate(gd["labels"]):
    candidates=[]
    for j,(d,t,g) in enumerate(gd["features"]):
        if d==di and g==32. and np.max(G[clean_idx,j])<=1e-6 and t>=0:
            candidates.append((float(t),j))
    if not candidates:
        print(label, "none"); continue
    t,j=min(candidates)
    safe_detectors.append((di,t,j))
    print("\n",di,label,"threshold",t,"clean_min/max",
          np.round([G[clean_idx,j].min(),G[clean_idx,j].max()],7),
          "raw_min/mean/max",np.round([G[raw_idx,j].min(),G[raw_idx,j].mean(),G[raw_idx,j].max()],5))
    for cat in sorted(set(cats)):
        ids=[raw_idx[i] for i,c in enumerate(cats) if c==cat]
        print(" ",cat, np.round(G[ids,j],4).tolist())
print("\nGate design feature cosines:")
dirs=gd["directions"]
print(np.round((dirs @ dirs.T).numpy(),3))
print("\nCurrent context readout files and saved gate weights:")
for dn in ("fwd/probe_25_weights","fwd/probe_23_weights","fwd/probe_13_weights"):
    p=W/dn
    print(dn,[(z.name,z.stat().st_size) for z in list(p.iterdir())[:6]] if p.exists() else "missing")
print("Top input coordinate contrast-to-clean ratios:")
D=C[:48:2]-C[1:48:2]
rmsd=D.square().mean(0).sqrt()
rmsc=C[clean_idx].square().mean(0).sqrt()
ix=(rmsd/(rmsc+0.003)).topk(20).indices
print([(int(i),round(float(rmsd[i]),4),round(float(rmsc[i]),4))
       for i in ix])
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 29: rank24 oblique edit development rollout\n"
            "96 paired episodes: original50/96, edited50/96; 0pp, paired SE3.0pp; "
            "4 wrong->right and4 right->wrong. No significant success gain; not validated. "
            "11 passes remain. Continuing with existing token-detector readouts; preserve "
            "independent tagged, strip and language validation budgets.\n")
```
