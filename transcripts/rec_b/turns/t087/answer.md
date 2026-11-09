The positional embeddings contain seven exceptional patches: the largest centered norm is 311 versus a median of 17, and patch 253 is among the outliers. These may be useful internal “registers,” so their size alone does not justify removal. I will check whether expert attention repeatedly targets these fixed locations before testing a partial attenuation.
```python
outlier_idx = torch.where(pn > 1.5*pn.median())[0].tolist()
print("Positional outlier patches:", outlier_idx, "uniform expected mass", round(len(outlier_idx)/256,4))

for k in [2,3,4,7,11,14,27]:
    rp=W/f"probe_{k}"/"results.json"
    if not rp.exists(): continue
    rr=json.loads(rp.read_text())
    if not rr or "attention" not in rr[0]: continue
    print("\nProbe",k)
    for r in rr:
        if r["arm"] != "orig": continue
        values=[]
        for ls,ap in r.get("attention",{}).items():
            a=np.load(local_path(ap))
            # Reduce all axes other than head and key.
            if a.ndim==4: a=a.mean(0)
            if a.ndim==3: a=a.mean(1)
            for h in range(a.shape[0]):
                vis=float(a[h,:512].sum())
                if vis < .25: continue
                oi=np.array(outlier_idx)
                frac=float((a[h,oi].sum()+a[h,oi+256].sum())/vis)
                top=int(a[h,:512].argmax())
                if frac > .2:
                    values.append((int(ls),h,round(vis,2),round(frac,2),top))
        print("input",r["idx"],"concentrated (L,h,visual,outlier_fraction,top):",values[:15])

# A local-neighbor reference retains the ordinary smooth spatial embedding.
pos_grid=pos.reshape(16,16,-1)
anchors={}
for idx in outlier_idx:
    row,col=divmod(idx,16)
    neigh=[]
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        rr,cc=row+dr,col+dc
        if 0 <= rr <16 and 0 <= cc <16 and rr*16+cc not in outlier_idx:
            neigh.append(pos_grid[rr,cc])
    anchors[idx]=torch.stack(neigh).median(0).values
print("\nOutlier residual norm after local interpolation:",
      [(i,round(float((pos[i]-anchors[i]).norm()),1)) for i in outlier_idx])

torch.save({"indices":outlier_idx, "anchors":anchors}, W/"position_register_analysis.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Upstream spatial outliers\n"
            "SigLIP positional embeddings have centered norms median16.88,max310.94. "
            "Seven patches exceed1.5xmedian:13,47,48,193,225,243,253. "
            "Previously observed wrist attention fixation509 corresponds to patch253. "
            "These may be functional implicit registers, not necessarily defects; "
            "test causal effects before judging.\n")
```
