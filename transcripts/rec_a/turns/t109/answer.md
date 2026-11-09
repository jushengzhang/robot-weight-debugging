I will check the transition changes and trace whether the perturbation is amplified into the gripper direction downstream. This will distinguish a real change to early decision feedback from another edit that only perturbs arm motion.
```python
pa34=probe_actions(34)
total=np.zeros(4); by=defaultdict(list)
for i,m in enumerate(selected):
    a,b=pa34[i]["orig"],pa34[i]["edit"]
    row=[np.sum((a[:,6]>0)!=(b[:,6]>0)),
         np.sum((a[:5,6]>0)!=(b[:5,6]>0)),
         np.mean((b[:,:3]-a[:,:3])**2),
         b[:5,6].mean()-a[:5,6].mean()]
    total+=row; by[(m["group"],m["direction"])].append(row)
    print(m["id"],m["step"],m["group"],m["direction"],
          "".join("+" if z>0 else "-" for z in a[:,6]),
          "".join("+" if z>0 else "-" for z in b[:,6]),
          "XYZ RMS",round(math.sqrt(row[2]),4))
print("Overall: sign changes/all/first5",int(total[0]),int(total[1]),"XYZ RMS",round(math.sqrt(total[2]/16),4))
for key,rows in by.items():
    ar=np.array(rows)
    print(key,"signs",ar[:,:2].sum(0).astype(int).tolist(),
          "mean grip change",round(ar[:,3].mean(),4))
for L in [6,7,8,11,15,17]:
    diffs=[]
    for i in range(16):
        a=np.load(W/"probe_34"/f"probe_orig_{i}_acts.npz")[f"calls::expert.{L}.resid::action"][0]
        b=np.load(W/"probe_34"/f"probe_edit_{i}_acts.npz")[f"calls::expert.{L}.resid::action"][0]
        diffs.append(unit_eff[0].numpy()@(b-a))
    print("L",L,"first-call perturbation RMS pose/grip",
          round(float(np.sqrt(np.mean(np.array(diffs)[:,:6]**2))),3),
          round(float(np.sqrt(np.mean(np.array(diffs)[:,6]**2))),3))
```
