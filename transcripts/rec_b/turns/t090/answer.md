I will measure the attention change and the size of the action perturbation in failures versus controls. If the edit changes the proposed mechanism without grossly disrupting the controls, I will test it on the full 96-episode diagnostic panel rather than the smaller panel that previously overfit.
```python
r34=json.loads((W/"probe_34"/"results.json").read_text())
by34={(r["idx"],r["arm"]):r for r in r34}
oi=np.array(outlier_idx)
for i in range(4):
    ro,re=by34[i,"orig"],by34[i,"edit"]
    ao,ae=np.load(local_path(ro["action"])),np.load(local_path(re["action"]))
    print("Input",i,"pose_RMS_change",round(float(np.sqrt(np.mean((ae[:,:6]-ao[:,:6])**2))),4),
          "grip_mean",round(float(ao[:,6].mean()),3),"->",round(float(ae[:,6].mean()),3))
    for layer,head in [(2,7),(7,7),(14,3),(16,3)]:
        vals=[]
        for r in [ro,re]:
            a=np.load(local_path(r["attention"][str(layer)])).mean(1)[head]
            vm=a[:512].sum()
            vals.append({"visual":round(float(vm),3),
                         "outlier_frac":round(float((a[oi].sum()+a[oi+256].sum())/max(vm,1e-8)),3),
                         "top":int(a[:512].argmax()),
                         "wrist253":round(float(a[509]),3)})
        print(" L/h",layer,head,vals)

diag_episodes=[{"id":r["id"],"grip":r["grip"],"seed":r["seed"]} for r in base_rows]
(W/"diagnostic_episodes.json").write_text(json.dumps(diag_episodes))
print("96-episode diagnostic scene IDs:", sorted(set(r["id"] for r in base_rows)))
print("Rollout request:")
print(json.dumps({"type":"rollout","model":"both","episodes":diag_episodes},separators=(",",":")))
```
