The final readout contrast generalizes poorly—only 64.3% pairwise separation—so I will not edit it. The earlier visual-value cue was much more selective. I will inspect the last layer's visual-reading heads to see whether that cue can reach the gripper through a shorter, testable route.
```python
for k in [3,31]:
    rr=json.loads((W/f"probe_{k}"/"results.json").read_text())
    print("Probe",k)
    for r in rr:
        if r["arm"]!="orig" or "17" not in r.get("attention",{}): continue
        aa=np.load(r["attention"]["17"]).mean(1)
        print("input",r["idx"],"base/wrist/lang per head:",
              np.round(np.stack([aa[:,:256].sum(1),aa[:,256:512].sum(1),aa[:,768:].sum(1)],1),2).tolist())
# Effective direct gripper-vs-pose coupling at the final attention branch.
O17=load_original(E+"layers.17.self_attn.o_proj.weight")
g17a=torch.from_numpy(gm["L17_ga"]).float()
eff17=torch.einsum("od,td,td,dh->toh",Wo[:7],1+fs,g17a,O17)
print("E17 attention coupling per head:")
for h in range(8):
    en=eff17[:,:,h*256:(h+1)*256].square().mean(0).sum(1)
    print(h,"grip",round(float(en[6]),5),"pose",round(float(en[:6].sum()),5))
print("Value direction keys:",{k:list(v) for k,v in torch.load(W/"physical_grip_value_directions.pt",weights_only=False).items()})
with (W/"notes.md").open("a") as f:
    f.write("\n## Final-readout cue rejected before editing\n"
            "- Protecting existing readout rows and control means left a weak final-residual physical cue: scene-disjoint separation 64.3%. No edit made.\n")
```
