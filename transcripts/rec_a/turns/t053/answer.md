Amplifying the visual direction also hurt the screen, so I will revert it. The two-sided test does not support that repair. I will now examine action-token attention: some heads attend mostly to the model’s own action proposal, which could reinforce a grasp sequence without enough corrective input from the scene.
```python
for n in list(edited()):
    drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 18: visual-gripper value amplification rejected\n"
            "- Same screen, gain 1.5: 31/64 -> 27/64, -6.2 pp (SE 3.8), 1 rescue / 5 losses; loop failures 21 -> 20.\n"
            "- Both attenuation and amplification damaged solved scenes. Reverted all edits.\n"
            "- Next hypothesis: action-token self-conditioning rather than inadequate raw gripper-state visibility.\n")

for k in (2,3,5,8,10):
    rec = probe_records(k)
    print("Probe",k,"nrecords",len(rec),"first-record fields",list(rec[0]))
    print("Attention fields:",str(rec[0].get("attention"))[:500])
    print("First input",local_path(rec[0]["input"]).parent.name,local_path(rec[0]["input"]).name)

# Parse all original saved attention maps and print B/W/language/self mass.
head_attn = {}
for k in (2,3,5):
    rec = [r for r in probe_records(k) if r["arm"]=="orig"]
    for r in rec:
        for layer,fp in r.get("attention",{}).items():
            a = np.load(local_path(fp))
            m = a.mean(axis=1)
            mass = np.stack([m[:,:256].sum(-1),m[:,256:512].sum(-1),
                             m[:,768:].sum(-1),1-m.sum(-1)],axis=-1)
            head_attn[(k,r["idx"],int(layer))]=mass
    print("PROBE",k,"inputs:",[(r["idx"],local_path(r["input"]).name) for r in rec])
    for layer in sorted({l for kk,i,l in head_attn if kk==k}):
        print("L",layer)
        for h in range(8):
            rows=[head_attn[(k,r["idx"],layer)][h] for r in rec if (k,r["idx"],layer) in head_attn]
            print("H",h,"; ".join("/".join(f"{v:.2f}" for v in row) for row in rows))
```
