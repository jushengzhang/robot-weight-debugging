I will recover the per-head feature scores and then run a reference-only intervention through the same value-output direction. The reference is the nearly common-mode normalized channel 674. Comparing the two at the attention output will let me center the feature using normal-state activation statistics, without fitting to task success or action targets.
```python
p41=json.loads((W/"probe_41"/"results.json").read_text())
b41={(r["idx"],r["arm"]):r for r in p41}
Croute=torch.stack([O[:,h*256:(h+1)*256]@dv for h in range(8)],dim=1).double()
Cpinv=torch.linalg.pinv(Croute)
deltas41=[]
for i in range(60):
    ao=np.load(b41[i,"orig"]["activations"])["calls::expert.17.attn_out::action"][0]
    ae=np.load(b41[i,"edit"]["activations"])["calls::expert.17.attn_out::action"][0]
    deltas41.append(ae-ao)
D41=torch.from_numpy(np.stack(deltas41)).double()
scores41=D41@Cpinv.T
recon41=scores41@Croute.T
print("rank-one attention decomposition relative error",
      round(float((D41-recon41).norm()/D41.norm()),5))
for label,idxs in [
    ("source_empty",[2*j for j in range(22) if j in source_indices]),
    ("source_contact",[2*j+1 for j in range(22) if j in source_indices]),
    ("check_empty",[2*j for j in range(22) if j not in source_indices]),
    ("check_contact",[2*j+1 for j in range(22) if j not in source_indices]),
    ("open",list(range(44,60)))]:
    print(label,"H1/H7 weighted scores",
          [round(float(scores41[idxs,h].mean()),3) for h in [1,7]])
torch.save({"scores":scores41.float(),"routing_matrix":Croute.float(),
            "decomposition_error":float((D41-recon41).norm()/D41.norm())},
           W/"clearance_attention_scores.pt")
save_edit(vname,load_original(vname).clone()+torch.outer(dv,ref))
print("reference-only relative V change",
      round(float(torch.outer(dv,ref).norm()/load_original(vname).norm()),4))
```
