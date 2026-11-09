I will center the feature separately for the two targeted heads, using only the 14 source contact states and 16 open-gripper controls. The correction is a ratio of attention-feature means, not an optimization against success or action error. It should remove the observed normal-state bias while retaining the empty-closure contrast; I will verify that prediction in one final fixed-input probe.
```python
p42=json.loads((W/"probe_42"/"results.json").read_text())
b42={(r["idx"],r["arm"]):r for r in p42}
D42=[]
for i in range(60):
    ao=np.load(b42[i,"orig"]["activations"])["calls::expert.17.attn_out::action"][0]
    ae=np.load(b42[i,"edit"]["activations"])["calls::expert.17.attn_out::action"][0]
    D42.append(ae-ao)
D42=torch.from_numpy(np.stack(D42)).double()
scores42=D42@Cpinv.T
print("reference attention decomposition error",
      round(float((D42-scores42@Croute.T).norm()/D42.norm()),5))
betas={h:-float(scores41[normal_source_indices,h].mean()/scores42[normal_source_indices,h].mean())
       for h in [1,7]}
print("head-wise centering coefficients", {h:round(v,4) for h,v in betas.items()})
B=design["readout_constraints"].double()
Bpinv=torch.linalg.pinv(B)
parts={}
for h in [1,7]:
    yh=torch.zeros(56,dtype=torch.float64);yh[7*h+2]=-1
    parts[h]=(Bpinv@yh*(0.60/src_score)).float()
combined=parts[1]+parts[7]
print("original route reconstruction relative error",round(float((combined-dv).norm()/dv.norm()),7))
offset=sum(betas[h]*parts[h] for h in [1,7])
centered_delta=torch.outer(combined,d)+torch.outer(offset,ref)
save_edit(vname,load_original(vname).clone()+centered_delta)
for label,idxs in [
    ("source_empty",[2*j for j in range(22) if j in source_indices]),
    ("source_contact",[2*j+1 for j in range(22) if j in source_indices]),
    ("check_empty",[2*j for j in range(22) if j not in source_indices]),
    ("check_contact",[2*j+1 for j in range(22) if j not in source_indices]),
    ("open",list(range(44,60)))]:
    print(label,"centered head feature means",
          [round(float((scores41[idxs,h]+betas[h]*scores42[idxs,h]).mean()),3) for h in [1,7]])
torch.save({"feature":d,"reference":ref,"parts":parts,"betas":betas,
            "delta":centered_delta,"nominal":0.60,"source_control_indices":normal_source_indices},
           W/"centered_clearance_design.pt")
cc=W/"candidate_centered_clearance_060";cc.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file():shutil.copy2(p,cc/p.name)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes41-42: correct attention-weighting mismatch\n"
            "Rank-one attn_out differences decomposed via known o_proj writes; reconstruction error0.115%. "
            "Despite zero token-mean scores, actual normal-control H1/H7 scores are~-0.53/-0.41. "
            "Reference-only channel674 probe measures per-head common-mode scores. "
            f"Source-control centering ratios: {betas}. "
            "Final candidate is rank2 VLM17 v_proj change: empty feature routing plus head-specific "
            "reference subtraction, geometric readout still targets onlyZ throughH1/H7. "
            "No success labels in centering. Needs final probe and fresh confirmation; no validated repair yet.\n")
```
