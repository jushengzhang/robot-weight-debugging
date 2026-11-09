I will check both the local activation change and the final action change. This separates a projection that suppresses an internal signal from one that actually makes the policy more viewpoint-invariant.
```python
r8 = json.loads((W/"probe_8"/"results.json").read_text())
orig_actions = np.stack([np.load(r["action"]) for r in r2]).reshape(19,3,32,7)
edit_actions8 = np.stack([np.load(r["action"]) for r in r8]).reshape(19,3,32,7)
def action_probe_summary(a0, ae):
    p0 = np.sqrt(np.mean((a0[:,1,:10,:3]-a0[:,0,:10,:3])**2,(1,2)))
    pe = np.sqrt(np.mean((ae[:,1,:10,:3]-ae[:,0,:10,:3])**2,(1,2)))
    out = {"original_camera_pair_rms":float(p0.mean()), "edited_camera_pair_rms":float(pe.mean()),
           "pair_rms_change_pct":float(100*(pe.mean()/p0.mean()-1)),
           "pairs_improved":int((pe<p0).sum()),
           "edit_xyz_rms_by_group":[float(x) for x in np.sqrt(np.mean((ae[:,:,:10,:3]-a0[:,:,:10,:3])**2,(0,2,3)))],
           "edit_gripper_disagreement_by_group":[float(x) for x in (np.sign(a0[:,:,:10,6])!=np.sign(ae[:,:,:10,6])).mean((0,2))]}
    return out
ps8 = action_probe_summary(orig_actions, edit_actions8)
print(json.dumps(ps8,indent=2))
for l in [11,16,29]:
    B = np.stack([np.load(r["activations"])[f"calls::action.{l}.attn_out::action"] for r in r8]).reshape(19,3,10,1024)
    db = B[:,1,0]-B[:,0,0]
    d0 = D[:,l,0]
    print("L",l,"first-call paired activation RMS original/edited",
          round(float(np.sqrt((d0**2).mean())),4),round(float(np.sqrt((db**2).mean())),4),
          "top original direction RMS original/edited",
          round(float(np.sqrt((d0@bases[l][0])**2 .mean())) if False else float(np.sqrt(np.mean((d0@bases[l][0])**2))),4),
          round(float(np.sqrt(np.mean((db@bases[l][0])**2))),4))
(W/"probe_8_summary.json").write_text(json.dumps(ps8,indent=2))
ledger = json.loads((W/"pass_ledger.json").read_text())
ledger.append({"pass":8,"type":"probe","edit":"l11_rank1_center_half","inputs":57,"noise_seed":46001,**ps8})
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
```
