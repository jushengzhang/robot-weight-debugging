The current protection space may be removing the very directions through which camera changes affect actions. I’ll check that explicitly: compare camera-sensitive residual directions with clean-task variation and action readout sensitivity. Any replacement will be a fixed spectral edit, not a fit to episode success labels.
```python
# Analyze saved activations only; no network execution or weight fitting.
delta_action_all = torch.from_numpy((Ac[:,1]-Ac[:,0]).mean(1)).double()
print("All32 normalized-action vs final-residual readout correlation:",
      [round(float(np.corrcoef(delta_action_all[:,j], delta_linear[:,j])[0,1]),3) for j in range(7)])
print("Readout delta cosine:", float(torch.nn.functional.cosine_similarity(
    delta_linear.flatten(), -delta_action_all.flatten(), dim=0)))
print("Relative delta-linear vs action norm:", float(torch.linalg.norm(delta_linear)/torch.linalg.norm(delta_action_all)))

enc = load_original("mot.mixtures.action.action_encoder.weight").double()
encb = load_original("mot.mixtures.action.action_encoder.bias").double()[:,None]
PE, _ = torch.linalg.qr(torch.cat([enc,encb],1), mode="reduced")
clean = R[:,2]
clean_mean = clean.mean(0)
Cc = clean-clean_mean
CcE = Cc-(Cc@PE)@PE.T
_,sc,vc = torch.linalg.svd(CcE,full_matrices=False)
print("Clean centered energy ranks:", {r:round(float(sc[:r].square().sum()/sc.square().sum()),4) for r in (1,2,4,8,16,24,32,48)})
print("Camera/clean RMS:", float(D.square().mean().sqrt()),float(Cc.square().mean().sqrt()))
rows_out=[]
basis_options={}
for clean_rank in (0,2,4,8,16,32):
    base = torch.cat([PE,vc[:clean_rank].T],1)
    Q,_=torch.linalg.qr(base,mode="reduced")
    Dq=D-(D@Q)@Q.T
    _,ss,vv=torch.linalg.svd(Dq,full_matrices=False)
    u=vv[:2].T
    perr=(D@u)@(W0@u).T
    cerr=(Cc@u)@(W0@u).T
    cca=R[:,2]@W0.T
    cca-=cca.mean(0)
    _,_,vaq=torch.linalg.svd(Dq[task_split==0],full_matrices=False)
    _,_,vbq=torch.linalg.svd(Dq[task_split==1],full_matrices=False)
    principal=torch.linalg.svdvals(vaq[:2]@vbq[:2].T)
    row={"protect_clean_rank":clean_rank,
         "camera_energy":round(float(ss[:2].square().sum()/ss.square().sum()),3),
         "readout_cam_fraction":round(float(perr.square().sum()/delta_linear.square().sum()),3),
         "readout_cam_cos":round(float(torch.nn.functional.cosine_similarity(perr.flatten(),delta_linear.flatten(),dim=0)),3),
         "clean_readout_rms_ratio":round(float(torch.linalg.norm(cerr)/torch.linalg.norm(cca)),3),
         "weight_rel":round(float(torch.linalg.norm(W0@u)/torch.linalg.norm(W0)),3),
         "principal_cosines":np.round(principal.numpy(),3).tolist()}
    rows_out.append(row)
    basis_options[clean_rank]=(Q,u,ss)
print(json.dumps(rows_out,indent=2))
(root / "readout_protection_analysis.json").write_text(json.dumps(rows_out,indent=2))
```
