The channel-674 edit failed fresh validation: 75 versus 83 successes, a 6.5-point loss. I will remove it. I will now test a more selective idea: whether a late-expert component seen during loops can be separated from normal denoising, current action coordinates, and successful recovery states, then used to trigger clearance motion rather than a global sensitivity change.
```python
for n in list(edited()):drop_edit(n)
# Geometric analysis only: no model execution or fitting a weight-loss objective.
L=16
k=f"calls::expert.{L}.mlp_out::action"
C19=[]
for r in res19:
    with np.load(r["activations"]) as z:C19.append(z[k].astype(np.float64))
C20=[]
for r in res20:
    with np.load(r["activations"]) as z:C20.append(z[k].astype(np.float64))
C19=np.stack(C19).reshape(16,4,10,1024)
C20=np.stack(C20)
trainpairs=np.array(list(range(6))+list(range(8,14)))
testpairs=np.array([6,7,14,15])
normal=C19[trainpairs][:,[0,1,2]].reshape(-1,1024)
normal_unit=normal/(np.linalg.norm(normal,axis=1,keepdims=True)+1e-12)
_,sv,vt=np.linalg.svd(normal_unit,full_matrices=False)
rank=32
print("normal subspace rank",rank,"energy retained",round(float((sv[:rank]@sv[:rank])/(sv@sv)),4))
ao=load_original("action_out_proj.weight").numpy()[:7]
sink=np.eye(1024)[[364,885]]
protect=np.concatenate([vt[:rank],ao,sink],axis=0).T
Q,_=np.linalg.qr(protect)
bads=np.concatenate([C19[trainpairs,3].reshape(-1,1024),
                     C20[trainpairs].reshape(-1,1024)])
dm=bads.mean(0)
uv=dm-Q@(Q.T@dm)
print("mean loop component retained",round(float(np.linalg.norm(uv)/np.linalg.norm(dm)),4))
uv/=np.linalg.norm(uv)
def show_scores(ind,label):
    si=(C19[ind,:2]@uv).mean(-1)
    sn=(C19[ind,2]@uv).mean(-1)
    b50=(C20[ind]@uv).mean(-1)
    b250=(C19[ind,3]@uv).mean(-1)
    print(label,"initial mean/sd",np.round([si.mean(),si.std()],3).tolist(),
          "normal50 mean/sd",np.round([sn.mean(),sn.std()],3).tolist(),
          "loop50 mean/sd",np.round([b50.mean(),b50.std()],3).tolist(),
          "loop250 mean/sd",np.round([b250.mean(),b250.std()],3).tolist())
    if label=="feature-check":
        for j,n,b,c in zip(ind,sn,b50,b250):
            print(" id",pair_sids[j],"normal/loop50/loop250",np.round([n,b,c],3).tolist())
show_scores(trainpairs,"direction-source")
show_scores(testpairs,"feature-check")
torch.save({"layer":L,"u":torch.tensor(uv,dtype=torch.float32),
            "protected_rank":rank,"source_pairs":trainpairs.tolist(),
            "check_pairs":testpairs.tolist()},W/"selective_loop_detector_L16.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass28: independent validation rejected sink edit\n"
            "124 valid paired episodes:75 vs83 successes (-6.5+/-3.0pp),3 wins/11 losses, "
            "loops30 vs22. Removed. Validation batch1 is now inspected; do not reuse as fresh confirmation.\n"
            "Next analysis: isolate L16 MLP loop component outside dominant normal denoising/initial/success "
            "subspace, physical readout span, and sink axes. Twelve matched pairs define geometry; four "
            "pairs are reserved for checking feature selectivity before any edit. Intended intervention "
            "would route a selective off-normal component into clearance/lift, not scale all actions.\n")
```
