I will algebraically recover the time-dependent residual multipliers from the recorded branch outputs and residual increments, then check the reconstruction on the other half of the observations. These are diagnostic coefficients only; no weights are being fitted or trained.
```python
# Arrange identical-input records from the three unedited probes.
def arrays_for_paths(k, paths):
    rr={str(local_path(r["input"])):r for r in probe_records(k) if r["arm"]=="orig"}
    return [np.load(local_path(rr[p]["activations"])) for p in paths]
paths4=[str(local_path(r["input"])) for r in p4]
a4=arrays_for_paths(4,paths4)
a13=arrays_for_paths(13,paths4)
a20=arrays_for_paths(20,paths4)
g_np=g.numpy().astype(np.float64)
grip4=np.stack([np.load(local_path(r["action"])) for r in p4])[:,:,6].mean(1)

def recover_branch_multipliers(A,M,R,indices):
    # R = ga*A + gm*M, independently at each time/channel.
    a,m,r=A[indices],M[indices],R[indices]
    aa=(a*a).sum(0); mm=(m*m).sum(0); am=(a*m).sum(0)
    ar=(a*r).sum(0); mr=(m*r).sum(0)
    det=aa*mm-am*am
    safe=det>np.maximum(aa*mm*1e-8,1e-16)
    ga=np.where(safe,(ar*mm-mr*am)/np.maximum(det,1e-30),0)
    gm=np.where(safe,(mr*aa-ar*am)/np.maximum(det,1e-30),0)
    return ga,gm

branch_data={}
print("L: held-out reconstruction error%; first-call grip std: total/attention/MLP; corr grip: attention/MLP")
for l in range(1,18):
    R=np.stack([a[f"calls::expert.{l}.resid::action"]-
                a[f"calls::expert.{l-1}.resid::action"] for a in a4]).astype(np.float64)
    A=np.stack([a[f"calls::expert.{l}.attn_out::action"] for a in a20]).astype(np.float64)
    M=np.stack([a[f"calls::expert.{l}.mlp_out::action"] for a in a13]).astype(np.float64)
    ga8,gm8=recover_branch_multipliers(A,M,R,np.arange(0,16,2))
    pred=A[1::2]*ga8+M[1::2]*gm8
    err=np.sqrt(np.mean((pred-R[1::2])**2)/np.mean(R[1::2]**2))
    ga,gm=recover_branch_multipliers(A,M,R,np.arange(16))
    AR=A*ga; MR=M*gm
    rg=R@g_np; ag=AR@g_np; mg=MR@g_np
    print(l,round(100*err,2),
          "/".join(f"{x[:,0].std():.2f}" for x in (rg,ag,mg)),
          round(float(np.corrcoef(ag[:,0],grip4)[0,1]),3),
          round(float(np.corrcoef(mg[:,0],grip4)[0,1]),3))
    branch_data[l]={"ga":ga,"gm":gm,"A":A,"M":M,"R":R,"validation_error":err}
np.savez_compressed(W/"inferred_residual_multipliers.npz",
                    **{f"L{l}_{typ}":d[typ] for l,d in branch_data.items() for typ in ("ga","gm")})
print("Late branch multiplier RMS (first/last denoising call):")
for l in (14,15,16,17):
    d=branch_data[l]
    print(l, "attention",np.round(np.sqrt(np.mean(d["ga"]**2,axis=1))[[0,9]],2).tolist(),
          "MLP",np.round(np.sqrt(np.mean(d["gm"]**2,axis=1))[[0,9]],2).tolist())
```
