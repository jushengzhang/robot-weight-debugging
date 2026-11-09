I will construct candidate video-input directions from the residual summaries, while protecting the light-control mean, camera difference, and leading light-task variation. This addresses the main weakness of the rejected action-output edit: removing texture variation without sufficiently protecting useful task features.
```python
r14=[json.loads(s) for s in (W/"roll_14.jsonl").read_text().splitlines()]
vr={}
for r in r14:
    with np.load(r["activations"]) as z:
        vr[(r["suite"],r["id"])]={
            k:z[k].astype(np.float32) for k in z.files if k.startswith("mean::video.")
        }
def standardized_summary(x):
    x=np.asarray(x,dtype=np.float32)
    c=x-x.mean(-1,keepdims=True)
    return c/np.sqrt(np.mean(c*c,axis=-1,keepdims=True)+1e-6)
vres_data={}; vres_info={}
for L in range(1,30):
    prev=L-1
    keys=[f"mean::video.{prev}.resid::{cam}" for cam in ["agent","wrist"]]
    B=np.stack([[vr[(p["suite"],p[z])][k] for z in ["high","low"] for k in keys] for p in validpairs])
    B=standardized_summary(B)
    D=(B[:,:2]-B[:,2:]).reshape(-1,3072)
    Y=standardized_summary(np.stack([[vr[t][k] for k in keys] for t in lightkeys]))
    cm=Y.mean(0); C=(Y-cm).reshape(-1,3072)
    _,sc,vtc=np.linalg.svd(C,full_matrices=False)
    # Protect normal means and the 8 leading light-task directions.
    protected=np.column_stack([cm[0],cm[1],np.ones(3072,dtype=np.float32),vtc[:8].T])
    Q=np.linalg.qr(protected, mode="reduced")[0].astype(np.float32)
    Dp=D-(D@Q)@Q.T
    _,s,vt=np.linalg.svd(Dp,full_matrices=False)
    U=vt[:2].T.copy()
    de=np.mean((D@U)**2,0); ce=np.mean((C@U)**2,0)
    Ua=np.linalg.svd(Dp[::4],full_matrices=False)[2][:2].T
    Ub=np.linalg.svd(Dp[2::4],full_matrices=False)[2][:2].T
    stab=np.linalg.svd(Ua.T@Ub,compute_uv=False)
    vres_data[f"U{L}"]=U; vres_data[f"Q{L}"]=Q
    vres_data[f"mean{L}"]=Y.mean((0,1))
    info={"ratio":(de/(ce+1e-12)).tolist(),
          "contrast_fraction":float(de.sum()/np.mean(np.sum(D*D,1))),
          "light_fraction":float(ce.sum()/np.mean(np.sum(C*C,1))),
          "split_cosines":stab.tolist(),
          "protected_light_fraction":float(np.sum(sc[:8]**2)/np.sum(sc**2))}
    vres_info[L]=info
    if L in list(range(1,11))+[14,15,18,20,22,25,28,29]:
        print("Input V",L,"ratios",np.round(info["ratio"],1).tolist(),
              "D fraction",round(info["contrast_fraction"],3),
              "light fraction",round(info["light_fraction"],4),
              "split",np.round(stab,2).tolist(),
              "protected light",round(info["protected_light_fraction"],3))
np.savez(W/"video_value_nuisance_data.npz",**vres_data)
(W/"video_value_nuisance_info.json").write_text(json.dumps(vres_info,indent=2))
```