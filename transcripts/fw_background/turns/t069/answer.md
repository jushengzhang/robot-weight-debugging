The probe does not support video-qualified layer identifiers, and the failed request cost no pass. I will use the existing video attention-output summaries instead, explicitly treating their trajectory averaging as a confound. I will look for texture directions that are also reproducible across pair subsets and small on light-task variation before testing them causally.
```python
video_acts={}
for r in r2:
    with np.load(r["activations"]) as z:
        video_acts[(r["suite"],r["id"])]={
            k:z[k].astype(np.float32) for k in z.files if k.startswith("mean::video.")
        }
validpairs=[p for p in pairs if p.get("low") is not None
            and (p["suite"],p["high"]) in video_acts and (p["suite"],p["low"]) in video_acts]
print("Video contrast task pairs:",len(validpairs))
lightkeys=[(r["suite"],r["id"]) for r in r2 if r["category"]=="Light Conditions"]
vdata={}; vinfo={}
for L in range(30):
    keys=[f"mean::video.{L}.attn_out::{cam}" for cam in ["agent","wrist"]]
    D=np.stack([video_acts[(p["suite"],p["high"])][k]-
                video_acts[(p["suite"],p["low"])][k] for p in validpairs for k in keys])
    Y=np.stack([[video_acts[t][k] for k in keys] for t in lightkeys])
    # Preserve normal inter-camera contrast by projecting it out of the nuisance estimate.
    cm=Y.mean(0); cd=cm[0]-cm[1]; cd=cd/(np.linalg.norm(cd)+1e-12)
    Dp=D-(D@cd)[:,None]*cd[None,:]
    _,s,vt=np.linalg.svd(Dp,full_matrices=False)
    U=vt[:4].T.copy()
    C=(Y-cm[None,:,:]).reshape(-1,3072)
    de=np.mean((D@U)**2,axis=0)
    ce=np.mean((C@U)**2,axis=0)
    u1=np.linalg.svd(Dp[::4],full_matrices=False)[2][:2].T
    u2=np.linalg.svd(Dp[2::4],full_matrices=False)[2][:2].T
    stability=np.linalg.svd(u1.T@u2,compute_uv=False)
    vdata[f"U{L}"]=U; vdata[f"mean{L}"]=Y.mean((0,1))
    vinfo[L]={"ratios":(de/(ce+1e-12)).tolist(),
              "rank2_contrast_fraction":float(np.sum(de[:2])/np.mean(np.sum(D**2,axis=1))),
              "rank2_light_fraction":float(np.sum(ce[:2])/np.mean(np.sum(C**2,axis=1))),
              "split_cosines":stability.tolist()}
    if L<10 or L in [10,15,20,25,29]:
        print("V",L,"contrast/light",np.round(de[:2]/(ce[:2]+1e-12),1).tolist(),
              "contrast frac",round(vinfo[L]["rank2_contrast_fraction"],3),
              "light frac",round(vinfo[L]["rank2_light_fraction"],3),
              "split",np.round(stability,2).tolist())
np.savez(W/"video_attention_contrast_data.npz",**vdata)
(W/"video_attention_contrast_info.json").write_text(json.dumps(vinfo,indent=2))
print("Video gating learned components (not full conditioned gate):")
for L in [0,1,2,3,4,5,10,20]:
    n=f"mot.mixtures.video.blocks.{L}.modulation"
    if n in NAMES:
        t=load_original(n).reshape(6,-1)
        print(L, [round(float(t[i].square().mean().sqrt()),3) for i in [2,5]])
```