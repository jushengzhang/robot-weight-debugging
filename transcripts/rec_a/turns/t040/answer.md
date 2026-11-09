There is another important confound: averaging across denoising steps makes early-layer residuals resemble the final action because later calls already contain a partly denoised action chunk. I will separate the first and last denoising calls. The first call is the cleaner place to locate visual formation of the gripper decision; the last shows how that decision is reinforced.
```python
noise_g4=np.array([np.asarray(torch.load(p["input"],map_location="cpu",weights_only=False)["noise"])[:,6].mean() for p in p4])
print("Saved noise mean vs final grip correlation:",round(corr(noise_g4,g_targets),3))
print("Per-layer correlations: first residual with grip/noise; last residual with grip; first MLP with grip; first increment std")
prev=None
for l in range(18):
    R=np.stack([x[f"calls::expert.{l}.resid::action"] for x in res4])
    M=np.stack([x[f"calls::expert.{l}.mlp_out::action"] for x in mlp13_orig])
    r0=R[:,0]@gdir.numpy(); r9=R[:,-1]@gdir.numpy()
    m0=M[:,0]@gdir.numpy()
    inc=(R[:,0]-prev[:,0])@gdir.numpy() if prev is not None else r0
    print(l,"r0",round(corr(r0,g_targets),3),"noise",round(corr(r0,noise_g4),3),
          "r9",round(corr(r9,g_targets),3),"m0",round(corr(m0,g_targets),3),
          "inc_gstd",round(float(inc.std()),3),"inc_corr",round(corr(inc,g_targets),3))
    prev=R
# Causal change caused by attenuating head 0/6, at the first call only.
orig12={p["idx"]:np.load(p["activations"]) for p in p12 if p["arm"]=="orig"}
edit12={p["idx"]:np.load(p["activations"]) for p in p12 if p["arm"]=="edit"}
print("L0H6 causal residual effect, first call: norm and grip-axis covariance")
for l in [0,3,6,9,12,15,17]:
    k=f"calls::expert.{l}.resid::action"
    D=np.stack([edit12[i][k][0]-orig12[i][k][0] for i in range(16)])
    print(l,"RMS",round(float(np.sqrt((D*D).mean())),4),
          "gstd",round(float((D@gdir.numpy()).std()),4),
          "corr_grip",round(corr(D@gdir.numpy(),g_targets),3))
```
