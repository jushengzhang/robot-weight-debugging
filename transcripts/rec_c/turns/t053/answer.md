I will recompute the contrasts at equal policy time and check them separately for open- and closed-start successes. If a direction remains consistent, I will test a rank-one suppression in that layer's MLP output, protecting the successful-recovery mean direction rather than weakening the whole layer.
```python
res20=json.loads((W/"probe_20/results.json").read_text())
z20=[np.load(r["activations"]) for r in res20]
tm={}
print("Expert equal-time contrasts: relative size / LOO / grip-split / initial cos / old-late cos")
for L in range(18):
    k=f"mean::expert.{L}.mlp_out::action"
    good=X[k].reshape(-1,4,1024)[:,2]
    bad=np.stack([z[k] for z in z20]).astype(float)
    dif=bad-good
    d=dif.mean(0)
    loo=np.mean([cos(dif[i],dif.sum(0)-dif[i]) for i in range(len(dif))])
    gc=cos(dif[succ_grips=="open"].mean(0),dif[succ_grips=="close"].mean(0))
    rel=np.linalg.norm(d)/np.sqrt(np.mean(np.sum(np.concatenate([good,bad])**2,axis=1)))
    tm[L]={"difference":d,"success_mean":good.mean(0),"loop_mean":bad.mean(0),"pair_difference":dif}
    print(L," ".join(f"{v:.3f}" for v in
                    [rel,loo,gc,cos(d,contrast[k]["initial"]),cos(d,contrast[k]["later"])]))
# Fixed noise cancellation check from saved data, not model execution.
noise_good=[];noise_bad=[]
for r in mf20:
    ig=r["success_index19"]
    noise_good.append(np.asarray(torch.load(mf[ig]["source"],weights_only=False)["noise"]))
    noise_bad.append(np.asarray(torch.load(r["source"],weights_only=False)["noise"]))
nd=np.stack(noise_bad)-np.stack(noise_good)
print("RMS mean matched noise difference",float(np.sqrt(np.mean(nd.mean(0)**2))),
      "mean individual difference",float(np.sqrt(np.mean(nd**2))))
# Inspect the strongest cross-start late candidates after protecting the successful centroid.
for L in [4,5,6,7,14,15,16]:
    v=tm[L]; d=v["difference"];m=v["success_mean"]
    u=d-m*np.dot(m,d)/np.dot(m,m);u/=np.linalg.norm(u)
    good=X[f"mean::expert.{L}.mlp_out::action"].reshape(-1,4,1024)[:,2]
    bad=good+v["pair_difference"]
    print("L",L,"retained contrast",round(abs(cos(u,d)),3),
          "success projected SD",round(float(np.std(good@u)),3),
          "loop projected mean",round(float(np.mean(bad@u)),3),
          "mean loop/success norm",round(float(np.linalg.norm(bad.mean(0))/np.linalg.norm(good.mean(0))),3))
torch.save(tm,W/"time_matched_mlp_contrasts.pt")
for z in z20:z.close()
```
