I will compare matched MLP contrasts across layers, checking whether each direction is consistent across gripper starts and across the initial and later observations. A contrast that disappears or reverses under these checks would be a poor target for a weight edit.
```python
mf=json.loads((W/"mlp_diagnostic_manifest.json").read_text())
res19=json.loads((W/"probe_19/results.json").read_text())
zs=[np.load(r["activations"]) for r in res19]
keys=[k for k in zs[0].files if k.startswith("mean::")]
X={k:np.stack([z[k] for z in zs]).astype(np.float64) for k in keys}
pair_sids=[mf[i]["id"] for i in range(0,len(mf),4)]
succ_grips=np.array([mf[i]["grip"] for i in range(0,len(mf),4)])
suites=np.array([mf[i]["suite"] for i in range(0,len(mf),4)])
def cos(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-20))
stats=[]
contrast={}
for k,x in X.items():
    x4=x.reshape(-1,4,x.shape[-1])
    di=x4[:,1]-x4[:,0]
    dl=x4[:,3]-x4[:,2]
    d0,d1=di.mean(0),dl.mean(0)
    # Leave-one-pair-out consistency, avoiding the self-contribution to a mean direction.
    loo=np.mean([cos(dl[i],dl.sum(0)-dl[i]) for i in range(len(dl))])
    splitcos=cos(dl[succ_grips=="open"].mean(0),dl[succ_grips=="close"].mean(0))
    rms=np.sqrt(np.mean(x*x))
    score=np.linalg.norm(d1)/np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
    stats.append((score,k,np.linalg.norm(d0)/(rms*np.sqrt(x.shape[-1])),
                  np.linalg.norm(d1)/(rms*np.sqrt(x.shape[-1])),cos(d0,d1),loo,splitcos))
    contrast[k]={"initial":d0,"later":d1,"pair_later":dl}
print("key: normalized separation, initial/late relative magnitude, phase-cos, LOO-cos, success-grip split-cos")
for s in sorted(stats,reverse=True)[:22]:
    print(s[1].replace("mean::","")," ".join(f"{v:.3f}" for v in (s[0],*s[2:])))
print("all expert layers")
ao=load_original("action_out_proj.weight").numpy()
for s in stats:
    if "expert." in s[1]:
        d=contrast[s[1]]["later"]
        print(s[1].split("::")[1], "rel %.3f loo %.3f split %.3f"%(s[3],s[5],s[6]),
              "readout cos xyz/grip",",".join(f"{cos(d,ao[j]):.3f}" for j in [0,1,2,6]),
              "top1 energy",round(float(np.max(d*d)/np.sum(d*d)),3))
acts=np.stack([np.load(r["action"]) for r in res19]).mean(1).reshape(-1,4,7)
for i,lab in enumerate(["success-initial","loop-initial","success-later","loop-later"]):
    a=acts[:,i]
    print(lab,"mean xyz/grip",np.round(a[:,[0,1,2,6]].mean(0),3).tolist(),
          "close fraction",round(float((a[:,6]>0).mean()),3))
np.savez(W/"mlp_contrasts.npz",**{k+"::"+phase:v[phase]
          for k,v in contrast.items() for phase in ["initial","later","pair_later"]})
for z in zs:z.close()
```
