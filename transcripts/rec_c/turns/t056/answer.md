I will summarize the paired action changes and verify attenuation on the first denoising call, before later calls can diverge. That will tell us whether this edit meaningfully affects the proposed pathway before spending a rollout on it.
```python
rp=json.loads((W/"probe_21/results.json").read_text())
p21=defaultdict(dict)
for r in rp:p21[r["idx"]][r["arm"]]=r
u=torch.load(W/"loop_mlp_direction_L7.pt",weights_only=False)["u"].numpy()
summary=[]
for i in sorted(p21):
    a,b=p21[i]["orig"],p21[i]["edit"]
    aa,bb=np.load(a["action"]),np.load(b["action"])
    za,zb=np.load(a["activations"]),np.load(b["activations"])
    ca=za["calls::expert.7.mlp_out::action"][0]
    cb=zb["calls::expert.7.mlp_out::action"][0]
    delta=bb-aa
    summary.append([np.sqrt(np.mean(delta[:,:3]**2)),np.sqrt(np.mean(delta[:,3:6]**2)),
                    np.sqrt(np.mean(delta[:,6]**2)),aa[:,6].mean(),bb[:,6].mean(),
                    ca@u,cb@u])
    za.close();zb.close()
s=np.array(summary)
for sel,lab in [(slice(0,16),"success"),(slice(16,32),"loop")]:
    v=s[sel]
    print(lab,"mean RMS xyz/rotation/grip",np.round(v[:,:3].mean(0),4).tolist(),
          "max xyz",round(float(v[:,0].max()),4),
          "grip sign changes",int(np.sum((v[:,3]>0)!=(v[:,4]>0))))
print("first-call direction ratio median",float(np.median(s[:,6]/s[:,5])))
print("largest changes")
for i in np.argsort(s[:,0])[-8:][::-1]:
    j=i if i<16 else i-16
    print("success" if i<16 else "loop","id",mf20[j]["id"],"xyzRMS",round(float(s[i,0]),4),
          "grip",np.round(s[i,3:5],3).tolist())
```
