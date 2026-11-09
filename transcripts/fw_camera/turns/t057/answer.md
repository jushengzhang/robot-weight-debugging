I will measure each block’s contribution to the final camera-dependent residual and its overlap with the action readout. This should identify behaviorally relevant directions that the earlier raw-attention analysis missed.
```python
r10 = json.loads((W/"probe_10"/"results.json").read_text())
z0 = np.load(r10[0]["activations"])
print("Residual keys/shapes:", [(k,z0[k].shape) for k in z0.files[:3]])
R = np.empty((57,30,10,1024),np.float32)
for i,r in enumerate(r10):
    with np.load(r["activations"]) as z:
        for l in range(30):
            R[i,l] = z[f"calls::action.{l}.resid::action"]
RG = R.reshape(19,3,30,10,1024)
DR = RG[:,1]-RG[:,0]
inc = DR.copy()
inc[:,1:] -= DR[:,:-1]
head = load_original("mot.mixtures.action.head.weight")
_,_,head_basis = torch.linalg.svd(head, full_matrices=False)
V = head_basis.numpy()
summ = []
for l in range(30):
    x = torch.from_numpy(inc[:,l,0])
    _,s,vh = torch.linalg.svd(x,full_matrices=False)
    last = DR[:,-1,0]
    align = np.sum(inc[:,l,0]*last)/np.sum(last**2)
    r = DR[:,l,0]
    read_fraction = np.sum((r@V.T)**2)/np.sum(r**2)
    summ.append([l,float(np.sqrt(np.mean(r*r))),float(np.sqrt(np.mean(x.numpy()**2))),
                 float(align),float(s[0]**2/s.square().sum()),float(read_fraction)])
print("L | total contrast RMS | increment RMS | fraction of final contrast | increment top1 | readout-subspace fraction")
for row in summ:
    print(int(row[0]),*[round(v,4) for v in row[1:]])
xlast = torch.from_numpy(DR[:,-1,0])
_,s,vh = torch.linalg.svd(xlast,full_matrices=False)
print("Final residual contrast singular energy first8:",np.round((s[:8].square()/s.square().sum()).numpy(),4).tolist())
print("Top residual-PC head row-space overlaps:",
      np.round((vh[:8]@head_basis.T).square().sum(1).numpy(),4).tolist())
np.savez(W/"residual_diagnosis.npz", stats=np.array(summ), final_basis=vh.numpy(), final_singular=s.numpy())
ledger = json.loads((W/"pass_ledger.json").read_text())
ledger.append({"pass":10,"type":"probe","edit":"original","inputs":57,"what":"resid","noise_seed":46001})
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
```
