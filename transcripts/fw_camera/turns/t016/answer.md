I will broaden the query/key coupling check to every head, rather than only the candidates selected by query rank. That guards against a selection bias in the weight analysis and gives us a complete reference for the eventual activation probes. I will leave the checkpoint unchanged while waiting for the simulator.```python
t0=time.time()
qk_all={"top1":[],"top4":[],"frobenius":[],"raw_top4":[],"bias_logit":[]}
for l in range(30):
    q=load_original(nm("action",l,"self_attn.q.weight")).reshape(24,128,1024)
    k=load_original(nm("video",l,"self_attn.k.weight")).reshape(24,128,3072)
    qg=load_original(nm("action",l,"self_attn.norm_q.weight")).reshape(24,128,1)
    kg=load_original(nm("video",l,"self_attn.norm_k.weight")).reshape(24,128,1)
    qb=load_original(nm("action",l,"self_attn.q.bias")).reshape(24,128)*qg.squeeze(-1)
    kb=load_original(nm("video",l,"self_attn.k.bias")).reshape(24,128)*kg.squeeze(-1)
    rq=torch.linalg.qr((q*qg).transpose(1,2),mode="r").R
    rk=torch.linalg.qr((k*kg).transpose(1,2),mode="r").R
    sv=torch.linalg.svdvals(rq@rk.transpose(1,2))
    qk_all["top1"].append((sv[:,0].square()/sv.square().sum(1)).numpy())
    qk_all["top4"].append((sv[:,:4].square().sum(1)/sv.square().sum(1)).numpy())
    qk_all["frobenius"].append(sv.norm(dim=1).numpy())
    qk_all["bias_logit"].append(((qb*kb).sum(1)/math.sqrt(128)).numpy())
    rq0=torch.linalg.qr(q.transpose(1,2),mode="r").R
    rk0=torch.linalg.qr(k.transpose(1,2),mode="r").R
    sv0=torch.linalg.svdvals(rq0@rk0.transpose(1,2))
    qk_all["raw_top4"].append((sv0[:,:4].square().sum(1)/sv0.square().sum(1)).numpy())
    del q,k,qg,kg,qb,kb,rq,rk,sv,rq0,rk0,sv0
qk_all={k:np.stack(v) for k,v in qk_all.items()}
np.savez(W/"all_head_qk_coupling.npz",**qk_all)
print_top_heads("All-head q/k top-4 fraction after gains",qk_all["top4"],16)
print_top_heads("All-head q/k coupling Frobenius proxy",qk_all["frobenius"],16)
print("L29 (head, raw top4, gained top4, gained Frobenius):")
for h in range(24):
    print(h,round(float(qk_all["raw_top4"][29,h]),4),
          round(float(qk_all["top4"][29,h]),4),
          round(float(qk_all["frobenius"][29,h]),2))
print("Elapsed:",round(time.time()-t0,1),"s. Waiting 180 s for backend startup.")
time.sleep(180)
print("No edits applied. Weight-analysis artifacts saved in WORK.")
```
