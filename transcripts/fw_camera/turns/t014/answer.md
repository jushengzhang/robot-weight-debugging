The backend is still unavailable. I will finish one useful weight-only check: whether the dominant action-query and video-key subspaces actually align, and how much of their coupling lies in the spatial coordinate bands. I will then allow a short startup interval rather than repeatedly submit rollout requests.```python
t0=time.time()
coupling=[]
for l,h in weight_candidates:
    mats={}
    biases={}
    for ex,p in (("action","q"),("video","k")):
        a=load_original(nm(ex,l,f"self_attn.{p}.weight"))[h*128:(h+1)*128]
        b=load_original(nm(ex,l,f"self_attn.{p}.bias"))[h*128:(h+1)*128]
        g=load_original(nm(ex,l,f"self_attn.norm_{p}.weight"))[h*128:(h+1)*128]
        mats[ex]=a*g[:,None]
        biases[ex]=b*g
    q,k=mats["action"],mats["video"]
    gq=q@q.T
    gk=k@k.T
    _,uq=torch.linalg.eigh(gq)
    _,uk=torch.linalg.eigh(gk)
    overlap=float((uq[:,-4:].T@uk[:,-4:]).square().sum()/4)
    # QR factors give the exact nonzero singular values of Q^T K without materializing it.
    rq=torch.linalg.qr(q.T,mode="r").R
    rk=torch.linalg.qr(k.T,mode="r").R
    sv=torch.linalg.svdvals(rq@rk.T)
    qbias,kbias=biases["action"],biases["video"]
    d={"layer":l,"head":h,
       "qk_top4_subspace_overlap":round(overlap,4),
       "qk_top4_energy":round(float(sv[:4].square().sum()/sv.square().sum()),4),
       "bias_only_logit_proxy":round(float(qbias@kbias/math.sqrt(128)),4),
       "q_bias_key_weight_coupling":round(float((qbias@k).norm()/math.sqrt(128)),4),
       "k_bias_query_weight_coupling":round(float((kbias@q).norm()/math.sqrt(128)),4)}
    coupling.append(d)
    del mats,biases,q,k,gq,gk,uq,uk,rq,rk,sv,qbias,kbias
print("Query/key coupling of leading weight-only candidates:")
for d in coupling[:12]:
    print(d)
(W/"weight_qk_coupling.json").write_text(json.dumps(coupling,indent=2))
plan = {
    "diagnosis": "Matched task/seed camera-light rollouts; first-input probes to avoid trajectory confounding.",
    "localization": "Compare layer shifts and per-head camera attention; candidate list is not fixed by weight norms.",
    "causality": "Reversible localized edits at several doses, plus a matched negative-control edit.",
    "selection": "Use discovery variants only; preserve untouched validation variants.",
    "confirmation": "Paired original/edited camera rollouts, then all six noncamera categories and additional seeds.",
    "forbidden": "No optimization, training, local network forward, schedule changes, or whole-tensor scaling."
}
(W/"analysis_plan.json").write_text(json.dumps(plan,indent=2))
print("Analysis elapsed:",round(time.time()-t0,1),"s; allowing 90 s for backend startup.")
time.sleep(90)
print("Startup interval complete. No weights changed.")
```
