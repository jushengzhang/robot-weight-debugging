A potentially precise repair is to reduce only the excessive query/key gain products in the four anomalous heads, leaving their values and output projections intact. I will calculate how several such caps change the weight-level coupling, without applying them or running the network. This will give us bounded, interpretable doses to test once the backend is available.```python
gain_cap_analysis=[]
for h in (4,7,8,17):
    l=29
    q=load_original(nm("action",l,"self_attn.q.weight"))[h*128:(h+1)*128]
    k=load_original(nm("video",l,"self_attn.k.weight"))[h*128:(h+1)*128]
    qg=load_original(nm("action",l,"self_attn.norm_q.weight"))[h*128:(h+1)*128]
    kg=load_original(nm("video",l,"self_attn.norm_k.weight"))[h*128:(h+1)*128]
    rk=torch.linalg.qr((k*kg[:,None]).T,mode="r").R
    for spatial_only in (False,True):
        for cap in (1.0,2.0,4.0,6.0):
            prod=(qg*kg).abs()
            mask=prod>cap
            if spatial_only:
                mask[:44]=False
            new=qg.clone()
            new[mask]*=cap/prod[mask]
            rq=torch.linalg.qr((q*new[:,None]).T,mode="r").R
            sv=torch.linalg.svdvals(rq@rk.T)
            gain_cap_analysis.append({
                "head":h,"spatial_bands_only":spatial_only,"gain_product_cap":cap,
                "changed_coordinates":int(mask.sum()),
                "relative_gain_delta":round(float((new-qg).norm()/qg.norm()),4),
                "qk_frobenius_proxy":round(float(sv.norm()),2),
                "qk_top4_fraction":round(float(sv[:4].square().sum()/sv.square().sum()),4)
            })
print("Candidate coordinate-cap effects (weights only):")
for x in gain_cap_analysis:
    print(x)
(W/"prospective_gain_cap_metrics.json").write_text(json.dumps(gain_cap_analysis,indent=2))
print("Actual edits:",list(edited()))
print("Waiting a further 600 seconds; an empirical diagnosis is required before editing.")
time.sleep(600)
print("Backend wait completed.")
```
