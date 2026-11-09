The largest late-layer gains are concentrated in a few channels rather than uniformly across entire heads. I will locate those channels and compare the action/video normalization profiles. If observation probes later implicate them, a selective channel or head edit could preserve far more of the policy than broad attention damping.```python
gain_details = {}
for l in (0,5,12,18,24,28,29):
    gain_details[l] = {}
    for expert in ("action","video"):
        for qk in ("q","k"):
            g=load_original(nm(expert,l,f"self_attn.norm_{qk}.weight")).reshape(24,128)
            gain_details[l][expert+"_"+qk] = g.numpy().copy()
    ga = gain_details[l]["action_q"]
    gv = gain_details[l]["video_k"]
    print("L%02d: q/k channel corr %.4f, q abs>3=%d, k abs>3=%d" % (
        l, np.corrcoef(ga.ravel(),gv.ravel())[0,1], (np.abs(ga)>3).sum(), (np.abs(gv)>3).sum()))
    if l == 29:
        for h in (4,7,8,17,0):
            inds=np.argsort(np.abs(ga[h]))[-10:][::-1]
            print(" H%02d top coordinate:(q,k)"%h,
                  [(int(i), round(float(ga[h,i]),3),round(float(gv[h,i]),3)) for i in inds])
print("\nWeight directions in candidate heads (singular concentration, no model execution):")
candidate_spectral = []
for l,h in [(29,4),(29,7),(29,8),(29,17),(29,0),(18,10),(5,4),(12,21)]:
    vals = {"layer":l, "head":h}
    for expert, proj in (("action","q"),("video","k"),("video","v")):
        wt=load_original(nm(expert,l,f"self_attn.{proj}.weight"))[128*h:128*(h+1)].contiguous()
        gram=wt@wt.T
        eig=torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
        vals[expert+"_"+proj+"_top4_fraction"]=round(float(eig[:4].sum()/eig.sum()),4)
        vals[expert+"_"+proj+"_condition_top_to_median"]=round(float((eig[0]/eig[64]).sqrt()),3)
        del wt,gram,eig
    candidate_spectral.append(vals)
    print(vals)
(W/"weight_candidates.json").write_text(json.dumps(candidate_spectral,indent=2))
# A pass ledger is maintained separately so failed/offline requests are not counted.
ledger = []
(W/"pass_ledger.json").write_text(json.dumps(ledger,indent=2))
print("Current edits:", list(edited()))
```
