The contact-cue gain produced only 9 wins versus 7 losses and did not reduce loop failures, so I will not retain it as a repair. I will check a narrower intervention: whether the same contact distinction survives in the late VLM features and could selectively influence a clearance command through the final prefix-value connection, without changing the visual encoder.
```python
for name in list(edited()): drop_edit(name)
with (W/"notes.md").open("a") as f:
    f.write("\nPass34 contact-projector gain2:72 vs70/126,9 wins/7 losses (+1.6+/-3.2pp), "
            "loops46 vs46, L2+6,L3-2,L4-2. Removed; no demonstrated recovery gain.\n")
# Analyze saved VLM features only; this is not a model forward.
p15=json.loads((W/"probe_15"/"results.json").read_text())
for L in [2,8,15]:
    gamma=1+load_original(f"{VLM}.layers.{min(L+1,17)}.input_layernorm.weight").clone()
    def analysis_repr(v):
        x=torch.as_tensor(v).float()
        return (x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt())*gamma
    X=torch.stack([analysis_repr(np.load(p32[i]["activations"])[f"mean::vlm.{L}.resid::wrist"])
                   for i in range(44)])
    I=torch.stack([analysis_repr(np.load(r["activations"])[f"mean::vlm.{L}.resid::wrist"])
                   for r in p15])
    # Initial observations in probe15 alternate open, close.
    aperture=(I[1::2]-I[0::2]).mean(0)
    E=X[0::2]; C=X[1::2]
    sink=torch.zeros(2048); sink[674]=1
    nuisance=torch.stack([aperture,C[source_mask].mean(0),sink],1)
    Q,_=torch.linalg.qr(nuisance)
    d=(E[source_mask]-C[source_mask]).mean(0)
    d=d-Q@(Q.T@d); d=d/d.norm()
    pe,pc=E@d,C@d
    dp=pe-pc
    print("L",L,"source/check paired positive",int((dp[source_mask]>0).sum()),int((dp[~source_mask]>0).sum()),
          "means E,C source",round(float(pe[source_mask].mean()),3),round(float(pc[source_mask].mean()),3),
          "check",round(float(pe[~source_mask].mean()),3),round(float(pc[~source_mask].mean()),3),
          "check paired mean/std",round(float(dp[~source_mask].mean()),3),round(float(dp[~source_mask].std()),3),
          "initial mean/std",round(float((I@d).mean()),3),round(float((I@d).std()),3))
```
