The clean-span projection caused substantial clean and paraphrase drift, so the 30-band readout did not capture enough token-level detail. I will remove it and improve the measurement: use a direction that spreads clean tokens apart, then sweep the selectors across diffusion steps. This should yield many more context measurements per pass without running the model locally.
```python
clear_edits()
for k in range(6):
    ss=torch.einsum("nbd,d->nb",Xt,Vhc[k])
    print("clean PC",k,"all-band quantiles",
          [round(float(z),3) for z in torch.quantile(ss.flatten(),torch.tensor([0.,.01,.1,.5,.9,.99,1.]))])
# A fixed mixture of clean PCs spreads ordinary words, unlike the suffix-specific detector.
u_scan=(Vhc[0]+.7*Vhc[1]+.5*Vhc[2]+.3*Vhc[3])
u_scan=u_scan-(u_scan@Qprop)@Qprop.T
u_scan/=u_scan.norm()
scan_scores=torch.einsum("nbd,d->nb",Xt,u_scan)
print("SCAN SCORE QUANTILES:",[round(float(z),3) for z in torch.quantile(scan_scores.flatten(),torch.tensor([0.,.01,.1,.5,.9,.99,1.]))])
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass12 clean manifold rejected\nGap ratio1.3476, clean drift.0653, paraphrase.1018. The30 prior-detector bands miss useful individual-token directions even though they preserve band means. All edits removed.\nPlan refined acquisition:300 selectors via30 layers x10 diffusion-step-dependent fixed selectors, using a clean-PC direction to separate ordinary words. Measurement only; final candidate must be independently validated.\n")
```
