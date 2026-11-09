The revised candidate also failed safety, especially on paraphrases. I’ll now turn the detector off while leaving its wiring in place. This ablation will distinguish false tag detections from damage caused by repurposing text features and attention coordinates.
```python
s31 = paired_rollout(31)
tb = load(V+"text_embedding.0.bias").clone()
tb[hidden] = -12800.
save_edit(V+"text_embedding.0.bias",tb)
print("Pass31:",s31)
print("Detector disabled; other candidate weights unchanged.")
print("Original query-bias signal-to-variation ratios, by layer:")
for i in [0,10,29]:
    p=V+f"blocks.{i}."
    qw=load_original(p+"cross_attn.q.weight").double()
    qb=load_original(p+"cross_attn.q.bias").double()
    ng=load_original(p+"norm3.weight").double()
    nb=load_original(p+"norm3.bias").double()
    A=qw*ng[None,:]
    A-=A.mean(dim=1,keepdim=True)
    c=qw@nb+qb
    snrs=[]
    for h in range(24):
        ah=A[128*h:128*(h+1)]
        ch=c[128*h:128*(h+1)]
        cov=ah@ah.T
        reg=1.e-5*torch.trace(cov)/128
        sol=torch.linalg.solve(cov+reg*torch.eye(128,dtype=torch.float64),ch)
        snrs.append(float(torch.sqrt(torch.clamp(ch@sol,min=0))))
    print(i, [round(float(t),3) for t in np.quantile(snrs,[0,.25,.5,.75,1])])
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass31 null-carrier+payload safety FAILED\n"+json.dumps(s31)+
            "\nClean59->57/80; language31->26/48. Detector disabled for causal safety ablation "
            "(only6 text-entry biases changed to-12800); must restore before repair evaluation.\n")
```
