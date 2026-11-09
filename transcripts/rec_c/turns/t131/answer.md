The saved centering coefficients are signed additions, so my subtraction convention was wrong. I will verify the sign directly from the saved weight delta and recompute the temporal summaries before drawing conclusions.
```python
dplus = sum(torch.outer(p, cd["feature"] + cd["betas"][h]*cd["reference"])
            for h,p in parts.items())
dminus = sum(torch.outer(p, cd["feature"] - cd["betas"][h]*cd["reference"])
             for h,p in parts.items())
print("Design reconstruction errors:",
      {"plus":float((dplus-cd["delta"]).norm()/cd["delta"].norm()),
       "minus":float((dminus-cd["delta"]).norm()/cd["delta"].norm())})
print("S41 first-call vs saved scores:",
      float((S41[:,0]-designs["clearance_attention_scores.pt"]["scores"]).norm()))
for group in ["source_empty","source_contact","check_empty","check_contact","control_open_control"]:
    inds = [i for i,r in enumerate(m) if group_key(r)==group]
    print("\n"+group)
    for h in [1,7]:
        centered = S41[inds,:,h] + cd["betas"][h]*S42[inds,:,h]
        print("head",h, [round(float(x),3) for x in centered.mean(0)])
```
