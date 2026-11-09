The measured tag directions have a large margin over clean context, so a small bank may suffice. I will form four category-balanced prototype directions—background, numeric/view-state, light, and layout—and check their coverage. This should disturb fewer text features than a large novelty detector.
```python
proto_groups=[["Background Textures"],["Camera Viewpoints","Robot Initial States","Sensor Noise"],
              ["Light Conditions"],["Objects Layout"]]
proto=[]
for group in proto_groups:
    per=[]
    for i,p in enumerate(splits["diagnostic"][:24]):
        if p["category"] not in group:continue
        mask=(maxabs[i]>.7)&(norms[i]>1)
        xx=Xt[i,mask]
        rr=xx-(xx@Qclean)@Qclean.T
        rr=rr/(rr.norm(dim=1,keepdim=True)+1e-9)
        per.append(rr.mean(0))
    u=torch.stack(per).mean(0)
    u=u/u.norm()
    proto.append(u)
gate_dirs=torch.stack(proto)
ps=torch.einsum("nbd,kd->nbk",Xt,gate_dirs)
print("PROTOTYPE COSINES:",np.round((gate_dirs@gate_dirs.T).numpy(),3).tolist())
for cat in tagcats:
    ids=[i for i,p in enumerate(splits["diagnostic"][:24]) if p["category"]==cat]
    cov=ps[ids].amax(-1)
    print(cat,"max score low/mid/high median",
          [round(float(cov[:,a:b].median()),3) for a,b in ((0,12),(12,18),(18,30))],
          "high min",round(float(cov[:,18:].min()),3))
print("clean/lang max prototype score:",float(ps[24:].max()))
torch.save({"directions":gate_dirs,"groups":proto_groups},W/"prototype_gate_geometry.pt")
```
