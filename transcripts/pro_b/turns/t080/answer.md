The unused action inputs are already almost orthogonal to the real action coordinates, and the large final-layer MLP component is mostly time-dependent rather than scene-dependent. Neither supports an intervention. I will instead check two more specific visual explanations: whether the causally identified heads rely on exceptional positional tokens, and whether object-identity directions are weak in the image-to-language projection.
```python
posname=next(n for n in NAMES if "vision_model.embeddings.position_embedding.weight" in n)
pe=load_original(posname)
pn=pe.norm(dim=1)
top=torch.topk(pn,12)
print("Position embedding median/max",round(float(pn.median()),3),round(float(pn.max()),3))
print("Top positional rows:",[(int(i),round(float(v),2)) for v,i in zip(top.values,top.indices)])
print("Grounding-head peak positional-row norms:")
for j in [1,193,114,72,91,225,241,242,247,251,253,255]:
    print(j,round(float(pn[j]),3),"relative median",round(float(pn[j]/pn.median()),2))
projw=next(n for n in NAMES if "multi_modal_projector.linear.weight" in n)
projb=next(n for n in NAMES if "multi_modal_projector.linear.bias" in n)
M=load_original(projw); b=load_original(projb)
print("Projector shape",tuple(M.shape),"weight norm",round(float(M.norm()),2),"bias norm",round(float(b.norm()),2))
bn=torch.topk(b.abs(),6)
print("Projector bias peaks",[(int(i),round(float(b[i]),2)) for i in bn.indices])
# An object-identity subspace defined solely by existing vocabulary weights.
objects=["alphabet soup","bbq sauce","butter","cream cheese","chocolate pudding",
         "orange juice","milk","ketchup","tomato sauce","salad dressing",
         "wine bottle","black bowl","white bowl","plate","ramekin",
         "white mug","red mug","yellow mug","porcelain mug","cup","moka pot",
         "book","basket","wooden tray","caddy","cabinet","drawer","stove"]
syntax=["pick","up","the","and","place","put","in","on","into","left","right",
        "front","back","behind","between","next","top","bottom","of"]
E=load_original(embname)
def phrase_vec(txt):
    return E[sp.encode(" "+txt)].mean(0)
objvec=torch.stack([phrase_vec(s) for s in objects],dim=1)
protect=torch.stack([phrase_vec(s) for s in syntax]+[E[sp.bos_id()],b,objvec.mean(1)],dim=1)
# Preserve the exceptional residual channels as well.
for d in [674,738,604,50]:
    v=torch.zeros(2048); v[d]=1
    protect=torch.cat([protect,v[:,None]],dim=1)
P=torch.linalg.qr(protect,mode="reduced").Q
C=objvec-objvec.mean(1,keepdim=True)
C=C-P@(P.T@C)
u,s,_=torch.linalg.svd(C,full_matrices=False)
objU=u[:,:16].contiguous()
print("Object contrast rank16 retained energy",round(float(s[:16].square().sum()/s.square().sum()),3))
print("Projector weight energy in object contrast span",round(float((objU.T@M).square().sum()/M.square().sum()),5))
print("Bias overlap norm",round(float((objU.T@b).norm()),7))
torch.save({"U":objU,"objects":objects,"protected":syntax},W/"subspaces/visual_object_identity.pt")
del E
```
