The early content-head edit also failed to improve success, so I will remove it. The repeated negative tests argue against simple attention-head gain as the repair. I will now examine the visual projector and positional embeddings for a more upstream source of scene or fixed-location bias, while keeping geometric information distinct from scene-wide features.
```python
clear_edits()
r10=read_jsonl(W/"roll_10.jsonl")
report_pairs(r10)
proj_name="paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
P=load_original(proj_name)
uP,sP,vhP=torch.linalg.svd(P,full_matrices=False)
print("Projector singular values, first16:",rounded(sP[:16],3))
print("Top1/4/16 energy fractions:",[round(float(sP[:k].square().sum()/sP.square().sum()),4) for k in [1,4,16]])
prn=P.norm(dim=1)
print("Projector output row norm median",round(float(prn.median()),3),
      "top",[(int(i),round(float(prn[i]),3)) for i in prn.topk(10).indices])
print("Top left singular vector channels:",
      [[(int(i),round(float(uP[i,j]),3)) for i in uP[:,j].abs().topk(6).indices] for j in range(3)])
# Orthonormal 2D DCT decomposition is an analytic basis projection, not fitted weights.
n=16
coord=torch.arange(n,dtype=torch.float64)
freq=torch.arange(n,dtype=torch.float64)
C=torch.cos(math.pi/n*(coord[None,:]+0.5)*freq[:,None])
C[0]*=1/math.sqrt(n); C[1:]*=math.sqrt(2/n)
pe=load_original(pe_name).double().reshape(16,16,1152)
coef=torch.einsum("ai,ijc,bj->abc",C,pe,C)
power=coef.square().sum(2)
pn=pe.reshape(256,1152).norm(dim=1)
print("Position embedding row norm min/median/max:",rounded(torch.stack([pn.min(),pn.median(),pn.max()]),3))
print("DCT cumulative lowfrequency square energy fractions:",
      {str(k):round(float(power[:k,:k].sum()/power.sum()),4) for k in [1,2,3,4,6,8,12,16]})
print("Top DCT modes:",[(int(i)//16,int(i)%16,round(float(power.flatten()[i]/power.sum()),4))
                         for i in power.flatten().topk(8).indices])
# Late vision pooled vectors: common vs scene-varying energy.
Xv=[]
for r in r1:
    a=np.load(r["activations"])
    Xv.append(a["mean::vision.26.resid::base"])
Xv=torch.tensor(np.stack(Xv),dtype=torch.float32)
_,sv,vhv=torch.linalg.svd(Xv,full_matrices=False)
_,svc,vhvc=torch.linalg.svd(Xv-Xv.mean(0),full_matrices=False)
print("Vision pooled uncentered top4 energy",round(float(sv[:4].square().sum()/sv.square().sum()),3),
      "centered top4 energy",round(float(svc[:4].square().sum()/svc.square().sum()),3))
print("Projector top singular direction alignment with pooled-vision PCs:",
      rounded((vhP[:4]@vhv[:4].T).abs().flatten(),3))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 10: candidate D, L4H3/H7 output x1.5\n"
            "dev128 69->68; paired -0.8 +/-1.8pp SE; task/position unchanged; "
            "environment17->15, semantic21->22, object unchanged. REJECTED.\n"
            "Four head-gain variants gave no meaningful target gains. Pivot to upstream visual scene/location encoding.\n")
```
