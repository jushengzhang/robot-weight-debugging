No head is an obvious global magnitude outlier. I will also check the observation patch embedding and the spectrum of the strongest coupling blocks. This will help distinguish a narrow feature direction from a broadly important head if the activation probes later implicate one of them.

```python
P = load_original("mot.mixtures.video.patch_embedding.weight").reshape(3072,48,4)
# Orthonormal 2x2 spatial modes; weight analysis only, not a model forward.
H = torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], dtype=torch.float32)/2
Pm = torch.einsum("dcp,mp->dcm", P, H)
energy = Pm.square().sum((0,1))
print("Patch spatial energy fractions [DC,horizontal,vertical,diagonal]:",
      np.round((energy/energy.sum()).numpy(),4).tolist())
s = torch.linalg.svdvals(P.reshape(3072,192))
print("Patch singular values [max,median,min]:",
      [round(float(s.max()),4), round(float(s.median()),4), round(float(s.min()),4)])
print("Patch top 1/4/16/48 energy fractions:",
      [round(float(s[:r].square().sum()/s.square().sum()),4) for r in [1,4,16,48]])
print("Action/video Q,K gain cosine and RMS ratio by representative layer:")
for l in [0,3,6,9,12,15,18,21,24,27,29]:
    vals = []
    for suffix in ["self_attn.norm_q.weight", "self_attn.norm_k.weight"]:
        a = load_original(pname("action",l,suffix))
        v = load_original(pname("video",l,suffix))
        vals.append((round(float(torch.nn.functional.cosine_similarity(a,v,dim=0)),6),
                     round(float(a.norm()/v.norm()),4)))
    print(l, vals)

rank_stats = []
for l in [0,3,6,9,12,15,18,21,24,27,28,29]:
    ss = head_stats[24*l:24*(l+1)]
    hs = sorted(range(24),key=lambda h:ss[h]["video_coupling"],reverse=True)[:2]
    ow = load_original(pname("action",l,"self_attn.o.weight"))
    vw = load_original(pname("video",l,"self_attn.v.weight"))
    for h in hs:
        Oh = ow[:,h*128:(h+1)*128]
        Vh = vw[h*128:(h+1)*128,:]
        # Nonzero spectrum of O_h V_h via reduced QR.
        _, ro = torch.linalg.qr(Oh,mode="reduced")
        _, rv = torch.linalg.qr(Vh.T,mode="reduced")
        sv = torch.linalg.svdvals(ro @ rv.T)
        e = sv.square()
        rank_stats.append((l,h,round(float(e[:1].sum()/e.sum()),3),
                           round(float(e[:8].sum()/e.sum()),3),
                           round(float(e.sum()**2/e.square().sum()),1)))
print("Coupling spectrum (layer,head,top1 fraction,top8 fraction,effective rank):")
print(rank_stats)
np.savez(W/"patch_weight_analysis.npz", spatial_energy=energy.numpy(), singular_values=s.numpy())
del P,Pm,ow,vw,Oh,Vh,ro,rv
```
