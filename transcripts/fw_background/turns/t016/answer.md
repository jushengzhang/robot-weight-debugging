I will examine the first layer's linear sensitivity to the four spatial modes inside each latent patch. This is not a forward pass and cannot identify semantic background features, but it can test whether any head is unusually sensitive to local high-frequency variation—a possible texture mechanism to check against the eventual probes.

```python
P=load_original("mot.mixtures.video.patch_embedding.weight").reshape(3072,48,4)
H=torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=torch.float32)/2
Pm=torch.einsum("dcp,mp->dcm",P,H).reshape(3072,192)
K=load_original(pname("video",0,"self_attn.k.weight"))
V=load_original(pname("video",0,"self_attn.v.weight"))
O=load_original(pname("action",0,"self_attn.o.weight")).reshape(1024,24,128).permute(1,0,2)
KP=(K@Pm).reshape(24,128,48,4)
VP=(V@Pm).reshape(24,128,48,4)
OVP=torch.bmm(O,VP.reshape(24,128,192)).reshape(24,1024,48,4)
ke=KP.square().sum((1,2)).numpy()
ve=VP.square().sum((1,2)).numpy()
oe=OVP.square().sum((1,2)).numpy()
r=oe[:,1:].sum(1)/np.maximum(oe[:,0],1e-12)
print("First-layer head, coupled HF/DC, K HF/DC, V HF/DC")
for h in np.argsort(-r)[:12]:
    print(int(h),round(float(r[h]),3),
          round(float(ke[h,1:].sum()/ke[h,0]),3),round(float(ve[h,1:].sum()/ve[h,0]),3))
print("Coupled HF/DC range:",np.round([r.min(),np.median(r),r.max()],3).tolist())
np.savez(W/"first_layer_spatial_mode_sensitivity.npz",
         key_energy=ke,value_energy=ve,action_read_energy=oe)
del P,Pm,K,V,O,KP,VP,OVP

# Build all future practice selections without inspecting any simulation outcomes.
validation_manifests={}
for ci,cat in enumerate(sorted(bycat)):
    pool=[v for v in bycat[cat] if v["_split"]=="validation"]
    rr=random.Random(31000+ci)
    rr.shuffle(pool)
    validation_manifests[cat]=[
        [{"suite":v["suite"],"id":v["id"],"seed":901+b} for v in pool[128*b:128*(b+1)]]
        for b in range(2)]
(W/"validation_manifests.json").write_text(json.dumps(validation_manifests))
print("Reserved validation batches:",{k:[len(x) for x in bs] for k,bs in validation_manifests.items()})
note("- Preselected two 128-variant disjoint validation batches per category in validation_manifests.json, seeds 901/902. "
     "No outcomes used to make these selections.")
```
