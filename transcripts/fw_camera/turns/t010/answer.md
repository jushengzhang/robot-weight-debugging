While the backend starts, I will examine which attention heads have the strongest direct connection to the action outputs. I will also measure query/key anisotropy across all heads, so that the later activation analysis can distinguish a genuinely camera-sensitive reader from a head that merely has large normalization gains.```python
t0=time.time()
head_out=load_original("mot.mixtures.action.head.weight")
print("Action head row norms (7 coordinates):", np.round(head_out.norm(dim=1).numpy(),4).tolist())
direct_readers=[]
for l in range(30):
    out=load_original(nm("action",l,"self_attn.o.weight"))
    direct=(head_out@out).reshape(7,24,128)
    direct_readers.append(direct.square().sum(2).sqrt().T.numpy())
    del out,direct
direct_readers=np.stack(direct_readers)
np.save(W/"direct_action_readers.npy",direct_readers)
print_top_heads("Direct translation readout proxy",np.linalg.norm(direct_readers[:,:,:3],axis=2),10)
print_top_heads("Direct rotation readout proxy",np.linalg.norm(direct_readers[:,:,3:6],axis=2),8)
print("L29 candidate readout strength by action coordinate:")
for h in (4,7,8,17,0):
    print(h,np.round(direct_readers[29,h],5).tolist())

spectral_stats={}
for expert,proj in (("action","q"),("video","k")):
    frac4=[]; frac1=[]; n_eff=[]; gain_frac4=[]
    for l in range(30):
        q=load_original(nm(expert,l,f"self_attn.{proj}.weight")).reshape(24,128,-1)
        eig=torch.linalg.eigvalsh(q@q.transpose(1,2)).clamp_min(0).flip(1)
        frac4.append((eig[:,:4].sum(1)/eig.sum(1)).numpy())
        frac1.append((eig[:,0]/eig.sum(1)).numpy())
        n_eff.append((eig.sum(1).square()/eig.square().sum(1)).numpy())
        g=load_original(nm(expert,l,f"self_attn.norm_{proj}.weight")).reshape(24,128,1)
        qg=q*g
        eg=torch.linalg.eigvalsh(qg@qg.transpose(1,2)).clamp_min(0).flip(1)
        gain_frac4.append((eg[:,:4].sum(1)/eg.sum(1)).numpy())
        del q,eig,qg,eg,g
    for label,data in (("top4",frac4),("top1",frac1),("effective_rank",n_eff),("gained_top4",gain_frac4)):
        spectral_stats[expert+"_"+proj+"_"+label]=np.stack(data)
np.savez(W/"head_spectral_stats.npz",**spectral_stats)
print_top_heads("Action q top-4 spectral energy",spectral_stats["action_q_top4"],15)
print_top_heads("Video k top-4 spectral energy",spectral_stats["video_k_top4"],15)
print("Elapsed seconds:",round(time.time()-t0,1))
```
