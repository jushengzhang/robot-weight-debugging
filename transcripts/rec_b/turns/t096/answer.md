I will construct the visual-change directions by subtracting matched-task camera averages, removing the observed language and camera-common span, and taking four principal directions. This is an analysis of activations, not a fit to outcomes. I will then measure each value projection’s gain on that subspace before choosing a causal test.
```python
r37=json.loads((W/"probe_37"/"results.json").read_text())
z37=[np.load(local_path(r["activations"])) for r in r37]
content_bases={}
gain_rows=[]
for layer in range(1,18):
    prev=layer-1
    groups={}
    scale=1+load_original(LP+f"layers.{layer}.input_layernorm.weight")
    for cam in ["base","wrist","lang"]:
        x=torch.tensor(np.stack([z[f"mean::vlm.{prev}.resid::{cam}"] for z in z37]))
        # Normalize recorded group means only for comparable direction statistics.
        x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-8)
        groups[cam]=x*scale
    nuisance=torch.cat([groups["lang"],
                         groups["base"].mean(0,keepdim=True),
                         groups["wrist"].mean(0,keepdim=True)],0)
    _,sn,vn=torch.linalg.svd(nuisance,full_matrices=False)
    bn=vn[sn > sn.max()*1e-5].T
    diff=torch.cat([groups[c][0::2]-groups[c][1::2] for c in ["base","wrist"]],0)
    clean=diff-(diff@bn)@bn.T
    _,sd,vd=torch.linalg.svd(clean,full_matrices=False)
    b=vd[:4].T.contiguous()
    v=load_original(LP+f"layers.{layer}.self_attn.v_proj.weight")
    k=load_original(LP+f"layers.{layer}.self_attn.k_proj.weight")
    vg=float((v@b).norm()/2/(v.norm()/math.sqrt(2048)))
    kg=float((k@b).norm()/2/(k.norm()/math.sqrt(2048)))
    captured=float(sd[:4].square().sum()/sd.square().sum())
    remaining=float(clean.norm()/diff.norm())
    content_bases[layer]=b
    gain_rows.append((layer,round(vg,3),round(kg,3),round(captured,3),round(remaining,3)))
print("L | V_gain/isotropic | K_gain/isotropic | top4_variance | retained_after_nuisance")
for row in gain_rows:
    print(row)

print("\nVision shared-mean fraction and paired-change RMS:")
for layer in [0,5,10,15,20,26]:
    for cam in ["base","wrist"]:
        x=np.stack([z[f"mean::vision.{layer}.resid::{cam}"] for z in z37])
        fraction=float(np.sum(x.mean(0)**2)/np.mean(np.sum(x*x,axis=1)))
        change=float(np.sqrt(np.mean((x[0::2]-x[1::2])**2)))
        print(layer,cam,round(fraction,4),round(change,3))
torch.save(content_bases,W/"visual_change_bases.pt")
(W/"visual_projection_gains.json").write_text(json.dumps(gain_rows))
```
