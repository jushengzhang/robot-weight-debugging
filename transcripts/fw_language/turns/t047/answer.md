The tool exposes action-to-video attention maps, not text-attention maps. I will use them to check whether failed paraphrases redirect visual grounding, while revisiting the cross-attention weight statistics to select a narrowly targeted causal test rather than ablating a broad visual feature.
```python
pr5 = json.loads((W/"probe_5/results.json").read_text())
print("Probe record:", compact_obj(pr5[0],2400))
for fn in ["cross_attention_temperature_screen.json","ov_spectral_screen.json",
           "all_cross_attention_spectra.json","rank_one_template_summary.json"]:
    obj=json.loads((W/fn).read_text())
    print("\n",fn, type(obj).__name__,compact_obj(obj,1200))
print("\nAttention changes (all token mass may be <1 due to action self-attention):")
for l in [10,11,13,15]:
    maps=[]
    for rec in pr5:
        ad=rec.get("attention",{})
        path=ad.get(str(l), ad.get(l)) if isinstance(ad,dict) else None
        if isinstance(path,dict): print("Unexpected map metadata",path); break
        if path: maps.append(np.load(path))
    if len(maps)!=4: print(l,"map lookup failed");continue
    print("Layer",l,"shape",maps[0].shape)
    for pair in range(2):
        a,b=maps[2*pair:2*pair+2]
        am,bm=a.mean(axis=1),b.mean(axis=1)
        rel=np.sum(np.abs(bm-am),axis=-1)/(np.sum((am+bm)*.5,axis=-1)+1e-8)
        hi=np.argsort(rel)[-5:][::-1]
        print(" pair",pair,"top heads",[(int(h),round(float(rel[h]),3),
                    round(float(am[h,:].sum()),3),round(float(bm[h,:].sum()),3)) for h in hi])
print("\nL13 cross-attention per-head norms:")
for e in ["action","video"]:
    pfx=f"mot.mixtures.{e}.blocks.13.cross_attn."
    for s in ["norm_q.weight","norm_k.weight","q.weight","k.weight","v.weight","o.weight"]:
        t=load_original(pfx+s)
        hn=(t.reshape(24,128,-1).square().sum((1,2)).sqrt()
            if s!="o.weight" else t.reshape(t.shape[0],24,128).square().sum((0,2)).sqrt())
        inds=torch.argsort(hn,descending=True)[:5].tolist()
        print(e,s, "median",round(hn.median().item(),3),
              "top", [(h,round(hn[h].item(),3)) for h in inds])
```
