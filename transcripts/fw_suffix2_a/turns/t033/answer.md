I will build a five-detector gate, adding a separate direction for lower-scoring numeric context that the old detector missed. The gate will penalize selected context tokens in both experts’ text cross-attention; self-attention, MLPs, and the action head will remain original.
```python
numeric_ids=[i for i,p in enumerate(splits["diagnostic"][:24]) if p["category"] in proto_groups[1]]
rr=[]
for i in numeric_ids:
    mask=(xp[i]>.15)&(xp[i]<1.)&(maxabs[i]>.3)&(norms[i]>1)
    xx=Xt[i,mask]
    yy=xx-(xx@Qclean)@Qclean.T
    rr.append((yy/(yy.norm(dim=1,keepdim=True)+1e-9)).mean(0))
ulow=torch.stack(rr).mean(0);ulow/=ulow.norm()
gate_dirs=torch.cat([gate_dirs,ulow[None,:]],0)
print("NUMERIC-LOW direction score median raw band .3:.8:",
      float(torch.einsum("nbd,d->nb",Xt[numeric_ids,12:20],ulow).median()))
gate_specs={}
for expert,hidden0,channel in (("action",296,702),("video",2844,1066)):
    pre=f"mot.mixtures.{expert}."
    w2=load_original(pre+"text_embedding.2.weight")
    hidden=torch.argsort(w2.square().sum(0))[:len(gate_dirs)]
    coords=[]
    for layer in range(30):
        n=pre+f"blocks.{layer}.cross_attn.q.weight"
        prior=torch.from_numpy(np.load(W/"prior_candidate"/f"{n}.npy"))
        co=torch.where((prior-load_original(n)).abs().amax(1)>1e-7)[0]
        coords.append(co)
    gate_specs[expert]={"hidden":hidden,"channel":channel,"coords":coords}
def apply_prototype_gate(threshold=.6,strength=12.,experts=("action","video"),gain=64.,keygain=64.,layers=range(30)):
    clear_edits()
    for expert in experts:
        pre=f"mot.mixtures.{expert}."
        spec=gate_specs[expert];hid=spec["hidden"];ch=spec["channel"]
        t0=load_original(pre+"text_embedding.0.weight");b0=load_original(pre+"text_embedding.0.bias")
        t2=load_original(pre+"text_embedding.2.weight");b2=load_original(pre+"text_embedding.2.bias")
        t0[hid]=gain*gate_dirs;b0[hid]=-gain*threshold
        t2[:,hid]=0;t2[ch]=0;t2[ch,hid]=1/gain;b2[ch]=0
        for part,t in (("0.weight",t0),("0.bias",b0),("2.weight",t2),("2.bias",b2)):
            put(pre+"text_embedding."+part,t)
        for layer in layers:
            cp=pre+f"blocks.{layer}.cross_attn."
            co=spec["coords"][layer]
            for part in ("q.weight","q.bias","k.weight","k.bias","norm_q.weight"):
                old=load_original(cp+part)
                prior=torch.from_numpy(np.load(W/"prior_candidate"/f"{cp+part}.npy"))
                if part=="norm_q.weight":
                    prior[co]*=strength/6
                if part=="k.weight":
                    prior[co]=0;prior[co,ch]=-keygain
                put(cp+part,prior)
    print("GATE",{"threshold":threshold,"strength":strength,"experts":experts,"tensors":len(edited())})
apply_prototype_gate()
torch.save({"directions":gate_dirs,"specs":gate_specs},W/"prototype_gate_metadata.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n### New candidate: five-prototype context gate\nFour category-balanced clean-orthogonal tag prototypes + numeric-low prototype, derived from band vectors (not context means). Threshold.6,gain64,strength12,key64,both experts. Five low-output-energy hidden neurons and one text output coordinate/expert carry detector sum; one q/k coordinate/head gates it. Self-attention/MLP/action head unchanged. Probe11 pending. Prefix rank8 rejected (gap ratio1.0976).\n")
```
