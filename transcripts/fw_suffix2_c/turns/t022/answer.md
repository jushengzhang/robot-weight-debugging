The action-only ablation leaves most of the tag gap intact, confirming that the video text path is the larger contributor. I will now use temporary weight-only instrumentation to read the video text MLP's feature statistics: an unweighted context mean, a mean with high-detector tokens suppressed, and a mean emphasizing those tokens. This can distinguish suffix-token features from changes that remain in the instruction tokens; all instrumentation will be removed before testing a repair.
```python
# Temporary measurement circuit. It uses the forward tool, not local network execution.
# Layer-0/1/2 action residual differences expose three context-weighted feature means.
clear_edits()
vd="mot.mixtures.video."
ad="mot.mixtures.action."
v_w1=load_original(vd+"text_embedding.0.weight")
v_b1=load_original(vd+"text_embedding.0.bias")
v_w2=load_original(vd+"text_embedding.2.weight")
feature_cost=v_w2.square().sum(0)*v_w1.square().sum(1)
excluded_v=feature_cost.argsort()[:3].tolist()
feature_ids=torch.tensor([i for i in range(3072) if i not in excluded_v])
slices_v=feature_ids.reshape(3,1023)
det_u=gate_meta["action"]["detector"].clone()
instrument_names=[]
def put(n,t):
    save_edit(n,t.contiguous())
    instrument_names.append(n)
def instrument_features(slice_index, expert="video"):
    clear_edits()
    instrument_names.clear()
    ids=slices_v[slice_index]
    w=torch.zeros_like(load_original(ad+"text_embedding.0.weight"))
    b=torch.zeros_like(load_original(ad+"text_embedding.0.bias"))
    w[:1023]=v_w1[ids]; b[:1023]=v_b1[ids]
    w[1023]=32*det_u; b[1023]=-32.0  # conservative prior token detector
    put(ad+"text_embedding.0.weight",w); put(ad+"text_embedding.0.bias",b)
    t2=torch.eye(1024); t2[1023,1023]=1/32
    put(ad+"text_embedding.2.weight",t2)
    bb=torch.zeros(1024); bb[1023]=1e-9
    put(ad+"text_embedding.2.bias",bb)
    for suff in ("weight","bias"):
        n=ad+"action_encoder."+suff
        t=load_original(n); t[:1023]=0
        put(n,t)
    coords=torch.arange(24)*128
    for l in range(3):
        base=ad+f"blocks.{l}."
        for module in ("self_attn.o","ffn.2"):
            for suff in ("weight","bias"):
                n=base+module+"."+suff
                t=load_original(n); t[:1023]=0
                put(n,t)
        cw=load_original(base+"cross_attn.q.weight")
        cb=load_original(base+"cross_attn.q.bias")
        cw[coords]=0; cb[coords]=100
        put(base+"cross_attn.q.weight",cw); put(base+"cross_attn.q.bias",cb)
        nq=torch.zeros(3072); nq[coords]=[1.0,-128.0,32.0][l]
        put(base+"cross_attn.norm_q.weight",nq)
        kw=torch.zeros(3072,1024)
        kb=torch.ones(3072)
        if l==0:
            kw[-1]=load_original(base+"cross_attn.k.weight")[-1]
        else:
            kw[coords,1023]=1.0
        kb[coords]=0
        put(base+"cross_attn.k.weight",kw); put(base+"cross_attn.k.bias",kb)
        put(base+"cross_attn.norm_k.weight",torch.ones(3072))
        vw=torch.zeros(3072,1024); vw[:1024]=torch.eye(1024)
        vb=torch.zeros(3072); vb[-1]=1
        put(base+"cross_attn.v.weight",vw); put(base+"cross_attn.v.bias",vb)
        ow=load_original(base+"cross_attn.o.weight"); ow[:1023]=0
        ow[torch.arange(1023),torch.arange(1023)]=1
        ob=load_original(base+"cross_attn.o.bias"); ob[:1023]=0
        put(base+"cross_attn.o.weight",ow); put(base+"cross_attn.o.bias",ob)
    print("INSTRUMENT SLICE",slice_index,"original video hidden features",ids.min().item(),ids.max().item(),
          "edits",len(edited()),"omitted low-energy features",excluded_v)
instrument_features(0)
with (W/"notes.md").open("a") as f:
    f.write("\nCausal localization: video text ablation shrank raw-clean gap .06163->.02500 but clean drift.1862; "
            "action text ablation left gap.05477, clean drift.05406. Neither retained.\n"
            "TEMPORARY INSTRUMENTATION for passes8-10: transplant disjoint1023-feature slices of original video text GELU into action text MLP; "
            "three controlled cross-attention layers read uniform / prior-detector-masked / detector-emphasized means. "
            "Original video MLP features3069/3072; exclude3 lowest weight-energy features. Not a candidate, remove all diagnostic edits after acquisition.\n")
```
