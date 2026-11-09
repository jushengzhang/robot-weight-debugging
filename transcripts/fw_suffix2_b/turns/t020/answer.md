The readouts are diffusion-step invariant, which confirms the isolation worked. Their small norms suggest the initial queries are still averaging many tokens, so I’ll sharpen the fixed queries and record their scores as a numerical check. This should reveal more token-specific variation than the earlier mean-only analysis.```python
# Sharpen fixed queries; add score coordinates to the isolated readout.
value_dim = read_dim + 14
temperatures = [128. if i < 7 else 512. for i in range(14)]
instrument_meta.update(temperatures=temperatures, mask_strength=4096., gain=1024.,
                       value_dim=value_dim)
torch.save(instrument_meta, W/"readout_instrument_meta.pt")
for suffix in ["weight","bias"]:
    n = f"mot.mixtures.action.action_encoder.{suffix}"
    t = load(n); t[:value_dim] = 0; save_edit(n,t)
for layer,(direction,masked) in enumerate(query_specs):
    base = f"mot.mixtures.action.blocks.{layer}."
    for module in ["self_attn.o","ffn.2"]:
        for suffix in ["weight","bias"]:
            n = base+module+"."+suffix
            t = load(n); t[:value_dim] = 0; save_edit(n,t)
    p = base+"cross_attn."
    kw = load(p+"k.weight"); kw[coord]=0
    if direction is not None:
        kw[coord,score_start+direction] = temperatures[direction]
    if masked:
        kw[coord,det_col] = -4096.
    save_edit(p+"k.weight",kw)
    vw=load(p+"v.weight"); vb=load(p+"v.bias")
    vw[:value_dim]=0; vw[torch.arange(value_dim),torch.arange(value_dim)]=1.
    vb[:value_dim]=0
    ow=load(p+"o.weight"); ob=load(p+"o.bias")
    ow[:value_dim]=0
    ow[torch.arange(value_dim),torch.arange(value_dim)]=1. if layer%2==0 else -1.
    ob[:value_dim]=0
    for nm,t in [("v.weight",vw),("v.bias",vb),("o.weight",ow),("o.bias",ob)]:
        save_edit(p+nm,t)
def set_readout_slice(start):
    dim = min(read_dim,4096-start)
    P = torch.zeros(read_dim,4096)
    P[:dim,start:start+dim] = torch.eye(dim)
    P -= (P@Qprop)@Qprop.T
    linear = torch.cat([P,dirs],0)
    p = "mot.mixtures.action.text_embedding."
    w0=torch.zeros_like(load_original(p+"0.weight"))
    b0=torch.zeros_like(load_original(p+"0.bias"))
    w2=torch.zeros_like(load_original(p+"2.weight"))
    b2=load_original(p+"2.bias")
    nf=linear.shape[0]
    w0[:nf]=linear; w0[nf:2*nf]=-linear
    w2[torch.arange(nf),torch.arange(nf)]=1.
    w2[torch.arange(nf),torch.arange(nf)+nf]=-1.
    w0[2*nf]=1024*u; b0[2*nf]=-512.
    w2[det_col,2*nf]=1/1024
    b2[:det_col+1]=0
    for nm,t in [("0.weight",w0),("0.bias",b0),("2.weight",w2),("2.bias",b2)]:
        save_edit(p+nm,t)
    return dim
def decode_readout(k,start):
    vals, scores, time_err = [], [], []
    for rec in probe_records(k):
        z=np.load(rec["activations"])
        per_layer=np.stack([z[f"calls::action.{i}.resid::action"][:,:value_dim]
                            for i in range(30)],0)
        dif=np.diff(np.concatenate([np.zeros_like(per_layer[:1]),per_layer],0),axis=0)
        dif[1::2]*=-1
        vals.append(dif.mean(1)[:,:read_dim])
        scores.append(dif.mean(1)[:,read_dim:value_dim])
        time_err.append(rms(dif-dif.mean(1,keepdims=True)))
    out=np.stack(vals); sc=np.stack(scores)
    np.save(readout_dir/f"sharp_slice_{start:04d}.npy",out)
    np.save(readout_dir/f"sharp_scores_{start:04d}.npy",sc)
    print("Pass",k,"slice",start,"context RMS",round(rms(out),6),
          "step drift max",max(time_err),"score RMS",round(rms(sc),6))
    return out,sc
set_readout_slice(0)
print("Sharpened readout slice0:480 ready.")
with (W/"notes.md").open("a") as f:
    f.write("\nPass7 pilot readout is exactly step-invariant but queries too diffuse. "
            "Final query bank sharpened to temperatures128/512 with high-confidence mask, "
            "and14 query-score coordinates also isolated for cross-slice consistency checks. "
            "Restart9-slice acquisition at pass8; first pilot not combined with final slices.\n")
```
